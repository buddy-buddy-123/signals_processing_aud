#!/usr/bin/env python3
"""
API REQUEST PARALLEL PROCESSOR — AUDIO TRANSCRIPTIONS

This script parallelizes requests to the OpenAI Audio Transcriptions endpoint
while throttling to stay under rate limits and retrying on transient errors.

Key differences vs. the text/embeddings version:
- Uses multipart/form-data with a file upload (`file_path` in each request item)
- Targets the /v1/audio/transcriptions endpoint by default
- Token throttling is not applied (audio endpoints are minute-based billing);
  we only throttle by requests/minute. You can still set a tokens budget, but
  it's ignored for audio jobs.
- Saves each result alongside its original request payload

Example usage:
    python api_request_parallel_processor_audio.py \
      --requests_filepath ./example_audio_requests.jsonl \
      --save_filepath ./example_audio_results.jsonl \
      --request_url https://api.openai.com/v1/audio/transcriptions \
      --max_requests_per_minute 60 \
      --max_attempts 5 \
      --logging_level 20
"""

import aiohttp
import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, AsyncGenerator, List

DEFAULT_REQUEST_URL = "https://api.openai.com/v1/audio/transcriptions"

# -------------------------
# Utility & data structures
# -------------------------

def api_endpoint_from_url(request_url: str) -> str:
    # Normalize known endpoints
    lowered = request_url.lower().rstrip("/")
    if lowered.endswith("/audio/transcriptions"):
        return "audio_transcriptions"
    return "unknown"

def append_to_jsonl(filepath: str, obj: Any):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0

@dataclass
class APIRequest:
    task_id: int
    request_json: Dict[str, Any]
    attempts_left: int
    input_audio_path: Optional[str] = field(default=None)
    result: Optional[Dict[str, Any]] = field(default=None)
    error: Optional[Any] = field(default=None)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: Dict[str, str],
        status: StatusTracker,
    ):
        """Calls the OpenAI API for audio transcription and stores the result or error."""
        logging.info(f"Starting request #{self.task_id}")
        self.error = None

        try:
            endpoint = api_endpoint_from_url(request_url)
            if endpoint != "audio_transcriptions":
                raise NotImplementedError(f"Unsupported endpoint: {request_url}")

            # Expect "file_path" in request_json; the rest of keys are sent as fields.
            file_path = self.request_json.get("file_path")
            model = self.request_json.get("model")
            if not file_path or not model:
                raise ValueError("Each request must include 'model' and 'file_path'.")

            form = aiohttp.FormData()
            form.add_field("model", str(model))

            # Optional fields commonly used with transcriptions
            for k in ["prompt", "response_format", "temperature", "language"]:
                if k in self.request_json and self.request_json[k] is not None:
                    form.add_field(k, str(self.request_json[k]))

            # Attach the audio file
            try:
                f = open(file_path, "rb")
            except Exception as e:
                raise FileNotFoundError(f"Could not open audio file at '{file_path}': {e}")

            form.add_field("file", f, filename=os.path.basename(file_path))

            async with session.post(request_url, headers=request_header, data=form) as resp:
                # The transcriptions endpoint returns JSON by default
                response = await resp.json(content_type=None)

            # Clean up file handle
            try:
                f.close()
            except Exception:
                pass

            if isinstance(response, dict) and "error" in response:
                msg = response["error"]
                logging.warning(f"Request {self.task_id} failed with error {msg}")
                status.num_api_errors += 1
                self.error = response
                # crude rate-limit detection
                if "rate limit" in str(msg).lower():
                    status.time_of_last_rate_limit_error = time.time()
                    status.num_rate_limit_errors += 1
                    status.num_api_errors -= 1  # counted separately
            else:
                # success
                self.result = response
                status.num_tasks_succeeded += 1
                logging.info(f"Request #{self.task_id} succeeded")

        except Exception as e:
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status.num_other_errors += 1
            self.error = {"exception": str(e)}

# -------------------------
# Streaming requests
# -------------------------

async def request_generator(filepath: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Yields dicts parsed from a JSONL file (one JSON object per line)."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logging.error(f"Skipping malformed JSONL line: {e}")
                continue
            yield obj

def task_id_generator():
    i = 0
    while True:
        yield i
        i += 1

# -------------------------
# Main loop
# -------------------------

async def process_requests(
    requests_filepath: str,
    save_filepath: Optional[str],
    request_url: str,
    api_key: Optional[str],
    max_requests_per_minute: float,
    max_attempts: int,
    logging_level: int,
):
    logging.getLogger().setLevel(logging_level)

    if not save_filepath:
        base = os.path.splitext(os.path.basename(requests_filepath))[0]
        save_filepath = os.path.join(os.path.dirname(requests_filepath), base + "_results.jsonl")

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No API key provided. Use --api_key or set OPENAI_API_KEY env var.")

    status = StatusTracker()
    pending: List[APIRequest] = []

    # capacity state
    available_request_capacity = max_requests_per_minute
    last_update_time = time.time()

    # HTTP session
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=600)

    gen = request_generator(requests_filepath)
    get_next = True
    file_not_finished = True
    id_gen = task_id_generator()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            # fetch next request if we're ready and file isn't exhausted
            if get_next and file_not_finished:
                try:
                    req_json = await gen.__anext__()
                    task = APIRequest(
                        task_id=next(id_gen),
                        request_json=req_json,
                        attempts_left=max_attempts,
                        input_audio_path=req_json.get("file_path"),
                    )
                    pending.append(task)
                    status.num_tasks_started += 1
                    get_next = False  # only pull one per loop to interleave capacity updates
                except StopAsyncIteration:
                    file_not_finished = False

            # update capacity bucket
            now = time.time()
            elapsed = now - last_update_time
            available_request_capacity = min(
                available_request_capacity + max_requests_per_minute * elapsed / 60.0,
                max_requests_per_minute,
            )
            last_update_time = now

            # if we have pending task and capacity, fire it
            if pending and available_request_capacity >= 1:
                task = pending.pop(0)
                available_request_capacity -= 1

                await task.call_api(session, request_url, headers, status)

                # Save result (success or error)
                record = {"request": task.request_json, "response": (task.result or task.error)}
                append_to_jsonl(save_filepath, record)

                # Decide whether to retry
                if task.error and task.attempts_left > 1:
                    task.attempts_left -= 1
                    # Backoff on rate limits
                    if status.time_of_last_rate_limit_error and (now - status.time_of_last_rate_limit_error) < 15:
                        await asyncio.sleep(10)
                    pending.append(task)
                elif task.error:
                    status.num_tasks_failed += 1

                # allow next request to be pulled
                get_next = True

            # Break condition: no more work
            if not pending and not file_not_finished:
                break

            # Gentle pacing to avoid a tight loop
            await asyncio.sleep(0.01)

    logging.info(
        f"Done. Started: {status.num_tasks_started}, "
        f"Succeeded: {status.num_tasks_succeeded}, Failed: {status.num_tasks_failed}, "
        f"API errors: {status.num_api_errors}, Rate-limit errors: {status.num_rate_limit_errors}, Other errors: {status.num_other_errors}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath", type=str, required=True, help="Path to JSONL of transcription jobs (one JSON object per line). Each must include 'model' and 'file_path'.")
    parser.add_argument("--save_filepath", type=str, default=None, help="Where to write results JSONL.")
    parser.add_argument("--request_url", type=str, default=DEFAULT_REQUEST_URL, help="API endpoint URL (defaults to /v1/audio/transcriptions).")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key; if omitted, reads from OPENAI_API_KEY.")
    parser.add_argument("--max_requests_per_minute", type=float, default=60.0, help="Target requests per minute.")
    parser.add_argument("--max_attempts", type=int, default=5, help="Retries per request.")
    parser.add_argument("--logging_level", type=int, default=20, help="Python logging level (e.g., 10=DEBUG, 20=INFO).")
    args = parser.parse_args()

    asyncio.run(
        process_requests(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            max_requests_per_minute=args.max_requests_per_minute,
            max_attempts=args.max_attempts,
            logging_level=args.logging_level,
        )
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure FreeToken scheduler times and local streaming latency on an idle server."""

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


def stream_events(response):
    if response.headers.get_content_type() != "text/event-stream":
        raise ValueError("Expected a streaming chat response.")
    lines = []
    for wire in response:
        line = wire.decode().rstrip("\r\n")
        if line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
        elif not line and lines:
            data = "\n".join(lines)
            if data == "[DONE]":
                return
            yield json.loads(data)
            lines.clear()
    raise ValueError("The stream ended before [DONE].")


def measure(url, model, item, disable_prompt_cache=False):
    body = {
        "model": model, "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0, "max_tokens": 64, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if disable_prompt_cache:
        body["cache_prompt"] = False
    request = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    parts = []
    first = last = finish_reason = None
    usage = {}
    timings = {}
    start = time.monotonic()
    with urlopen(request, timeout=1200) as response:
        for event in stream_events(response):
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage.update(event.get("usage") or {})
            timings.update(event.get("timings") or {})
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice["delta"]
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    last = time.monotonic()
                    if first is None:
                        first = last
                    parts.append(delta.get("content") or "")
    end = time.monotonic()
    answer = "".join(parts)
    if not re.match(r"\W*" + re.escape(item["city"]) + r"\b", answer, re.IGNORECASE):
        raise ValueError(f"Unexpected answer for {item['id']}: {answer!r}")
    if not timings:
        raise ValueError("Restart FreeToken with --enable-timings-report.")
    if timings["cache_n"] != 0:
        raise ValueError("A request reused cached prompt tokens.")
    n_in, n_out = usage["prompt_tokens"], usage["completion_tokens"]
    if not (n_in == timings["prompt_n"] > 0 and n_out == timings["predicted_n"] >= 2):
        raise ValueError(f"Token counts differ or are too short: {usage!r}, {timings!r}")
    if first is None or last <= first or min(timings["prompt_ms"], timings["predicted_ms"]) <= 0:
        raise ValueError("Expected positive durations for prefill and decode.")
    return {
        "id": item["id"], "input_tokens": n_in, "output_tokens": n_out, "text": answer,
        "finish_reason": finish_reason, "timings": timings, "first_s": first - start,
        "output_span_s": last - first, "request_s": end - start,
        "prefill_tokens_per_s": n_in * 1000 / timings["prompt_ms"],
        "decode_tokens_per_s": (n_out - 1) * 1000 / timings["predicted_ms"],
        "client_input_tokens_per_ttft_s": n_in / (first - start),
        "client_decode_tokens_per_s": (n_out - 1) / (last - first),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1919", help="Server root URL")
    parser.add_argument("--model", default="qwen", help="Served model name")
    parser.add_argument("--disable-prompt-cache", action="store_true",
                        help="Explicitly disable prompt reuse on a llama.cpp reference server")
    args = parser.parse_args()
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())
    if [p["id"] for p in prompts] != ["warmup", "run1", "run2", "run3"]:
        raise ValueError("Expected one warm-up followed by three measured prompts.")
    results = []
    for item in prompts:
        row = measure(args.url.rstrip("/"), args.model, item, args.disable_prompt_cache)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if item["id"] != "warmup":
            results.append(row)
    fields = ("prefill_tokens_per_s", "decode_tokens_per_s",
              "client_input_tokens_per_ttft_s", "client_decode_tokens_per_s")
    print(json.dumps({"id": "median", **{f: statistics.median(r[f] for r in results) for f in fields}}))


if __name__ == "__main__":
    main()

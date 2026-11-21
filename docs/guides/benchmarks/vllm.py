#!/usr/bin/env python3
"""Measure the two native vLLM phases on an otherwise idle local server."""
import argparse
import json
import re
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


METRIC = re.compile(
    r"^vllm:request_(prefill|decode)_time_seconds_(count|sum)(?:\{[^}]*\})?\s+(\S+)"
)


def counters(url):
    values = {}
    with urlopen(url + "/metrics", timeout=10) as response:
        for line in response:
            match = METRIC.match(line.decode())
            if match:
                phase, kind, value = match.groups()
                key = (phase, kind)
                values[key] = values.get(key, 0.0) + float(value)
    return values


def phase_seconds(url, before):
    deadline = time.monotonic() + 15
    while True:
        after = counters(url)
        counts = [after.get((p, "count"), 0) - before.get((p, "count"), 0)
                  for p in ("prefill", "decode")]
        if any(n < 0 or n > 1 for n in counts):
            raise ValueError("The server restarted or served another request during measurement.")
        if counts == [1, 1]:
            seconds = {p: after[p, "sum"] - before.get((p, "sum"), 0)
                       for p in ("prefill", "decode")}
            if min(seconds.values()) <= 0:
                raise ValueError("Expected positive durations for both phases.")
            return seconds
        if time.monotonic() >= deadline:
            raise ValueError("Expected one observation in each vLLM phase histogram.")
        time.sleep(0.5)


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
    raise ValueError("The server closed the stream before [DONE].")


def measure(url, model, item):
    body = {
        "model": model, "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0, "max_tokens": 64, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    before = counters(url)
    parts = []
    usage = {}
    with urlopen(req, timeout=1200) as response:
        for event in stream_events(response):
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage.update(event.get("usage") or {})
            for choice in event.get("choices", []):
                parts.append(choice["delta"].get("content") or "")
    text = "".join(parts)
    if not re.match(r"\W*" + re.escape(item["city"]) + r"\b", text, re.IGNORECASE):
        raise ValueError(f"Unexpected answer for {item['id']}: {text!r}")
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):
        raise ValueError("A request reused cached prompt tokens.")
    n_in, n_out = usage["prompt_tokens"], usage["completion_tokens"]
    if n_in < 1 or n_out < 2:
        raise ValueError(f"Insufficient tokens for {item['id']}: {usage!r}")
    seconds = phase_seconds(url, before)
    return {
        "id": item["id"], "input_tokens": n_in, "output_tokens": n_out,
        "prefill_s": seconds["prefill"], "decode_s": seconds["decode"],
        "prefill_tokens_per_s": n_in / seconds["prefill"],
        "decode_tokens_per_s": (n_out - 1) / seconds["decode"], "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server root URL")
    parser.add_argument("--model", default="qwen", help="Served model name")
    args = parser.parse_args()
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())
    if [p["id"] for p in prompts] != ["warmup", "run1", "run2", "run3"]:
        raise ValueError("Expected one warm-up followed by three measured prompts.")
    results = []
    for item in prompts:
        row = measure(args.url.rstrip("/"), args.model, item)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if item["id"] != "warmup":
            results.append(row)
    median = {key: statistics.median(row[key] for row in results)
              for key in ("prefill_tokens_per_s", "decode_tokens_per_s")}
    print(json.dumps({"id": "median", **median}))


if __name__ == "__main__":
    main()

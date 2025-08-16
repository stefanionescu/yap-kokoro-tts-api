#!/usr/bin/env python3
"""
Simple local client to call the remote Kokoro TTS API and save streamed audio.

Usage examples:
  python client.py --host <RUNPOD_PUBLIC_IP> --port 8000 \
    --text "Hello there" --voice female --out hello.pcm

Optionally pass an API key via env RUNPOD_API_KEY (sent as Bearer token) or
--api-key to send a custom header X-API-Key.
"""
import argparse
import os
import sys
import time
import requests


def main():
    parser = argparse.ArgumentParser(description="Call remote Kokoro TTS API and save streamed audio")
    parser.add_argument("--host", default="localhost", help="API host (RunPod public IP or hostname)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--voice", choices=["female", "male"], default="female", help="Voice to use")
    parser.add_argument("--text", required=True, help="Input text to synthesize")
    parser.add_argument("--out", default="output.pcm", help="Output file path (.pcm)")
    parser.add_argument("--api-key", default=None, help="Optional API key sent as X-API-Key header")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/v1/audio/speech/stream"

    headers = {"Content-Type": "application/json", "Accept-Encoding": "identity", "Cache-Control": "no-store"}
    # Prefer Bearer token from env if present
    bearer = os.getenv("RUNPOD_API_KEY") or os.getenv("API_TOKEN")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    # Require explicit sampling params
    if args.voice == "female":
        temperature, top_p, rep = 0.5, 0.95, 1.15
    else:
        temperature, top_p, rep = 0.3, 0.95, 1.12
    payload = {
        "input": args.text,
        "voice": args.voice,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": rep,
    }

    print(f"Calling {url}…")
    t0 = time.time()
    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            first = True
            total = 0
            with open(args.out, "wb") as f:
                for chunk in r.iter_content(chunk_size=1):
                    if not chunk:
                        continue
                    if first:
                        print(f"TTFB: {1000*(time.time()-t0):.0f}ms")
                        first = False
                    f.write(chunk)
                    total += len(chunk)
            print(f"Saved {total} bytes to {args.out}")
    except requests.HTTPError as e:
        print(f"HTTP error: {e} - body: {getattr(e.response,'text', '')}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Request error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



"""Ask the running API a question and print the streamed answer with its sources.

    python scripts/ask.py "How much parental leave do I get?"
    python scripts/ask.py --user <user-uuid> "What is the hotel cap in Paris?"

Standard library only, so it runs from any Python 3.10+ without installing anything.
"""

import argparse
import json
import sys
import urllib.request


def stream_events(url: str, body: dict, headers: dict):
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(request) as response:
        event, data = None, []
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
            elif not line and event:
                yield event, json.loads("".join(data))
                event, data = None, []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--user", help="act as this user id (dev mode only)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    headers = {"X-User-Id": args.user} if args.user else {}
    sources: list[dict] = []
    for event, data in stream_events(f"{args.url}/ask", {"question": args.question}, headers):
        if event == "sources":
            sources = data["sources"]
        elif event == "text":
            print(data["text"], end="", flush=True)
        elif event == "citation":
            print(f"[{data['source']}]", end="", flush=True)
        elif event == "error":
            print(f"\n\nError {data['status']}: {data['message']}")
            return 1
        elif event == "done":
            if data["status"] == "no_relevant_sources":
                print(data["answer"], end="")
            cited = {c["source"] for s in data["segments"] for c in s["citations"]}
            print("\n\nSources:")
            for source in sources:
                if source["number"] in cited:
                    print(f"  {source['label']}")
            usage, timings = data["usage"], data["timings_ms"]
            cost = f"${data['cost_usd']:.4f}" if data["cost_usd"] is not None else "n/a"
            print(
                f"\n{data['status']} | {data['model'] or 'no model call'} | "
                f"{usage['input_tokens']} in / {usage['output_tokens']} out tokens | {cost} | "
                f"first token {timings.get('llm_first_token', '-')} ms, "
                f"total {timings.get('llm_total', '-')} ms"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

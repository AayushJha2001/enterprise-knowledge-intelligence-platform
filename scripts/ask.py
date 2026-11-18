"""Ask the running API a question and print the streamed answer with its sources.

    python scripts/ask.py --email hana@northwind.example "How much parental leave do I get?"

Prompts for the password (or reads EKIP_PASSWORD). Standard library only, so it runs from
any Python 3.10+ without installing anything.
"""

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def login(url: str, email: str, password: str) -> str:
    form = urllib.parse.urlencode({"username": email, "password": password}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{url}/auth/token", form)) as r:
            return json.load(r)["access_token"]
    except urllib.error.HTTPError as exc:
        sys.exit(f"Login failed ({exc.code}): {json.load(exc).get('detail')}")


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
    parser.add_argument("--email", required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    password = os.environ.get("EKIP_PASSWORD") or getpass.getpass(f"Password for {args.email}: ")
    headers = {"Authorization": f"Bearer {login(args.url, args.email, password)}"}
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
            if data.get("cached"):
                timing = f"from cache in {timings['cache_lookup']} ms"
            else:
                timing = (
                    f"first token {timings.get('llm_first_token', '-')} ms, "
                    f"total {timings.get('llm_total', '-')} ms"
                )
            print(
                f"\n{data['status']} | {data['model'] or 'no model call'} | "
                f"{usage['input_tokens']} in / {usage['output_tokens']} out tokens | {cost} | "
                f"{timing}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

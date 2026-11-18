"""Evaluation harness: measure retrieval and answer quality against a labelled dataset.

    python -m evals.run --setup                 # upload + share the sample corpus first
    python -m evals.run                         # retrieval + answer suites, writes a report
    python -m evals.run --check evals/thresholds.json   # exit 1 if any metric regresses

Runs against the HTTP API (the whole system, as users see it), as each question's user,
with `Cache-Control: no-cache` so latencies are real. Standard library only.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.metrics import (
    citation_precision,
    fact_recall,
    first_relevant_rank,
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals" / "datasets" / "northwind.jsonl"
DEMO_USERS = ROOT / "samples" / "demo-users.json"
CORPUS_DIR = ROOT / "samples" / "corpus"
NO_CACHE = {"Cache-Control": "no-cache"}

RETRIEVAL_CONFIGS: dict[str, dict[str, Any]] = {
    "dense": {"mode": "dense", "rerank": False},
    "keyword": {"mode": "keyword", "rerank": False},
    "hybrid": {"mode": "hybrid", "rerank": False},
    "hybrid+rerank": {"mode": "hybrid", "rerank": True},
}
TOP_K = 10


# ---------------------------------------------------------------- dataset


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    user: str = "eli"
    expected_sections: tuple[str, ...] = ()
    expected_facts: tuple[str, ...] = ()
    forbidden_documents: tuple[str, ...] = ()

    @property
    def answerable(self) -> bool:
        return bool(self.expected_sections)


def load_dataset(path: Path) -> list[Question]:
    questions = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            questions.append(
                Question(
                    id=raw["id"],
                    category=raw["category"],
                    question=raw["question"],
                    user=raw.get("user", "eli"),
                    expected_sections=tuple(raw.get("expected_sections", ())),
                    expected_facts=tuple(raw.get("expected_facts", ())),
                    forbidden_documents=tuple(raw.get("forbidden_documents", ())),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{number}: missing field {exc}") from exc
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate question ids")
    return questions


# ---------------------------------------------------------------- HTTP


class ApiError(Exception):
    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class Api:
    """Minimal JSON/form/multipart client over urllib. Tests subclass `_send`."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _send(self, method: str, path: str, data: bytes | None, headers: dict) -> tuple[int, Any]:
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw)
            except ValueError:
                return exc.code, raw.decode(errors="replace")

    def call(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        form: dict[str, str] | None = None,
        file: tuple[str, bytes] | None = None,
        token: str | None = None,
        headers: dict[str, str] | None = None,
        ok: tuple[int, ...] = (200, 201, 202, 204),
    ) -> Any:
        all_headers = dict(headers or {})
        if token:
            all_headers["Authorization"] = f"Bearer {token}"
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode()
            all_headers["Content-Type"] = "application/json"
        elif form is not None:
            data = urllib.parse.urlencode(form).encode()
            all_headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif file is not None:
            boundary = uuid.uuid4().hex
            name, content = file
            data = (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                    f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
                ).encode()
                + content
                + f"\r\n--{boundary}--\r\n".encode()
            )
            all_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        status, body = self._send(method, path, data, all_headers)
        if status not in ok:
            raise ApiError(status, body)
        return body

    def login(self, email: str, password: str) -> str:
        body = self.call("POST", "/auth/token", form={"username": email, "password": password})
        return body["access_token"]


def demo_credentials() -> dict[str, tuple[str, str]]:
    """Short name ('hana') -> (email, password), from samples/demo-users.json."""
    spec = json.loads(DEMO_USERS.read_text(encoding="utf-8"))
    return {u["email"].split("@")[0]: (u["email"], u["password"]) for u in spec["users"]}


# ---------------------------------------------------------------- setup


def setup_corpus(api: Api, admin_token: str, wait_s: int = 300) -> None:
    """Upload samples/corpus, create demo groups/users if missing, share per the demo spec."""
    spec = json.loads(DEMO_USERS.read_text(encoding="utf-8"))
    for path in sorted(CORPUS_DIR.iterdir()):
        try:
            api.call("POST", "/documents", file=(path.name, path.read_bytes()), token=admin_token)
            print(f"uploaded {path.name}")
        except ApiError as exc:
            if exc.status != 409:
                raise
            print(f"present  {path.name}")

    groups = {g["name"]: g["id"] for g in api.call("GET", "/admin/groups", token=admin_token)}
    for group in spec["groups"]:
        if group["name"] not in groups:
            created = api.call("POST", "/admin/groups", json_body=group, token=admin_token)
            groups[group["name"]] = created["id"]

    existing = {u["email"] for u in api.call("GET", "/admin/users?limit=200", token=admin_token)}
    for user in spec["users"]:
        if user["email"] not in existing:
            api.call(
                "POST",
                "/admin/users",
                token=admin_token,
                json_body={
                    "email": user["email"],
                    "display_name": user["name"],
                    "password": user["password"],
                    "is_admin": user.get("is_admin", False),
                    "group_ids": [groups[g] for g in user.get("groups", [])],
                },
            )

    deadline = time.monotonic() + wait_s
    while True:
        docs = api.call("GET", "/documents?limit=200", token=admin_token)["items"]
        pending = [d["title"] for d in docs if d["status"] in ("pending", "processing")]
        if not pending:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"documents still processing: {pending}")
        time.sleep(2)
    failed = [d["title"] for d in docs if d["status"] == "failed"]
    if failed:
        raise RuntimeError(f"ingestion failed for: {failed}")

    by_title = {d["title"]: d["id"] for d in docs}
    for title, group_names in spec["document_access"].items():
        api.call(
            "PUT",
            f"/admin/documents/{by_title[title]}/groups",
            json_body={"group_ids": [groups[g] for g in group_names]},
            token=admin_token,
        )
    print(f"corpus ready: {len(docs)} documents")


# ---------------------------------------------------------------- suites


@dataclass
class RetrievalRow:
    question: str
    config: str
    rank: int | None
    ndcg: float
    latency_ms: float
    leaked: list[str] = field(default_factory=list)


def run_retrieval(
    api: Api, questions: list[Question], tokens: dict[str, str]
) -> list[RetrievalRow]:
    rows = []
    for config_name, options in RETRIEVAL_CONFIGS.items():
        for q in questions:
            if not q.answerable and not q.forbidden_documents:
                continue
            body = api.call(
                "POST",
                "/search",
                json_body={"query": q.question, "top_k": TOP_K, **options},
                token=tokens[q.user],
                headers=NO_CACHE,
            )
            paths = [r["section_path"] for r in body["results"]]
            titles = {r["document_title"] for r in body["results"]}
            rows.append(
                RetrievalRow(
                    question=q.id,
                    config=config_name,
                    rank=first_relevant_rank(paths, q.expected_sections) if q.answerable else None,
                    ndcg=ndcg_at_k(paths, q.expected_sections, TOP_K) if q.answerable else 0.0,
                    latency_ms=body["timings_ms"].get("total", 0.0),
                    leaked=sorted(titles & set(q.forbidden_documents)),
                )
            )
        print(f"retrieval: {config_name} done")
    return rows


@dataclass
class AnswerRow:
    question: str
    category: str
    status: str
    answer: str
    fact_recall: float | None
    grounded: bool
    citation_precision: float | None
    abstained: bool
    leaked: list[str]
    latency_ms: float
    cost_usd: float
    model: str | None


def run_answers(api: Api, questions: list[Question], tokens: dict[str, str]) -> list[AnswerRow]:
    rows = []
    for q in questions:
        started = time.perf_counter()
        body = api.call(
            "POST",
            "/ask",
            json_body={"question": q.question, "stream": False},
            token=tokens[q.user],
            headers=NO_CACHE,
        )
        latency = (time.perf_counter() - started) * 1000
        sources = {s["number"]: s for s in body["sources"]}
        cited = [
            sources[c["source"]]
            for segment in body["segments"]
            for c in segment["citations"]
            if c["source"] in sources
        ]
        rows.append(
            AnswerRow(
                question=q.id,
                category=q.category,
                status=body["status"],
                answer=body["answer"],
                fact_recall=fact_recall(body["answer"], q.expected_facts) if q.answerable else None,
                grounded=body["grounded"],
                citation_precision=citation_precision(
                    [s["section_path"] for s in cited], q.expected_sections
                )
                if q.answerable
                else None,
                abstained=body["status"] == "no_relevant_sources",
                leaked=sorted(
                    {s["document_title"] for s in body["sources"]} & set(q.forbidden_documents)
                ),
                latency_ms=round(latency, 1),
                cost_usd=body.get("cost_usd") or 0.0,
                model=body.get("model"),
            )
        )
    print("answers: done")
    return rows


# ---------------------------------------------------------------- summaries


def summarize_retrieval(rows: list[RetrievalRow], questions: list[Question]) -> dict[str, Any]:
    category = {q.id: q.category for q in questions}
    answerable = {q.id for q in questions if q.answerable}
    summary: dict[str, Any] = {}
    for config in RETRIEVAL_CONFIGS:
        config_rows = [r for r in rows if r.config == config]
        scored = [r for r in config_rows if r.question in answerable]
        by_category: dict[str, list[float]] = defaultdict(list)
        for r in scored:
            by_category[category[r.question]].append(hit_at_k(r.rank, 3))
        latencies = [r.latency_ms for r in config_rows]
        summary[config] = {
            "questions": len(scored),
            "hit@1": round(mean([hit_at_k(r.rank, 1) for r in scored]), 4),
            "hit@3": round(mean([hit_at_k(r.rank, 3) for r in scored]), 4),
            "hit@5": round(mean([hit_at_k(r.rank, 5) for r in scored]), 4),
            "mrr@10": round(mean([reciprocal_rank(r.rank) for r in scored]), 4),
            "ndcg@10": round(mean([r.ndcg for r in scored]), 4),
            "hit@3_by_category": {c: round(mean(v), 4) for c, v in sorted(by_category.items())},
            "latency_p50_ms": round(percentile(latencies, 50), 1),
            "latency_p95_ms": round(percentile(latencies, 95), 1),
            "acl_leaks": sum(bool(r.leaked) for r in config_rows),
            "misses": sorted(r.question for r in scored if r.rank is None or r.rank > 3),
        }
    return summary


def summarize_answers(rows: list[AnswerRow], questions: list[Question]) -> dict[str, Any]:
    by_id = {q.id: q for q in questions}
    answerable = [r for r in rows if by_id[r.question].answerable]
    unanswerable = [
        r for r in rows if not by_id[r.question].answerable
    ]  # includes ACL questions whose only relevant document is forbidden
    precisions = [r.citation_precision for r in answerable if r.citation_precision is not None]
    latencies = [r.latency_ms for r in rows]
    return {
        "questions": len(rows),
        "model": next((r.model for r in rows if r.model), None),
        "fact_recall": round(mean([r.fact_recall for r in answerable]), 4),
        "fully_correct_rate": round(mean([float(r.fact_recall == 1.0) for r in answerable]), 4),
        "grounded_rate": round(mean([float(r.grounded) for r in answerable]), 4),
        "citation_precision": round(mean(precisions), 4),
        "false_abstention_rate": round(mean([float(r.abstained) for r in answerable]), 4),
        "abstention_accuracy": round(mean([float(r.abstained) for r in unanswerable]), 4),
        "acl_leaks": sum(bool(r.leaked) for r in rows),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "total_cost_usd": round(sum(r.cost_usd for r in rows), 6),
        "incorrect": sorted(r.question for r in answerable if r.fact_recall < 1.0),
        "wrongly_answered": sorted(r.question for r in unanswerable if not r.abstained),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation report: {report['dataset']}",
        "",
        f"{report['created_at']} | {report['question_count']} questions | {report['url']}",
        "",
    ]
    retrieval = report.get("retrieval")
    if retrieval:
        lines += [
            "## Retrieval (top 10, section-level relevance)",
            "",
            "| config | hit@1 | hit@3 | hit@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms | ACL leaks |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, s in retrieval.items():
            lines.append(
                f"| {name} | {s['hit@1']:.2f} | {s['hit@3']:.2f} | {s['hit@5']:.2f} | "
                f"{s['mrr@10']:.3f} | {s['ndcg@10']:.3f} | {s['latency_p50_ms']:.0f} | "
                f"{s['latency_p95_ms']:.0f} | {s['acl_leaks']} |"
            )
        lines += ["", "hit@3 by question type:", ""]
        categories = sorted({c for s in retrieval.values() for c in s["hit@3_by_category"]})
        lines.append("| config | " + " | ".join(categories) + " |")
        lines.append("|---|" + "---|" * len(categories))
        for name, s in retrieval.items():
            values = " | ".join(f"{s['hit@3_by_category'].get(c, 0):.2f}" for c in categories)
            lines.append(f"| {name} | {values} |")
        for name, s in retrieval.items():
            if s["misses"]:
                lines.append(f"\nMissed at top 3 by **{name}**: {', '.join(s['misses'])}")
        lines.append("")
    answers = report.get("answers")
    if answers:
        a = answers
        lines += [
            f"## Answers ({a['model']})",
            "",
            "| metric | value |",
            "|---|---|",
            f"| fact recall (answerable) | {a['fact_recall']:.2f} |",
            f"| fully correct (all facts present) | {a['fully_correct_rate']:.2f} |",
            f"| grounded (has citations) | {a['grounded_rate']:.2f} |",
            f"| citation precision | {a['citation_precision']:.2f} |",
            f"| false abstention (answerable, said not found) | {a['false_abstention_rate']:.2f} |",
            f"| abstention accuracy (unanswerable) | {a['abstention_accuracy']:.2f} |",
            f"| ACL leaks | {a['acl_leaks']} |",
            f"| latency p50 / p95 (ms) | {a['latency_p50_ms']:.0f} / {a['latency_p95_ms']:.0f} |",
            f"| total cost (USD) | {a['total_cost_usd']:.4f} |",
            "",
        ]
        if a["incorrect"]:
            lines.append(f"Missing facts: {', '.join(a['incorrect'])}\n")
        if a["wrongly_answered"]:
            lines.append(f"Answered but should not have: {', '.join(a['wrongly_answered'])}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------- thresholds


def check_thresholds(report: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Return human-readable violations. Keys ending in `_max` are upper bounds."""
    violations = []
    for suite, targets in thresholds.items():
        if suite.startswith("_"):
            continue
        results = report.get(suite)
        if results is None:
            violations.append(f"{suite}: suite was not run")
            continue
        flat = (
            {f"{config}.{m}": v for config, metrics in targets.items() for m, v in metrics.items()}
            if suite == "retrieval"
            else targets
        )
        for key, limit in flat.items():
            is_upper_bound = key.endswith("_max")
            actual: Any = results
            for part in key.removesuffix("_max").split("."):
                actual = actual.get(part) if isinstance(actual, dict) else None
            if actual is None:
                violations.append(f"{suite}.{key}: metric missing")
            elif is_upper_bound and actual > limit:
                violations.append(f"{suite}.{key}: {actual} > {limit}")
            elif not is_upper_bound and actual < limit:
                violations.append(f"{suite}.{key}: {actual} < {limit}")
    return violations


# ---------------------------------------------------------------- main


def run(
    api: Api,
    dataset: Path,
    suites: set[str],
    credentials: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    questions = load_dataset(dataset)
    users = {q.user for q in questions}
    missing = users - credentials.keys()
    if missing:
        raise ValueError(f"no credentials for users: {sorted(missing)}")
    tokens = {name: api.login(*credentials[name]) for name in users}

    report: dict[str, Any] = {
        "dataset": dataset.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": api.base_url,
        "question_count": len(questions),
    }
    if "retrieval" in suites:
        rows = run_retrieval(api, questions, tokens)
        report["retrieval"] = summarize_retrieval(rows, questions)
        report["retrieval_rows"] = [r.__dict__ for r in rows]
    if "answers" in suites:
        rows = run_answers(api, questions, tokens)
        report["answers"] = summarize_answers(rows, questions)
        report["answer_rows"] = [r.__dict__ for r in rows]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--suites", default="retrieval,answers")
    parser.add_argument("--setup", action="store_true", help="upload and share the corpus first")
    parser.add_argument("--check", type=Path, help="thresholds JSON; exit 1 on regression")
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "reports")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    api = Api(args.url)
    credentials = demo_credentials()
    if args.setup:
        setup_corpus(api, api.login(*credentials["admin"]))

    report = run(api, args.dataset, set(args.suites.split(",")), credentials)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (args.out / f"report-{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    (args.out / "latest.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)

    if args.check:
        violations = check_thresholds(report, json.loads(args.check.read_text(encoding="utf-8")))
        if violations:
            print("\nREGRESSIONS:\n  " + "\n  ".join(violations))
            return 1
        print("\nAll thresholds met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

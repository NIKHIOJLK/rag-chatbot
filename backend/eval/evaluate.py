"""Measure retrieval quality and tune the guardrail. Run from the backend folder:

    python -m eval.evaluate            # all configurations (downloads 2 embedding models the first time)
    python -m eval.evaluate --quick    # only the current (v2) configuration

For each labelled question we check where the right documentation section lands in the results:
  hit@1 / hit@3 - the right section is the 1st result / in the top 3
  MRR           - mean reciprocal rank: 1 for rank 1, 0.5 for rank 2, 0.33 for rank 3 ... (top 5)
The off-topic questions are used to pick the similarity threshold below which the bot refuses.
Results are printed and written to eval/results.md.
"""
import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

from app.config import INDEX_DIR
from app.ingest import BASE_URL, DOC_PAGES, chunk_by_sections, chunk_fixed, load_pages
from app.retriever import FastEmbedder, Retriever, build_collection, open_collection

EVAL_DIR = Path(__file__).resolve().parent
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-small-en-v1.5"

# (name, chunking, embedding model, hybrid search) - the first row is the original v1 design
CONFIGS = [
    ("v1: 500-word chunks + MiniLM (original)", "fixed", MINILM, False),
    ("Section chunks + MiniLM", "sections", MINILM, False),
    ("Section chunks + bge-small", "sections", BGE, False),
    ("Section chunks + bge-small + BM25 hybrid (v2)", "sections", BGE, True),
]


def load_questions() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in (EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if not r.get("out_of_scope")], [r for r in rows if r.get("out_of_scope")]


def is_relevant(chunk: dict, q: dict) -> bool:
    """Right page, and the chunk mentions what the question is about."""
    text = chunk["text"].lower()
    return chunk["source"].startswith(BASE_URL + q["page"]) and any(k.lower() in text for k in q["keywords"])


def build_retriever(pages, strategy: str, model: str, hybrid: bool, embedders: dict) -> Retriever:
    chunks = chunk_by_sections(pages) if strategy == "sections" else chunk_fixed(pages)
    if model not in embedders:
        print(f"  loading {model} ...")
        embedders[model] = FastEmbedder(model)
    name = f"eval_{strategy}_{model.split('/')[-1].replace('.', '_')}"
    collection = open_collection(str(INDEX_DIR), name)
    if collection.count() != len(chunks):  # embed once, reuse on later runs
        collection = open_collection(str(INDEX_DIR), name, reset=True)
        build_collection(collection, chunks, embedders[model])
    return Retriever(chunks, embedders[model], collection, hybrid=hybrid)


def evaluate(retriever: Retriever, in_scope: list[dict], out_scope: list[dict]) -> dict:
    ranks, latencies, in_best, out_best = [], [], [], []
    for q in in_scope:
        start = time.perf_counter()
        hits, best = retriever.search(q["question"], k=5)
        latencies.append((time.perf_counter() - start) * 1000)
        in_best.append(best)
        rank = next((i + 1 for i, h in enumerate(hits) if is_relevant(h.chunk, q)), None)
        ranks.append(rank)
    for q in out_scope:
        _, best = retriever.search(q["question"], k=5)
        out_best.append(best)
    n = len(in_scope)
    return {
        "hit1": sum(r == 1 for r in ranks) / n,
        "hit3": sum(r is not None and r <= 3 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "p50": statistics.median(latencies),
        "p95": sorted(latencies)[int(0.95 * (n - 1))],
        "chunks": len(retriever.chunks),
        "misses": [q["question"] for q, r in zip(in_scope, ranks) if not r or r > 3],
        "in_best": in_best,
        "out_best": out_best,
    }


def tune_threshold(in_best: list[float], out_best: list[float]) -> tuple[float, float]:
    """Pick the similarity cut-off that best separates real questions from off-topic ones."""
    best_t, best_acc = 0.0, -1.0
    for step in range(20, 96):
        t = step / 100
        correct = sum(s >= t for s in in_best) + sum(s < t for s in out_best)
        acc = correct / (len(in_best) + len(out_best))
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="only evaluate the v2 configuration")
    args = parser.parse_args()

    pages = load_pages()
    if len(pages) < len(DOC_PAGES):
        raise SystemExit("Some pages aren't downloaded yet - run `python -m app.ingest` first.")
    in_scope, out_scope = load_questions()
    configs = CONFIGS[-1:] if args.quick else CONFIGS
    print(f"{len(in_scope)} questions + {len(out_scope)} off-topic questions\n")

    embedders: dict = {}
    rows = []
    for name, strategy, model, hybrid in configs:
        print(f"Evaluating: {name}")
        rows.append((name, evaluate(build_retriever(pages, strategy, model, hybrid, embedders), in_scope, out_scope)))

    final = rows[-1][1]
    threshold, accuracy = tune_threshold(final["in_best"], final["out_best"])

    lines = [
        f"# Retrieval evaluation ({date.today().isoformat()})",
        "",
        f"{len(in_scope)} labelled questions over {len(pages)} Python docs pages; {len(out_scope)} off-topic questions for the guardrail.",
        "",
        "| Configuration | Chunks | hit@1 | hit@3 | MRR@5 | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in rows:
        lines.append(f"| {name} | {r['chunks']} | {r['hit1']:.0%} | {r['hit3']:.0%} | {r['mrr']:.2f} | {r['p50']:.0f} | {r['p95']:.0f} |")
    lines += [
        "",
        f"**Guardrail:** refusing when the best similarity is below **{threshold:.2f}** classifies "
        f"**{accuracy:.0%}** of questions correctly (answer real questions, refuse off-topic ones). "
        f"Set `MIN_SCORE={threshold:.2f}` in `backend/.env`.",
        "",
        "Questions the v2 configuration still misses (right section not in the top 3):",
        "",
        *([f"- {q}" for q in final["misses"]] or ["- none"]),
    ]
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    (EVAL_DIR / "results.md").write_text(report, encoding="utf-8")
    print("Saved eval/results.md")


if __name__ == "__main__":
    main()

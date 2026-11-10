"""Run the eval cases: python -m evals [--embedding NAME] [--sweep] [--misses]

Uses a real embedding model, from an embedding profile in configs/embeddings/ (the
threshold depends on the model), and the engine config from configs/default.toml,
without Django or a database. Ollama profiles need Ollama running.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from adapters.embeddings.factory import embedder_from_profile
from dataclasses import replace

from brain.config import load_embedding_profile, load_engine_config, with_min_score

from .cases import load_scenarios
from .runner import line, prepare, run, summarize, sweep

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals", description="Measure retrieval against the eval cases.")
    parser.add_argument("--cases", default=str(ROOT / "evals" / "cases"),
                        help="a case file or a folder of them (default: evals/cases)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    parser.add_argument("--embedding", default="bge-m3",
                        help="an embedding profile in configs/embeddings/ (default: bge-m3)")
    parser.add_argument("--ollama-url", default=None, help="Ollama server for Ollama profiles")
    parser.add_argument("--sweep", action="store_true",
                        help="try min_score around the profile's and print the trade-off")
    parser.add_argument("--grid", default=None, metavar="LOW,HIGH,STEP",
                        help="thresholds for --sweep (default: the profile's +/- 0.10, step 0.02)")
    parser.add_argument("--misses", action="store_true",
                        help="list every query that missed or recalled noise")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. messages.w_keyword=0.6 (repeatable).")
    parser.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    profile = load_embedding_profile(ROOT / "configs" / "embeddings" / f"{args.embedding}.toml")
    if args.fake_embedder:  # for tests: no model download
        from tests.fakes import FakeEmbedder
        embedder = FakeEmbedder()
    else:
        embedder = embedder_from_profile(profile, base_url=args.ollama_url)

    config = replace(with_min_score(load_engine_config(args.config), profile.min_score),
                     tag_match_score=profile.tag_match_score)
    from .turns import with_settings

    config = with_settings(config, args.set)
    scenarios = load_scenarios(args.cases)
    started = time.monotonic()
    prepared = [prepare(s, embedder, config.attachment_chunk_chars, config.attachment_overlap) for s in scenarios]
    queries = sum(len(s.queries) for s in scenarios)
    print(f"{len(scenarios)} scenario(s), {queries} queries, embedding model {embedder.model} "
          f"(loaded in {time.monotonic() - started:.1f}s)\n")

    if args.sweep:
        print("min_score")
        current = profile.min_score
        low, high, step = (
            map(float, args.grid.split(",")) if args.grid
            else (max(0.0, current - 0.10), min(1.0, current + 0.10), 0.02)
        )
        thresholds = [round(low + step * i, 3) for i in range(int(round((high - low) / step)) + 1)]
        for threshold, metrics in sweep(prepared, embedder, config, thresholds):
            mark = "  <- current" if abs(threshold - current) < 1e-9 else ""
            print(f"  {threshold:.2f}     {line(metrics)}{mark}")
        return 0

    results = run(prepared, embedder, config)
    print(f"min_score {config.messages.min_score:.2f}: {line(summarize(results))}")
    if args.misses:
        for r in results:
            wrong = (not r.negative and not r.hit) or (r.negative and r.kept) or r.rejected
            if wrong:
                label = "NOISE" if r.negative else ("REJECTED" if r.rejected else "MISS")
                print(f"\n{label}  [{r.scenario}] {r.query.text}")
                print(f"  expected {list(r.query.expect) or 'nothing'}; got {list(r.kept) or 'nothing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

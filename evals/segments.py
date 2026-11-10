"""Episode boundaries by change of subject (PIPELINE.md step 3): measured on sessions
of messages labeled with their topic.

    python -m evals.segments                     # the shipped settings
    python -m evals.segments --stats             # similarities, same topic vs new
    python -m evals.segments --set drift_threshold=0.5

Each session is walked as a turn would: a message is compared with the average of the
episode's last `drift_window` messages (brain.segmentation.drift_similarity), the
two-strike rule decides (drift_step), and a split starts the new episode at the first
drifting message, which moves with it. A true boundary is where the topic label
changes. Scored: boundaries found (recall), splits that were right (precision), and
splits inside a topic (the costly mistake: the conversation loses its thread).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from brain.config import EngineConfig, load_engine_config
from brain.ports import Embedder
from brain.segmentation import drift_similarity, drift_step

from .turns import RememberingEmbedder, with_settings

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Session:
    name: str
    texts: tuple[str, ...]
    topics: tuple[str, ...]

    @property
    def boundaries(self) -> set[int]:
        return {i for i in range(1, len(self.topics)) if self.topics[i] != self.topics[i - 1]}


def load_sessions(path: str | Path) -> list[Session]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Session(s["name"], tuple(m["text"] for m in s["messages"]), tuple(m["topic"] for m in s["messages"]))
            for s in data["sessions"]]


def predict(session: Session, vectors: Sequence[Sequence[float]], config: EngineConfig) -> set[int]:
    """Where new episodes start, as the engine would decide."""
    start, strikes, pending = 0, 0, None
    splits: set[int] = set()
    for i, text in enumerate(session.texts):
        # The subject so far: the episode's messages before any pending (drifting) one.
        before = list(range(start, pending if pending is not None else i))
        recent = [vectors[j] for j in before[-config.drift_window:]]
        similarity = drift_similarity(vectors[i], recent) if len(before) >= config.drift_min_messages else None
        decision, strikes = drift_step(text, similarity, strikes, config)
        if decision == "strike" and pending is None:
            pending = i
        elif decision == "same":
            pending = None
        elif decision == "split":
            start = pending if pending is not None else i
            splits.add(start)
            pending = None
    return splits


def score(sessions: Sequence[Session], embedder: Embedder, config: EngineConfig) -> dict:
    found = right = predicted = truth = inside = 0
    details = []
    for session in sessions:
        vectors = [e.vector for e in embedder.embed(list(session.texts))]
        splits = predict(session, vectors, config)
        truth += len(session.boundaries)
        predicted += len(splits)
        right += len(splits & session.boundaries)
        found += len(session.boundaries & splits)
        wrong = splits - session.boundaries
        inside += len(wrong)
        details.append((session.name, sorted(session.boundaries), sorted(splits)))
    return {"recall": found / truth if truth else 1.0, "precision": right / predicted if predicted else 1.0,
            "splits_inside_a_topic": inside, "boundaries": truth, "splits": predicted, "details": details}


def stats(sessions: Sequence[Session], embedder: Embedder, config: EngineConfig) -> str:
    """Similarity of each message to the true episode so far: same topic vs a new one."""
    same, new = [], []
    for session in sessions:
        vectors = [e.vector for e in embedder.embed(list(session.texts))]
        start = 0
        for i in range(1, len(vectors)):
            if i in session.boundaries:
                recent = vectors[max(start, i - config.drift_window):i]
                if i - start >= config.drift_min_messages:
                    new.append(drift_similarity(vectors[i], recent))
                start = i
                continue
            if i - start >= config.drift_min_messages and len(session.texts[i].split()) >= config.drift_min_words:
                same.append(drift_similarity(vectors[i], vectors[max(start, i - config.drift_window):i]))
    rows = []
    for name, xs in (("same topic", same), ("new topic", new)):
        xs.sort()
        rows.append(f"{name:11} n={len(xs):3}  min {xs[0]:.2f}  p10 {xs[len(xs) // 10]:.2f}  "
                    f"median {statistics.median(xs):.2f}  max {xs[-1]:.2f}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.segments", description="Episode boundaries by subject.")
    parser.add_argument("--sessions", default=str(ROOT / "evals" / "segment_cases" / "sessions.json"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    parser.add_argument("--embedding", default="bge-m3")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.fake_embedder:
        from tests.fakes import FakeEmbedder
        inner: Embedder = FakeEmbedder()
    else:
        from adapters.embeddings.factory import embedder_from_profile
        from brain.config import load_embedding_profile
        inner = embedder_from_profile(load_embedding_profile(ROOT / "configs" / "embeddings" / f"{args.embedding}.toml"))
    embedder = RememberingEmbedder(inner)
    config = with_settings(load_engine_config(args.config), args.set)
    sessions = load_sessions(args.sessions)
    if args.stats:
        print(stats(sessions, embedder, config))
        return 0
    result = score(sessions, embedder, config)
    print(f"{len(sessions)} sessions, {result['boundaries']} topic changes, drift_threshold "
          f"{config.drift_threshold:.2f}: found {result['recall']:.0%}, right {result['precision']:.0%} "
          f"of {result['splits']} splits, {result['splits_inside_a_topic']} inside a topic")
    if args.details:
        for name, truth, splits in result["details"]:
            mark = "ok  " if truth == splits else "DIFF"
            print(f"[{mark}] {name}: topic changes {truth}, splits {splits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

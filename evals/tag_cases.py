"""Give eval episodes the tags consolidation would give them.

    python -m evals.tag_cases            # episodes without tags
    python -m evals.tag_cases --all      # every episode again

Each episode goes through the real `consolidate_episode` (background model, embedding
model, tag merging), in file order and in one shared store, so a new tag that means
the same as an earlier one reuses it, as in the app. The tags are written
into the case file as `"tags": [...]` after the episode's key; the rest of the file
is left as written. Tags written by the model, not by hand: whoever writes the cases
knows the questions, and hand-picked tags would flatter the tag signal.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from adapters.stores.in_memory import InMemoryStore
from brain.config import load_embedding_profile, load_engine_config, load_model_profile, with_embedding
from brain.consolidation import consolidate_episode
from brain.types import Role

from .cases import load_scenarios

ROOT = Path(__file__).resolve().parent.parent
T0 = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


def write_tags(file: Path, tags: dict[str, list[str]]) -> None:
    """Put each episode's tags on the line after its key, replacing old ones."""
    text = file.read_text(encoding="utf-8")
    for key, names in tags.items():
        pattern = re.compile(rf'(\n(\s*)"key": {re.escape(json.dumps(key))},)(\n\s*"tags": \[[^\]]*\],)?')
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"{file}: episode {key!r} not found")
        line = f'\n{match.group(2)}"tags": {json.dumps(names, ensure_ascii=False)},'
        text = text[:match.start()] + match.group(1) + line + text[match.end():]
    file.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.tag_cases", description="Give eval episodes the tags consolidation would give them.")
    parser.add_argument("--cases", default=str(ROOT / "evals" / "cases"))
    parser.add_argument("--model", default="qwen3.5-4b-background")
    parser.add_argument("--embedding", default="bge-m3")
    parser.add_argument("--all", action="store_true", help="Tag episodes that have tags too.")
    args = parser.parse_args(argv)

    from adapters.embeddings.factory import embedder_from_profile
    from adapters.llm.factory import provider_from_profile

    profile = load_model_profile(ROOT / "configs" / "models" / f"{args.model}.toml")
    embedding = load_embedding_profile(ROOT / "configs" / "embeddings" / f"{args.embedding}.toml")
    config = with_embedding(load_engine_config(ROOT / "configs" / "default.toml"), embedding)
    llm, embedder = provider_from_profile(profile), embedder_from_profile(embedding)

    store = InMemoryStore()
    day = 0
    for scenario in load_scenarios(args.cases):
        tags: dict[str, list[str]] = {}
        for case in scenario.episodes:
            day += 1
            start = T0 + timedelta(days=day)
            episode = store.open_episode(start)
            for i, (role, text) in enumerate(case.messages):
                store.add_message(episode.id, Role(role), text, start + timedelta(minutes=i))
            store.close_episode(episode.id, start + timedelta(minutes=len(case.messages)))
            if case.tags and not args.all:
                for name in case.tags:  # there to be merged with, as in the app
                    tag = store.upsert_tag(name, embedder.embed([name])[0])
                    if tag is not None:
                        store.tag_episode(episode.id, tag.id, 1.0)
                continue
            report = consolidate_episode(
                episode.id, store=store, llm=llm, embedder=embedder, profile=profile,
                now=start + timedelta(hours=1), tag_same_score=config.tag_same_score,
            )
            assert report is not None
            tags[case.key] = report.tags
            merged = "".join(f" ({new} -> {old})" for new, old in report.merged)
            print(f"{scenario.name}/{case.key}: {', '.join(report.tags)}{merged}", flush=True)
        if tags:
            write_tags(Path(scenario.source), tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

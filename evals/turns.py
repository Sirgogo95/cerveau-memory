"""Multi-turn eval: short conversations over the eval episodes, measuring what each
turn recalls (docs/ACTIVATION.md, step 1).

    python -m evals.turns
    python -m evals.turns --details      # every turn, what it kept

A case file names a scenario from evals/cases (its episodes are the past) and lists
conversations, each a few user messages with the memories expected at each turn:

    {
      "scenario": "life",
      "conversations": [
        {"name": "knee, then running",
         "turns": [
           {"text": "My knee is bothering me again.", "expect": ["knee"]},
           {"text": "Can I still go running this weekend?", "expect": ["knee"]}
         ]},
        {"name": "knee, then coffee",
         "turns": [
           {"text": "My knee is bothering me again."},
           {"text": "Anyway, where should I get coffee?", "switch": true,
            "expect": ["coffee"], "reject": ["knee"]}
         ]}
      ]
    }

The first turn of a conversation is an opener; later turns are follow-ups, or
switches when marked `"switch": true` (the subject changed: what came before must
not come along). Each conversation runs on a fresh store, through brain.recall,
the function a real turn uses; the current conversation is an open episode, so its
own messages are left out as in a real turn. Recent conversations are turned off,
as in the single-question eval.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

from brain.config import EngineConfig, load_embedding_profile, load_engine_config, with_min_score
from brain.ports import Embedder
from brain.recall import recall
from brain.types import Embedding, Role

from .cases import CaseError, QueryCase, Scenario, load_scenarios
from .runner import T0, QueryResult, line, prepare, summarize

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("opener", "follow-up", "switch")


@dataclass(frozen=True, slots=True)
class Turn:
    text: str
    kind: str
    expect: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Conversation:
    name: str
    scenario: Scenario
    turns: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class TurnResult:
    conversation: str
    index: int
    kind: str
    result: QueryResult


def parse_conversations(data: dict, scenarios: dict[str, Scenario], source: str = "") -> list[Conversation]:
    try:
        scenario = scenarios[data["scenario"]]
        conversations = [
            Conversation(
                str(c["name"]), scenario,
                tuple(
                    Turn(str(t["text"]),
                         "opener" if i == 0 else "switch" if t.get("switch") else "follow-up",
                         tuple(t.get("expect", ())), tuple(t.get("reject", ())))
                    for i, t in enumerate(c["turns"])
                ),
            )
            for c in data["conversations"]
        ]
    except (KeyError, TypeError) as exc:
        raise CaseError(f"{source}: malformed conversation file ({exc})") from exc
    valid = scenario.refs()
    for conversation in conversations:
        if not conversation.turns:
            raise CaseError(f"{source}: {conversation.name!r} has no turns")
        for turn in conversation.turns:
            for ref in (*turn.expect, *turn.reject):
                if ref not in valid:
                    raise CaseError(f"{source}: {conversation.name!r} refers to unknown {ref!r}")
    return conversations


def load_conversations(path: str | Path, cases: str | Path) -> list[Conversation]:
    scenarios = {s.name: s for s in load_scenarios(cases)}
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob("*.json"))
    conversations = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        conversations.extend(parse_conversations(data, scenarios, str(file)))
    return conversations


class RememberingEmbedder:
    """Embeds each text once for the whole run: every conversation rebuilds its store
    from the same episodes."""

    def __init__(self, inner: Embedder):
        self.inner, self.model = inner, inner.model
        self._known: dict[str, Embedding] = {}

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._known]
        if missing:
            self._known.update(zip(missing, self.inner.embed(missing)))
        return [self._known[t] for t in texts]


def run_conversation(conversation: Conversation, embedder: Embedder, config: EngineConfig) -> list[TurnResult]:
    prepared = prepare(conversation.scenario, embedder)
    store = prepared.store
    now = T0 + timedelta(days=len(conversation.scenario.episodes) + 30)
    episode = store.open_episode(now)
    results = []
    for i, turn in enumerate(conversation.turns):
        at = now + timedelta(minutes=2 * i)
        [query] = embedder.embed([turn.text])
        store.add_message(episode.id, Role.USER, turn.text, at, query)
        recalled = recall(turn.text, query, store, config, embedder=embedder, episode_id=episode.id, now=at)
        kept = tuple(
            prepared.refs[(m.candidate.kind, m.candidate.id)]
            for m in recalled.memories if (m.candidate.kind, m.candidate.id) in prepared.refs
        )
        case = QueryCase(turn.text, turn.expect, turn.reject)
        results.append(TurnResult(conversation.name, i, turn.kind, QueryResult(conversation.scenario.name, case, kept)))
    return results


def run(conversations: Sequence[Conversation], embedder: Embedder, config: EngineConfig) -> list[TurnResult]:
    return [r for c in conversations for r in run_conversation(c, embedder, config)]


def report(results: Sequence[TurnResult]) -> str:
    rows = [f"{'all turns':10} {line(summarize([r.result for r in results]))}"]
    for kind in KINDS:
        chosen = [r.result for r in results if r.kind == kind]
        if chosen:
            rows.append(f"{kind:10} {line(summarize(chosen))}")
    return "\n".join(rows)


def setting_value(raw: str, current: object) -> object:
    """`raw` as the type of the setting it replaces: true/false for switches (not
    any word as true), numbers for numbers (negative ones too), text otherwise."""
    if isinstance(current, bool):
        if raw.lower() not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise ValueError(f"{raw!r} is not true or false")
        return raw.lower() in {"true", "1", "yes", "on"}
    if isinstance(current, (int, float)):
        return type(current)(float(raw))
    return raw


def with_settings(config: EngineConfig, settings: Sequence[str]) -> EngineConfig:
    """`key=value` overrides: a field of EngineConfig, or `section.field` for one
    source's SourceConfig or another settings group (entity_block)."""
    for setting in settings:
        key, _, raw = setting.partition("=")
        if "." in key:
            section, name = key.split(".", 1)
            group = getattr(config, section)
            config = replace(config, **{section: replace(group, **{name: setting_value(raw, getattr(group, name))})})
        else:
            config = replace(config, **{key: setting_value(raw, getattr(config, key))})
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.turns", description="Multi-turn recall evaluation.")
    parser.add_argument("--turns", default=str(ROOT / "evals" / "turn_cases"))
    parser.add_argument("--cases", default=str(ROOT / "evals" / "cases"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    parser.add_argument("--embedding", default="bge-m3")
    parser.add_argument("--details", action="store_true", help="Print every turn and what it kept.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. context_weight=0.5 or episodes.w_tag=0.1 (repeatable).")
    parser.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    profile = load_embedding_profile(ROOT / "configs" / "embeddings" / f"{args.embedding}.toml")
    if args.fake_embedder:  # for tests: no model download
        from tests.fakes import FakeEmbedder
        inner: Embedder = FakeEmbedder()
    else:
        from adapters.embeddings.factory import embedder_from_profile
        inner = embedder_from_profile(profile)
    config = replace(with_min_score(load_engine_config(args.config), profile.min_score),
                     tag_match_score=profile.tag_match_score, recent_episodes=0)
    config = with_settings(config, args.set)
    conversations = load_conversations(args.turns, args.cases)
    results = run(conversations, RememberingEmbedder(inner), config)
    if args.details:
        for r in results:
            mark = "ok  " if r.result.hit or (r.result.negative and not r.result.kept) else "MISS"
            if r.result.rejected:
                mark = "DRAG"
            print(f"[{mark}] {r.conversation} #{r.index} ({r.kind}) {r.result.query.text}\n"
                  f"       kept: {', '.join(r.result.kept) or '-'}")
    turns = len(results)
    print(f"{len(conversations)} conversations, {turns} turns, {inner.model}, min_score {config.messages.min_score:.2f}")
    print(report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

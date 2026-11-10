"""Eval cases: scenarios of past conversations, and the questions that should (or
should not) recall parts of them.

A scenario file (JSON) looks like this:

    {
      "name": "family",
      "episodes": [
        {
          "key": "sister",
          "tags": ["family", "relocation"],
          "messages": [
            ["user", "My sister Lucía just moved to Montreal."],
            ["assistant", "Noted."]
          ],
          "summary": "The user's sister Lucía moved to Montreal.",
          "entities": [{"name": "Lucía", "type": "person", "aliases": ["my sister"]}]
        }
      ],
      "queries": [
        {"text": "Where does my sister live?", "expect": ["sister:0", "sister"]},
        {"text": "Explain recursion.", "expect": []}
      ]
    }

References: "sister:0" is message 0 of episode "sister"; "sister" alone is that
episode's summary; "lease/deposit is 60,000" is a passage of the shared file "lease"
containing that phrase. An empty `expect` means nothing should be recalled: every hit
counts as noise. `reject` lists memories that must not come back (optional).
`summary` (and optional `summary_long`) stand for a consolidated episode; `entities`
are recorded with their mentions found in the messages, as consolidation would.
`tags` (optional) are the episode's topics, written by `python -m evals.tag_cases`
with the real background model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROLES = ("user", "assistant")


class CaseError(ValueError):
    """A case file is malformed or refers to something that doesn't exist."""


@dataclass(frozen=True, slots=True)
class EntityCase:
    name: str
    type: str = ""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AttachmentCase:
    """A file shared in the episode's first message, read from `file` (next to the
    case file). Questions expect its passages by a phrase they contain:
    "lease/deposit is 60,000", whatever the passage size."""

    key: str
    filename: str
    text: str


@dataclass(frozen=True, slots=True)
class EpisodeCase:
    key: str
    messages: tuple[tuple[str, str], ...]
    summary: str = ""
    summary_long: str = ""
    entities: tuple[EntityCase, ...] = ()
    tags: tuple[str, ...] = ()
    attachments: tuple[AttachmentCase, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryCase:
    text: str
    expect: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    source: str
    episodes: tuple[EpisodeCase, ...]
    queries: tuple[QueryCase, ...] = field(default_factory=tuple)

    def refs(self) -> set[str]:
        """Every valid reference: each message, and each episode with a summary."""
        valid = set()
        for episode in self.episodes:
            valid.update(f"{episode.key}:{i}" for i in range(len(episode.messages)))
            if episode.summary:
                valid.add(episode.key)
        return valid


def parse_scenario(data: dict, source: str = "") -> Scenario:
    try:
        episodes = tuple(
            EpisodeCase(
                key=str(e["key"]),
                messages=tuple((str(r), str(t)) for r, t in e["messages"]),
                summary=str(e.get("summary", "")),
                summary_long=str(e.get("summary_long", "")),
                entities=tuple(
                    EntityCase(str(n["name"]), str(n.get("type", "")), tuple(n.get("aliases", ())))
                    for n in e.get("entities", ())
                ),
                tags=tuple(str(t) for t in e.get("tags", ())),
                attachments=tuple(
                    AttachmentCase(str(a["key"]), Path(a["file"]).name,
                                   (Path(source).parent / a["file"]).read_text(encoding="utf-8"))
                    for a in e.get("attachments", ())
                ),
            )
            for e in data["episodes"]
        )
        queries = tuple(
            QueryCase(str(q["text"]), tuple(q.get("expect", ())), tuple(q.get("reject", ())))
            for q in data.get("queries", ())
        )
        scenario = Scenario(str(data.get("name", source)), source, episodes, queries)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise CaseError(f"{source}: malformed case ({exc})") from exc

    keys = [e.key for e in episodes]
    if len(keys) != len(set(keys)):
        raise CaseError(f"{source}: episode keys must be unique")
    for episode in episodes:
        for role, _ in episode.messages:
            if role not in ROLES:
                raise CaseError(f"{source}: unknown role {role!r} in episode {episode.key!r}")
    valid = scenario.refs()
    files = {a.key: a.text.casefold() for e in episodes for a in e.attachments}
    for query in queries:
        for ref in (*query.expect, *query.reject):
            key, _, phrase = ref.partition("/")
            if phrase and key in files:
                if phrase.casefold() not in files[key]:
                    raise CaseError(f"{source}: {ref!r}: the phrase isn't in that file")
            elif ref not in valid:
                raise CaseError(f"{source}: query {query.text!r} refers to unknown {ref!r}")
    return scenario


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Every scenario in a file, or in a folder and its subfolders (*.json). A `chats`
    folder is skipped: the personal chats eval keeps its own files there (the chosen
    chats, the questions, the answers), which are not scenarios."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(
        f for f in root.rglob("*.json") if "chats" not in f.relative_to(root).parts[:-1])
    scenarios = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaseError(f"{file}: cannot read ({exc})") from exc
        scenarios.append(parse_scenario(data, str(file)))
    return scenarios

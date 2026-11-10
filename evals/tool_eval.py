"""Tool-use evaluation: real turns with the real chat model, tools on and off.

    python -m evals.tool_eval [--cases PATH] [--runs N] [--only on|off]

Retrieval alone is measured by `python -m evals`; this measures whole turns. Each
question runs in a fresh conversation on a fresh copy of the scenario's memory
(nothing one question says leaks into the next), once with the memory tools offered
and once without, and records whether the model searched, what it answered, whether
the answer holds an expected word, and how long it took.

Question kinds, and what "right" means:

- general: world knowledge. Shouldn't search; right if the answer has an expected
  word (when the case gives some).
- in_context: the answer is in a recent conversation the prompt always shows.
- memory: the answer is in an older conversation, reached by automatic recall or
  a search.
- absent: nothing is stored. Right if the answer says it doesn't remember.
- near_miss: the subject is stored, but not the detail asked for ("who is Pedro
  marrying?" when the bride was never named). Right if the answer says it doesn't
  know, and has none of the `wrong` words (a plausible invention, like the dentist's
  name for the knee doctor). Where a small model is most tempted to fill the gap.

Answers are checked by words (accents and case ignored): crude, but deterministic.
Needs Ollama running (chat model and embedding model).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from brain.config import EngineConfig, load_embedding_profile, load_engine_config, load_model_profile, with_embedding
from brain.text import normalize_key
from brain.turn import run_turn

from .cases import parse_scenario
from .runner import prepare
from .turns import with_settings

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("general", "in_context", "memory", "absent", "near_miss")
# "I don't remember" in the three languages, normalized (no accents, lowercase).
DONT_REMEMBER = (
    "dont remember", "do not remember", "dont recall", "do not recall", "no record",
    "dont have any", "do not have any", "dont have information", "no information",
    "nothing about", "not something weve", "havent talked", "have not talked",
    "havent discussed", "have not discussed", "no recuerdo", "no tengo", "no hemos hablado",
    "no me has", "ne me souviens", "je n ai pas", "nous n avons pas", "ne detiens pas", "ne dispose pas",
    "isnt in my notes", "is not in my notes", "not in my notes", "dont have the",
    "didnt mention", "did not mention", "didnt tell", "did not tell", "havent told",
    "have not told", "not mentioned", "dont know", "do not know", "never mentioned",
    "no mencionaste", "no me dijiste", "no lo se", "no se ", "no sabemos",
    "ne m as pas", "ne mavez pas", "ne m avez pas", "pas mentionne", "je ne sais pas",
    "isnt mentioned", "wasnt mentioned", "dont specify", "do not specify", "doesnt say",
    "not specified", "dont mention", "do not mention", "doesnt mention", "no especifica",
    "no se menciona", "no mencionan", "ne precise pas", "ne mentionne pas",
    "no mention", "havent mentioned", "have not mentioned", "je nai pas", "ne trouve pas",
    "ne connais pas", "no aparece", "not in my notes", "do not have information",
    "dont have the", "do not have the", "dont have information", "doesnt specify",
    "does not specify", "dont have your", "do not have your", "dont have that", "do not have that", "dont have specific", "do not have specific", "no encuentro",
)


@dataclass(frozen=True)
class Question:
    text: str
    kind: str
    answer: tuple[str, ...] = ()
    wrong: tuple[str, ...] = ()


@dataclass(frozen=True)
class Result:
    question: Question
    tools: bool
    searched: tuple[str, ...]     # tool calls, as "name(arguments)"
    answer: str
    right: bool | None            # None: nothing to check
    seconds: float


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario = parse_scenario({"name": data["name"], "episodes": data["episodes"]}, str(path))
    questions = [
        Question(q["text"], q["kind"], tuple(q.get("answer", ())), tuple(q.get("wrong", ())))
        for q in data["questions"]
    ]
    for q in questions:
        if q.kind not in KINDS:
            raise ValueError(f"{path}: unknown kind {q.kind!r} for {q.text!r}")
    return scenario, questions, datetime.fromisoformat(data["now"])


def is_right(question: Question, answer: str) -> bool | None:
    said = normalize_key(answer.replace("'", "").replace("’", ""))
    if question.kind in ("absent", "near_miss"):
        invented = any(normalize_key(word) in said for word in question.wrong)
        return any(phrase in said for phrase in DONT_REMEMBER) and not invented
    if not question.answer:
        return None
    return any(normalize_key(word) in said for word in question.answer)


class RememberingEmbedder:
    """An embedder that computes each text once, so rebuilding a scenario's memory
    for every question costs nothing after the first time."""

    def __init__(self, inner):
        self.inner = inner
        self.model = inner.model
        self._known: dict[str, object] = {}

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def embed(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self._known]
        if missing:
            self._known.update(zip(missing, self.inner.embed(missing)))
        return [self._known[t] for t in texts]


def ask(question: Question, scenario, now: datetime, *, llm, embedder, profile, config: EngineConfig,
        tools: bool) -> Result:
    store = prepare(scenario, embedder).store  # fresh memory: each question starts the same
    started = time.monotonic()
    answer = "".join(c.content for c in run_turn(
        question.text, now, store=store, llm=llm, embedder=embedder, profile=profile,
        config=replace(config, memory_tools=tools),
    ))
    seconds = time.monotonic() - started
    current = store.current_episode()
    calls = store.get_tool_calls(current.id) if current else []
    searched = tuple(f"{c.name}({json.dumps(dict(c.arguments), ensure_ascii=False)})" for c in calls)
    return Result(question, tools, searched, answer, is_right(question, answer), seconds)


def report(results: list[Result]) -> str:
    lines = []
    for tools in (True, False):
        mine = [r for r in results if r.tools is tools]
        if not mine:
            continue
        lines.append(f"\nTools {'on' if tools else 'off'}:")
        lines.append(f"  {'kind':11} {'right':>9} {'searched':>9} {'seconds':>8}")
        for kind in KINDS:
            rows = [r for r in mine if r.question.kind == kind]
            if not rows:
                continue
            checked = [r for r in rows if r.right is not None]
            right = f"{sum(r.right is True for r in checked)}/{len(checked)}" if checked else "-"
            searched = f"{sum(bool(r.searched) for r in rows)}/{len(rows)}"
            seconds = statistics.median(r.seconds for r in rows)
            lines.append(f"  {kind:11} {right:>9} {searched:>9} {seconds:>7.1f}s")
        checked = [r for r in mine if r.right is not None]
        lines.append(f"  {'all':11} {sum(r.right is True for r in checked)}/{len(checked):<6} "
                     f"{sum(bool(r.searched) for r in mine)}/{len(mine):<6}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.tool_eval", description="Tool-use evaluation.")
    parser.add_argument("--cases", default=str(ROOT / "evals" / "tool_cases" / "january.json"))
    parser.add_argument("--runs", type=int, default=1, help="Runs per question and setting.")
    parser.add_argument("--only", choices=("on", "off"), help="Only tools on, or only off.")
    parser.add_argument("--chat-model", default="qwen3.5-4b")
    parser.add_argument("--embedding", default="bge-m3")
    parser.add_argument("--details", action="store_true", help="Print every answer and search.")
    parser.add_argument("--kinds", default="", help="Only these question kinds, comma-separated.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override an engine setting, e.g. --set fallback_search=false.")
    args = parser.parse_args(argv)

    from adapters.embeddings.factory import embedder_from_profile
    from adapters.llm.factory import provider_from_profile

    profile = load_model_profile(ROOT / "configs" / "models" / f"{args.chat_model}.toml")
    embedding = load_embedding_profile(ROOT / "configs" / "embeddings" / f"{args.embedding}.toml")
    config = with_settings(with_embedding(load_engine_config(ROOT / "configs" / "default.toml"), embedding), args.set)
    llm, embedder = provider_from_profile(profile), RememberingEmbedder(embedder_from_profile(embedding))

    scenario, questions, now = load(Path(args.cases))
    if args.kinds:
        questions = [q for q in questions if q.kind in args.kinds.split(",")]
    settings = {"on": (True,), "off": (False,)}.get(args.only or "", (True, False))
    results = []
    for question in questions:
        for tools in settings:
            for _ in range(args.runs):
                result = ask(question, scenario, now, llm=llm, embedder=embedder, profile=profile,
                             config=config, tools=tools)
                results.append(result)
                if args.details:
                    mark = {True: "RIGHT", False: "WRONG", None: "  -  "}[result.right]
                    print(f"[{mark}] tools {'on ' if tools else 'off'} {question.kind:10} "
                          f"{question.text}\n        searched: {', '.join(result.searched) or 'no'}"
                          f"\n        {' '.join(result.answer.split())[:220]}")
    print(f"{profile.name}, {embedding.model}: {len(questions)} questions x {args.runs} run(s)")
    print(report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

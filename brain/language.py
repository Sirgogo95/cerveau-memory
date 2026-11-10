"""Which language a message is written in, among the user's languages.

Used to tell the model which language to answer in: in a conversation that mixes
languages, or with memories in another language, a small model drifts to the
language of what it just read (measured 2026-10-17: about one reply in four). A hint
at the end of the user's message fixes it; a system-prompt rule alone doesn't.

Common short words decide, so it needs no model and no dependency. It answers only
when the message is clear; otherwise None, and no hint is given.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

# Frequent words that are rare in the other two languages' running text. Words two
# languages share (la, de, en) count for both and cancel out; words common in both
# but listed for one only ("me") would tip the balance, so they're left out.
COMMON = {
    "English": {
        "the", "is", "are", "was", "and", "you", "your", "what", "how", "did", "do",
        "does", "i", "my", "me", "it", "of", "to", "in", "for", "with", "about", "that",
        "this", "have", "can", "we", "when", "where", "who", "tell", "remind",
    },
    "Spanish": {
        "el", "la", "los", "las", "es", "y", "que", "qué", "tu", "mi", "de", "en", "por",
        "para", "con", "sobre", "cómo", "como", "una", "un", "del", "lo", "hay", "está",
        "fue", "cuándo", "dónde", "quién", "cuál", "cuánto", "dime", "recuerdas", "yo",
        "puedo", "tengo", "voy", "él", "ella",
    },
    "French": {
        "le", "la", "les", "est", "et", "que", "qui", "tu", "mon", "ma", "mes", "de",
        "des", "du", "en", "pour", "avec", "sur", "comment", "une", "un", "je", "j",
        "c", "ce", "cette", "vous", "nous", "quand", "où", "pas", "ai", "dis", "moi",
        "combien", "quel", "quelle", "quels", "pourquoi", "dois", "suis", "il", "elle",
    },
}
MIN_EVIDENCE = 2
_WORD = re.compile(r"[a-zà-ÿñ]+")


def detect_language(text: str) -> str | None:
    """"English", "Spanish", or "French" when the message clearly is; else None."""
    words = _WORD.findall(text.lower())
    counts = {name: sum(word in common for word in words) for name, common in COMMON.items()}
    if "¿" in text or "¡" in text:
        counts["Spanish"] += MIN_EVIDENCE
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top < MIN_EVIDENCE or top == second:
        return None
    return best


def main_language(texts: Sequence[str]) -> str | None:
    """The language most of the user's messages clearly are in; None when none is clear
    or two tie. Consolidation is told it instead of letting the model name it: on the
    user's imported chats, gemma4:e4b wrote 37 of 77 summaries and 16 of 100 facts of
    English conversations in Portuguese (2026-11-06)."""
    counts = Counter(filter(None, map(detect_language, texts)))
    if not counts:
        return detect_language(" ".join(texts))      # short messages: together they may be clear
    ranked = counts.most_common(2)
    if len(ranked) == 2 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]

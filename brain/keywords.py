"""Exact terms: the words in a message that only match literally.

Meaning finds most things, even identifiers, but not always: "INV-2026-00417" asked
about alone missed the message it came from, a long traceback whose vector says
"an error" more than "this invoice". So the distinctive terms of a message are also
looked up literally (SearchStore.keyword_candidates), and a memory containing them
gets the keyword signal (brain/retrieval.py).

A term is either something in quotes ("Royal Canin"), or a word that looks like an
identifier: at least 4 characters with a digit, an underscore, or a dot or slash
between letters or digits (INV-2026-00417, show_full_result_count, 0004,
settings.py). A hyphen alone doesn't count ("follow-up", "e-mail"), and years are
left out: "in 2026" says nothing exact. Ordinary words are left to meaning, so
this never competes with it.
"""

from __future__ import annotations

import re

# At most this many terms per message: a pasted log would otherwise list hundreds.
MAX_TERMS = 8

QUOTED = re.compile(r'["“”«»]([^"“”«»]{3,60})["“”«»]')
WORD = re.compile(r"[\w][\w.\-/]*[\w]")
INNER_SEPARATOR = re.compile(r"\w[./]\w")
YEAR = re.compile(r"(19|20)\d\d")


def is_identifier(word: str) -> bool:
    if len(word) < 4 or YEAR.fullmatch(word):
        return False
    return (any(c.isdigit() for c in word) or "_" in word
            or bool(INNER_SEPARATOR.search(word)))


def keyword_terms(text: str) -> list[str]:
    """The message's exact terms, in order, without repeats (case ignored)."""
    terms = [q.strip() for q in QUOTED.findall(text)]
    terms += [w for w in WORD.findall(text) if is_identifier(w)]
    unique: dict[str, str] = {}
    for term in terms:
        unique.setdefault(term.casefold(), term)
    return list(unique.values())[:MAX_TERMS]

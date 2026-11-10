"""Text normalization and the matching keys shared by the engine and the stores.

A key is lowercase words without accents or punctuation: "Dr. Marta Peña" and
"dr marta pena" both become "dr marta pena". Messages are split into the same words,
so a name matches wherever it appears, however it was typed.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

# A title before a person's name is not a name part: "Dr." alone must not find Dr. Peña.
TITLES = frozenset({"dr", "dra", "mr", "mrs", "ms", "miss", "prof", "sr", "sra", "srta", "don", "dona"})
# Name parts shorter than this don't find an entity on their own (initials, "de", "la").
MIN_PART_LENGTH = 3
# Longest phrase looked up in a message: "Hospital General de Santo Domingo" is five
# words, but four covers nearly every name and keeps the lookup small.
MAX_PHRASE_WORDS = 4


def words(text: str) -> list[str]:
    """Lowercase words, accents removed, punctuation dropped."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.findall(r"\w+", stripped.casefold())


def normalize_key(text: str) -> str:
    """The matching key: "José-María, Jr." becomes "jose maria jr"."""
    return " ".join(words(text))


def contains_phrase(text: str, phrase: str) -> bool:
    """Whether `phrase` appears in `text` as whole words, after normalizing both.
    "Ana" is found in "I met Ana today" but not in "banana"."""
    key = normalize_key(phrase)
    return bool(key) and f" {key} " in f" {normalize_key(text)} "


def phrase_keys(text: str, max_words: int = MAX_PHRASE_WORDS) -> set[str]:
    """Every run of 1 to `max_words` consecutive words in `text`, as keys: what a
    message could be naming. Looked up against entities' search keys."""
    tokens = words(text)
    return {
        " ".join(tokens[i:i + n])
        for n in range(1, max_words + 1)
        for i in range(len(tokens) - n + 1)
    }


def identity_keys(name: str, aliases: Iterable[str] = ()) -> list[str]:
    """Keys that identify an entity: its name and aliases. Used to decide whether an
    extracted entity is one already known (so they merge)."""
    keys = (normalize_key(n) for n in (name, *aliases))
    return list(dict.fromkeys(k for k in keys if k))


def search_keys(name: str, aliases: Iterable[str] = (), type: str = "") -> list[str]:
    """Keys that find an entity in a message: its identity keys, plus, for a person,
    each name part ("Marta" and "Peña" for "Dr. Marta Peña"), since people are
    usually called by part of their name. Never used for merging: two Martas are
    different people. Not for places or projects, where a part is often an ordinary
    word ("Santo Domingo" must not match "domingo", Spanish for Sunday)."""
    keys = identity_keys(name, aliases)
    if type == "person":
        parts = [w for w in words(name) if w not in TITLES and len(w) >= MIN_PART_LENGTH]
        keys += [p for p in parts if p not in keys]
    return keys

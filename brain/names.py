"""Names with typos: "Lucai" finds Lucía, "Almonti" finds Dr. Almonte.

Exact matching (brain.text.phrase_keys + match_entities) misses a name typed with one
wrong, missing, extra, or swapped letter. This finds those, cautiously, because a
wrong match is costly: it puts that person's memories in the prompt ("Ramen" is one
letter from "Ramón"). So, in a message:

- only capitalized words count, and not the first word of a sentence (people
  capitalize names, even misspelled ones);
- only words of at least MIN_FUZZY_LENGTH letters, compared with keys starting with
  the same letter (typos rarely hit the first one);
- at most one edit for names up to 7 letters, two for longer ones.

When a name is given on purpose (the search tool's `entities`), capitalization isn't
required. Pure Python: at personal scale, comparing a few words with every entity's
keys is instant.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .ports import EntityStore
from .text import normalize_key, phrase_keys, search_keys
from .types import Entity

MIN_FUZZY_LENGTH = 4
_TOKEN = re.compile(r"[^\W\d_]+")
_SENTENCE_END = re.compile(r"[.!?¿¡:;\n]\s*$")


def edit_distance(a: str, b: str) -> int:
    """Edits between two words, a swap of two neighbouring letters counting as one
    ("lucai" and "lucia" are 1 apart): optimal string alignment distance."""
    if a == b:
        return 0
    rows = [list(range(len(b) + 1))]
    for i in range(1, len(a) + 1):
        row = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            row[j] = min(rows[-1][j] + 1, row[j - 1] + 1, rows[-1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                row[j] = min(row[j], rows[-2][j - 2] + 1)
        rows.append(row)
    return rows[-1][-1]


def allowed_edits(key: str) -> int:
    return 1 if len(key) <= 7 else 2


def close_enough(word: str, key: str) -> bool:
    """A typo of `key`: same first letter, long enough, within the allowed edits,
    and not the same word (exact matches are found elsewhere)."""
    if word == key or len(word) < MIN_FUZZY_LENGTH or not key or word[0] != key[0]:
        return False
    if abs(len(word) - len(key)) > allowed_edits(key):
        return False
    return edit_distance(word, key) <= allowed_edits(key)


def name_words(text: str, require_capital: bool = True) -> list[str]:
    """The words of `text` that could be names, normalized, plus each pair of
    neighbouring ones (for two-word names): capitalized words not starting a
    sentence, unless `require_capital` is False."""
    picked: list[tuple[int, str]] = []
    for position, match in enumerate(_TOKEN.finditer(text)):
        token = match.group()
        if require_capital:
            starts_sentence = match.start() == 0 or bool(_SENTENCE_END.search(text[:match.start()]))
            if not token[0].isupper() or starts_sentence:
                continue
        picked.append((position, normalize_key(token)))
    found = [w for _, w in picked]
    found += [f"{a} {b}" for (i, a), (j, b) in zip(picked, picked[1:]) if j == i + 1]
    return found


def named_entities(text: str, store: EntityStore, fuzzy: bool = True) -> list[Entity]:
    """The entities a message names: exactly (brain.text.phrase_keys), plus, when
    `fuzzy`, those typed with a typo."""
    exact = store.match_entities(phrase_keys(text))
    if not fuzzy:
        return exact
    return exact + fuzzy_names(text, store.active_entities(), exclude=[e.id for e in exact])


def fuzzy_names(
    text: str, entities: Iterable[Entity], *, require_capital: bool = True,
    exclude: Sequence[int] = (),
) -> list[Entity]:
    """Entities `text` names with a typo, in the order given (none already in
    `exclude`)."""
    candidates = name_words(text, require_capital)
    if not candidates:
        return []
    skip = set(exclude)
    found = []
    for entity in entities:
        if entity.id in skip:
            continue
        keys = search_keys(entity.name, entity.aliases, entity.type)
        if any(close_enough(word, key) for word in candidates for key in keys
               if word.count(" ") == key.count(" ")):
            found.append(entity)
    return found

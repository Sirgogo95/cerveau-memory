"""Which entities a fact is about: the people, places, and projects its statement
names, found with the same keys recall uses for a message (brain/text.py), exact
only (a typo-matched link would put a fact in the wrong person's block).

A fact's source messages also have mentions, but those say who was named in the
message, not who the fact is about ("Caroline: Melanie's pottery is lovely" gives a
fact about Melanie): the statement is what's matched."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .ports import MemoryStore
from .text import phrase_keys
from .types import Entity, Fact, MemoryKind


def link_fact(store: MemoryStore, fact: Fact) -> list[Entity]:
    """Link one fact to the entities its statement names; returns them."""
    entities = store.match_entities(phrase_keys(fact.statement))
    if entities:
        store.link_fact_entities(fact.id, [e.id for e in entities])
    return entities


def without_names(text: str, entities: Sequence[Entity]) -> str:
    """`text` with these entities' names and aliases taken out (and a possessive
    's), so what's left is what the question asks about them."""
    for entity in entities:
        for name in sorted({entity.name, *entity.aliases}, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(name)}(?:'s|\u2019s)?\b", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def link_entity_facts(store: MemoryStore, entities: Iterable[Entity], limit: int = 500) -> int:
    """Link an entity's existing facts (new entities, or new aliases): facts whose
    statement contains its name or an alias literally, confirmed by the same exact
    key match as `link_fact`. Returns the number of links made."""
    made = 0
    for entity in entities:
        names = [entity.name, *entity.aliases]
        for candidate in store.keyword_candidates(MemoryKind.FACT, names, limit):
            [fact] = store.get_facts([candidate.id]) or [None]
            if fact is not None and entity.id in {e.id for e in store.match_entities(phrase_keys(fact.statement))}:
                store.link_fact_entities(fact.id, [entity.id])
                made += 1
    return made

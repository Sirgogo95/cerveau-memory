"""Entity blocks (docs/features/ENTITY_PROFILES.md, step 1): facts linked to the
entities their statements name, a named entity's facts recalled as a block, shown
under its card in date order; off by default."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from adapters.stores.in_memory import InMemoryStore
from brain.config import ConfigError, EngineConfig, EntityBlockConfig, load_engine_config
from brain.consolidation import ExtractedEntity, own_aliases
from brain.entity_links import link_entity_facts, link_fact, without_names
from brain.hashing import text_hash
from brain.prompt import fit_memory
from brain.retrieval import retrieve
from brain.types import Embedding, Entity, MemoryKind
from tests.fakes import FakeEmbedder
from tests.test_store_contract import at

EMBEDDER = FakeEmbedder()


def vector(text: str) -> Embedding:
    return Embedding(EMBEDDER.vector(text), EMBEDDER.model, text_hash(text))


def config(**block: Any) -> EngineConfig:
    base = EngineConfig()
    return replace(base, facts=replace(base.facts, min_score=0.3, max_items=1),
                   messages=replace(base.messages, max_items=0), episodes=replace(base.episodes, max_items=0),
                   entity_block=EntityBlockConfig(**{"enabled": True, **block}))


class Memory:
    def __init__(self):
        self.store = InMemoryStore()
        melanie = self.store.upsert_entity("Melanie", "person", ["Mel"])
        caroline = self.store.upsert_entity("Caroline", "person")
        assert melanie is not None and caroline is not None
        self.melanie: Entity = melanie
        self.caroline: Entity = caroline

    def fact(self, statement: str, minutes: int):
        fact = self.store.add_fact(statement, "person", episode_id=None, message_ids=[], at=at(minutes),
                                   embedding=vector(statement))
        link_fact(self.store, fact)
        return fact


class LinkTests(unittest.TestCase):
    def test_a_fact_links_to_the_entities_its_statement_names(self):
        memory = Memory()
        fact = memory.store.add_fact("Mel and Caroline went hiking", "person", episode_id=None, message_ids=[],
                                     at=at(0), embedding=vector("Mel and Caroline went hiking"))
        linked = link_fact(memory.store, fact)
        self.assertEqual({e.name for e in linked}, {"Melanie", "Caroline"})   # by alias too
        nobody = memory.store.add_fact("The city held a parade", "event", episode_id=None, message_ids=[],
                                       at=at(1), embedding=vector("The city held a parade"))
        self.assertEqual(link_fact(memory.store, nobody), [])

    def test_earlier_facts_link_when_their_entity_appears(self):
        store = InMemoryStore()
        fact = store.add_fact("Jolene adopted a snake named Susie", "person", episode_id=None, message_ids=[],
                              at=at(0), embedding=vector("Jolene adopted a snake named Susie"))
        self.assertEqual(link_fact(store, fact), [])                  # nobody known yet
        jolene = store.upsert_entity("Jolene", "person")
        assert jolene is not None
        self.assertEqual(link_entity_facts(store, [jolene]), 1)
        self.assertEqual([c.id for c in store.entity_fact_candidates([jolene.id], vector("jolene"), 5)], [fact.id])


class AliasTests(unittest.TestCase):
    def test_an_alias_belonging_to_someone_else_is_dropped(self):
        memory = Memory()
        caroline = ExtractedEntity("Caroline", "person", ("Caro", "Mel", "Melanie"))
        melanie = ExtractedEntity("Melanie", "person", ("Mel",))
        # Melanie is already known (with "Mel"): neither her name nor her alias goes to Caroline.
        self.assertEqual(own_aliases(caroline, [caroline], memory.store), ["Caro"])
        # The same within one reply, before either is stored.
        fresh = InMemoryStore()
        self.assertEqual(own_aliases(caroline, [caroline, melanie], fresh), ["Caro"])
        self.assertEqual(own_aliases(melanie, [caroline, melanie], fresh), [])   # "Mel" claimed by both: dropped
        self.assertEqual(own_aliases(ExtractedEntity("Melanie", "person", ("my friend",)), [], memory.store),
                         ["my friend"])

    def test_names_are_taken_out_of_a_question(self):
        memory = Memory()
        self.assertEqual(without_names("What activities does Melanie's family enjoy?", [memory.melanie]),
                         "What activities does family enjoy?")
        self.assertEqual(without_names("Where did Mel go?", [memory.melanie]), "Where did go?")


class RecallTests(unittest.TestCase):
    def test_the_block_is_ranked_partly_without_the_name(self):
        memory = Memory()
        moved = memory.fact("Melanie moved from Sweden", 0)
        for i in range(3):
            memory.fact(f"Melanie Melanie likes Melanie things {i}", i + 1)
        asked, bare = vector("where did melanie move from"), vector("where did move from")
        as_asked = retrieve(asked, memory.store, config(max_items=1, without_name_weight=0.0, min_score_offset=-0.3),
                            entities=[memory.melanie])
        mixed = retrieve(asked, memory.store, config(max_items=1, without_name_weight=1.0, min_score_offset=-0.3),
                         entities=[memory.melanie], entity_query=bare)
        block = lambda found: [s.candidate.id for s in found if s.candidate.parent_id is not None]
        self.assertEqual(block(mixed), [moved.id])
        self.assertEqual(len(block(as_asked)), 1)

    def test_a_named_entity_brings_its_facts_as_a_block(self):
        memory = Memory()
        camping = memory.fact("Melanie went camping with her family", 10)
        pottery = memory.fact("Melanie does pottery with her family", 20)
        memory.fact("Caroline went camping alone", 15)
        query = vector("melanie family activities")
        plain = retrieve(query, memory.store, replace(config(), entity_block=EntityBlockConfig()),
                         entities=[memory.melanie])
        blocked = retrieve(query, memory.store, config(), entities=[memory.melanie])
        plain_facts = [s.candidate.id for s in plain if s.candidate.kind is MemoryKind.FACT]
        block = [s.candidate for s in blocked if s.candidate.kind is MemoryKind.FACT]
        self.assertEqual(len(plain_facts), 1)                          # one slot, as configured
        self.assertEqual({c.id for c in block}, {camping.id, pottery.id})
        self.assertTrue(all(c.parent_id == memory.melanie.id for c in block))
        self.assertEqual(len(block), len({c.id for c in block}))        # found twice, shown once

    def test_the_block_keeps_its_threshold_and_its_cap(self):
        memory = Memory()
        for i in range(4):
            memory.fact(f"Melanie painted picture number {i}", i)
        memory.fact("Melanie likes zebras in winter", 9)
        query = vector("melanie painted picture")
        found = retrieve(query, memory.store, config(max_items=2), entities=[memory.melanie])
        block = [s for s in found if s.candidate.kind is MemoryKind.FACT]
        self.assertEqual(len(block), 2)
        strict = retrieve(query, memory.store, config(min_score_offset=0.0), entities=[memory.melanie])
        self.assertNotIn("zebras", " ".join(s.candidate.text for s in strict))

    def test_only_the_first_entities_named_get_a_block(self):
        memory = Memory()
        memory.fact("Melanie went camping", 0)
        memory.fact("Caroline went camping", 1)
        found = retrieve(vector("camping"), memory.store, config(max_entity_blocks=1),
                         entities=[memory.caroline, memory.melanie])
        owners = {s.candidate.parent_id for s in found
                  if s.candidate.kind is MemoryKind.FACT and s.candidate.parent_id is not None}
        self.assertEqual(owners, {memory.caroline.id})


class ExpansionTests(unittest.TestCase):
    def test_the_second_search_finds_the_rest_of_a_list(self):
        memory = Memory()
        camping = memory.fact("Melanie went camping at the beach", 0)
        swimming = memory.fact("Melanie went swimming at the beach", 1)
        memory.fact("Melanie dislikes winter coats", 2)
        query = vector("what activities does melanie camping")
        one = config(max_items=1, min_score_offset=-0.3)
        plain = retrieve(query, memory.store, one, entities=[memory.melanie])
        expanded = retrieve(query, memory.store, replace(one, entity_block=replace(one.entity_block, expand_seeds=1,
                                                                                    expand_items=2)),
                            entities=[memory.melanie])
        ids = lambda found: {s.candidate.id for s in found if s.candidate.parent_id is not None}
        self.assertEqual(ids(plain), {camping.id})
        self.assertEqual(ids(expanded), {camping.id, swimming.id})             # the coats aren't close to camping


class PromptTests(unittest.TestCase):
    def test_the_block_sits_under_its_card_in_date_order(self):
        memory = Memory()
        later = memory.fact("Melanie does pottery with her family", 20)
        earlier = memory.fact("Melanie went camping with her family", 10)
        found = retrieve(vector("melanie family"), memory.store, config(), entities=[memory.melanie])
        lines = fit_memory(found, 2000, 4.0)
        card = next(i for i, line in enumerate(lines) if line.startswith(f"- [N{memory.melanie.id}]"))
        self.assertTrue(lines[card + 1].startswith(f"  - [F{earlier.id} "))
        self.assertTrue(lines[card + 2].startswith(f"  - [F{later.id} "))


class ConfigTests(unittest.TestCase):
    def test_off_by_default_validated_and_read_from_toml(self):
        self.assertFalse(EngineConfig().entity_block.enabled)
        bad_settings: list[dict[str, Any]] = [{"max_items": -1}, {"min_score_offset": 0.2}, {"max_entity_blocks": -1}]
        for bad in bad_settings:
            with self.assertRaises(ConfigError):
                EntityBlockConfig(**bad)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "c.toml"
            path.write_text("[entity_block]\nenabled = true\nmax_items = 6\n", encoding="utf-8")
            block = load_engine_config(path).entity_block
            self.assertEqual((block.enabled, block.max_items, block.max_entity_blocks), (True, 6, 2))
            path.write_text("[entity_block]\nmin_score = 0.4\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_engine_config(path)


if __name__ == "__main__":
    unittest.main()

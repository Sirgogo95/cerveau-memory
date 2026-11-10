"""brain.activation: neighbours from shared episodes, and one turn of fading,
boosting, and spreading."""

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from adapters.stores.in_memory import InMemoryStore
from brain.activation import FLOOR, neighbours, update_activation
from brain.config import EngineConfig
from brain.types import MemoryKind, Node, Role

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
CONFIG = EngineConfig(activation_turn_decay=0.3, activation_half_life_hours=6.0, activation_spread=0.5)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.tags: dict[str, int] = {}
        # health+injury twice, health+diet once, injury with an entity once.
        self.episode(["health", "injury"])
        self.episode(["health", "injury"])
        self.episode(["health", "diet"])
        self.episode(["injury"], entity="Dr. Almonte")

    def episode(self, names, entity=None):
        episode = self.store.open_episode(T0)
        message = self.store.add_message(episode.id, Role.USER, "something", T0)
        self.store.close_episode(episode.id, T0)
        for name in names:
            found = self.store.upsert_tag(name)
            assert found is not None
            self.tags[name] = found.id
            self.store.tag_episode(episode.id, found.id, 1.0)
        if entity:
            person = self.store.upsert_entity(entity, "person")
            assert person is not None
            self.entity = (MemoryKind.ENTITY, person.id)
            self.store.add_mention(person.id, message.id)

    def node(self, name) -> Node:
        return (MemoryKind.TAG, self.tags[name])

    def test_neighbours_are_weighted_by_shared_episodes(self):
        links = dict(neighbours([self.node("health")], self.store)[self.node("health")])
        # health: episodes 1, 2, 3; injury: 1, 2, 4 -> 2 of 4; diet: 3 -> 1 of 3.
        self.assertAlmostEqual(links[self.node("injury")], 2 / 4)
        self.assertAlmostEqual(links[self.node("diet")], 1 / 3)
        injury = dict(neighbours([self.node("injury")], self.store)[self.node("injury")])
        self.assertIn(self.entity, injury)          # entities are nodes too

    def test_matched_nodes_are_boosted_and_spread_to_neighbours(self):
        values = update_activation({}, {self.node("health"): 1.0}, self.store, T0, CONFIG)
        self.assertEqual(values[self.node("health")], 1.0)
        # health has 2 neighbours: injury gets 0.5 x 0.5 x 1.0 / 2, diet 0.5 x 1/3 / 2.
        self.assertAlmostEqual(values[self.node("injury")], 0.125)
        self.assertAlmostEqual(values[self.node("diet")], 0.5 / 3 / 2)

    def test_values_fade_with_turns_and_hours_and_drop_below_the_floor(self):
        state = {self.node("diet"): (0.8, T0)}
        same_hour = update_activation(state, {}, self.store, T0, CONFIG)
        self.assertAlmostEqual(same_hour[self.node("diet")], 0.8 * 0.7)
        later = update_activation(state, {}, self.store, T0 + timedelta(hours=6), CONFIG)
        self.assertAlmostEqual(later[self.node("diet")], 0.8 * 0.5 * 0.7)
        faded = update_activation({self.node("diet"): (FLOOR, T0)}, {}, self.store, T0, CONFIG)
        self.assertEqual(faded, {})

    def test_values_are_capped_at_one(self):
        state = {self.node("health"): (0.9, T0)}
        no_decay = replace(CONFIG, activation_turn_decay=0)
        values = update_activation(state, {self.node("health"): 1.0}, self.store, T0, no_decay)
        self.assertEqual(values[self.node("health")], 1.0)

    def test_forgotten_nodes_are_not_neighbours(self):
        self.store.set_status(MemoryKind.TAG, self.tags["diet"], "Z")
        links = dict(neighbours([self.node("health")], self.store)[self.node("health")])
        self.assertNotIn(self.node("diet"), links)


if __name__ == "__main__":
    unittest.main()

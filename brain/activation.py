"""Activation (PIPELINE.md step 5, docs/ACTIVATION.md idea C): what is on your mind.

Nodes are tags and entities. Each turn, every node's activation fades (per turn and
over hours), the nodes the message matched (names it said, topics close to it) are
boosted, and part of that boost spreads to their neighbours: nodes that appear in
the same episodes, weighted by how often (Jaccard), divided by how many neighbours
the boosted node has (the fan effect, so a broad tag like `work` can't warm up
everything). Memories whose episode carries an active node get a boost in scoring.

Two simplifications of the README's design, both measured as unnecessary so far:
edges are computed from the episodes when needed instead of kept in a table (so
they always match what is remembered and forgotten), and values are capped at 1
instead of squashed with a sigmoid (which lifts every node to about 0.08, so
nothing would ever fade out). Energy spreads only from the nodes boosted this turn,
so two linked nodes can't keep each other warm forever.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime

from .config import EngineConfig
from .ports import GraphStore
from .types import Node

# Below this, a node has faded from mind and is dropped.
FLOOR = 0.01


def neighbours(nodes: Sequence[Node], store: GraphStore) -> dict[Node, list[tuple[Node, float]]]:
    """Each node's neighbours with their Jaccard weight: episodes with both / episodes
    with either."""
    episodes_of = store.node_episodes(nodes)
    linked = set().union(*episodes_of.values()) if episodes_of else set()
    per_episode = store.episode_nodes(sorted(linked))
    others = {m for found in per_episode.values() for m in found} - set(episodes_of)
    episodes_of = {**episodes_of, **store.node_episodes(sorted(others))}
    result: dict[Node, list[tuple[Node, float]]] = {}
    for node in nodes:
        mine = episodes_of.get(node, set())
        near = {m for e in mine for m in per_episode.get(e, set())} - {node}
        result[node] = sorted(
            ((m, len(mine & episodes_of[m]) / len(mine | episodes_of[m])) for m in near),
            key=lambda pair: pair[1], reverse=True,
        )
    return result


def update_activation(
    state: Mapping[Node, tuple[float, datetime]],
    matched: Mapping[Node, float],
    store: GraphStore,
    now: datetime,
    config: EngineConfig,
) -> dict[Node, float]:
    """The activation after this turn: `state` (value and when it was set) faded,
    `matched` nodes boosted by their match strength, and the boost spread one hop."""
    values: dict[Node, float] = {}
    for node, (value, at) in state.items():
        hours = max((now - at).total_seconds(), 0) / 3600
        values[node] = value * 0.5 ** (hours / config.activation_half_life_hours) * (1 - config.activation_turn_decay)
    for node, strength in matched.items():
        values[node] = min(1.0, values.get(node, 0.0) + strength)
    received: dict[Node, float] = defaultdict(float)
    for node, links in neighbours(list(matched), store).items():
        for other, weight in links:
            received[other] += config.activation_spread * weight * matched[node] / len(links)
    for node, energy in received.items():
        values[node] = min(1.0, values.get(node, 0.0) + energy)
    return {node: value for node, value in values.items() if value >= FLOOR}

"""Tags and entities: the human-readable layer consolidation builds from episodes.

Tags are topics (`programming`, `health`); entities are specific named things (a
person, a project, a place). Both are curated memory: audited, soft-deleted, and
visible to the engine only while active. "Forgotten" (Z) is never auto-reactivated.
"""

from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.db import models

from kernel.base_app.models import SoftDeleteModel


class Tag(SoftDeleteModel):
    API_FIELDS = ("id", "name")
    CHANGE_FIELDS = (("name", "Name"),)

    name = models.CharField(max_length=100)
    # The matching key (brain.text.normalize_key): "Databases" and "databases" are one tag.
    key = models.CharField(max_length=100, unique=True)
    parent = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL, related_name="children"
    )

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    def get_created_text(self):
        return f"Tag created: {self.name}"

    def get_updated_text(self, deactivated=False, reactivated=False, archived=False):
        return f"Tag updated: {self.name}"

    def get_deleted_text(self):
        return f"Tag deleted: {self.name}"


class Entity(SoftDeleteModel):
    API_FIELDS = ("id", "name", "type", "aliases")
    CHANGE_FIELDS = (("name", "Name"), ("type", "Type"), ("aliases", "Aliases"))

    name = models.CharField(max_length=200)
    type = models.CharField(max_length=30, blank=True)
    aliases = ArrayField(models.CharField(max_length=200), default=list, blank=True)
    # Identity keys (brain.text.identity_keys): the name and every alias, normalized.
    # Decide whether an extracted entity is this one.
    match_keys = ArrayField(models.CharField(max_length=200), default=list, blank=True)
    # Search keys (brain.text.search_keys): identity keys plus, for people, name parts.
    # Find this entity when a message names it; never used for merging.
    search_keys = ArrayField(models.CharField(max_length=200), default=list, blank=True)

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "entities"
        indexes = [
            GinIndex(fields=["match_keys"], name="entity_match_keys_gin"),
            GinIndex(fields=["search_keys"], name="entity_search_keys_gin"),
        ]

    def __str__(self):
        return self.name

    def get_created_text(self):
        return f"Entity learned: {self.name}"

    def get_updated_text(self, deactivated=False, reactivated=False, archived=False):
        return f"Entity updated: {self.name}"

    def get_deleted_text(self):
        return f"Entity deleted: {self.name}"


class EpisodeTag(models.Model):
    """Join table: plain model, no audit snapshots."""

    episode = models.ForeignKey("cerveau.Episode", on_delete=models.CASCADE, related_name="episode_tags")
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE, related_name="episode_tags")
    relevance = models.FloatField(default=1.0)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["episode", "tag"], name="uniq_episode_tag")]


class FactEntity(models.Model):
    """Which entities a fact is about (its statement names them), for entity blocks
    (docs/features/ENTITY_PROFILES.md). Derived and rewritten: plain model, no audit."""

    fact = models.ForeignKey("cerveau.Fact", on_delete=models.CASCADE, related_name="entity_links")
    entity = models.ForeignKey(Entity, on_delete=models.CASCADE, related_name="fact_links")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["fact", "entity"], name="uniq_fact_entity")]


class Activation(models.Model):
    """How much a tag or an entity is on your mind (brain/activation.py), and when
    that was set. Rewritten every turn: plain model, no audit snapshots."""

    kind = models.CharField(max_length=10)   # "tag" or "entity"
    node_id = models.BigIntegerField()
    value = models.FloatField()
    updated_at = models.DateTimeField()

    class Meta:
        constraints = [models.UniqueConstraint(fields=["kind", "node_id"], name="uniq_activation_node")]


class EntityMention(models.Model):
    """A message that mentions an entity. Join table: plain model, no audit snapshots."""

    entity = models.ForeignKey(Entity, on_delete=models.CASCADE, related_name="mentions")
    message = models.ForeignKey("cerveau.Message", on_delete=models.CASCADE, related_name="mentions")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["entity", "message"], name="uniq_entity_mention")
        ]


class Fact(SoftDeleteModel):
    """A short, standalone statement the user made (docs/FACTS.md). Curated memory:
    audited, never deleted. An outdated fact points to its replacement through
    `superseded_by` and stays as history; recall only sees active facts that aren't
    superseded. Confirmations are bookkeeping, written with update(), never audited."""

    API_FIELDS = ("id", "statement", "type")
    CHANGE_FIELDS = (("statement", "Statement"), ("type", "Type"), ("superseded_by", "Replaced by"))

    statement = models.TextField()
    type = models.CharField(max_length=30, default="other")
    # Only "user_stated" for now: facts the user said (docs/FACTS.md, decision 3).
    source = models.CharField(max_length=20, default="user_stated")
    episode = models.ForeignKey(
        "cerveau.Episode", null=True, blank=True, on_delete=models.SET_NULL, related_name="facts"
    )
    episode_id: int | None
    sources = models.ManyToManyField("cerveau.Message", blank=True, related_name="facts")
    superseded_by = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL, related_name="replaced"
    )
    superseded_by_id: int | None
    times_confirmed = models.PositiveIntegerField(default=1)
    # Set by the engine (when the fact was learned), not by the database.
    learned_at = models.DateTimeField()
    last_confirmed_at = models.DateTimeField()

    class Meta:
        ordering = ["-learned_at"]

    def __str__(self):
        return self.statement[:80]

    def get_created_text(self):
        return f"Fact learned: {self}"

    def get_updated_text(self, deactivated=False, reactivated=False, archived=False):
        if archived:
            return f"Fact forgotten: {self}"
        if deactivated:
            return f"Fact set aside: {self}"
        return f"Fact updated: {self}"

    def get_deleted_text(self):
        return f"Fact deleted: {self}"

"""Vectors, one table per kind of memory (EMBEDDINGS.md).

Unsized vector columns let models of any dimension share a table; every query
filters on `model`. Plain models: vectors are rewritten on every re-embed and must
never produce audit snapshots. CASCADE removes a memory's vectors, for every model,
when the memory is purged. Entity and skill tables arrive when something searches
them.
"""

from django.db import models
from pgvector.django import VectorField


class SingleEmbeddingBase(models.Model):
    """For kinds that embed one text per row."""

    model = models.CharField(max_length=100)
    embedding = VectorField()
    text_hash = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        abstract = True


class MessageEmbedding(SingleEmbeddingBase):
    message = models.ForeignKey(
        "cerveau.Message", on_delete=models.CASCADE, related_name="embeddings"
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["message", "model"], name="uniq_message_emb")
        ]


class TagEmbedding(SingleEmbeddingBase):
    """A tag's name, embedded: consolidation reuses "programming" for "coding"."""

    tag = models.ForeignKey("cerveau.Tag", on_delete=models.CASCADE, related_name="embeddings")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["tag", "model"], name="uniq_tag_emb")
        ]


class FactEmbedding(SingleEmbeddingBase):
    fact = models.ForeignKey("cerveau.Fact", on_delete=models.CASCADE, related_name="embeddings")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["fact", "model"], name="uniq_fact_emb")
        ]


class EpisodeEmbedding(models.Model):
    """Both summaries of an episode under one model; each has its own hash because
    they are rewritten independently."""

    episode = models.ForeignKey(
        "cerveau.Episode", on_delete=models.CASCADE, related_name="embeddings"
    )
    model = models.CharField(max_length=100)
    summary_short = VectorField(null=True)
    summary_short_hash = models.CharField(max_length=64, blank=True)
    summary_long = VectorField(null=True)
    summary_long_hash = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["episode", "model"], name="uniq_episode_emb")
        ]


class EmbeddingCache(models.Model):
    """Text hash to vector, per model: the same text is never embedded twice.
    Disposable: it can be cleared at any time and refills itself."""

    model = models.CharField(max_length=100)
    text_hash = models.CharField(max_length=64)
    embedding = VectorField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["model", "text_hash"], name="uniq_embedding_cache")
        ]

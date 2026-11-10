"""Cerveau's Django models, one module per area. Import each model here so Django
discovers it."""

from .attachments import (
    Attachment,
    AttachmentChunk,
    AttachmentChunkEmbedding,
    AttachmentMention,
    AttachmentTag,
    MessageAttachment,
)
from .conversation import Episode, ImportedMessage, Message, ToolCall
from .embeddings import EmbeddingCache, EpisodeEmbedding, FactEmbedding, MessageEmbedding, TagEmbedding
from .knowledge import Activation, Entity, EntityMention, EpisodeTag, Fact, FactEntity, Tag

__all__ = [
    "Activation",
    "Attachment",
    "AttachmentChunk",
    "AttachmentChunkEmbedding",
    "AttachmentMention",
    "AttachmentTag",
    "EmbeddingCache",
    "Entity",
    "EntityMention",
    "Episode",
    "EpisodeEmbedding",
    "EpisodeTag",
    "ImportedMessage",
    "FactEntity",
    "Fact",
    "FactEmbedding",
    "Message",
    "MessageAttachment",
    "MessageEmbedding",
    "Tag",
    "TagEmbedding",
    "ToolCall",
]

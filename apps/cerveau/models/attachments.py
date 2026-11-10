"""Files the user shared (docs/ATTACHMENTS.md).

An Attachment is curated memory: audited, soft-deleted, forgettable like episodes and
facts. It's stored once per content hash, on disk under CERVEAU_ATTACHMENTS_DIR
(named by hash, so a crafted filename can't escape the folder), and read in the
background into AttachmentChunk passages. Chunks, their vectors, mentions, tags, and
links are plain models: derived, high volume, never audited.
"""

from django.db import models

from kernel.base_app.models import SoftDeleteModel

from .embeddings import SingleEmbeddingBase


class Attachment(SoftDeleteModel):
    API_FIELDS = ("id", "filename", "media_type", "size_bytes", "title", "extraction_status")
    CHANGE_FIELDS = (("title", "Title"), ("summary_short", "Summary"))

    EXTRACTION_CHOICES = [("pending", "Being read"), ("done", "Read"), ("failed", "Couldn't be read"),
                          ("unsupported", "Not readable")]

    filename = models.CharField(max_length=255)
    media_type = models.CharField(max_length=120)
    size_bytes = models.PositiveBigIntegerField()
    content_hash = models.CharField(max_length=64, unique=True)
    storage_path = models.CharField(max_length=300)   # relative to CERVEAU_ATTACHMENTS_DIR
    added_at = models.DateTimeField(auto_now_add=True)
    # When and where it was first shared: dates its passages in the prompt.
    shared_at = models.DateTimeField(null=True, blank=True)
    episode = models.ForeignKey("cerveau.Episode", null=True, blank=True, on_delete=models.SET_NULL,
                                related_name="attachments")
    episode_id: int | None
    message = models.ForeignKey("cerveau.Message", null=True, blank=True, on_delete=models.SET_NULL,
                                related_name="first_shared")
    message_id: int | None

    title = models.CharField(max_length=200, blank=True)
    summary_short = models.TextField(blank=True)
    summary_long = models.TextField(blank=True)

    extraction_status = models.CharField(max_length=20, choices=EXTRACTION_CHOICES, default="pending")
    extraction_error = models.TextField(blank=True)
    page_count = models.PositiveIntegerField(null=True, blank=True)
    # Written with update() by the background jobs: bookkeeping, never audited.
    read_at = models.DateTimeField(null=True, blank=True)
    summarized_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-added_at"]

    def __str__(self):
        return self.title or self.filename

    @property
    def extension(self) -> str:
        """The file's type, as on its label in the viewer: "PDF", "MD", "DOCX"."""
        name, dot, ext = self.filename.rpartition(".")
        return ext[:4].upper() if dot and name else "FILE"

    def get_created_text(self):
        return f"File shared: {self.filename}"

    def get_updated_text(self, deactivated=False, reactivated=False, archived=False):
        return f"File updated: {self.filename}"

    def get_deleted_text(self):
        return f"File deleted: {self.filename}"


class MessageAttachment(models.Model):
    """A message that shared the file. Join table: plain model."""

    message = models.ForeignKey("cerveau.Message", on_delete=models.CASCADE, related_name="attachment_links")
    attachment = models.ForeignKey(Attachment, on_delete=models.CASCADE, related_name="message_links")
    message_id: int
    attachment_id: int

    class Meta:
        constraints = [models.UniqueConstraint(fields=["message", "attachment"], name="uniq_message_attachment")]


class AttachmentChunk(models.Model):
    """A passage of a file, sized for the embedding model, with its pages and heading."""

    attachment = models.ForeignKey(Attachment, on_delete=models.CASCADE, related_name="chunks")
    attachment_id: int
    position = models.PositiveIntegerField()
    text = models.TextField()
    page_start = models.PositiveIntegerField(null=True, blank=True)
    page_end = models.PositiveIntegerField(null=True, blank=True)
    heading = models.CharField(max_length=300, blank=True)

    class Meta:
        ordering = ["attachment", "position"]
        constraints = [models.UniqueConstraint(fields=["attachment", "position"], name="uniq_chunk_position")]


class AttachmentChunkEmbedding(SingleEmbeddingBase):
    chunk = models.ForeignKey(AttachmentChunk, on_delete=models.CASCADE, related_name="embeddings")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["chunk", "model"], name="uniq_chunk_emb")]


class AttachmentMention(models.Model):
    """A passage that names a known entity."""

    entity = models.ForeignKey("cerveau.Entity", on_delete=models.CASCADE, related_name="attachment_mentions")
    chunk = models.ForeignKey(AttachmentChunk, on_delete=models.CASCADE, related_name="mentions")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["entity", "chunk"], name="uniq_attachment_mention")]


class AttachmentTag(models.Model):
    attachment = models.ForeignKey(Attachment, on_delete=models.CASCADE, related_name="attachment_tags")
    tag = models.ForeignKey("cerveau.Tag", on_delete=models.CASCADE, related_name="attachment_tags")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["attachment", "tag"], name="uniq_attachment_tag")]

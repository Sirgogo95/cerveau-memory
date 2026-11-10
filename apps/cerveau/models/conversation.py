"""Episodes and their messages: the one continuous conversation, segmented."""

from django.db import models

from kernel.base_app.models import SoftDeleteModel


class Episode(SoftDeleteModel):
    """A segment of the conversation. Curated memory: audited, soft-deleted, and only
    visible to the engine while active.

    There is no `last_message_at` column: it changes with every message, so it is
    derived from the messages instead (DJANGO_SETUP.md, "Keep high-churn fields out of
    the audit trail"). The running summary is rewritten every ~10 messages, so it is
    written with update() and kept out of API_FIELDS: never audited.
    """

    API_FIELDS = ("id", "title", "summary_short", "started_at", "ended_at")
    CHANGE_FIELDS = (("title", "Title"), ("summary_short", "Short summary"))

    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(null=True, blank=True)
    title = models.CharField(max_length=200, blank=True)
    summary_short = models.TextField(blank=True)
    summary_long = models.TextField(blank=True)
    consolidated_at = models.DateTimeField(null=True, blank=True)
    # When facts were learned from it (brain/facts.py); null means not yet, so
    # `extract_facts` knows what to backfill. Written with update(), never audited.
    facts_learned_at = models.DateTimeField(null=True, blank=True)
    # The earlier part of a still-running episode, up to and including the message
    # whose id is summarized_until (brain/summary.py).
    running_summary = models.TextField(blank=True)
    summarized_until = models.PositiveBigIntegerField(null=True, blank=True)
    # What the last turns recalled, kept in mind for the next ones (brain/recall.py):
    # [{"kind": "message", "id": 12, "score": 0.61}, ...]. Rewritten every turn with
    # update(), never audited.
    carried = models.JSONField(default=list, blank=True)
    # Drift (brain/segmentation.py): consecutive messages unlike the conversation so
    # far, and the first of them. Written with update(), never audited.
    drift_strikes = models.PositiveSmallIntegerField(default=0)
    drift_pending_message_id = models.PositiveBigIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return self.title or f"Episode {self.pk} ({self.started_at:%Y-%m-%d %H:%M})"

    def get_created_text(self):
        return f"Episode started: {self}"

    def get_updated_text(self, deactivated=False, reactivated=False, archived=False):
        if deactivated:
            return f"Episode set aside: {self}"
        if reactivated:
            return f"Episode reactivated: {self}"
        if archived:
            return f"Episode forgotten: {self}"
        return f"Episode updated: {self}"

    def get_deleted_text(self):
        return f"Episode deleted: {self}"


class Message(models.Model):
    """One turn. High volume and append-only, so a plain model: no audit snapshots."""

    class Role(models.TextChoices):
        USER = "user"
        ASSISTANT = "assistant"

    episode = models.ForeignKey(Episode, on_delete=models.CASCADE, related_name="messages")
    episode_id: int  # declared for type checkers; Django creates it with the FK
    role = models.CharField(max_length=20, choices=Role.choices)
    content = models.TextField()
    # Set by the engine (the time the message arrived), not by the database.
    created_at = models.DateTimeField()
    # Pinned by the user: stays in the prompt after the recent window moves past it.
    pinned = models.BooleanField(default=False)

    class Meta:
        ordering = ["id"]  # insertion order: the conversation's order, whatever the clock
        indexes = [models.Index(fields=["episode", "created_at"])]

    def __str__(self):
        return f"{self.role}: {self.content[:60]}"


class ToolCall(models.Model):
    """One memory-tool call made during a turn (brain/tools.py): what the model asked,
    what it got back, how long it took. A plain append-only log, for debugging and
    for showing past searches on the chat page. Consolidation never reads it."""

    episode = models.ForeignKey(Episode, on_delete=models.CASCADE, related_name="tool_calls")
    # The reply the call led to; null if the turn ended without one.
    message = models.ForeignKey(
        Message, on_delete=models.SET_NULL, null=True, blank=True, related_name="tool_calls"
    )
    message_id: int | None  # declared for type checkers; Django creates it with the FK
    name = models.CharField(max_length=50)
    arguments = models.JSONField(default=dict)
    result = models.JSONField(default=dict)
    duration_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField()

    class Meta:
        ordering = ["id"]
        indexes = [models.Index(fields=["episode", "id"])]

    def __str__(self):
        return f"{self.name}({self.arguments})"


class ImportedMessage(models.Model):
    """A message that came from an imported chat history (brain/importing.py), by the
    export's own ids: a chat already imported is skipped, and an eval question that
    points at an export's message finds the stored one. Bookkeeping: never recalled,
    not audited, deleted with its message."""

    source = models.CharField(max_length=20)            # "chatgpt", "claude"
    conversation_id = models.CharField(max_length=100)
    external_id = models.CharField(max_length=100)      # the message's id in the export
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name="imports")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["source", "external_id"], name="unique_imported_message")]
        indexes = [models.Index(fields=["source", "conversation_id"])]

    def __str__(self):
        return f"{self.source} {self.conversation_id} {self.external_id}"

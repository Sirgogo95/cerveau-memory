"""The admin is the memory inspector: it shows every row, including set-aside (I) and
forgotten (Z) memory, which the engine never sees."""

from django.contrib import admin
from django.db.models import Count

from .models import (
    Attachment,
    AttachmentChunk,
    Entity,
    EntityMention,
    Episode,
    EpisodeTag,
    Fact,
    Message,
    Tag,
    ToolCall,
)


class MessageInline(admin.TabularInline):
    model = Message
    fields = ("created_at", "role", "content")
    readonly_fields = fields
    extra = 0
    can_delete = False
    show_change_link = True


class ToolCallInline(admin.TabularInline):
    """What Cerveau searched during the episode, read-only."""

    model = ToolCall
    fields = ("created_at", "name", "arguments", "duration_ms", "message")
    readonly_fields = fields
    extra = 0
    can_delete = False
    show_change_link = True


class FactInline(admin.TabularInline):
    """The facts learned from the episode, read-only (edit them on their own page)."""

    model = Fact
    fk_name = "episode"
    fields = ("statement", "type", "status", "times_confirmed", "superseded_by")
    readonly_fields = fields
    extra = 0
    can_delete = False
    show_change_link = True


class EpisodeTagInline(admin.TabularInline):
    model = EpisodeTag
    fields = ("tag", "relevance")
    autocomplete_fields = ("tag",)
    extra = 0


@admin.register(Episode)
class EpisodeAdmin(admin.ModelAdmin):
    list_display = (
        "id", "__str__", "status", "started_at", "ended_at", "message_count", "consolidated_at"
    )
    list_filter = ("status", ("consolidated_at", admin.EmptyFieldListFilter))
    search_fields = ("title", "summary_short", "summary_long")
    readonly_fields = (
        "started_at", "ended_at", "consolidated_at", "running_summary", "summarized_until",
        "created_at", "updated_at",
    )
    fields = (
        "title", "summary_short", "summary_long", "status",
        "running_summary", "summarized_until",
        "started_at", "ended_at", "consolidated_at", "created_at", "updated_at",
    )
    inlines = [EpisodeTagInline, FactInline, MessageInline, ToolCallInline]

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(_messages=Count("messages"))

    @admin.display(description="Messages", ordering="_messages")
    def message_count(self, obj):
        return obj._messages


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "episode", "role", "preview", "created_at")
    list_filter = ("role",)
    search_fields = ("content",)
    list_select_related = ("episode",)

    @admin.display(description="Content")
    def preview(self, obj):
        return obj.content[:100]


@admin.register(ToolCall)
class ToolCallAdmin(admin.ModelAdmin):
    """The memory tools' log: each call, its arguments, what it returned."""

    list_display = ("id", "episode", "name", "arguments", "duration_ms", "created_at")
    list_filter = ("name",)
    list_select_related = ("episode",)
    readonly_fields = (
        "episode", "message", "name", "arguments", "result", "duration_ms", "created_at",
    )

    def has_add_permission(self, request):
        return False


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ("name", "status", "parent", "episode_count")
    list_filter = ("status",)
    search_fields = ("name",)
    readonly_fields = ("key", "created_at", "updated_at")

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(_episodes=Count("episode_tags"))

    @admin.display(description="Episodes", ordering="_episodes")
    def episode_count(self, obj):
        return obj._episodes


class EntityMentionInline(admin.TabularInline):
    model = EntityMention
    fields = ("message",)
    readonly_fields = fields
    extra = 0
    can_delete = False


@admin.register(Entity)
class EntityAdmin(admin.ModelAdmin):
    list_display = ("name", "type", "aliases", "status", "mention_count")
    list_filter = ("status", "type")
    search_fields = ("name", "aliases")
    readonly_fields = ("match_keys", "created_at", "updated_at")
    inlines = [EntityMentionInline]

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(_mentions=Count("mentions"))

    @admin.display(description="Mentions", ordering="_mentions")
    def mention_count(self, obj):
        return obj._mentions


@admin.register(Fact)
class FactAdmin(admin.ModelAdmin):
    """Facts (docs/FACTS.md): statement and type can be corrected here; status is
    Forget (Z) / Set aside (I). Replaced facts stay, pointing to their replacement."""

    list_display = ("id", "statement", "type", "status", "times_confirmed", "replaced", "learned_at")
    list_filter = ("status", "type", ("superseded_by", admin.EmptyFieldListFilter))
    search_fields = ("statement",)
    list_select_related = ("superseded_by",)
    fields = (
        "statement", "type", "status", "superseded_by", "episode", "sources",
        "times_confirmed", "learned_at", "last_confirmed_at", "source",
    )
    readonly_fields = (
        "superseded_by", "episode", "sources", "times_confirmed", "learned_at",
        "last_confirmed_at", "source",
    )

    @admin.display(description="Replaced", boolean=True, ordering="superseded_by")
    def replaced(self, obj):
        return obj.superseded_by_id is not None


class AttachmentChunkInline(admin.TabularInline):
    model = AttachmentChunk
    fields = ("position", "pages", "heading", "text")
    readonly_fields = fields
    extra = 0
    can_delete = False

    @admin.display(description="Pages")
    def pages(self, obj):
        if obj.page_start is None:
            return ""
        return obj.page_start if obj.page_end in (None, obj.page_start) else f"{obj.page_start}-{obj.page_end}"


@admin.register(Attachment)
class AttachmentAdmin(admin.ModelAdmin):
    """Shared files (docs/ATTACHMENTS.md): title and summary can be corrected here;
    status is Forget (Z) / Set aside (I). Passages are read-only."""

    list_display = ("id", "filename", "title", "media_type", "extraction_status", "status", "page_count",
                    "shared_at")
    list_filter = ("extraction_status", "status", "media_type")
    search_fields = ("filename", "title", "summary_short")
    fields = ("filename", "title", "summary_short", "summary_long", "status", "media_type", "size_bytes",
              "extraction_status", "extraction_error", "page_count", "shared_at", "episode", "storage_path",
              "read_at", "summarized_at")
    readonly_fields = ("filename", "media_type", "size_bytes", "extraction_status", "extraction_error",
                       "page_count", "shared_at", "episode", "storage_path", "read_at", "summarized_at")
    inlines = [AttachmentChunkInline]

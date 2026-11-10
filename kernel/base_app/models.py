from django.db import models
from collections import Counter

from django.contrib.auth.models import User
from .enums import GeneralStatusTypes, ActionEnum
from django.apps import apps

# ---------------------------------------------------------------------------
# Abstract base — shared audit/soft-delete fields (matches existing convention)
# ---------------------------------------------------------------------------


class ClassProperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func(owner)


class SoftDeleteManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(status="A")


class SoftDeleteModel(models.Model):
    API_FIELDS = ("id",)
    CHANGE_FIELDS = tuple()  # example: (("due_date", "Vencimiento"),)
    status = models.CharField(
        max_length=255, choices=GeneralStatusTypes.choices(), default="A", db_index=True
    )
    updated_at = models.DateTimeField(auto_now=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_created_by",  # <- this makes related_name unique per model
    )
    updated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_updated_by",  # <- this makes related_name unique per model
    )

    objects = models.Manager()
    active_objects = SoftDeleteManager()

    class Meta:
        abstract = True  # <--- This is critical!

    def reactivate(self, user=None):
        self.status = "A"
        if hasattr(self, "updated_by") and user:
            self.updated_by = user
        self.save()

    def set_inactive(self, user=None):
        self.status = "I"
        if hasattr(self, "updated_by") and user:
            self.updated_by = user
        self.save()

    def set_archived(self, user=None):
        """Intentional history (ARCHIVED 'Z'), distinct from a soft-delete (INACTIVE 'I'). Used
        for superseded records: kept for the record, excluded from live (`active_objects`).
        """
        self.status = GeneralStatusTypes.ARCHIVED
        if hasattr(self, "updated_by") and user:
            self.updated_by = user
        self.save()

    def delete(self, *args, **kwargs):
        self.set_inactive(kwargs.get("user", None))

    def save(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        if user:
            if not self.pk and hasattr(self, "created_by"):
                self.created_by = user
            if hasattr(self, "updated_by"):
                self.updated_by = user
        super().save(*args, **kwargs)

    @property
    def is_active(self):
        return self.status == "A"

    @property
    def is_archived(self):
        """Intentionally archived for history (e.g. a superseded version), distinct from a
        soft-deleted (INACTIVE) record. Both are excluded from `active_objects`."""
        return self.status == GeneralStatusTypes.ARCHIVED

    def api_to_dict(self):
        data = {}

        for field in self.API_FIELDS:
            value = getattr(self, field)

            # ManyToMany
            if hasattr(value, "all"):
                data[field] = list(value.all())

            else:
                data[field] = value

        return data

    def serialize_instance(self, serializer_function: callable):
        return serializer_function(self)

    def get_created_text(self):
        raise NotImplementedError("get_created_text must be implemented")

    def get_updated_text(self, deactivated=False, reactivated=False, archived=False):
        raise NotImplementedError("get_updated_text must be implemented")

    def get_deleted_text(self):
        raise NotImplementedError("get_deleted_text must be implemented")

    @classmethod
    def prefetch_settings(cls, queryset):
        return queryset.select_related("created_by", "updated_by")

    @ClassProperty
    def objects_with_prefetch(cls):
        return cls.prefetch_settings(cls.objects)

    @classmethod
    def serialize_queryset(cls, queryset, serializer_function: callable):
        queryset = cls.prefetch_settings(queryset)
        return [item.serialize_instance(serializer_function) for item in queryset]


# ---------------------------------------------------------------------------
# Audit trail  (shared ChangeLog)
# ---------------------------------------------------------------------------


class ChangeLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    action = models.CharField(max_length=10, choices=ActionEnum.choices())
    model_name = models.CharField(max_length=50)
    instance_id = models.PositiveIntegerField()
    previous_data = models.JSONField(null=True, blank=True)
    current_data = models.JSONField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [models.Index(fields=["model_name", "instance_id"])]

    def __str__(self):
        return f"{self.model_name} {self.action} by {self.user}"

    def _model_class(self):
        """The model class this log entry refers to, resolved by class name across all
        installed apps (audited models may live outside base_app). Returns None if unknown.
        If two apps define a model with the same name, an audited (SoftDeleteModel) match
        wins; otherwise the first match is used."""
        matches = [m for m in apps.get_models() if m.__name__ == self.model_name]
        if not matches:
            return None
        for model in matches:
            if issubclass(model, SoftDeleteModel):
                return model
        return matches[0]

    def _get_model_instance(self):
        cls = self._model_class()
        return cls.objects.filter(id=self.instance_id).first() if cls else None

    @staticmethod
    def _fmt_value(value):
        """Readable value for a change line: dates without the midnight suffix the serializer
        appends, empty/None as (vacío)."""
        if value is None or value == "":
            return "(vacío)"
        text = str(value)
        if text.endswith("T00:00:00"):
            return text[: -len("T00:00:00")]
        return text

    def _compute_changes(self):
        """A human-readable, field-level diff for an UPDATE, limited to the model's CHANGE_FIELDS
        allowlist. Each entry is a field name or a (field, label) pair. Returns None when the
        model declares no CHANGE_FIELDS or nothing tracked changed. Values are read from the
        serialized change data, so a field must be in the model's API_FIELDS to be diffable.
        """
        cls = self._model_class()
        if not cls or not self.previous_data or not self.current_data:
            return None
        changes = []
        for spec in getattr(cls, "CHANGE_FIELDS", ()):
            field, label = spec if isinstance(spec, (tuple, list)) else (spec, spec)
            prev = self.previous_data.get(field)
            cur = self.current_data.get(field)
            if isinstance(prev, (list, tuple)) and isinstance(cur, (list, tuple)):
                try:
                    unchanged = Counter(prev) == Counter(cur)
                except TypeError:  # unhashable members (e.g. serialized dicts)
                    unchanged = list(prev) == list(cur)
                if unchanged:
                    continue
            elif prev == cur:
                continue
            changes.append(
                "%s: %s → %s" % (label, self._fmt_value(prev), self._fmt_value(cur))
            )
        return ", ".join(changes) or None

    @property
    def description(self):
        if self.action == ActionEnum.CREATED:
            instance = self._get_model_instance()
            if instance:
                if isinstance(instance, SoftDeleteModel):
                    return instance.get_created_text()
                return f"Created {instance.__str__()}"
            else:
                return f"Created a {self.model_name}"
        elif self.action == ActionEnum.UPDATED:
            instance = self._get_model_instance()
            if instance:
                if isinstance(instance, SoftDeleteModel):
                    if self.previous_data and self.current_data:
                        if (
                            self.previous_data.get("status") == "A"
                            and self.current_data.get("status") == "I"
                        ):
                            return instance.get_updated_text(deactivated=True)
                        if (
                            self.previous_data.get("status") == "I"
                            and self.current_data.get("status") == "A"
                        ):
                            return instance.get_updated_text(reactivated=True)
                    return self._compute_changes() or instance.get_updated_text()
                return f"Updated a {instance.__str__()}"
            else:
                return f"Updated a {self.model_name}"
        elif self.action == ActionEnum.DELETED:
            instance = self._get_model_instance()
            if instance:
                if isinstance(instance, SoftDeleteModel):
                    return instance.get_deleted_text()
                return f"Deleted {instance.__str__()}"
            else:
                return f"Deleted a {self.model_name}"
        else:
            return "Unknown"

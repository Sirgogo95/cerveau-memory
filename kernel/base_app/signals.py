"""Audit trail wiring.

Every domain model (SoftDeleteModel subclass) gets pre_save / post_save / post_delete
receivers that record a ChangeLog entry. Because deletes are soft (delete() -> set_inactive
-> save()), the post_delete hook rarely fires; soft-deletes are captured as "updated" with
the status flipping to "I", and archiving (set_archived, e.g. a superseded record) as
"updated" with the status flipping to "Z". ChangeLog.description reads those transitions
back into human-readable text via each model's get_updated_text().

Connection happens in connect_audit_signals(), called from BaseAppConfig.ready() so the
receivers are registered exactly once at app startup.
"""

from django.apps import apps
from django.db.models.signals import post_delete, post_save, pre_save

from .middleware import get_current_user
from .utilities import log_change

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def capture_changes(sender, instance, **kwargs):
    """pre_save: stash the pre-change row so post_save can diff against it."""
    if kwargs.get("raw", False):
        return
    db = kwargs.get("using") or "default"
    if hasattr(instance, "_original_data"):
        instance._previous_data = instance._original_data
    elif instance.pk is not None:
        instance._previous_data = (
            sender.objects.using(db).filter(pk=instance.pk).first()
        )
    else:
        instance._previous_data = None


def log_save_changes(sender, instance, **kwargs):
    """post_save: write a created/updated ChangeLog entry."""
    if kwargs.get("raw", False):
        return
    db = kwargs.get("using") or "default"
    action = "created" if kwargs.get("created", False) else "updated"
    previous_data = getattr(instance, "_previous_data", None)

    current_user = get_current_user()
    user = current_user if (current_user and current_user.is_authenticated) else None

    log_change(user, action, instance, previous_data, db)


def log_delete_changes(sender, instance, **kwargs):
    """post_delete: write a deleted ChangeLog entry (only fires on a hard delete)."""
    if kwargs.get("raw", False):
        return
    db = kwargs.get("using") or "default"
    previous_data = getattr(instance, "_previous_data", None)

    current_user = get_current_user()
    user = current_user if (current_user and current_user.is_authenticated) else None

    log_change(user, "deleted", instance, previous_data, db)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _audited_models():
    """All domain models to audit: every concrete SoftDeleteModel subclass in any
    installed app, at any inheritance depth (so a model built on an intermediate
    abstract base is still audited). apps.get_models() only yields concrete models,
    and Django matches signal senders by exact class, so abstract bases are skipped.
    ChangeLog itself is a plain Model and is therefore excluded (it would recurse)."""
    from .models import SoftDeleteModel

    return [m for m in apps.get_models() if issubclass(m, SoftDeleteModel)]


def connect_audit_signals():
    """Register the audit receivers for every audited model. Idempotent via dispatch_uid."""
    for model in _audited_models():
        label = model._meta.label_lower
        pre_save.connect(
            capture_changes, sender=model, dispatch_uid=f"audit_pre_{label}"
        )
        post_save.connect(
            log_save_changes, sender=model, dispatch_uid=f"audit_post_{label}"
        )
        post_delete.connect(
            log_delete_changes, sender=model, dispatch_uid=f"audit_del_{label}"
        )

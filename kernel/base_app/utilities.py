import time
from datetime import date, datetime
from decimal import Decimal
from functools import wraps
from django.contrib.auth import get_user_model
from .models import (
    ChangeLog,
    SoftDeleteModel,
)
from django.http import HttpResponseForbidden
from django.core.cache import cache
from django.db import models
from django.db.models.fields.files import FieldFile
from django.forms.models import model_to_dict

from pathlib import Path

User = get_user_model()


def serialize_model_data(instance):

    data = model_to_dict(instance)

    for field_name, value in data.items():
        if hasattr(value, "pk"):
            data[field_name] = value.pk
        elif isinstance(value, list):
            data[field_name] = [item.pk for item in value]

    return data


def serialize_model_data_recursive(instance, serialize_decimal=False, _user_cache=None):
    if _user_cache is None:
        _user_cache = {}

    if isinstance(instance, User):
        pk = instance.pk
        if pk in _user_cache:
            return _user_cache[pk]
        payload = {
            "id": pk,
            "username": instance.username,
            "email": instance.email,
            "first_name": instance.first_name,
            "last_name": instance.last_name,
        }
        _user_cache[pk] = payload
        return payload

    data = instance.api_to_dict()

    if isinstance(instance, SoftDeleteModel):
        created_by_id = getattr(instance, "created_by_id", None)
        updated_by_id = getattr(instance, "updated_by_id", None)

        data["created_by"] = (
            serialize_model_data_recursive(
                instance.created_by, serialize_decimal, _user_cache
            )
            if created_by_id is not None
            else "Unknown"
        )
        data["updated_by"] = (
            serialize_model_data_recursive(
                instance.updated_by, serialize_decimal, _user_cache
            )
            if updated_by_id is not None
            else "Unknown"
        )
        data["created_at"] = instance.created_at.isoformat()
        data["updated_at"] = instance.updated_at.isoformat()
        data["status"] = instance.status

    for field_name, value in data.items():
        if hasattr(value, "pk"):
            data[field_name] = value.pk
        elif isinstance(value, list):
            data[field_name] = [
                serialize_model_data_recursive(item, serialize_decimal, _user_cache)
                for item in value
            ]
        elif isinstance(value, Decimal):
            if serialize_decimal:
                data[field_name] = str(value)
        elif isinstance(value, datetime):
            # Must precede the date branch: datetime is a subclass of date.
            data[field_name] = value.isoformat()
        elif isinstance(value, date):
            data[field_name] = str(value) + "T00:00:00"
        elif isinstance(value, FieldFile):
            if value and getattr(value, "url", None):
                data[field_name] = value.url
            else:
                data[field_name] = value.name or None
    return data


def log_change(user, action, instance, previous_data=None, db_alias="default"):
    model_name = instance.__class__.__name__
    instance_id = instance.pk

    try:
        current_data = serialize_model_data_recursive(instance, serialize_decimal=True)
    except Exception as e:
        current_data = {
            "id": instance_id,
            "model": model_name,
            "error": f"Serialization failed: {str(e)}",
        }

    if previous_data and isinstance(previous_data, models.Model):
        try:
            previous_data = serialize_model_data_recursive(
                previous_data, serialize_decimal=True
            )
        except Exception as e:
            previous_data = {
                "id": instance_id,
                "model": model_name,
                "error": f"Serialization failed: {str(e)}",
            }
    qs = ChangeLog.objects.using(db_alias)

    qs.create(
        user=user,
        action=action,
        model_name=model_name,
        instance_id=instance_id,
        previous_data=previous_data,
        current_data=current_data if action != "deleted" else None,
    )


def throttle_by_ip(limit=5, window=60, ban_time=300):
    """
    Rate-limits a view based on IP.

    Args:
        limit (int): Number of requests allowed in the time window.
        window (int): Time window in seconds for rate limiting.
        ban_time (int): Duration (in seconds) the IP is banned after exceeding the limit.
    """

    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            ip = request.META.get("REMOTE_ADDR", "")
            ban_key = f"throttle:ban:{ip}"
            history_key = f"throttle:history:{ip}"

            # Check if IP is currently banned
            if cache.get(ban_key):
                return HttpResponseForbidden(
                    "You are temporarily banned due to too many requests."
                )

            now = time.time()
            history = cache.get(history_key, [])
            history = [t for t in history if now - t < window]

            if len(history) >= limit:
                # Exceeded limit — ban the IP
                cache.set(ban_key, True, timeout=ban_time)
                cache.delete(history_key)
                return HttpResponseForbidden(
                    "Rate limit exceeded. You are temporarily banned."
                )

            # Update history
            history.append(now)
            cache.set(history_key, history, timeout=window)
            return view_func(request, *args, **kwargs)

        return _wrapped_view

    return decorator


def delete_old_files(path, keep=30, verbose=False):
    folder = Path(path)

    # Get only files (ignore subfolders)
    files = [f for f in folder.iterdir() if f.is_file()]

    # Sort by modification time (oldest first)
    files.sort(key=lambda f: f.stat().st_mtime)

    # Files to delete
    to_delete = files[:-keep] if len(files) > keep else []

    for f in to_delete:
        f.unlink()
        if verbose:
            print(f"Deleted: {f.name}")
    if verbose:
        print(f"Remaining files: {len(files) - len(to_delete)}")

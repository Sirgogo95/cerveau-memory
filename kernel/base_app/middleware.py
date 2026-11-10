import cProfile
import os
import pstats
import threading
import time
import uuid

from django.conf import settings
from django.templatetags.static import static
from django.utils.html import escape

from .utilities import delete_old_files

_thread_locals = threading.local()


def get_current_user():
    return getattr(_thread_locals, "user", None)


class CurrentUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        _thread_locals.user = getattr(request, "user", None)
        try:
            response = self.get_response(request)
        finally:
            if hasattr(_thread_locals, "user"):
                del _thread_locals.user
        return response


profile_lock = threading.Lock()


class CProfileMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request_id = uuid.uuid4().hex[:8]
        # print(
        #     f"[PROFILE ENTER] {request_id} {request.method} {request.get_full_path()}"
        # )

        with profile_lock:
            pr = cProfile.Profile()
            pr.enable()

            try:
                response = self.get_response(request)
                return response

            finally:
                pr.disable()

                # Anchor to the repo root, not CWD, so a process started from
                # elsewhere (cron, an IDE runner) still writes next to manage.py.
                profiles_dir = settings.BASE_DIR / "profiles"
                profiles_dir.mkdir(exist_ok=True)

                ts = time.strftime("%Y%m%d-%H%M%S")
                safe_path = (
                    request.get_full_path()
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace("?", "_")
                    .replace("&", "_")
                    .replace("=", "-")
                )[:120] or "root"

                base = str(profiles_dir / f"{ts}_{request_id}_{safe_path}")
                prof_path = f"{base}.prof"
                txt_path = f"{base}.txt"

                pr.dump_stats(prof_path)

                with open(txt_path, "w", encoding="utf-8") as f:
                    stats = pstats.Stats(prof_path, stream=f)
                    stats.sort_stats(pstats.SortKey.TIME).print_stats(100)

                delete_old_files(profiles_dir, keep=40)
                # print(
                #     f"[PROFILE SAVED] {request_id} {request.get_full_path()} -> {txt_path}"
                # )


# ---------------------------------------------------------------------------
# Dev-only HTML injectors. Both splice a snippet before </body> on HTML responses
# (the trick django-debug-toolbar uses), so nothing needs to be added to templates
# by hand. Each can be scoped to specific URL prefixes:
#   - <NAME>_INCLUDE_PREFIXES: if set, inject ONLY on these path prefixes.
#   - <NAME>_EXCLUDE_PREFIXES: never inject on these (wins over include).
# Both default to skipping the Django admin and allauth.
# ---------------------------------------------------------------------------

# Default path prefixes both injectors skip when their EXCLUDE setting is unset.
_DEFAULT_EXCLUDE_PREFIXES = ("/admin/", "/accounts/")


def _path_allowed(request, include, exclude):
    """Apply the include/exclude prefix filter to request.path."""
    path = request.path
    if include is not None and not any(path.startswith(p) for p in include):
        return False
    return not any(path.startswith(p) for p in exclude)


def _inject_before_body(response, snippet: bytes):
    """Splice `snippet` just before the first </body> and fix Content-Length."""
    body = response.content
    response.content = body.replace(b"</body>", snippet + b"</body>", 1)
    if response.has_header("Content-Length"):
        response["Content-Length"] = str(len(response.content))


def _is_injectable_html(response):
    return (
        not getattr(response, "streaming", False)
        and "text/html" in response.get("Content-Type", "")
        and b"</body>" in getattr(response, "content", b"")
    )


class AnnotationScriptMiddleware:
    """Inject the annotation script before </body> (dev-only, self-disables when
    DEBUG is off). Scope with ANNOTATIONS_INCLUDE_PREFIXES / ANNOTATIONS_EXCLUDE_PREFIXES."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if not settings.DEBUG or not _is_injectable_html(response):
            return response
        if not _path_allowed(
            request,
            getattr(settings, "ANNOTATIONS_INCLUDE_PREFIXES", None),
            getattr(settings, "ANNOTATIONS_EXCLUDE_PREFIXES", _DEFAULT_EXCLUDE_PREFIXES),
        ):
            return response
        _inject_before_body(
            response, f'<script src="{static("annotations.js")}"></script>'.encode()
        )
        return response


class DebugBannerMiddleware:
    """Inject a fixed diagonal corner ribbon on HTML pages when the active environment
    has `debug_banner` on, so a test/staging build is never mistaken for prod. Scope
    with DEBUG_BANNER_INCLUDE_PREFIXES / DEBUG_BANNER_EXCLUDE_PREFIXES."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if not getattr(settings, "DEBUG_BANNER", False):
            return response
        if not _is_injectable_html(response):
            return response
        if not _path_allowed(
            request,
            getattr(settings, "DEBUG_BANNER_INCLUDE_PREFIXES", None),
            getattr(settings, "DEBUG_BANNER_EXCLUDE_PREFIXES", _DEFAULT_EXCLUDE_PREFIXES),
        ):
            return response
        text = escape(getattr(settings, "DEBUG_BANNER_TEXT", "") or "")
        ribbon = (
            '<div style="position:fixed;top:0;right:0;width:150px;height:150px;'
            "overflow:hidden;pointer-events:none;z-index:99998;\">"
            '<div style="position:absolute;top:38px;right:-42px;transform:rotate(45deg);'
            "width:200px;background:linear-gradient(135deg,#fb7185,#e11d48);color:#fff;"
            "text-align:center;font:700 12px/26px system-ui;letter-spacing:.08em;"
            'text-transform:uppercase;box-shadow:0 2px 8px rgba(0,0,0,.35);">'
            f"{text}</div></div>"
        ).encode()
        _inject_before_body(response, ribbon)
        return response

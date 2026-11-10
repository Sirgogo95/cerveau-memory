"""Dev-only UI annotation store.

Receives annotations POSTed from the browser (see `static/annotations.js`) and
writes them to `.annotations/pending.json` at the repo root, keyed by template
origin so re-annotating an element overwrites its stale note (last-write-wins).
Claude reads that file on demand. DEBUG-gated: the route is registered only under
DEBUG (see core/urls.py) and the view refuses to run if DEBUG is off, so nothing
here ships to production.
"""

import json
import os
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path

from django.conf import settings
from django.http import HttpResponseNotFound, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

# Where pending annotations live. Repo-relative, dev-only, gitignored.
STORE = Path(settings.BASE_DIR) / ".annotations" / "pending.json"


def _read():
    if not STORE.exists():
        return {}
    try:
        return json.loads(STORE.read_text() or "{}")
    except json.JSONDecodeError:
        return {}


def _write_atomic(data):
    # Temp file in the same dir, then rename: atomic on one filesystem, so a
    # half-written file is never read (matters under concurrent POSTs).
    STORE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(STORE.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, STORE)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _build_prompt(data):
    """Compose ONE well-structured prompt covering all pending annotations, sent to the
    Claude Code channel. Lives here (not in the channel) so the channel stays a generic
    prompt relay any local app can POST to; this view owns the annotation-specific
    prompt. Each annotation becomes a numbered block with its source, note, element, and
    page, followed by the shared workflow.
    """
    n = len(data)
    cap = 7  # enumerate at most this many inline; the rest are read from the file
    blocks = []
    for i, (key, v) in enumerate(list(data.items())[:cap], start=1):
        loc = (v.get("tpl") or key or "").strip() or "(unknown source)"
        note = " ".join((v.get("note") or "").split()) or "(no note)"
        text = " ".join((v.get("text") or "").split())
        selector = (v.get("selector") or "").strip()
        url = (v.get("url") or "").strip()
        rows = [
            f"{i}. {loc}",
            f"   note:    {note}",
            f"   element: {text[:80] or '(no visible text)'}"
            + (f"   (selector: {selector})" if selector else ""),
        ]
        if url:
            rows.append(f"   page:    {url}")
        blocks.append("\n".join(rows))
    if n > cap:
        blocks.append(
            f"(+{n - cap} more not shown here; read them from .annotations/pending.json)"
        )
    body = "\n\n".join(blocks) if blocks else "(none)"
    return (
        f"The user left {n} UI annotation(s) in the browser (Alt+click notes on rendered "
        "pages) and wants them addressed now. Handle all of them this session, following "
        "AGENTS.md.\n\n"
        f"Annotations ({n}):\n\n{body}\n\n"
        'For each: open the source template from its `tpl` (format "template:line"; if '
        "`tpl` is empty, locate the element from its selector and visible text) and make "
        "the change the note describes. The full records live in .annotations/pending.json, "
        "which is also where you record progress: remove ONLY the keys you handled and "
        "write the file back atomically (temp file + os.replace); never blank the whole "
        "file, since new annotations may arrive while you work. When finished, briefly "
        "report which annotations you handled and which (if any) you skipped and why. Do "
        "not run `manage.py annotations clear`."
    )


def _channel_url():
    """Resolve the channel URL. ANNOTATIONS_CHANNEL_URL enables the push; if the
    channel recorded a live port in `.annotations/channel.port` (it auto-moves off the
    default when that port is busy), target that port instead so we still reach it."""
    url = os.environ.get("ANNOTATIONS_CHANNEL_URL")
    if not url:
        return None
    try:
        live = (STORE.parent / "channel.port").read_text().strip()
    except OSError:
        live = ""
    if live.isdigit():
        parsed = urllib.parse.urlparse(url)
        url = f"{parsed.scheme or 'http'}://{parsed.hostname or '127.0.0.1'}:{live}"
    return url


def _notify_channel(data):
    """Best-effort push to a live Claude Code channel (opt-in, dev only).

    Builds the annotation prompt here and POSTs it to the channel, which just relays
    whatever prompt it receives. Fire-and-forget in a thread so the Send response is
    never blocked; if no channel is running the POST just fails and the file stays the
    source of truth. Enable by setting ANNOTATIONS_CHANNEL_URL and a matching
    ANNOTATIONS_CHANNEL_SECRET. The channel itself is a small standalone MCP server
    run at user scope; it is not vendored in this template (see README).
    """
    url = _channel_url()
    if not url:
        return
    secret = os.environ.get("ANNOTATIONS_CHANNEL_SECRET", "change-me-annotations")
    payload = json.dumps(
        {"prompt": _build_prompt(data), "meta": {"count": str(len(data))}}
    ).encode()

    def _post():
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Channel-Secret": secret,
                    },
                    method="POST",
                ),
                timeout=2,
            )
        except OSError:
            pass  # channel not running / unreachable; ignore

    threading.Thread(target=_post, daemon=True).start()


@csrf_exempt  # dev-only endpoint: DEBUG-gated below and unregistered in production
@require_POST
def add_annotation(request):
    if not settings.DEBUG:
        return HttpResponseNotFound()

    payload = json.loads(request.body or "{}")

    # Accept a batch {"annotations": [record, ...]} (the browser sends the whole queue
    # in one request), a bare list, or a single record (back-compat). Writing once and
    # notifying the channel once keeps a Send of N notes to a single consolidated prompt.
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        incoming = payload["annotations"]
    elif isinstance(payload, list):
        incoming = payload
    else:
        incoming = [payload]

    data = _read()
    for ann in incoming:
        if not isinstance(ann, dict):
            continue
        # Key by template origin; fall back to url+selector when data-tpl is absent.
        key = ann.get("tpl") or f"{ann.get('url', '')}::{ann.get('selector', '')}"
        data[key] = {
            "note": ann.get("note", ""),
            "selector": ann.get("selector", ""),
            "text": ann.get("text", ""),
            "tpl": ann.get("tpl", ""),
            "url": ann.get("url", ""),
            "ts": ann.get("ts"),
        }

    _write_atomic(data)
    _notify_channel(data)
    return JsonResponse({"status": "queued", "pending": len(data)})

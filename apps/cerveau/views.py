"""The chat page and the streaming endpoint behind it."""

from __future__ import annotations

import json
from collections.abc import Iterator

from django.conf import settings
from django.http import HttpRequest, HttpResponseBadRequest, JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_POST
from django.views.generic import TemplateView

from brain.ports import ChatProviderError, EmbedderError
from brain.turn import run_turn, utcnow
from brain.types import Attachment, Message

from .attachments import UploadRefused, read_now, store_upload
from .engine import get_engine
from .tasks import queue_attachment, queue_consolidation, queue_running_summary, summarize_attachment

# How much of the current episode the page shows on load.
PAGE_MESSAGES = 200


class ChatView(TemplateView):
    template_name = "cerveau/chat.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        engine = get_engine()
        episode = engine.store.current_episode()
        context["episode"] = episode
        messages = engine.store.get_messages(episode.id, last=PAGE_MESSAGES) if episode else []
        # What Cerveau searched before each reply, for the page's recall trace (drawn
        # by chat.js from the same arguments as live tool events).
        searched: dict[int, list[dict]] = {}
        for run in engine.store.get_tool_calls(episode.id) if episode else []:
            if run.message_id is not None:
                searched.setdefault(run.message_id, []).append(
                    {"tool": run.name, "arguments": dict(run.arguments)}
                )
        shared = engine.store.message_attachments([m.id for m in messages])
        context["turns"] = [
            {"message": m, "tools": json.dumps(searched[m.id]) if m.id in searched else "",
             "files": shared.get(m.id, [])}
            for m in messages
        ]
        context["model_name"] = engine.profile.name
        context["max_attachment_mb"] = settings.CERVEAU_MAX_ATTACHMENT_MB
        return context


@require_POST
def pin_message(request: HttpRequest, pk: int):
    """Pin a message (pinned=1) or unpin it (pinned=0): a pinned message stays in the
    prompt after the conversation's window moves past it."""
    store = get_engine().store
    if store.get_message(pk) is None:
        return JsonResponse({"error": "Unknown message."}, status=404)
    pinned = request.POST.get("pinned") == "1"
    store.set_pinned(pk, pinned)
    return JsonResponse({"id": pk, "pinned": pinned})


def file_json(attachment: Attachment) -> dict:
    """A file's state for the chat page's chips."""
    return {"id": attachment.id, "filename": attachment.filename, "status": attachment.extraction_status,
            "title": attachment.title, "error": attachment.error}


def file_status(request: HttpRequest, pk: int):
    """A file's reading state, polled by the chat page until it's read."""
    attachment = get_engine().store.get_attachment(pk)
    if attachment is None:
        return JsonResponse({"error": "Unknown file."}, status=404)
    return JsonResponse(file_json(attachment))


def _event(**data) -> bytes:
    return (json.dumps(data) + "\n").encode()


@require_POST
def send_message(request: HttpRequest):
    """Run one turn and stream it as newline-delimited JSON events:

    {"episode": id, "new": bool}  once, when the turn's episode is known
    {"files": [...]}              once, the files the message shared, with their state
    {"notice": "..."}             about a file that wasn't kept (forgotten before)
    {"ids": {"user": id, "reply": id}}  once the reply is stored (so the page can pin them)
    {"delta": "..."}              for each piece of the reply
    {"done": true}                when the reply is complete and stored
    {"error": "..."}              instead of "done" if the model fails

    With think=1 (the page's Think switch, off by default), the turn is answered by the
    chat model with its thinking on (engine.thinking_llm): slower to start, more careful.
    """
    text = request.POST.get("text", "").strip()
    uploads = request.FILES.getlist("files")
    if not text and not uploads:
        return HttpResponseBadRequest("Empty message.")

    engine = get_engine()
    llm, profile = engine.llm, engine.profile
    if request.POST.get("think") == "1" and engine.thinking_llm is not None and engine.thinking_profile is not None:
        llm, profile = engine.thinking_llm, engine.thinking_profile
    # Files are stored before the turn (refused ones stop it, so nothing half-sent
    # is saved); they're linked to the message once it exists.
    stored, forgotten = [], []
    try:
        for upload in uploads:
            result = store_upload(upload.name or "file", upload.read(), engine.store, utcnow())
            (stored.append(result.attachment) if result.attachment else forgotten.append(result.filename))
    except UploadRefused as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    if not text:
        text = "(shared " + ", ".join(u.name or "a file" for u in uploads) + ")"
    files: list[Attachment] = []
    saved: list[Message] = []

    def link(message: Message) -> None:
        saved.append(message)
        for attachment in stored:
            engine.store.link_attachment(message.id, attachment.id)
            if read_now(attachment):
                # Small text: read before the reply, so it can already be quoted.
                summarize_attachment.enqueue(attachment.id)
            elif attachment.extraction_status == "pending":
                queue_attachment(attachment.id)
            files.append(engine.store.get_attachment(attachment.id) or attachment)
    current = engine.store.current_episode()
    previous_id = current.id if current else None

    def events() -> Iterator[bytes]:
        announced = False
        try:
            turn = run_turn(
                text, utcnow(),
                store=engine.store, llm=llm, embedder=engine.embedder,
                profile=profile, config=engine.config,
                on_episode_closed=queue_consolidation,
                on_summary_due=queue_running_summary,
                on_user_message=link,
                reranker=engine.reranker,
            )
            for chunk in turn:
                if not announced:
                    # run_turn has settled the episode before the first chunk.
                    episode = engine.store.current_episode()
                    episode_id = episode.id if episode else None
                    yield _event(episode=episode_id, new=episode_id != previous_id)
                    if files:
                        yield _event(files=[file_json(a) for a in files])
                    for name in forgotten:
                        yield _event(notice=f"You asked Cerveau to forget {name} before, so it wasn't kept.")
                    announced = True
                for call in chunk.tool_calls:
                    # The model is searching its memory: the page can say so.
                    yield _event(tool=call.name, arguments=dict(call.arguments))
                if chunk.content:
                    yield _event(delta=chunk.content)
            if saved:
                current = engine.store.get_message(saved[0].id)
                episode_id = current.episode_id if current else None
                [reply] = engine.store.get_messages(episode_id, last=1) if episode_id else [None]
                yield _event(ids={"user": saved[0].id,
                                  "reply": reply.id if reply and reply.role.value == "assistant" else None})
            yield _event(done=True)
        except (ChatProviderError, EmbedderError) as exc:
            yield _event(error=str(exc))

    return StreamingHttpResponse(events(), content_type="application/x-ndjson")

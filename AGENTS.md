# AGENTS.md

Instructions for any AI coding agent working in this repository. This file is tool-agnostic; tool-specific entry points (for example `CLAUDE.md`) point here so the guidance stays in one place.

## Project

**The Cerveau Project**: a brain-inspired, model-agnostic memory system for an AI assistant (one continuous conversation, automatic episodes, long-term memory in Postgres + pgvector). See [README.md](README.md) for the architecture and [docs/STATUS.md](docs/STATUS.md) for what is built and the decisions made so far.

It is built on the `base_django_setup` template (environment profiles, soft-delete models, audit trail, request profiling, utilities), documented in [docs/base_django_setup.md](docs/base_django_setup.md). How Cerveau uses that template is in [docs/DJANGO_SETUP.md](docs/DJANGO_SETUP.md).

## Architecture (MANDATORY)

- **`brain/`**: all core logic, plus every Protocol the engine depends on (`brain/ports.py`: stores and `ChatProvider`). Pure Python: never import Django, `kernel/`, `apps/`, `adapters/`, or any vendor SDK here.
- **`adapters/`**: pure-Python implementations of `brain/` Protocols (`adapters/llm/` for LLM providers, `adapters/stores/` for `InMemoryStore`). Imports `brain/` only, never Django. Unrelated to `kernel/base_app/adapters.py` (allauth).
- **`apps/cerveau/`**: the single Django app wrapping the brain (models, `DjangoMemoryStore`, views, tasks, admin). May import `brain/`, `adapters/`, `kernel/`. Do not split it into more apps; add modules inside it instead.

When you implement or change something, update [docs/STATUS.md](docs/STATUS.md) in the same commit, and [docs/GUIDE.md](docs/GUIDE.md) where it describes the code you changed.

## Git workflow (MANDATORY)

All git operations in this repository, creating branches, committing, merging, and opening pull requests, MUST follow the rules in [docs/github_workflow.md](docs/github_workflow.md). Read that document before performing any git action. It is not optional.

In short (the linked doc is authoritative):

- Never commit or push directly to `main` or `development`; do work on a `feature/*`, `fix/*`, or `sandbox/*` branch.
- Branch names are lowercase, hyphenated, and prefixed.
- Every commit message starts with a tag (`[ADD]`, `[REF]`, `[DEL]`, `[WIP]`, `[FIX]`, `[DOC]`, `[TEST]`, `[UPDT]`); a commit may carry multiple tags, one tagged line per logical change.
- Pull requests target `development`, not `main`.

When a user instruction conflicts with these rules, point out the conflict and ask before deviating.

## Project structure (MANDATORY)

Templates and static files belong to the app that uses them. Only genuinely shared
assets live at the project root.

- **App-owned**: `<app>/templates/<app_label>/...` and `<app>/static/<app_label>/...`.
  The inner folder repeats the app label on purpose: Django flattens every app's
  template and static directories into one namespace, so without it two apps shipping
  an `index.html` or a `main.css` silently shadow each other. Reference them by the
  namespaced name, e.g. `{% include "invoices/components/row.html" %}` and
  `{% static 'invoices/invoices.css' %}`.
- **Project root** (`templates/`, `static/`): only what more than one app uses, the
  layout skeleton (`base.html`) and shared partials, error pages (`404.html`,
  `500.html`), site-wide CSS, logos, favicons. If exactly one app uses it, it is not
  common: move it into that app.
- **The test is who uses a file, not who serves it.** `static/annotations.js` stays at
  the root even though `AnnotationScriptMiddleware` (part of `base_app`) is what injects
  it, because it is injected into every page of the project. A template only
  `base_app`'s own views render would instead live in
  `kernel/base_app/templates/base_app/`. Kernel apps otherwise follow the same rule as
  domain apps.

Why: an app stays deletable and portable. You can copy `apps/invoices/` into another
project, or delete it from this one, without hunting through root folders for its
leftovers. [apps/cerveau/](apps/cerveau/) follows it: its templates live in
`apps/cerveau/templates/cerveau/` and its assets in `apps/cerveau/static/cerveau/`.

When creating an app, make both folders up front (`<app>/templates/<app_label>/` and
`<app>/static/<app_label>/`), even if empty at first. When you find an app-specific file
at the root, move it into its app rather than adding to the pile.

## UI annotation feedback

Visual feedback on rendered pages is captured to `.annotations/pending.json` (dev-only, gitignored). The user Alt+clicks elements in the browser, adds a note, and hits "Send annotations". It is a JSON object keyed by template origin; each value is a record with the note, the element `selector`, its visible `text`, the page `url`, and a timestamp.

When asked to check annotations (or after making template/UI changes):

1. Read `.annotations/pending.json`. If it is empty (`{}`) or missing, there is nothing pending, say so and stop.
2. Note exactly which keys you are about to address.
3. Use each record's `tpl` (`template:line`) to go straight to the source template; if `tpl` is empty, locate the element on the page at `url` from its `selector` and `text`. Those two fields are there to disambiguate, not to guess from.
4. Make the fixes.
5. Remove ONLY the keys you addressed, do NOT empty the whole file, since new annotations may have arrived while you worked. Write the file back atomically (temp file + `os.replace`).
6. Briefly report which annotations you handled and which (if any) you left, and why.

Never blanket-clear the file (that is the human-only `python manage.py annotations clear`), and never create or commit it in production, it is dev-only.
</content>

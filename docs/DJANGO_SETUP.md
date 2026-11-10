# Cerveau on base_django_setup

How the Cerveau Project maps onto the reusable `base_django_setup` template: what to reuse as-is, what to apply selectively, what to change, and what to add.

See the [root README](../README.md) for the Cerveau architecture itself, and [base_django_setup.md](base_django_setup.md) for the template reference.

---

## 1. Summary

| Template feature | Use in Cerveau | Notes |
|---|---|---|
| Environment system (`core/environments.py`) | Yes | Every environment must use Postgres (pgvector has no SQLite support) |
| `SoftDeleteModel` + audit trail | Selectively | Curated memory only, never high-churn tables |
| `CHANGE_FIELDS` diffs | Yes | Readable history of memory edits (facts, entities, tags) |
| `CurrentUserMiddleware` | Yes, with a change | Switch thread-locals to `contextvars` for async streaming |
| Request profiler | Yes, in dev | Measure retrieval, activation, and LLM latency |
| Cache switching (LocMem / Redis) | Yes | Working-memory cache and priming |
| DB cloning | Yes | Snapshot memory before experiments and eval runs |
| Debug banner | Yes | Unchanged |
| UI annotation feedback | Yes, in dev | Useful while building the chat UI |
| Social authentication | No | Single-user local assistant; leave all providers `None` |
| Utilities (serialization, throttling) | As needed | Throttling is useful if the API is ever exposed |

---

## 2. Layout

The template uses `core/` for the Django project package (settings, environments, urls), so Cerveau's pure-Python engine is named **`brain/`**. Around it sit two shells: pure-Python `adapters/` and a single Django app, `apps/cerveau/`.

```
cerveau/
├── core/               # Django project (template): settings, environments, urls
├── kernel/             # template: base_app with SoftDeleteModel, ChangeLog, middleware
├── brain/              # Cerveau's pure-Python engine, no Django or SDK imports
│   ├── types.py
│   ├── ports.py        # every Protocol: stores, Embedder, ChatProvider
│   ├── turn.py         # run_turn, the pipeline's spine
│   ├── segmentation.py
│   ├── prompt.py
│   ├── retrieval.py
│   ├── activation.py
│   ├── hashing.py
│   └── config.py       # EngineConfig, SourceConfig, ModelProfile
├── adapters/           # pure-Python implementations of brain's Protocols
│   ├── llm/            # one ChatProvider per LLM provider
│   ├── embeddings/     # Embedder implementations
│   └── stores/
│       └── in_memory.py  # InMemoryStore for tests and evals
├── apps/
│   └── cerveau/        # the single Django app
│       ├── models/     # package, one module per area (episodes, facts, tags...)
│       ├── store.py    # DjangoMemoryStore
│       ├── tasks.py    # consolidation jobs
│       ├── views/      # chat UI, streaming
│       ├── admin.py    # memory inspector
│       ├── templates/cerveau/
│       └── static/cerveau/
├── evals/              # LoCoMo runner, personal eval set
├── configs/            # engine presets per model
└── tests/
```

### The dependency rule

```
brain/          imports nothing from adapters/, apps/, kernel/, or Django
adapters/       imports brain/ only
apps/cerveau/   may import brain/, adapters/, and kernel/
kernel/         imports nothing from Cerveau
```

`brain/` talks to storage and LLMs only through the Protocols in `brain/ports.py`. `apps/cerveau/store.py` implements the store Protocols with the Django ORM; `adapters/` holds the implementations that need no Django (LLM providers, `InMemoryStore`), so tests and evals can run without starting Django. This is what keeps the engine testable and portable.

### Why one Django app

A Django app is a unit of reuse: something you could drop into another project or delete on its own. Chat, memory, and consolidation are neither: chat reads memory, consolidation writes it, and all three exist only to wrap `brain/`. Splitting them would add cross-app imports and migration dependencies without real independence. Structure inside the app comes from modules (a `models/` package, `store.py`, `tasks.py`, `views/`). Split out a second app only if a piece becomes separately deployable or reusable.

The root `adapters/` package is unrelated to `kernel/base_app/adapters.py`, which holds the allauth social-account adapters.

---

## 3. Models: which base class

The template's audit signals write a `ChangeLog` row with a full before/after snapshot on **every** save of a `SoftDeleteModel`. That is valuable for memory the user curates and edits, and harmful for tables that change every turn.

### Inherit `SoftDeleteModel` (curated memory)

| Model | Why | Suggested `CHANGE_FIELDS` |
|---|---|---|
| `Fact` | User-visible, edited, superseded | `statement`, `type`, `confidence` |
| `Entity` | Names and aliases get corrected | `name`, `type`, `aliases` |
| `Tag` | Renamed, merged, re-parented | `name`, `parent` |
| `Skill` | Edited occasionally | `name`, `description`, `file_path` |
| `Episode` | Titles and summaries get corrected | `title`, `summary_short` |
| `PersonalityTrait` (later) | The `ChangeLog` doubles as the trait history | `value`, `locked` |

Using `ChangeLog` for personality traits removes the need for the separate `personality_history` table described in the README.

### Plain `models.Model` (high churn or join tables)

| Model | Why |
|---|---|
| `Message` | High volume, append-only |
| `Activation` | Rewritten every turn |
| `TagEdge` | Weights updated at every consolidation |
| `EpisodeTag`, `FactTag`, `EntityMention`, `EpisodeLink` | Join tables |
| `EpisodeEmbedding`, `MessageEmbedding`, `FactEmbedding`, `EntityEmbedding`, `TagEmbedding`, `SkillEmbedding` | Vectors, rewritten on every re-embed ([EMBEDDINGS.md](EMBEDDINGS.md)) |
| `EmbeddingCache` | Pure cache |

Prefer `bulk_create` / `bulk_update` for activation and edge updates. Bulk operations skip signals, which is the desired behavior here.

### Vectors live in their own tables

Content models carry no `VectorField`. Embeddings are stored in one table per kind of memory, with a real foreign key and `ON DELETE CASCADE`, so several embedding models can coexist and purging a memory removes its vectors ([EMBEDDINGS.md](EMBEDDINGS.md)). A side benefit: audit snapshots of curated memory never contain hundreds of floats.

```python
class Fact(SoftDeleteModel):
    API_FIELDS = ("id", "statement", "type", "source", "confidence", "superseded_by")
    CHANGE_FIELDS = (("statement", "Statement"), "type", "confidence")

    statement = models.TextField()
    type = models.CharField(max_length=50, blank=True)
    source = models.CharField(max_length=20)
    confidence = models.FloatField(default=0.5)
    source_episode = models.ForeignKey("Episode", null=True, on_delete=models.SET_NULL)
    superseded_by = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL)
    times_confirmed = models.PositiveIntegerField(default=1)
    last_confirmed_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    def get_created_text(self):
        return f"Fact learned: {self.statement}"

    def get_updated_text(self, **kw):
        return f"Fact updated: {self.statement}"

    def get_deleted_text(self):
        return f"Fact removed: {self.statement}"
```

### Keep high-churn fields out of the audit trail

Every `save()` of a `SoftDeleteModel` runs an extra query to load the previous row and writes a full before/after `ChangeLog` snapshot. Some fields on curated models change far more often than anyone edits them:

| Field | Changes | Handling |
|---|---|---|
| `Episode.last_message_at` | Every message | Not a column: derived from the episode's messages (latest `created_at`, indexed) |
| `Episode.drift_strikes`, `drift_pending_message_id` (v2) | Most messages | `Episode.objects.filter(pk=...).update(...)` |
| `Episode` running summary and the message it covers | Every ~10 messages | `QuerySet.update()` |
| `Fact.last_used_at`, `times_confirmed`, use counters (v2) | Every turn the fact is used | `QuerySet.update()`, or `F()` expressions for counters |

`QuerySet.update()` writes straight to the database without calling `save()`, so no signals fire and no snapshot is written. Keep `save()` for real edits: title and summary corrections, renames, status changes. That is what the audit trail is for.

### Visibility: the AI only ever sees active memory

- **`objects` stays all rows** (the template default). The admin shows everything, including inactive and archived memory, so anything can be inspected and restored.
- **`DjangoMemoryStore` is the single choke point.** Every retrieval method uses `active_objects`, and for facts also `superseded_by__isnull=True`. Since `brain/` can only reach data through the store, the engine never sees inactive memory.
- **Tests lock it in**: assert that inactive, archived, and superseded rows are never returned by any store method.

### Status meanings

The template's status lifecycle gets a precise meaning in Cerveau:

| Status | Meaning | Auto-reactivated? |
|---|---|---|
| `A` | Active, visible to the AI | n/a |
| `I` | Set aside **by the system** (merged, replaced, cleaned up, or deactivated in the admin) | Yes, if it comes up again |
| `Z` | **Forgotten by the user** | **Never.** Consolidation skips it and may flag it in the admin |

`Z` makes the user's intent explicit in the data: deliberately forgotten memory can't be silently resurrected when the topic comes up again in passing.

### Reactivate on create (status-aware upsert)

Creating a tag, entity, or skill that already exists reuses the existing row instead of hitting the unique constraint. The rule lives in the store's upsert, in one place:

```python
def upsert_tag(self, name):
    tag = Tag.objects.filter(name=name).first()
    if tag is None:
        return Tag.objects.create(name=name)
    if tag.status == "Z":
        return None
    if tag.status == "I":
        tag.reactivate()
    return tag
```

- **Entities**: same rule, matching on name and aliases.
- **Facts**: during dedup, newly extracted facts are also compared against `Z` facts. A match above the duplicate threshold (0.92) is dropped instead of saved, so forgotten facts aren't re-learned.
- **Callers** treat `None` as "skip": don't tag, link, or mention that node.

### Removal summary

| Situation | Mechanism | Reversible? |
|---|---|---|
| A fact is outdated | Supersede it (`superseded_by`) | History kept |
| Memory is wrong, merged, or cleaned up | `set_inactive()` → `I` | Yes, and may auto-reactivate |
| The user asks Cerveau to **forget** | `set_archived()` → `Z` | Yes, but only manually |
| The user wants it **erased** | `purge_memory` command | No |

### The purge command

Soft delete and archiving don't erase anything: the row stays, and `ChangeLog` snapshots keep copies of its content. Real erasure is a management command:

```bash
python manage.py purge_memory --status I Z --older-than 30 --dry-run
python manage.py purge_memory --status Z --older-than 0
```

It should:

1. **Hard-delete** matching rows, bypassing the soft-delete `delete()` override. Confirm how the template's queryset-level delete behaves before relying on it.
2. **Delete their `ChangeLog` rows**, which hold copies of the content.
3. **Clear related rows**: activation, entity mentions, join tables, embedding cache entries, and cache keys. Embedding rows go automatically through `ON DELETE CASCADE`.
4. **Handle derived memory**: facts whose only source was a purged episode are purged too (or marked for review).
5. Support **`--older-than`** as a grace period and **`--dry-run`** to list what would be erased.

Note: database clones and backups (including the `eval` database) keep their own copies; purge does not reach them.

**As built (2026-10-18):** [apps/cerveau/purge.py](../apps/cerveau/purge.py) and the `purge_memory` command, as above. Derived facts are purged (not marked for review), along with the history of every purged fact (otherwise an older fact it replaced would come back as current), and tool calls in other episodes whose stored results name a purged memory. Default: status Z, 30 days. Confirmation by typing `purge`, or `--no-input`. Since 2026-11-09 every purge also empties the whole embedding cache: some cached texts (search queries) are stored nowhere else, so their entries can't be matched to what was purged.

---

## 4. Environments

pgvector requires Postgres, so **every** environment, including dev, sets `database_engine="postgresql"`.

```python
dev_environment = Environment(
    name="dev",
    secret_key=os.environ.get("SECRET_KEY", "CHANGE_ME"),
    allowed_hosts=["*"],
    database_engine="postgresql",
    database_url=os.environ.get("DATABASE_URL__DEV"),
    activate_profiler=True,
    oidc_client=None,
)

eval_environment = Environment(
    name="eval",
    secret_key=os.environ.get("SECRET_KEY", "CHANGE_ME"),
    allowed_hosts=["*"],
    database_engine="postgresql",
    database_url=os.environ.get("DATABASE_URL__EVAL"),
    activate_profiler=False,
    port=8010,
    oidc_client=None,
)
```

The separate `eval` environment, on its own port and database, lets benchmark runs and experiments happen without touching real memory.

For local Postgres with pgvector preinstalled, use the official pgvector Docker image; the exact command is under "Development setup" in the [root README](../README.md#development-setup).

### Cerveau-specific settings

Keep engine tuning out of the `Environment` dataclass; it changes far more often than deployment settings. Point to a config file instead:

```ini
# .env
CERVEAU_ENGINE_CONFIG=configs/default.toml
CERVEAU_EMBEDDING_MODEL=all-MiniLM-L6-v2
CERVEAU_CHAT_MODEL=
CERVEAU_BACKGROUND_MODEL=
```

`brain/config.py` loads `EngineConfig` and `ModelProfile` from the file. Only the path lives in settings.

---

## 5. Postgres extensions

Add `django.contrib.postgres` to `INSTALLED_APPS`, and enable the extensions in the first `cerveau` migration:

```python
from django.contrib.postgres.operations import TrigramExtension, UnaccentExtension
from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        VectorExtension(),
        TrigramExtension(),
        UnaccentExtension(),
    ]
```

Trigram indexes go on the models' `Meta.indexes` (`GinIndex` with `opclasses=["gin_trgm_ops"]` from `django.contrib.postgres.indexes`). Vector columns are unsized, so v1 has no vector index; when searches slow down, add a partial HNSW index per embedding model on a cast expression, as described in [EMBEDDINGS.md section 6](EMBEDDINGS.md#6-indexing).

---

## 6. Middleware

### `CurrentUserMiddleware`: switch to `contextvars`

Thread-local storage is unreliable in async views, which the chat needs for streaming responses. A `ContextVar` works for both sync and async code:

```python
from contextvars import ContextVar

_current_user: ContextVar = ContextVar("current_user", default=None)


def get_current_user():
    return _current_user.get()


class CurrentUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        token = _current_user.set(getattr(request, "user", None))
        try:
            return self.get_response(request)
        finally:
            _current_user.reset(token)
```

An async-capable version also needs Django's `sync_and_async_middleware` handling. This change is worth upstreaming into the template itself.

Background consolidation jobs have no request, so `created_by` / `updated_by` will be empty for memory the system writes on its own. That is useful information in itself: it separates system-learned memory from user edits in the audit trail.

### Request profiler

Keep `activate_profiler=True` in dev. The stages worth watching are embedding, signal gathering, activation, candidate scoring, prompt assembly, and the LLM call.

---

## 7. Cache

| Cache | Where it lives |
|---|---|
| Embedding cache | Postgres `EmbeddingCache` table (durable, survives restarts) |
| Working-memory cache | Django cache (LocMem in dev, Redis in prod) |
| Priming prefetch | Django cache, short TTL |
| Provider prompt caching | At the LLM provider, driven by prompt ordering |

Include a version or `updated_at` in every cache key so rewritten summaries and superseded facts are never served stale:

```python
def episode_cache_key(episode):
    return f"episode:{episode.id}:{int(episode.updated_at.timestamp())}"
```

---

## 8. DB cloning for experiments

The template's cloning service fits Cerveau's evaluation workflow:

- Clone real memory into the `eval` database before trying a new engine config.
- Snapshot before changing consolidation logic, then compare the resulting facts and tags.
- Keep a frozen clone as a regression baseline for the personal eval set.

---

## 9. Additions the template doesn't provide

| Need | Suggested addition |
|---|---|
| Vector search | `pgvector` |
| Postgres driver | `psycopg` |
| Local embeddings | `sentence-transformers` |
| Background jobs | Django 6's `django.tasks` API with the `django-tasks-db` backend (jobs in Postgres, run by `python manage.py db_worker`); see [PIPELINE.md](PIPELINE.md#queuing-v1) |
| LLM providers | The provider SDKs you use, behind one interface in `adapters/llm/` |
| Token counting | Provider tokenizer or counting endpoint, with a chars ÷ 4 fallback |
| Config files | `tomllib` (standard library) for `configs/*.toml` |

LLM providers follow the same pattern as stores: a `ChatProvider` Protocol in `brain/ports.py`, one implementation per provider in `adapters/llm/`, so Cerveau stays model-agnostic. Each adapter imports its SDK inside its own module, so `brain/` never depends on any vendor package.

---

## 10. Setup checklist

1. Clone `base_django_setup` and rename the project to `cerveau`.
2. Create the `brain/`, `adapters/llm/`, `adapters/stores/`, `evals/`, and `configs/` directories, and the `apps/cerveau/` app with its `templates/cerveau/` and `static/cerveau/` folders.
3. Configure `dev` and `eval` environments with Postgres; disable social auth.
4. Start Postgres with pgvector.
5. Add `pgvector`, `psycopg`, `sentence-transformers`, and a task queue to `requirements.txt`.
6. Add `django.contrib.postgres` to `INSTALLED_APPS`.
7. Add the extensions migration as the first `cerveau` migration.
8. Write the v1 models, choosing `SoftDeleteModel` or plain `Model` per the table in section 3.
9. Register curated models in the admin; this is the memory inspector.
10. Switch `CurrentUserMiddleware` to `contextvars`.
11. Write `brain/ports.py` and `apps/cerveau/store.py` (`DjangoMemoryStore`).
12. Write `adapters/stores/in_memory.py` and the first tests for `brain/`.
13. Build the status-aware upserts (reactivate `I`, never `Z`) and the `purge_memory` command.
14. Start the personal eval set.

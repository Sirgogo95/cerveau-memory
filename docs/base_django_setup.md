# base_django_setup

> Reference for the Django starter template that Cerveau is built on. For the Cerveau project itself, see the [root README](../README.md); for how Cerveau uses this template, see [DJANGO_SETUP.md](DJANGO_SETUP.md).

A pre-wired Django 6 starter template with the boilerplate I reach for on every project already built in: environment configs, an audit trail, soft-delete models, per-request profiling, IP throttling, optional social authentication (OIDC lockdown, or Google and Microsoft sign-in), and a database cloning tool. Clone it, rename the project, and start building the domain instead of re-writing plumbing.

## What's inside

| Area | Location | What it gives you |
|------|----------|-------------------|
| **Environment system** | [core/environments.py](../core/environments.py) | Named, validated environment configs (dev / staging / prod) selected via one env var |
| **Soft-delete base model** | [kernel/base_app/models.py](../kernel/base_app/models.py) | `SoftDeleteModel` with audit fields, status lifecycle, and serialization helpers |
| **Audit trail** | [kernel/base_app/models.py](../kernel/base_app/models.py), [kernel/base_app/signals.py](../kernel/base_app/signals.py) | Automatic `ChangeLog` entries on every create / update / delete |
| **Current-user tracking** | [kernel/base_app/middleware.py](../kernel/base_app/middleware.py) | Thread-local access to the request user from anywhere (signals, services) |
| **Request profiler** | [kernel/base_app/middleware.py](../kernel/base_app/middleware.py) | Per-request cProfile dumps, toggled per environment |
| **Social authentication** | [core/environments.py](../core/environments.py), [core/settings.py](../core/settings.py), [kernel/base_app/adapters.py](../kernel/base_app/adapters.py) | Optional allauth login, OIDC (SSO lockdown) or Google / Microsoft (extra logins), per environment |
| **Utilities** | [kernel/base_app/utilities.py](../kernel/base_app/utilities.py) | Recursive serialization, IP throttling decorator, old-file cleanup |
| **DB cloning** | [kernel/base_app/services/clone_db.py](../kernel/base_app/services/clone_db.py) | Copy one database into another (Django or native `pg_dump`) |
| **Cache switching** | [core/settings.py](../core/settings.py) | LocMem in dev, Redis in prod, automatically |
| **Debug banner** | [kernel/base_app/middleware.py](../kernel/base_app/middleware.py) | Diagonal corner ribbon auto-injected on non-prod environments so you never confuse test and prod |
| **UI annotation feedback** (dev) | [kernel/base_app/views/annotations.py](../kernel/base_app/views/annotations.py), [static/annotations.js](../static/annotations.js) | Alt+click elements to leave notes (auto-tagged with source template:line) that Claude reads, or pushes into a live session |

## Requirements

- Python 3.12+ (uses `StrEnum`, `slots=True` dataclasses)
- Django 6.0
- `python-dotenv`, `dj-database-url`
- Optional: `django-allauth[socialaccount]` (OIDC login), `django-redis` + a Redis server (prod cache), `psycopg` (PostgreSQL)

All pinned dependencies live in [requirements.txt](../requirements.txt).

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (see below)
#    copy .env.example to .env and fill it in

# 4. Migrate and run
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

### Environment (`.env`)

Copy [.env.example](../.env.example) to `.env` and fill it in:

```ini
DJANGO_ENV=default          # which environment to activate
SECRET_KEY=your-secret-key      # required (non-debug environments validate this)
REDIS_URL=                      # optional; leave blank for LocMem cache
# DATABASE_URL=postgres://...   # only if an environment uses a non-sqlite engine

# OIDC (only needed when an environment enables oidc_client)
OIDC_PROVIDER_ID_DEV=oidc
OIDC_NAME_DEV=Example SSO
OIDC_CLIENT_ID_DEV=
OIDC_CLIENT_SECRET_DEV=
OIDC_SERVER_URL_DEV=http://localhost:8000/.well-known/openid-configuration
```

## The environment system

Instead of juggling `if DEBUG:` branches and a pile of environment variables, this template centralizes every environment difference into an `Environment` dataclass. `settings.py` reads from whichever environment is active:

```python
# core/settings.py
ACTIVE = env_registry.active_environment
ACTIVE.validate()
SECRET_KEY = ACTIVE.secret_key
DEBUG = ACTIVE.debug
DATABASES = ACTIVE.as_django_settings()["DATABASES"]
```

An environment carries everything that changes between environments: `secret_key`, `debug`, `allowed_hosts`, `csrf_trusted_origins`, `timezone`, the database engine and URL, the debug banner, the dev-server `port`, whether the request profiler is on, and the optional OIDC client.

### Debug banner

When an environment sets `debug_banner=True` (default in dev), `DebugBannerMiddleware` auto-injects a fixed diagonal corner ribbon (text from `debug_banner_text`, default "🐞 TESTING 🐞") into every HTML page, so a test or staging build is never mistaken for prod. No template markup needed. Scope it with `DEBUG_BANNER_INCLUDE_PREFIXES` / `DEBUG_BANNER_EXCLUDE_PREFIXES` (path prefixes); by default it skips `/admin/` and `/accounts/`. Turn it off per environment with `debug_banner=False`.

### Dev-server port

Each environment sets a `port` (default `8000`). It flows to `settings.APP_DEFAULT_PORT`, and [manage.py](../manage.py) uses it as the `runserver` default port, so `python manage.py runserver` with no address argument binds to the active environment's port. This lets several environments (or several projects) run side by side without colliding. Passing an explicit port still overrides it, e.g. `python manage.py runserver 9000`.

### Defining environments

Environments are declared at the bottom of [core/environments.py](../core/environments.py). One default environment ships enabled, and a commented production example shows the full shape:

```python
default_environment = Environment(
    name="default",
    secret_key=os.environ.get("SECRET_KEY", "CHANGE_ME"),
    allowed_hosts=["*"],
    activate_profiler=True,
    oidc_client=None,          # OIDC off by default; see below to enable
)

prod_environment = Environment(
    name="prod",
    debug=False,
    debug_banner=False,
    secret_key=os.environ.get("SECRET_KEY", "CHANGE_ME"),
    database_engine="postgresql",
    database_url=os.environ.get("DATABASE_URL__PROD"),
    csrf_trusted_origins=["https://myapp.example.com"],
    allowed_hosts=["myapp.example.com"],
    oidc_client=oidc_prod_settings,
)

env_registry.add_environments([default_environment, prod_environment])
env_registry.default_environment = "default"
env_registry.select_from_env()   # DJANGO_ENV picks the active one
```

`select_from_env()` reads `DJANGO_ENV`, so switching environments is a one-line change in `.env` (or a real environment variable on the host). `EnvironmentRegistry` is a singleton, so the active environment is consistent across the process.

### Validation

`Environment.validate()` runs at settings import time and fails fast on unsafe or incomplete combinations:

- a placeholder `SECRET_KEY` outside debug
- a `CSRF_TRUSTED_ORIGINS` entry missing its scheme
- a non-sqlite engine with no database URL
- OIDC enabled but missing `client_id`, `client_secret`, or `server_url`

### Dual-database support

Each environment can define a `target_database_url` (or `target_local=True`) in addition to its main database. This wires up a second `"target"` connection that the cloning tool uses.

## Social authentication (optional)

The template ships optional social login built on [django-allauth](https://docs.allauth.org/). It is **off by default** so the bare template boots without any allauth configuration, and it is gated entirely on environment fields: set `oidc_client`, `google_client`, or `microsoft_client` on the active environment and the matching stack turns on.

Three providers are supported in two modes. **OIDC** is a lockdown and is exclusive: `Environment.validate()` raises if it is combined with anything else. **Google** and **Microsoft** are additional login buttons and may be enabled together. The modes model two different intents:

| | **OIDC** (`oidc_client`) | **Google / Microsoft** (`google_client`, `microsoft_client`) |
|---|---|---|
| Intent | Full SSO lockdown | Extra login options |
| Django's default login | Disabled (`SOCIALACCOUNT_ONLY`) | Kept, each provider is an added button |
| `LoginRequiredMiddleware` | On (every view needs login) | Off |
| `LOGIN_URL` | The provider login route | Django default (`/accounts/login/`) |
| New accounts | Auto-created ("shadow" users) | Never, existing users only |
| Role mapping | Yes (see adapter) | No |

Both modes share the same base: allauth apps + `AccountMiddleware`, silent auto-signup/linking, no allauth email verification (the provider is the email authority), and auto-connect to an existing local user by verified email. Credentials come from env vars, so `validate()` fails fast if an enabled provider is missing any.

### OIDC single sign-on (portal lockdown)

When `oidc_client` is set, the app becomes SSO-only:

- `LoginRequiredMiddleware` is inserted, so **every view requires login** (opt a view out with `@login_not_required` from `django.contrib.auth.decorators`).
- `SOCIALACCOUNT_ONLY = True` disables allauth's local username/password login, signup, and password management, the provider is the only way in.
- `LOGIN_URL` is derived from the provider id (`/accounts/oidc/<provider_id>/login/`) so unauthenticated users go straight to the portal, and `SOCIALACCOUNT_LOGIN_ON_GET` skips the interstitial.
- The role-mapping adapter (`SOCIALACCOUNT_ADAPTER`) is registered and the role-sync signals are connected (`OIDC_ROLE_SYNC`).
- allauth's routes mount at `/accounts/`, plus the SSO logout at `/kernel/logout/` (see below).

#### Auto-provisioning and role mapping

A user who does not exist locally is **created automatically** on first login (linked to an existing user by verified email, or created as a new shadow user from the claims). But **privileges are not inherited by default**: allauth makes a plain user (`is_staff=False`, `is_superuser=False`) and does not translate provider roles into Django permissions.

To close that gap, [kernel/base_app/adapters.py](../kernel/base_app/adapters.py) provides `OIDCRoleAdapter`, which reads role/group claims and maps them onto `is_staff`, `is_superuser`, and groups. It runs on signup and re-syncs on every login (via the `social_account_added` / `social_account_updated` signals connected in `ready()`), so a role change at the provider propagates, including revoking access. The claim names and role values are placeholders you must set for your provider (inspect `sociallogin.account.extra_data` on a real login):

```python
# kernel/base_app/adapters.py
ROLE_CLAIMS = ("roles", "groups")     # which claim carries the roles
SUPERUSER_ROLES = {"superuser"}       # roles -> is_staff + is_superuser
STAFF_ROLES = {"staff", "admin"}      # roles -> is_staff
SYNC_GROUPS = True                    # mirror roles into Django groups
REVOKE_WHEN_ABSENT = True             # drop privileges when the role is gone
```

Prefer mapping to Django **groups** over `is_superuser` where you can. Note the `REVOKE_WHEN_ABSENT` caveat: a break-glass superuser who also logs in through the portal without the superuser role would have its Django privileges stripped. An account that only ever uses `/admin/login/` is never touched, since role sync runs on social login only.

#### Single sign-on logout

Clearing only the local session is not enough under SSO: the portal's session is still alive and `SOCIALACCOUNT_LOGIN_ON_GET` signs the user right back in. [kernel/base_app/views/account.py](../kernel/base_app/views/account.py) provides `sso_logout` (wired at `/kernel/logout/`, name `sso_logout`, in [kernel/base_app/urls/](../kernel/base_app/urls/)): it ends the local session, then redirects to the portal's `end_session_endpoint` (read from the OIDC discovery document, cached an hour). If no portal is configured or it is unreachable, it falls back to a plain local logout to `/`, so the endpoint is safe under any provider. It is `@require_POST` and `@login_not_required`:

```html
<form method="post" action="{% url 'sso_logout' %}">{% csrf_token %}
  <button type="submit">Log out</button>
</form>
```

#### Enabling OIDC

1. Fill the five `OIDC_*_DEV` vars in `.env` (provider id, name, client id, client secret, server url), the example `OIDCClientAppSettings` in [core/environments.py](../core/environments.py) reads them.
2. Attach it to an environment: `oidc_client=oidc_dev_settings`.
3. Install allauth if needed: `pip install "django-allauth[socialaccount]"`. `validate()` refuses to start if any field is missing.

### Google and Microsoft sign-in (additional logins)

When `google_client` and/or `microsoft_client` is set, each is an **extra** way to log in and Django's default login stays fully available:

- No `SOCIALACCOUNT_ONLY`, no `LoginRequiredMiddleware`, no `LOGIN_URL` override, `/accounts/login/` keeps its username/password form and the Django admin is unaffected. allauth's login page lists every enabled provider as a button, so with both set you get "Google" and "Microsoft" side by side.
- Existing users only: the additional-login branch registers `PortalSocialAccountAdapter` ([kernel/base_app/adapters.py](../kernel/base_app/adapters.py)), whose `is_open_for_signup()` returns `False`. A social login matched to an existing local user (by verified email) succeeds; an account with no matching user is refused, no self-provisioning. The same user may sign in with either provider as long as the addresses match.
- No role mapping: neither provider's tokens carry role claims here, and the role-sync signals stay disconnected (`OIDC_ROLE_SYNC` is unset), so these logins never change `is_staff`/`is_superuser`.
- Microsoft emails are trusted: Microsoft Graph returns no `email_verified` claim, so the Microsoft branch sets `VERIFIED_EMAIL: True`, otherwise allauth could never match the address to a local user. This is reasonable because the address is controlled by the tenant. To trust only your own domain(s), set it to a list such as `["example.com"]`; other addresses are then refused.

#### Enabling Google

1. Fill `GOOGLE_CLIENT_ID_DEV` / `GOOGLE_CLIENT_SECRET_DEV` in `.env` (the example `GoogleClientAppSettings` in [core/environments.py](../core/environments.py) reads them).
2. Attach it to an environment: `google_client=google_app_dev_settings` (and make sure `oidc_client` is `None`).
3. Install allauth if needed: `pip install "django-allauth[socialaccount]"`.
4. In Google Cloud, register the redirect URI `<origin>/accounts/google/login/callback/`.

#### Enabling Microsoft

1. In [Microsoft Entra admin center](https://entra.microsoft.com/), create an app registration. Under Authentication add a **Web** platform with redirect URI `<origin>/accounts/microsoft/login/callback/`, then create a client secret under Certificates & secrets.
2. Fill `MICROSOFT_CLIENT_ID_DEV` / `MICROSOFT_CLIENT_SECRET_DEV` in `.env`, and set `MICROSOFT_TENANT_DEV` to your tenant GUID to restrict sign-in to that organization. `organizations` (any work/school account), `consumers` (personal accounts) and `common` (both) are also accepted. The example `MicrosoftClientAppSettings` in [core/environments.py](../core/environments.py) reads them.
3. Attach it to an environment: `microsoft_client=microsoft_app_dev_settings`. It may sit next to `google_client`; `oidc_client` must stay `None`.
4. Install allauth if needed: `pip install "django-allauth[socialaccount]"`. The provider requests only the `User.Read` Graph scope.

### Django admin and break-glass

The Django admin keeps its **own** login, independent of social auth. Under either provider, `/admin/` redirects anonymous users to `/admin/login/` (the admin's username/password form), not to the social provider. This is deliberate break-glass access:

- Social-login users have no usable password, so they cannot reach `/admin/` unless the OIDC adapter granted them `is_staff`.
- A local superuser (from `createsuperuser`) can always sign in at `/admin/login/` directly, even when the identity provider is down. Keep at least one.

allauth's own login/callback views and the admin login page exempt themselves from the login gate, so there is no redirect loop.

## Soft-delete models and audit trail

Inherit domain models from `SoftDeleteModel` to get, for free:

- **Audit columns**: `created_at`, `updated_at`, `created_by`, `updated_by`
- **Status lifecycle**: `A` active, `I` inactive (soft-deleted), `Z` archived (kept for history)
- **Two managers**: `objects` (all rows) and `active_objects` (status `A` only)
- **Safe operations**: `delete()` soft-deletes, plus `set_inactive()`, `set_archived()`, `reactivate()`
- **Serialization**: `api_to_dict()`, `serialize_queryset()`, and prefetch helpers to avoid N+1 queries
- **Human-readable change log**: declare `CHANGE_FIELDS` to get field-level "before → after" diffs in the audit trail (see below)

```python
from kernel.base_app.models import SoftDeleteModel

class Invoice(SoftDeleteModel):
    API_FIELDS = ("id", "number", "total", "due_date")
    # Fields to diff in the audit log. Bare name or (field, label) for a friendly label.
    CHANGE_FIELDS = ("number", ("due_date", "Due date"))
    number = models.CharField(max_length=50)
    total = models.DecimalField(max_digits=12, decimal_places=2)
    due_date = models.DateField(null=True, blank=True)

    def get_created_text(self):
        return f"Invoice {self.number} created"
    def get_updated_text(self, **kw):
        return f"Invoice {self.number} updated"
    def get_deleted_text(self):
        return f"Invoice {self.number} removed"
```

Pass the acting user through `save()` to populate the audit columns automatically:

```python
invoice.save(user=request.user)   # sets created_by / updated_by
```

### How the audit log works

[kernel/base_app/signals.py](../kernel/base_app/signals.py) auto-registers `pre_save` / `post_save` / `post_delete` receivers for **every** `SoftDeleteModel` subclass at startup (from `BaseAppConfig.ready()`). Each change writes a `ChangeLog` row capturing the acting user, the action, and a full before/after snapshot of the instance. The user is pulled from `CurrentUserMiddleware`, so you do not have to thread `request` into your models or services.

Because deletes are soft (`delete()` calls `set_inactive()`), they are recorded as `updated` transitions with the status flipping to `I` or `Z`, which `ChangeLog.description` reads back into human-readable text.

### Field-level diffs (`CHANGE_FIELDS`)

`ChangeLog.description` turns a raw row into readable text. For updates it produces a field-level diff, but only for the fields a model opts in via `CHANGE_FIELDS`:

```python
CHANGE_FIELDS = ("number", ("due_date", "Due date"))
```

Each entry is either a bare field name or a `(field, label)` pair for a friendlier label. Given the example, changing an invoice's due date renders as:

```
Due date: 2026-01-15 → 2026-02-01
```

Notes on how it behaves:

- A field must also be in `API_FIELDS` to be diffable, since values are read from the serialized snapshot stored on the log row.
- Dates are shown without the serializer's midnight suffix, and empty/`None` values render as `(vacío)`.
- List and many-to-many values are compared order-insensitively (via `Counter`), so a reordering is not reported as a change.
- Status transitions are special-cased first: `A → I` calls `get_updated_text(deactivated=True)` and `I → A` calls `get_updated_text(reactivated=True)`. Any other update falls back to the computed diff, then to plain `get_updated_text()` if nothing tracked changed.
- With `CHANGE_FIELDS` empty (the default), updates just use `get_updated_text()`, no diff.

## Request profiling

Set `activate_profiler=True` on an environment and `CProfileMiddleware` writes a cProfile dump plus a readable `.txt` stats file to `environments/` for every request. The 40 most recent runs are kept and older ones are pruned automatically. The `environments/` directory is git-ignored.

See [core/environments.py](../core/environments.py) to tune which environment enables this.

## Utilities

From [kernel/base_app/utilities.py](../kernel/base_app/utilities.py):

- **`serialize_model_data_recursive(instance)`**: deep serialization that walks relations, unwraps files to URLs, handles `Decimal` and `date`, and caches users to avoid repeat work.
- **`throttle_by_ip(limit, window, ban_time)`**: view decorator for per-IP rate limiting backed by the cache. Exceeding the limit temporarily bans the IP.
- **`log_change(...)`**: the low-level writer behind the audit signals, usable directly if needed.
- **`delete_old_files(path, keep)`**: keep only the newest N files in a directory.

```python
from kernel.base_app.utilities import throttle_by_ip

@throttle_by_ip(limit=5, window=60, ban_time=300)
def login_view(request):
    ...
```

## Database cloning

The `clone_db` management command copies the `default` database into the `target` database defined by the active environment, picking the fastest available method:

- **PostgreSQL to PostgreSQL**: native `pg_dump` / `pg_restore` (fast)
- **Anything else**: Django `dumpdata` / `loaddata` (portable, cross-engine)

```bash
python manage.py clone_db            # prompts for confirmation
python manage.py clone_db --no-input # skip the prompt (scripts/CI)
python manage.py clone_db --force    # allow running when DEBUG is off
```

Useful for pulling a production snapshot into a local database for debugging. It **wipes the target**, so the command refuses to run until several guardrails pass:

- a `target` database must be configured on the environment (`target_database_url` or `target_local=True`);
- the target must not resolve to the same database as the source (no self-wipe);
- `DEBUG` must be on, unless `--force` is passed (so you cannot casually wipe a database from a production deployment);
- an interactive prompt requires you to type the target database name, unless `--no-input`.

The mechanics live in [kernel/base_app/services/clone_db.py](../kernel/base_app/services/clone_db.py); the guardrails and CLI in [kernel/base_app/management/commands/clone_db.py](../kernel/base_app/management/commands/clone_db.py). The PostgreSQL client binaries are discovered on `PATH` (with common install dirs as a fallback), and the DB password is passed via `PGPASSWORD` rather than the command line.

## Caching

[core/settings.py](../core/settings.py) selects the cache backend automatically: in-memory `LocMemCache` when `DEBUG` is on or no `REDIS_URL` is set, and Redis via `django-redis` otherwise. Keys are namespaced per environment via `get_redis_prefix()`, so environments never collide on a shared Redis instance.

## UI annotation feedback (dev)

A dev-only loop for annotating rendered pages in the browser and feeding those notes to Claude. Alt+click any element, type a note in a small popup, and hit "Send"; annotations POST to a Django view that writes `.annotations/pending.json` (gitignored, at the repo root). Claude reads that file, fixes the source, and removes only the keys it handled. The file is the source of truth, no extra server or port required.

Pieces (all DEBUG-gated off the active environment, nothing ships to production):

- **Store view** [kernel/base_app/views/annotations.py](../kernel/base_app/views/annotations.py): `add_annotation` at `/__annotations__/`, registered only under `DEBUG` in [core/urls.py](../core/urls.py). Writes atomically (temp file + `os.replace`), keyed by template origin (last-write-wins); accepts the whole queue in one batched POST.
- **Browser script** [static/annotations.js](../static/annotations.js): Alt+click opens a debug popup at the click point (editable note, Save/Cancel, Enter to save, Esc to cancel). The floating Send button is draggable, and the unsent queue persists in `localStorage` across reloads and page switches. Inspect it in the console via `window.annotations`.
- **Auto `data-tpl` tagging** [kernel/base_app/template_loaders.py](../kernel/base_app/template_loaders.py): dev-only template loaders inject `data-tpl="<template>:<line>"` into every rendered HTML tag (wired under `DEBUG` in [core/settings.py](../core/settings.py), which swaps `APP_DIRS` for the annotating loaders). So each note records the exact source template and line, with no hand-tagging.
- **Auto-injection** [kernel/base_app/middleware.py](../kernel/base_app/middleware.py): `AnnotationScriptMiddleware` splices the script before `</body>` on HTML responses under `DEBUG` (the same trick django-debug-toolbar uses), so no template needs to include it by hand. Scope it with `ANNOTATIONS_INCLUDE_PREFIXES` / `ANNOTATIONS_EXCLUDE_PREFIXES` (path prefixes); by default it skips `/admin/` and `/accounts/`.
- **`debug` context processor**: exposes `debug` for `{% if debug %}` in any template.
- **`manage.py annotations show|clear`**: inspect the store, or `clear` as a human-only reset (Claude never blanket-clears).
- **Demo harness**: the template's `apps/example/` app (removed in Cerveau) renders a dense dashboard at `/` to exercise the loop across many element types and template files (the script auto-injects, no include needed). It doubles as the worked example of the template/static layout convention. Delete the whole package when you build real UI.
- **Claude's instructions** live in the `## UI annotation feedback` block of [AGENTS.md](../AGENTS.md).

### Optional: push into a live Claude Code session

By default Claude reads the file when you ask. You can also have Send push a single consolidated prompt straight into a running Claude Code session via a local **prompt channel** (an MCP "channel" server, research preview). Set `ANNOTATIONS_CHANNEL_URL` (and a matching `ANNOTATIONS_CHANNEL_SECRET`, see [.env.example](../.env.example)); the view then POSTs the built prompt to that channel, which injects it into the session. The same path also backs voice dictation (transcribe, then POST the text). The channel program is a small standalone MCP server you run at user scope, not vendored in this template. Leave `ANNOTATIONS_CHANNEL_URL` blank to keep the plain file-based loop.

## Project layout

```
base_django_setup/
├── core/                       # Django project package
│   ├── settings.py             # reads from the active environment
│   ├── environments.py         # Environment + EnvironmentRegistry + OIDC/Google/Microsoft settings
│   ├── urls.py                 # admin + conditional allauth/base_app routes
│   ├── wsgi.py / asgi.py
├── kernel/                     # shared foundation (import-only dependency target)
│   └── base_app/               # reusable base app (label: base_app)
│       ├── models.py           # SoftDeleteModel, ChangeLog
│       ├── signals.py          # audit-trail auto-wiring
│       ├── adapters.py         # OIDC role mapping + Google/Microsoft no-signup adapter
│       ├── middleware.py       # CurrentUser, CProfile, AnnotationScript, DebugBanner
│       ├── template_loaders.py # dev-only auto data-tpl injection for annotations
│       ├── utilities.py        # serialization, throttling, file cleanup
│       ├── enums.py            # GeneralStatusTypes, ActionEnum
│       ├── context_processors.py  # navigation, debug flag
│       ├── views/              # views package (SSO logout, annotation store)
│       ├── urls/               # app URL routes (included by core/urls.py)
│       ├── management/commands/
│       │   ├── clone_db.py     # `manage.py clone_db` (guardrails + CLI)
│       │   └── annotations.py  # `manage.py annotations show|clear`
│       └── services/
│           └── clone_db.py     # database cloning mechanics
├── apps/                       # your domain apps (e.g. apps.invoices); import from kernel
│   └── example/                # throwaway demo app + worked layout example
│       ├── apps.py             # ExampleConfig (name = "apps.example")
│       ├── views.py            # IndexView -> "example/index.html"
│       ├── urls.py             # included at "/" by core/urls.py
│       ├── templates/example/  # app-owned templates, namespaced by app label
│       │   ├── index.html
│       │   └── components/
│       └── static/example/     # app-owned assets, namespaced by app label
│           └── example.css
├── templates/                  # PROJECT-WIDE templates only
│   └── base.html               # the layout skeleton every app extends
├── static/                     # PROJECT-WIDE assets only (site CSS, logos, favicon)
│   └── annotations.js          # dev-only browser annotation capture (every page)
├── docs/
│   └── github_workflow.md      # mandatory git workflow (branches, commit tags)
├── AGENTS.md                   # tool-agnostic agent guidance
├── CLAUDE.md                   # Claude Code entry point -> AGENTS.md
├── manage.py
├── requirements.txt
└── .env.example
```

New domain apps go under `apps/` (e.g. `apps/invoices/` with `AppConfig.name = "apps.invoices"`) and import shared building blocks from `kernel.base_app` (`SoftDeleteModel`, the audit trail, utilities, enums). Keep the dependency one-directional: `apps/* -> kernel`, never the reverse.

Note where the front-end files sit: **each app owns its own `templates/` and `static/`**, and only shared assets live at the project root. See [Templates and static files](#templates-and-static-files) below.

## Adding apps and shared code

### A new domain app (`apps/`)

Domain apps, the features specific to a given project, live under `apps/`:

```bash
mkdir apps/invoices
python manage.py startapp invoices apps/invoices
```

Point the app config at its dotted path and register it:

```python
# apps/invoices/apps.py
class InvoicesConfig(AppConfig):
    name = "apps.invoices"          # the dotted path, not just "invoices"

# core/settings.py
INSTALLED_APPS += ["apps.invoices"]
```

Then build on the kernel instead of re-inventing plumbing:

```python
# apps/invoices/models.py
from kernel.base_app.models import SoftDeleteModel

class Invoice(SoftDeleteModel):     # audit fields, soft delete, change log: all free
    ...
```

Any `SoftDeleteModel` subclass is picked up by the audit trail automatically (the signals connect to every subclass at startup), and `save(user=...)` fills the audit columns. The only rule: import from `kernel`, never the reverse.

### Templates and static files

**Templates and static files belong to the app that uses them.** Only genuinely shared assets live at the project root:

```
apps/invoices/
├── templates/invoices/         # app-owned templates
│   ├── invoice_list.html
│   └── components/row.html
└── static/invoices/            # app-owned CSS, JS, images
    └── invoices.css

templates/                      # project-wide only
└── base.html                   # layout skeleton, shared partials, 404.html / 500.html
static/                         # project-wide only
└── site.css                    # site-wide CSS, logos, favicons
```

The inner folder repeats the app label on purpose. Django flattens every app's template and static directories into a single namespace, so without it two apps that both ship an `index.html` or a `main.css` silently shadow each other, and which one wins depends on `INSTALLED_APPS` order. With the namespace, names are unambiguous:

```django
{% extends "base.html" %}                        {# project-wide #}
{% include "invoices/components/row.html" %}     {# app-owned, namespaced #}
<link rel="stylesheet" href="{% static 'invoices/invoices.css' %}">
```

Nothing needs configuring: `APP_DIRS` (and the annotating loader under `DEBUG`) already finds app templates, and staticfiles' `AppDirectoriesFinder` already finds app static. `TEMPLATES["DIRS"]` and `STATICFILES_DIRS` point only at the root folders, which is what makes them the "shared only" slot.

The payoff is that an app stays **deletable and portable**: copy `apps/invoices/` into another project, or delete it from this one, and nothing is left behind at the root. `apps/example/` (the template's demo app, removed in Cerveau) is the worked example, and `AGENTS.md` states the rule for AI agents.

The test is **who uses a file, not who serves it**. [static/annotations.js](../static/annotations.js) stays at the root even though the middleware that injects it belongs to `base_app`, because it is injected into every page of the project. A template only `base_app`'s own views render would live in `kernel/base_app/templates/base_app/` instead.

### Growing the kernel (shared foundation)

`kernel/` is for anything used by more than one domain app, so shared code does not become a hidden dependency between apps:

- **Essential / infrastructure apps**: when a concern is cross-cutting, add a sibling app under `kernel/` (e.g. `kernel/notifications/`, `kernel/audit/`, `kernel/auth/`). Create it like any app, set `AppConfig.name = "kernel.notifications"`, and add it to `INSTALLED_APPS`. Its app label defaults to the last path component (`notifications`).
- **Shared models**: models several apps depend on (a common `Attachment`, an `Organization`/tenant, shared lookup tables) belong in the kernel, either in `kernel.base_app` or a dedicated kernel app, rather than in one domain app that the others reach into. This keeps `apps/*` from importing each other.
- **Shared utilities, mixins, base classes**: same principle, if two apps would import it, it belongs in the kernel.

Rule of thumb: code specific to one feature stays in that `apps/` app; the moment a second app needs it, promote it into `kernel/`. Keep the arrow pointing one way (`apps/* -> kernel`), and never let a kernel app import from a domain app.

## Repository conventions and agent guidance

Contributor and AI-agent instructions are layered so the rules live in one place:

- **[docs/github_workflow.md](github_workflow.md)** is the authoritative git workflow, protected `main`/`development`, `feature/*` `fix/*` `sandbox/*` branch naming, and tagged commit messages (`[ADD]`, `[FIX]`, `[REF]`, ...; a commit may carry several tags, one tagged line per logical change). Follow it for every git operation.
- **[AGENTS.md](../AGENTS.md)** is the tool-agnostic entry point for any AI coding agent; it points at the workflow doc and summarizes the project.
- **[CLAUDE.md](../CLAUDE.md)** is Claude Code's entry point and defers to AGENTS.md. It also registers the `django-frontend-design` skill (in [.claude/skills/](../.claude/skills/)) for UI/UX and template work.

So the chain is `CLAUDE.md -> AGENTS.md -> docs/github_workflow.md`, with the frontend skill invoked automatically for UI tasks.

## Cleanup checklist when starting a new project

Old-project references have been cleaned up. The remaining steps are the usual per-project setup:

1. Set a real `SECRET_KEY` in `.env` and review the default environment's `allowed_hosts`.
2. Delete the demo app: remove `apps/example/`, its line in `INSTALLED_APPS`, and its `path("", include("apps.example.urls"))` in [core/urls.py](../core/urls.py). Keep [templates/base.html](../templates/base.html), it is the shared layout skeleton.
3. To enable social login, configure providers on an environment (`oidc_client` alone, or any mix of `google_client` and `microsoft_client`) and fill the matching env vars.
</content>

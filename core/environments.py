from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import dj_database_url
from dotenv import load_dotenv

# environments.py lives inside the core/ settings package, one level
# below the repo root that holds manage.py. We resolve that root once here so
# the SQLite DB path below is absolute and lands in the repo root (next to
# manage.py) - relying on CWD breaks any consumer spawned outside the project
# (cron jobs, the MCP server launched by Claude Desktop, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(slots=True)
class OIDCClientAppSettings:
    provider_id: str
    name: str
    client_id: str
    client_secret: str
    server_url: str

    def as_django_settings(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "client_id": self.client_id,
            "secret": self.client_secret,
            "settings": {
                "server_url": self.server_url,
            },
        }


@dataclass(slots=True)
class GoogleClientAppSettings:
    client_id: str
    client_secret: str

    def as_django_settings(self) -> dict:
        return {
            "client_id": self.client_id,
            "secret": self.client_secret,
            "key": "",
        }


@dataclass(slots=True)
class MicrosoftClientAppSettings:
    """Microsoft Entra ID (Azure AD) app registration.

    `tenant` scopes who may sign in: a tenant GUID (that one organization only),
    "organizations" (any work/school account), "consumers" (personal accounts), or
    "common" (both). Defaults to "organizations".
    """

    client_id: str
    client_secret: str
    tenant: str = "organizations"

    def as_django_settings(self) -> dict:
        return {
            "client_id": self.client_id,
            "secret": self.client_secret,
            "key": "",
            "settings": {
                "tenant": self.tenant or "organizations",
            },
        }


@dataclass(slots=True)
class Environment:
    """All per-environment configuration (dev / staging / prod) for the project."""

    # non-defaults first
    name: str
    secret_key: str

    # defaults after
    port: int = 8000
    database_engine: str = "sqlite"
    debug_banner: bool = True
    debug_banner_text: str = "🐞 TESTING 🐞"
    database_url: str | None = None
    target_database_url: str | None = None
    target_local: bool = False
    debug: bool = True
    allowed_hosts: list[str] = field(default_factory=list)
    timezone: str = "UTC"
    csrf_trusted_origins: list[str] = field(default_factory=list)
    attachment_folder: str = "attachments"
    activate_profiler: bool = False
    oidc_client: OIDCClientAppSettings | None = None
    google_client: GoogleClientAppSettings | None = None
    microsoft_client: MicrosoftClientAppSettings | None = None

    def get_redis_prefix(self) -> str:
        return f"app:{self.name}"

    @property
    def activate_oidc_client_auth(self) -> bool:
        return self.oidc_client is not None

    @property
    def activate_google_auth(self) -> bool:
        return self.google_client is not None

    @property
    def activate_microsoft_auth(self) -> bool:
        return self.microsoft_client is not None

    @property
    def activate_additional_login(self) -> bool:
        """Any "extra login button" provider (Google and/or Microsoft). These coexist
        with each other and with Django's default login; OIDC is the lockdown mode."""
        return self.activate_google_auth or self.activate_microsoft_auth

    def to_debug(self) -> Environment:
        self.debug = True
        self.allowed_hosts = ["*"]
        self.csrf_trusted_origins = []
        self.debug_banner = True
        return self

    def validate(self) -> None:
        """Raise on obviously unsafe/invalid combinations."""
        if not self.secret_key or self.secret_key == "CHANGE_ME" and not self.debug:
            raise ValueError(f'Environment "{self.name}" has an unsafe SECRET_KEY.')
        # Django >= 4.0 requires scheme in CSRF_TRUSTED_ORIGINS
        for origin in self.csrf_trusted_origins:
            if origin != "*" and "://" not in origin:
                raise ValueError(
                    f'CSRF origin "{origin}" in environment "{self.name}" must include '
                    'scheme (e.g., "https://example.com").'
                )
        if self.database_engine != "sqlite" and not self.database_url:
            raise ValueError(
                f'Environment "{self.name}" requires DATABASE_URL or database_url.'
            )
        # Social login: OIDC (SSO lockdown) is exclusive with the additional-login
        # providers. Google and Microsoft may be enabled together.
        if self.activate_oidc_client_auth and self.activate_additional_login:
            raise ValueError(
                f'Environment "{self.name}" enables OIDC together with Google/Microsoft; '
                "OIDC lockdown cannot be combined with additional login providers."
            )
        # OIDC: if enabled, the client app must be fully configured. These come
        # from env vars, so an unset var (None) or an empty .env line ("") would
        # otherwise pass silently and only fail at login time.
        if self.activate_oidc_client_auth:
            oidc = self.oidc_client
            missing = [
                field_name
                for field_name, value in (
                    ("provider_id", oidc.provider_id),
                    ("name", oidc.name),
                    ("client_id", oidc.client_id),
                    ("client_secret", oidc.client_secret),
                    ("server_url", oidc.server_url),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f'Environment "{self.name}" enables OIDC but is missing: '
                    f'{", ".join(missing)}.'
                )
        # Google: same fail-fast check on its (env-driven) credentials.
        if self.activate_google_auth:
            google = self.google_client
            missing = [
                field_name
                for field_name, value in (
                    ("client_id", google.client_id),
                    ("client_secret", google.client_secret),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f'Environment "{self.name}" enables Google auth but is missing: '
                    f'{", ".join(missing)}.'
                )
        # Microsoft: same fail-fast check.
        if self.activate_microsoft_auth:
            ms = self.microsoft_client
            missing = [
                field_name
                for field_name, value in (
                    ("client_id", ms.client_id),
                    ("client_secret", ms.client_secret),
                    ("tenant", ms.tenant),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f'Environment "{self.name}" enables Microsoft auth but is missing: '
                    f'{", ".join(missing)}.'
                )

    def as_django_settings(self) -> dict:
        """
        Minimal mapping you can unpack into settings.py.
        You can extend this as your project grows.
        """
        out = {
            "DEBUG": self.debug,
            "SECRET_KEY": self.secret_key,
            "TIME_ZONE": self.timezone,
            "ALLOWED_HOSTS": self.allowed_hosts,
            "CSRF_TRUSTED_ORIGINS": self.csrf_trusted_origins,
        }

        sqlite_path = str(_PROJECT_ROOT / "db.sqlite3")
        if self.database_engine == "sqlite":
            out["DATABASES"] = {
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": sqlite_path,
                }
            }
        else:
            # Leave parsing to settings.py to keep this utility decoupled,
            # or parse here if you prefer dj-database-url.
            target = None
            if self.target_local:
                target = {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": sqlite_path,
                }
            if self.target_database_url and not self.target_local:
                target = dj_database_url.config(
                    default=self.target_database_url, conn_max_age=600
                )

            out["DATABASES"] = {
                "default": dj_database_url.config(
                    default=self.database_url, conn_max_age=600
                ),
            }
            # Only declare "target" when one is configured: Django requires every
            # DATABASES entry to be a dict and crashes at startup on None. clone_db
            # already treats a missing "target" as "not configured".
            if target is not None:
                out["DATABASES"]["target"] = target

        if self.activate_oidc_client_auth:
            out["OIDC_CLIENT_APP"] = self.oidc_client.as_django_settings()
        if self.activate_google_auth:
            out["GOOGLE_CLIENT_APP"] = self.google_client.as_django_settings()
        if self.activate_microsoft_auth:
            out["MICROSOFT_CLIENT_APP"] = self.microsoft_client.as_django_settings()
        return out


class EnvironmentRegistry:
    """Process-wide registry of environments; tracks the default and active one."""

    _instance: EnvironmentRegistry | None = None
    # _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.environments: dict[str, Environment] = {}
        self._default_environment: Environment | None = None
        self._active_environment: Environment | None = None
        self._initialized = True

    # CRUD
    def add_environment(self, environment: Environment) -> None:
        if environment.name in self.environments:
            raise ValueError(f"Environment {environment.name} already exists.")
        self.environments[environment.name] = environment

    def add_environments(self, environments: list[Environment]) -> None:
        for environment in environments:
            self.add_environment(environment)

    def get_environment(self, name: str) -> Environment:
        try:
            return self.environments[name]
        except KeyError:
            raise ValueError(f"Environment {name} does not exist.") from None

    # default environment
    @property
    def default_environment(self) -> Environment:
        if self._default_environment is None:
            self._default_environment = self.get_environment("default")
        return self._default_environment

    @default_environment.setter
    def default_environment(self, environment: str | Environment) -> None:
        if isinstance(environment, str):
            self._default_environment = self.get_environment(environment)
        else:
            if environment.name not in self.environments:
                self.add_environment(environment)
            self._default_environment = environment

    # active environment
    @property
    def active_environment(self) -> Environment:
        if self._active_environment is None:
            self._active_environment = self.default_environment
        return self._active_environment

    @active_environment.setter
    def active_environment(self, environment: str | Environment) -> None:
        if isinstance(environment, str):
            self._active_environment = self.get_environment(environment)
        else:
            if environment.name not in self.environments:
                self.add_environment(environment)
            self._active_environment = environment

    def select_from_env(self, env_var: str = "DJANGO_ENV") -> None:
        """
        If DJANGO_ENV is set, switch active & default environment to it.
        Falls back to current default if env var missing.
        """
        name = os.environ.get(env_var)
        if name:
            self.active_environment = name

    def validate_all(self) -> None:
        """Run validation on all registered environments."""
        for env in self.environments.values():
            env.validate()


oidc_dev_settings = OIDCClientAppSettings(
    provider_id=os.environ.get("OIDC_PROVIDER_ID_DEV"),
    name=os.environ.get("OIDC_NAME_DEV"),
    client_id=os.environ.get("OIDC_CLIENT_ID_DEV"),
    client_secret=os.environ.get("OIDC_CLIENT_SECRET_DEV"),
    server_url=os.environ.get("OIDC_SERVER_URL_DEV"),
)

oidc_prod_settings = OIDCClientAppSettings(
    provider_id=os.environ.get("OIDC_PROVIDER_ID_PROD"),
    name=os.environ.get("OIDC_NAME_PROD"),
    client_id=os.environ.get("OIDC_CLIENT_ID_PROD"),
    client_secret=os.environ.get("OIDC_CLIENT_SECRET_PROD"),
    server_url=os.environ.get("OIDC_SERVER_URL_PROD"),
)

google_app_dev_settings = GoogleClientAppSettings(
    client_id=os.environ.get("GOOGLE_CLIENT_ID_DEV"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET_DEV"),
)

google_app_prod_settings = GoogleClientAppSettings(
    client_id=os.environ.get("GOOGLE_CLIENT_ID_PROD"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET_PROD"),
)

microsoft_app_dev_settings = MicrosoftClientAppSettings(
    client_id=os.environ.get("MICROSOFT_CLIENT_ID_DEV"),
    client_secret=os.environ.get("MICROSOFT_CLIENT_SECRET_DEV"),
    tenant=os.environ.get("MICROSOFT_TENANT_DEV") or "organizations",
)

microsoft_app_prod_settings = MicrosoftClientAppSettings(
    client_id=os.environ.get("MICROSOFT_CLIENT_ID_PROD"),
    client_secret=os.environ.get("MICROSOFT_CLIENT_SECRET_PROD"),
    tenant=os.environ.get("MICROSOFT_TENANT_PROD") or "organizations",
)

# --- Define environments from env ---
# Cerveau needs Postgres everywhere (pgvector, pg_trgm, unaccent have no SQLite
# equivalent). Social login stays off: Cerveau is a single-user local assistant.
dev_environment = Environment(
    name="dev",
    secret_key=os.environ.get("SECRET_KEY", "CHANGE_ME"),
    csrf_trusted_origins=[],
    allowed_hosts=["*"],
    database_engine="postgresql",
    database_url=os.environ.get("DATABASE_URL__DEV"),
    activate_profiler=True,
    oidc_client=None,
    google_client=None,
    microsoft_client=None,
)

# Benchmarks and experiments run here, on their own database and port, so they
# never touch real memory.
eval_environment = Environment(
    name="eval",
    secret_key=os.environ.get("SECRET_KEY", "CHANGE_ME"),
    csrf_trusted_origins=[],
    allowed_hosts=["*"],
    port=8010,
    database_engine="postgresql",
    database_url=os.environ.get("DATABASE_URL__EVAL"),
    activate_profiler=False,
    debug_banner_text="EVAL",
    oidc_client=None,
    google_client=None,
    microsoft_client=None,
)

# prod_environment = Environment(
#     name="prod",
#     debug=False,
#     debug_banner=False,
#     secret_key=os.environ.get("SECRET_KEY", "CHANGE_ME"),
#     database_engine="postgresql",
#     database_url=os.environ.get("DATABASE_URL__PROD"),
#     target_database_url=os.environ.get("DATABASE_URL__CLONE_TARGET"),
#     target_local=False,
#     csrf_trusted_origins=["https://myapp.example.com"],
#     allowed_hosts=["myapp.example.com"],
#     attachment_folder="attachments_prod",
#     oidc_client=oidc_prod_settings,
# )

# --- Build registry (per process) ---
env_registry = EnvironmentRegistry()
env_registry.add_environments([dev_environment, eval_environment])
env_registry.default_environment = "dev"
env_registry.select_from_env()

"""Database cloning mechanics.

Copies the SOURCE_DB ("default") into the TARGET_DB ("target"), replacing the
target's contents. This module holds only the how; the guardrails and CLI live in
the `clone_db` management command (base_app/management/commands/clone_db.py), which
is the intended entry point (kernel/base_app/management/commands/clone_db.py). All
output is routed through a `log` callable so the command can write through
`self.stdout`.
"""

import os
import shutil
import subprocess
import tempfile

from django.conf import settings
from django.core.management import call_command
from django.db import connections

SOURCE_DB = "default"
TARGET_DB = "target"

# Fallback locations checked when pg_dump / pg_restore are not on PATH.
_PG_FALLBACK_DIRS = (
    r"C:\Program Files\PostgreSQL\18\bin",
    r"C:\Program Files\PostgreSQL\17\bin",
    r"C:\Program Files\PostgreSQL\16\bin",
)


# -------------------------
# Option 1: Django method
# Slower, cross-database
# -------------------------


def wipe_postgres_schema(alias):
    with connections[alias].cursor() as cursor:
        cursor.execute("DROP SCHEMA public CASCADE;")
        cursor.execute("CREATE SCHEMA public;")
        cursor.execute("GRANT ALL ON SCHEMA public TO public;")


def wipe_sqlite(alias):
    path = settings.DATABASES[alias]["NAME"]
    connections[alias].close()

    if os.path.exists(path):
        os.remove(path)


def wipe_target(alias):
    engine = settings.DATABASES[alias]["ENGINE"]

    if "postgresql" in engine:
        wipe_postgres_schema(alias)
    elif "sqlite3" in engine:
        wipe_sqlite(alias)
    else:
        raise NotImplementedError(f"Unsupported DB engine: {engine}")


def clone_with_django(log=print):
    fixture_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as f:
            fixture_path = f.name

            log("Dumping source DB with Django...")
            call_command(
                "dumpdata",
                exclude=["contenttypes", "auth.Permission"],
                database=SOURCE_DB,
                stdout=f,
            )

        log("Wiping target DB...")
        wipe_target(TARGET_DB)

        log("Migrating target DB...")
        call_command("migrate", database=TARGET_DB, interactive=False)

        log("Loading data into target DB...")
        call_command("loaddata", fixture_path, database=TARGET_DB, verbosity=2)

        log("Done.")

    finally:
        if fixture_path and os.path.exists(fixture_path):
            os.remove(fixture_path)


# -------------------------
# Option 2: PostgreSQL method
# Much faster, Postgres only
# Requires pg_dump + pg_restore
# -------------------------


def _pg_binary(name):
    """Locate a PostgreSQL client binary on PATH, falling back to common install
    dirs. Raises with a clear message if it cannot be found."""
    found = shutil.which(name)
    if found:
        return found
    for directory in _PG_FALLBACK_DIRS:
        candidate = os.path.join(directory, name + (".exe" if os.name == "nt" else ""))
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        f"Could not find '{name}'. Install the PostgreSQL client tools or add them "
        "to PATH (or extend _PG_FALLBACK_DIRS)."
    )


def _pg_env(alias):
    """Environment for the pg subprocess with the password supplied out-of-band via
    PGPASSWORD, so it never appears in the process command line."""
    env = os.environ.copy()
    password = settings.DATABASES[alias].get("PASSWORD")
    if password:
        env["PGPASSWORD"] = password
    return env


def postgres_url(alias, with_password=False):
    db = settings.DATABASES[alias]

    name = db["NAME"]
    user = db.get("USER", "")
    password = db.get("PASSWORD", "")
    host = db.get("HOST") or "localhost"
    port = db.get("PORT") or "5432"

    auth = user
    if with_password and password:
        auth += f":{password}"

    return f"postgresql://{auth}@{host}:{port}/{name}"


def clone_with_pg_tools(log=print):
    # Password goes through PGPASSWORD (see _pg_env), not the URL/argv.
    source_url = postgres_url(SOURCE_DB)
    target_url = postgres_url(TARGET_DB)

    log("Cloning with pg_dump / pg_restore...")

    dump = subprocess.Popen(
        [
            _pg_binary("pg_dump"),
            "--format=custom",
            "--no-owner",
            "--no-privileges",
            source_url,
        ],
        stdout=subprocess.PIPE,
        env=_pg_env(SOURCE_DB),
    )

    restore = subprocess.Popen(
        [
            _pg_binary("pg_restore"),
            "--clean",
            "--if-exists",
            "--no-owner",
            "--no-privileges",
            "--dbname",
            target_url,
        ],
        stdin=dump.stdout,
        env=_pg_env(TARGET_DB),
    )

    dump.stdout.close()

    dump_return = dump.wait()
    restore_return = restore.wait()

    if dump_return != 0:
        raise RuntimeError("pg_dump failed")

    if restore_return != 0:
        raise RuntimeError("pg_restore failed")

    log("Done.")


def clone_db(log=print):
    source_engine = settings.DATABASES[SOURCE_DB]["ENGINE"]
    target_engine = settings.DATABASES[TARGET_DB]["ENGINE"]

    if "postgresql" in source_engine and "postgresql" in target_engine:
        log("Detected PostgreSQL -> PostgreSQL. Using fast pg_dump/pg_restore.")
        clone_with_pg_tools(log=log)
    else:
        log("Using Django dumpdata/loaddata.")
        clone_with_django(log=log)

"""Clone the source database into the target database.

    python manage.py clone_db

This WIPES the target database ("target") and replaces it with a copy of the
source ("default"). Because it is destructive, several guardrails must pass before
anything is touched:

  * a "target" database must be configured on the active profile;
  * the target must not resolve to the same database as the source;
  * DEBUG must be on, unless --force is given (wiping a DB from a production
    deployment is almost never intended);
  * an interactive typed confirmation of the target database name, unless
    --no-input is given.

The cloning mechanics live in kernel/base_app/services/clone_db.py.
"""

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from kernel.base_app.services import clone_db as service


def _db_identity(alias):
    """(engine, name, host, port) for a database alias, for comparison and display."""
    cfg = settings.DATABASES[alias]
    return (
        cfg.get("ENGINE", ""),
        str(cfg.get("NAME", "")),
        cfg.get("HOST") or "",
        str(cfg.get("PORT") or ""),
    )


def _describe(alias):
    engine, name, host, port = _db_identity(alias)
    short_engine = engine.rsplit(".", 1)[-1]
    location = f"{host}:{port}" if host else "local"
    return f"{short_engine} name={name} ({location})"


class Command(BaseCommand):
    help = "Clone the 'default' database into the 'target' database (destructive)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--no-input",
            "--noinput",
            action="store_false",
            dest="interactive",
            help="Do not prompt for confirmation (for scripts/CI).",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Allow running when DEBUG is off. Use with care.",
        )

    def handle(self, *args, **options):
        interactive = options["interactive"]
        force = options["force"]

        # Guardrail: a target must be configured.
        target = settings.DATABASES.get(service.TARGET_DB)
        if not target or not target.get("ENGINE"):
            raise CommandError(
                f'No "{service.TARGET_DB}" database is configured on the active '
                "profile. Set target_database_url (or target_local=True) on the "
                "profile before cloning."
            )

        # Guardrail: never let the target resolve to the source (self-wipe).
        if _db_identity(service.SOURCE_DB) == _db_identity(service.TARGET_DB):
            raise CommandError(
                "Source and target resolve to the SAME database; refusing to wipe it."
            )

        # Guardrail: don't wipe from a production deployment unless explicitly forced.
        if not settings.DEBUG and not force:
            raise CommandError(
                "DEBUG is off, refusing to wipe the target database. Re-run with "
                "--force if you are certain this is a disposable target."
            )

        _, target_name, _, _ = _db_identity(service.TARGET_DB)

        self.stdout.write(self.style.WARNING("This will WIPE the target database."))
        self.stdout.write(f"  source (read)  : {_describe(service.SOURCE_DB)}")
        self.stdout.write(f"  target (WIPED) : {_describe(service.TARGET_DB)}")

        # Guardrail: interactive typed confirmation of the target name.
        if interactive:
            answer = input(
                f'\nType the target database name ("{target_name}") to proceed, '
                "anything else to abort: "
            )
            if answer.strip() != target_name:
                raise CommandError("Aborted, confirmation did not match.")

        try:
            service.clone_db(log=self.stdout.write)
        except Exception as exc:  # surface mechanics failures as command errors
            raise CommandError(f"Clone failed: {exc}") from exc

        self.stdout.write(self.style.SUCCESS("Clone complete."))

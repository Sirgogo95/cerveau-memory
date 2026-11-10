from django.apps import AppConfig
from django.conf import settings


class BaseAppConfig(AppConfig):
    name = "kernel.base_app"

    def ready(self):
        # Register the ChangeLog audit receivers once at startup.
        from .signals import connect_audit_signals

        connect_audit_signals()

        # Connect the role-sync receivers only under OIDC. Google logins carry no
        # role claims, so running the sync there would (with REVOKE_WHEN_ABSENT)
        # strip privileges from local users; OIDC_ROLE_SYNC is set only in the
        # OIDC branch of settings.py.
        if getattr(settings, "OIDC_ROLE_SYNC", False):
            from .adapters import connect_oidc_role_sync

            connect_oidc_role_sync()

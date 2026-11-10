"""Optional OIDC role mapping.

allauth's default signup creates a plain local user (`is_staff=False`,
`is_superuser=False`) and never maps identity-provider roles onto Django
permissions. This adapter bridges that gap: it reads role/group claims from the
OIDC token and applies them to the local user, both on first signup and on every
subsequent login, so a privilege change at the provider propagates.

Wired only when OIDC is active: `settings.py` points `SOCIALACCOUNT_ADAPTER` at
this class inside the OIDC block, and `BaseAppConfig.ready()` connects the
re-sync receivers when allauth is installed. When OIDC is off this module is
never imported.

IMPORTANT: the claim names and role values below are placeholders. Set them to
whatever your identity provider actually emits. Inspect
`sociallogin.account.extra_data` on a real login to see the available claims.
"""

from __future__ import annotations

from allauth.socialaccount import signals
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.socialaccount.models import SocialLogin
from django.contrib.auth.models import Group
from django.http import HttpRequest

# --- Configure these for your identity provider -----------------------------

# Claim(s) in the OIDC token that carry the user's roles/groups. The first one
# holding a list wins.
ROLE_CLAIMS = ("roles", "groups")

# Role values that grant Django privileges.
SUPERUSER_ROLES = {"superuser"}
STAFF_ROLES = {"staff", "admin"}

# Mirror the provider's roles into Django groups (created on demand)? This uses
# .set(), so Django group membership becomes an exact mirror of the claim on
# every login (manually-assigned groups are replaced).
SYNC_GROUPS = True

# Remove staff/superuser when the granting role is no longer present on login?
# Keeps a revoked portal admin from retaining Django access. Note: this also
# affects a local break-glass superuser IF they log in through the portal
# without the superuser role. A break-glass account that only ever uses
# /admin/login/ is never touched, since this runs on social login only.
REVOKE_WHEN_ABSENT = True


def _roles_from_claims(extra_data: dict) -> set[str]:
    for claim in ROLE_CLAIMS:
        value = extra_data.get(claim)
        if isinstance(value, (list, tuple, set)):
            return {str(v) for v in value}
    return set()


def sync_user_from_claims(user, sociallogin: SocialLogin) -> None:
    """Apply the provider's role claims to `user`. Idempotent, so it is safe to
    call from both `save_user` and the login signals."""
    roles = _roles_from_claims(sociallogin.account.extra_data or {})

    want_superuser = bool(roles & SUPERUSER_ROLES)
    want_staff = want_superuser or bool(roles & STAFF_ROLES)

    changed = []
    if want_superuser != user.is_superuser and (want_superuser or REVOKE_WHEN_ABSENT):
        user.is_superuser = want_superuser
        changed.append("is_superuser")
    if want_staff != user.is_staff and (want_staff or REVOKE_WHEN_ABSENT):
        user.is_staff = want_staff
        changed.append("is_staff")
    if changed:
        user.save(update_fields=changed)

    if SYNC_GROUPS and roles:
        groups = [Group.objects.get_or_create(name=role)[0] for role in roles]
        user.groups.set(groups)


class OIDCRoleAdapter(DefaultSocialAccountAdapter):
    """Default allauth behaviour (create/link the local user) plus role mapping."""

    def save_user(self, request: HttpRequest, sociallogin: SocialLogin, form=None):
        user = super().save_user(request, sociallogin, form)
        sync_user_from_claims(user, sociallogin)
        return user


class PortalSocialAccountAdapter(DefaultSocialAccountAdapter):
    """Additional-login providers (Google, Microsoft) may log in existing users
    (email match via SOCIALACCOUNT_EMAIL_AUTHENTICATION) but never create new
    accounts. Shared by every provider enabled alongside Django's default login."""

    def is_open_for_signup(self, request, sociallogin):
        return False


def _resync_roles(sender, request, sociallogin, **kwargs):
    if sociallogin.user and sociallogin.user.pk:
        sync_user_from_claims(sociallogin.user, sociallogin)


def connect_oidc_role_sync() -> None:
    """Re-apply roles when a social account is first connected (`added`, which
    covers auto-connect to an existing local user by email) and on every
    subsequent login (`updated`). Idempotent via dispatch_uid."""
    signals.social_account_added.connect(
        _resync_roles, dispatch_uid="oidc_role_sync_added"
    )
    signals.social_account_updated.connect(
        _resync_roles, dispatch_uid="oidc_role_sync_updated"
    )

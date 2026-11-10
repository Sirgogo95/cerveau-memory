"""Account views: SSO logout.

Ending only the local session is not enough under single sign-on: the next request
bounces to the portal, whose own session is still alive, and SOCIALACCOUNT_LOGIN_ON_GET
silently signs the user right back in. So logout kills the local session first, then
sends the browser to the portal's RP-initiated logout endpoint. The portal redirects
back to "/", which the login-required middleware forwards to the portal login page."""

import json
import urllib.request

from django.conf import settings
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_not_required
from django.core.cache import cache
from django.shortcuts import redirect
from django.views.decorators.http import require_POST


def _portal_app():
    """The configured OIDC portal app, or None when OIDC is not the active provider
    (e.g. under Google, where this endpoint just does a plain local logout)."""
    providers = getattr(settings, "SOCIALACCOUNT_PROVIDERS", {})
    apps = (providers.get("openid_connect") or {}).get("APPS") or []
    return apps[0] if apps else None


def _end_session_endpoint():
    """The portal's end_session_endpoint, from its OIDC discovery document.

    Successful lookups are cached; a portal that is down, absent, or has no logout
    endpoint returns "" (caller falls back to local-only logout).
    """
    portal = _portal_app()
    if portal is None:
        return ""
    endpoint = cache.get("portal_end_session_endpoint")
    if endpoint is None:
        server_url = portal["settings"]["server_url"]
        try:
            with urllib.request.urlopen(server_url, timeout=5) as resp:
                endpoint = json.load(resp).get("end_session_endpoint", "")
        except OSError:
            return ""
        cache.set("portal_end_session_endpoint", endpoint, 3600)
    return endpoint


@login_not_required
@require_POST
def sso_logout(request):
    """Kill the local session, then the portal session (RP-initiated logout)."""
    logout(request)
    endpoint = _end_session_endpoint()
    if not endpoint:
        return redirect("/")
    return redirect(f"{endpoint}")

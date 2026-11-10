from django.conf import settings


def debug(request):
    """Expose `debug` to every template (so `{% if debug %}` works without
    INTERNAL_IPS). Used by the annotation script gate, but generally useful."""
    return {"debug": settings.DEBUG}


def navigation(request):
    """Expose navigation data to every template.

    Stub for the base template: returns an empty menu. Populate `nav_items` with
    the entries your app needs, e.g.
        {"label": "Dashboard", "url": reverse("dashboard")}
    """
    return {
        "nav_items": [],
    }

"""URL routes provided by base_app.

These are included from core/urls.py. The SSO logout route lives here rather than
in the project URLconf so the reusable app owns its own endpoints.
"""

from django.urls import path

from kernel.base_app.views.account import sso_logout

urlpatterns = [
    path("logout/", sso_logout, name="sso_logout"),
]

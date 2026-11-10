"""
URL configuration for core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("apps.cerveau.urls")),
]

# Wire up allauth's routes (login/callback/etc.) only when a social provider is
# active, so the bare template still imports when allauth isn't installed or
# configured. base_app's routes (SSO logout) are included before allauth so they
# own their paths.
if "allauth" in settings.INSTALLED_APPS:
    urlpatterns += [
        path("kernel/", include("kernel.base_app.urls")),
        path("accounts/", include("allauth.urls")),
    ]

# Dev-only UI annotation endpoint (see kernel/base_app/views/annotations.py).
# Registered only under DEBUG, so the route does not exist in production.
if settings.DEBUG:
    from kernel.base_app.views.annotations import add_annotation

    urlpatterns += [path("__annotations__/", add_annotation, name="add_annotation")]

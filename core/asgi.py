"""
ASGI config for core project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

application = get_asgi_application()

# Only web servers import this module, so this loads Cerveau's models exactly where
# they're needed: never for manage.py commands or tests. See apps/cerveau/warmup.py.
from apps.cerveau.warmup import start_warm_up  # noqa: E402

start_warm_up()

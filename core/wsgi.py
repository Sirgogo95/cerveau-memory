"""
WSGI config for core project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

application = get_wsgi_application()

# Only web servers import this module, so this loads Cerveau's models exactly where
# they're needed: never for manage.py commands or tests. See apps/cerveau/warmup.py.
from apps.cerveau.warmup import start_warm_up  # noqa: E402

start_warm_up()

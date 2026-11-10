#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import io
import os
import sys
from django.core.management.commands.runserver import Command as runserver
from django.conf import settings


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    runserver.default_port = settings.APP_DEFAULT_PORT
    # On Windows a redirected stdout is cp1252: a character it can't encode (a model
    # wrote a non-breaking hyphen in a summary) crashed `consolidate` mid-run. Escape
    # such characters in the output instead.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="backslashreplace")
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()

"""Dev-only template loaders that auto-inject a `data-tpl` attribute into every
HTML opening tag, tagging it with its source template and line, e.g.
`data-tpl="index.html:19"`. This lets the annotation tooling point Claude at the
exact source line with zero hand-tagging.

Wired only under DEBUG (see core/settings.py). It rewrites template *source* with
a regex, which is pragmatic but has known limits:

  * A literal `<letter` inside an inline <script>/<style> (e.g. `a<b` in JS)
    could be mis-tagged. Prefer external JS/CSS, or accept the stray attribute.
  * Only templates whose name ends in .html / .htm are touched (so text, JSON,
    and email templates are left alone).
  * Loop-rendered elements share one source line, so they share one data-tpl,
    which matches how the annotation store keys by line.
"""

import re

from django.conf import settings
from django.template.loaders.app_directories import Loader as AppDirectoriesLoader
from django.template.loaders.filesystem import Loader as FilesystemLoader

# Opening tag of a real element: "<" + a letter + tag name, followed by
# whitespace, "/" or ">". This skips </close>, <!doctype>, <!-- comments -->, and
# Django's {% %} / {{ }} (which contain no "<").
_OPEN_TAG = re.compile(r"<([a-zA-Z][\w-]*)(?=[\s/>])")


def annotate(template_name, contents):
    if not settings.DEBUG or not template_name:
        return contents
    if not template_name.endswith((".html", ".htm")):
        return contents

    def repl(m):
        line = contents.count("\n", 0, m.start()) + 1
        return f'<{m.group(1)} data-tpl="{template_name}:{line}"'

    return _OPEN_TAG.sub(repl, contents)


class _AnnotateMixin:
    def get_contents(self, origin):
        return annotate(origin.template_name, super().get_contents(origin))


class AnnotatingFilesystemLoader(_AnnotateMixin, FilesystemLoader):
    pass


class AnnotatingAppDirectoriesLoader(_AnnotateMixin, AppDirectoriesLoader):
    pass

"""Recompute tag and entity keys after keys stopped containing punctuation.

"Dr. Marta Peña" used to have the key "dr. marta pena", which a message split into
words ("dr marta pena") could never equal. This rewrites every Tag.key and
Entity.match_keys under the new rule, and fills the new Entity.search_keys.

The key functions are copied here, frozen, instead of imported from brain.text: a
migration must keep doing what it did when it was written, even if those rules
change later.
"""

import re
import unicodedata

from django.db import migrations

TITLES = {"dr", "dra", "mr", "mrs", "ms", "miss", "prof", "sr", "sra", "srta", "don", "dona"}


def _words(text):
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.findall(r"\w+", stripped.casefold())


def _key(text):
    return " ".join(_words(text))


def _identity(name, aliases):
    keys = (_key(n) for n in (name, *aliases))
    return list(dict.fromkeys(k for k in keys if k))


def _search(name, aliases, type_):
    keys = _identity(name, aliases)
    if type_ == "person":
        parts = [w for w in _words(name) if w not in TITLES and len(w) >= 3]
        keys += [p for p in parts if p not in keys]
    return keys


def recompute(apps, schema_editor):
    # Bulk updates on purpose: these are derived keys, not edits worth audit rows.
    Tag = apps.get_model("cerveau", "Tag")
    taken = set()
    for tag in Tag.objects.order_by("id"):
        key = _key(tag.name)
        if key in taken:  # two tags now share a key: keep the older one's
            key = f"{key} #{tag.pk}"
        taken.add(key)
        Tag.objects.filter(pk=tag.pk).update(key=key)

    Entity = apps.get_model("cerveau", "Entity")
    for entity in Entity.objects.all():
        Entity.objects.filter(pk=entity.pk).update(
            match_keys=_identity(entity.name, entity.aliases),
            search_keys=_search(entity.name, entity.aliases, entity.type),
        )


class Migration(migrations.Migration):
    dependencies = [("cerveau", "0006_entity_search_keys")]

    operations = [migrations.RunPython(recompute, migrations.RunPython.noop)]

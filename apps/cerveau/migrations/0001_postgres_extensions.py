from django.contrib.postgres.operations import TrigramExtension, UnaccentExtension
from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    """Enable the Postgres extensions Cerveau's memory relies on: vector (pgvector
    embeddings and HNSW indexes), pg_trgm (fuzzy entity matching), and unaccent
    (accent-insensitive matching). Creating extensions needs a superuser, which the
    Docker dev database user is."""

    initial = True

    dependencies = []

    operations = [
        VectorExtension(),
        TrigramExtension(),
        UnaccentExtension(),
    ]

from django.apps import AppConfig


class CerveauConfig(AppConfig):
    """The single Django app wrapping brain/: models, DjangoMemoryStore, chat views,
    consolidation tasks, and the admin as memory inspector."""

    name = "apps.cerveau"
    label = "cerveau"

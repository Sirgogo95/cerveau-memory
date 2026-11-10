"""Embed memory that has no vector for an embedding model yet: messages, episode
summaries, facts, and tags.

Used after switching models (re-embed everything under the new one) and to backfill
rows stored before embeddings existed. Safe to rerun: rows that already have a vector
for the model are skipped. Vectors of other models are kept, so switching back needs
no re-embedding.

    python manage.py embed_memory                          # the configured profile
    python manage.py embed_memory --profile all-MiniLM-L6-v2
    python manage.py embed_memory --dry-run
"""

from django.conf import settings
from django.core.management.base import BaseCommand

from adapters.embeddings.factory import embedder_from_profile
from apps.cerveau import models
from apps.cerveau.engine import embedding_profile
from apps.cerveau.store import DjangoMemoryStore
from brain.embedding import embed_cached


class Command(BaseCommand):
    help = "Embed messages and episode summaries missing a vector for an embedding model."

    def add_arguments(self, parser):
        parser.add_argument("--profile", default=settings.CERVEAU_EMBEDDING_MODEL,
                            help="An embedding profile in configs/embeddings/.")
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--dry-run", action="store_true", help="Only count what's missing.")

    def handle(self, *args, profile, batch_size, dry_run, **options):
        embedding = embedding_profile(profile)
        model = embedding.model
        messages = models.Message.objects.exclude(embeddings__model=model).order_by("id")
        episodes = (
            models.Episode.objects.exclude(summary_short="")
            .exclude(embeddings__model=model)
            .order_by("id")
        )
        facts = models.Fact.objects.exclude(embeddings__model=model).order_by("id")
        tags = models.Tag.objects.exclude(embeddings__model=model).order_by("id")
        total_messages, total_episodes, total_facts = messages.count(), episodes.count(), facts.count()
        total_tags = tags.count()
        self.stdout.write(
            f"{total_messages} message(s), {total_episodes} episode summary pair(s), "
            f"{total_facts} fact(s), and {total_tags} tag(s) without a {model} vector."
        )
        if dry_run or not (total_messages or total_episodes or total_facts or total_tags):
            return

        embedder = embedder_from_profile(embedding, base_url=settings.CERVEAU_OLLAMA_URL)
        store = DjangoMemoryStore()
        done, after = 0, 0
        # Walk by id rather than re-running "missing" each round: always ends, even if
        # a batch can't be written.
        while batch := list(messages.filter(id__gt=after)[:batch_size]):
            after = batch[-1].pk
            vectors = embed_cached([m.content for m in batch], embedder, store)
            models.MessageEmbedding.objects.bulk_create(
                [
                    models.MessageEmbedding(
                        message=m, model=v.model, embedding=list(v.vector), text_hash=v.text_hash
                    )
                    for m, v in zip(batch, vectors)
                ],
                ignore_conflicts=True,
            )
            done += len(batch)
            self.stdout.write(f"  messages {done}/{total_messages}")

        for episode in episodes:
            short, long = embed_cached(
                [episode.summary_short, episode.summary_long or episode.summary_short],
                embedder, store,
            )
            models.EpisodeEmbedding.objects.update_or_create(
                episode=episode, model=model,
                defaults={
                    "summary_short": list(short.vector), "summary_short_hash": short.text_hash,
                    "summary_long": list(long.vector), "summary_long_hash": long.text_hash,
                },
            )
        tag_rows = list(tags)
        for tag, vector in zip(tag_rows, embed_cached([t.name for t in tag_rows], embedder, store)):
            models.TagEmbedding.objects.get_or_create(
                tag=tag, model=model,
                defaults={"embedding": list(vector.vector), "text_hash": vector.text_hash},
            )
        fact_rows = list(facts)
        for fact, vector in zip(fact_rows, embed_cached([f.statement for f in fact_rows], embedder, store)):
            models.FactEmbedding.objects.get_or_create(
                fact=fact, model=model,
                defaults={"embedding": list(vector.vector), "text_hash": vector.text_hash},
            )
        self.stdout.write(self.style.SUCCESS(
            f"Embedded {done} message(s), {total_episodes} episode(s), {total_facts} fact(s), "
            f"and {len(tag_rows)} tag(s) with {model}."
        ))

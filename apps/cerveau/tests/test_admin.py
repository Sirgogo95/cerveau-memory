"""The admin as memory inspector: facts are listed, filterable, and editable, and
show under the episode they came from."""

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from apps.cerveau.store import DjangoMemoryStore
from brain.types import Embedding, Role
from tests.test_store_contract import T0, at


class FactAdminTests(TestCase):
    def setUp(self):
        self.client.force_login(User.objects.create_superuser("admin", "a@example.com", "pw"))
        store = DjangoMemoryStore()
        episode = store.open_episode(T0)
        message = store.add_message(episode.id, Role.USER, "My sister Lucía lives in Montreal.", at(1))
        store.close_episode(episode.id, at(2))
        self.episode = episode
        self.fact = store.add_fact(
            "Lucía lives in Montreal.", "person", episode_id=episode.id,
            message_ids=[message.id], at=at(3), embedding=Embedding([1.0, 0.0], "m", "h"),
        )

    def test_facts_list_and_change_page(self):
        listing = self.client.get(reverse("admin:cerveau_fact_changelist"))
        self.assertContains(listing, "Lucía lives in Montreal.")
        replaced = self.client.get(reverse("admin:cerveau_fact_changelist"), {"superseded_by__isnull": "False"})
        self.assertNotContains(replaced, "Lucía lives in Montreal.")
        page = self.client.get(reverse("admin:cerveau_fact_change", args=[self.fact.id]))
        self.assertContains(page, "Lucía lives in Montreal.")
        self.assertContains(page, "My sister Lucía lives in Montreal.")  # its source, read-only

    def test_facts_show_under_their_episode(self):
        page = self.client.get(reverse("admin:cerveau_episode_change", args=[self.episode.id]))
        self.assertContains(page, "Lucía lives in Montreal.")

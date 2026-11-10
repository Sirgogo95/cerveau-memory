"""The memory viewer: timeline, episode, people and things, the lens, and changing a
memory's status. Real Django store, fake embedder (no Ollama)."""

from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.cerveau import models
from apps.cerveau.engine import Engine
from apps.cerveau.memory_views import describe_call
from apps.cerveau.store import DjangoMemoryStore
from brain.config import EngineConfig, with_min_score
from brain.ports import EmbedderError
from brain.types import MemoryKind, Role, ToolRun
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE
from tests.test_store_contract import at


class MemoryViewTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()
        self.embedder = FakeEmbedder()
        self.engine = Engine(
            store=self.store, llm=FakeLLM(), embedder=self.embedder, profile=PROFILE,
            config=with_min_score(EngineConfig(), 0.6, search=0.3),
            background_llm=FakeLLM(), background_profile=PROFILE,
        )
        for target in ("apps.cerveau.memory_views.get_engine",):
            patcher = patch(target, return_value=self.engine)
            patcher.start()
            self.addCleanup(patcher.stop)

        # A past, consolidated episode about Lucía, with a stored search.
        self.past = self.store.open_episode(at(0))
        ask = self.say(self.past, Role.USER, "my sister lucia moved to montreal", 1)
        self.reply = self.say(self.past, Role.ASSISTANT, "Noted: **Lucía** lives in Montreal.", 2)
        self.store.close_episode(self.past.id, at(3))
        [vector] = self.embedder.embed(["sister lucia moved to montreal"])
        self.store.save_summary(
            self.past.id, title="Lucía moves", summary_short="Lucía moved to Montreal.",
            summary_long="The user's sister Lucía moved to Montreal.",
            short_embedding=vector, long_embedding=vector,
        )
        tag = self.store.upsert_tag("family")
        assert tag is not None
        self.store.tag_episode(self.past.id, tag.id, 1.0)
        lucia = self.store.upsert_entity("Lucía", "person", ["my sister"])
        assert lucia is not None
        self.lucia = lucia
        self.store.add_mention(lucia.id, ask.id)
        self.store.add_tool_calls(self.past.id, self.reply.id, [ToolRun(
            "search_memory", {"query": "sister", "after": "2026-10-01"},
            '{"query": "sister", "results": [{"id": "M1"}]}', 42, at(2),
        )])
        # The conversation going on now.
        self.now = self.store.open_episode(at(500))
        self.say(self.now, Role.USER, "the build uses docker", 501)

    def say(self, episode, role, text, minutes):
        [vector] = self.embedder.embed([text])
        return self.store.add_message(episode.id, role, text, at(minutes), vector)

    # -- timeline and episodes -----------------------------------------------------

    def test_timeline_lists_episodes_newest_first_with_their_facts(self):
        page = self.client.get(reverse("cerveau:timeline"))
        self.assertContains(page, "Lucía moves")
        self.assertContains(page, "This conversation (still going)")
        self.assertContains(page, "1 memory search")
        self.assertContains(page, "family")
        body = page.content.decode()
        self.assertLess(body.index("This conversation"), body.index("Lucía moves"))

    def test_timeline_filters_by_text_and_status(self):
        found = self.client.get(reverse("cerveau:timeline"), {"q": "montreal"})
        self.assertContains(found, "Lucía moves")
        self.assertNotContains(found, "still going")
        models.Episode.objects.get(pk=self.past.id).set_archived()
        active = self.client.get(reverse("cerveau:timeline"))
        self.assertNotContains(active, "Lucía moves")
        forgotten = self.client.get(reverse("cerveau:timeline"), {"status": "forgotten"})
        self.assertContains(forgotten, "Lucía moves")

    def test_episode_shows_summaries_names_transcript_and_searches(self):
        page = self.client.get(reverse("cerveau:episode", args=[self.past.id]))
        self.assertContains(page, "The user&#x27;s sister Lucía moved to Montreal.")
        self.assertContains(page, reverse("cerveau:entity", args=[self.lucia.id]))
        self.assertContains(page, 'data-markdown>Noted: **Lucía** lives in Montreal.')
        self.assertContains(page, "Searched memory for “sister” · from 2026-10-01")
        self.assertContains(page, "1 found")
        self.assertContains(page, "42 ms")

    def test_unknown_episode_is_404(self):
        self.assertEqual(self.client.get(reverse("cerveau:episode", args=[999])).status_code, 404)

    # -- people and things ----------------------------------------------------------

    def test_entities_and_their_mentions(self):
        listing = self.client.get(reverse("cerveau:entities"))
        self.assertContains(listing, "Lucía")
        self.assertContains(listing, "also my sister")
        self.assertContains(listing, "1 mention")
        detail = self.client.get(reverse("cerveau:entity", args=[self.lucia.id]))
        self.assertContains(detail, "my sister lucia moved to montreal")
        self.assertContains(detail, "Lucía moves")

    # -- lens -----------------------------------------------------------------------

    def test_lens_scores_memories_against_both_thresholds(self):
        page = self.client.get(reverse("cerveau:lens"), {"q": "sister lucia moved to montreal"})
        self.assertContains(page, "0.60 and above")
        self.assertContains(page, "0.30 and above")
        rows = page.context["rows"]
        self.assertTrue(rows)
        self.assertEqual(rows, sorted(rows, key=lambda r: r.score, reverse=True))
        best = rows[0]
        self.assertTrue(best.automatic and best.searchable)
        self.assertEqual([e.name for e in page.context["named"]], ["Lucía"])

    # -- on my mind -------------------------------------------------------------------

    def with_activation_on(self):
        config = replace(self.engine.config, episodes=replace(self.engine.config.episodes, w_act=0.05))
        return patch("apps.cerveau.memory_views.get_engine", return_value=replace(self.engine, config=config))

    def test_on_my_mind_lists_active_nodes_strongest_first_faded_to_now(self):
        tag = models.Tag.objects.get(name="family")
        now = timezone.now()
        self.store.save_activation({
            (MemoryKind.TAG, tag.pk): (0.4, now),
            (MemoryKind.ENTITY, self.lucia.id): (0.9, now - timedelta(hours=6)),   # halved: 0.45
            (MemoryKind.ENTITY, 999): (0.8, now),                                  # gone: skipped
        })
        with self.with_activation_on():
            page = self.client.get(reverse("cerveau:mind"))
        rows = page.context["rows"]
        self.assertEqual([r.name for r in rows], ["Lucía", "family"])
        self.assertAlmostEqual(rows[0].value, 0.45, places=2)
        self.assertEqual(rows[1].kind, "topic")
        self.assertContains(page, reverse("cerveau:entity", args=[self.lucia.id]))
        self.assertContains(page, "On my mind")

    def test_on_my_mind_says_when_nothing_is_on_it_or_activation_is_off(self):
        with self.with_activation_on():
            self.assertContains(self.client.get(reverse("cerveau:mind")), "Nothing on my mind")
        self.assertContains(self.client.get(reverse("cerveau:mind")), "Activation is off")

    def test_lens_shows_what_a_question_would_bring_to_mind_without_saving_it(self):
        with self.with_activation_on():
            page = self.client.get(reverse("cerveau:lens"), {"q": "how is lucia doing?"})
        self.assertEqual([m.name for m in page.context["on_mind"]][:1], ["Lucía"])
        self.assertContains(page, "On my mind with this question")
        self.assertEqual(self.store.load_activation(), {})

    def test_lens_shows_topics_close_to_the_question_and_their_boost(self):
        tag = models.Tag.objects.get(name="family")
        [vector] = self.embedder.embed(["family"])
        models.TagEmbedding.objects.create(tag=tag, model=vector.model, embedding=list(vector.vector),
                                           text_hash=vector.text_hash)
        config = replace(self.engine.config, tag_match_score=0.5,
                         episodes=replace(self.engine.config.episodes, w_tag=0.1))
        with patch("apps.cerveau.memory_views.get_engine", return_value=replace(self.engine, config=config)):
            page = self.client.get(reverse("cerveau:lens"), {"q": "family lucia"})
        self.assertContains(page, "Topics close to your question")
        self.assertEqual([t.name for t, _ in page.context["topics"]], ["family"])
        [episode] = [r for r in page.context["rows"] if r.candidate.id == self.past.id and r.memory_id[0] == "E"]
        self.assertIsNotNone(episode.candidate.topical)

    def test_lens_leaves_out_the_current_conversation_unless_asked(self):
        query = {"q": "the build uses docker"}
        default = self.client.get(reverse("cerveau:lens"), query).context["rows"]
        self.assertNotIn(self.now.id, [r.candidate.episode_id for r in default])
        included = self.client.get(reverse("cerveau:lens"), {**query, "current": "1"}).context["rows"]
        self.assertIn(self.now.id, [r.candidate.episode_id for r in included if r.current])

    def test_lens_applies_a_period_like_a_real_turn(self):
        page = self.client.get(reverse("cerveau:lens"), {"q": "what did we talk about in July 2020?"})
        self.assertContains(page, "July 2020")
        self.assertEqual(page.context["rows"], [])  # nothing is from then

    def test_lens_reports_an_unreachable_embedder(self):
        with patch.object(self.embedder, "embed", side_effect=EmbedderError("Ollama is unreachable")):
            page = self.client.get(reverse("cerveau:lens"), {"q": "anything new"})
        self.assertContains(page, "The embedding model didn")
        self.assertContains(page, "Ollama is unreachable")

    def test_empty_lens_just_shows_the_form(self):
        page = self.client.get(reverse("cerveau:lens"))
        self.assertContains(page, 'name="q"')
        self.assertNotIn("rows", page.context)

    # -- status -----------------------------------------------------------------------

    def post_status(self, kind, pk, action, next_url="/memory/"):
        return self.client.post(
            reverse("cerveau:status", args=[kind, pk]), {"action": action, "next": next_url}
        )

    def test_forget_set_aside_and_restore(self):
        url = reverse("cerveau:episode", args=[self.past.id])
        response = self.post_status("episode", self.past.id, "forget", url)
        self.assertRedirects(response, url, fetch_redirect_response=False)
        self.assertEqual(models.Episode.objects.get(pk=self.past.id).status, "Z")
        self.post_status("episode", self.past.id, "restore")
        self.assertEqual(models.Episode.objects.get(pk=self.past.id).status, "A")
        self.post_status("entity", self.lucia.id, "set_aside")
        self.assertEqual(models.Entity.objects.get(pk=self.lucia.id).status, "I")

    def test_a_forgotten_episode_leaves_the_engine_s_memory(self):
        self.post_status("episode", self.past.id, "forget")
        [vector] = self.embedder.embed(["my sister lucia moved to montreal"])
        hits = self.store.semantic_candidates(models_kind("message"), vector, 10)
        self.assertNotIn(self.past.id, [h.episode_id for h in hits])

    def test_status_changes_are_audited(self):
        from kernel.base_app.models import ChangeLog
        before = ChangeLog.objects.count()
        self.post_status("entity", self.lucia.id, "forget")
        self.assertGreater(ChangeLog.objects.count(), before)

    def test_bad_requests(self):
        self.assertEqual(self.post_status("message", 1, "forget").status_code, 404)
        self.assertEqual(self.post_status("episode", self.past.id, "delete").status_code, 404)
        self.assertEqual(
            self.client.get(reverse("cerveau:status", args=["episode", self.past.id])).status_code, 405
        )
        outside = self.post_status("episode", self.past.id, "set_aside", "https://evil.example/")
        self.assertEqual(outside["Location"], "/memory/")

    # -- facts ---------------------------------------------------------------------------

    def add_facts(self):
        [old_v] = self.embedder.embed(["lucia lives in montreal"])
        old = self.store.add_fact("Lucía lives in Montreal.", "person", episode_id=self.past.id,
                                  message_ids=[self.reply.id - 1], at=at(3), embedding=old_v)
        [new_v] = self.embedder.embed(["lucia lives in toronto"])
        new = self.store.add_fact("Lucía lives in Toronto.", "person", episode_id=self.past.id,
                                  message_ids=[self.reply.id - 1], at=at(400), embedding=new_v)
        self.store.supersede_fact(old.id, new.id)
        return old, new

    def test_facts_list_hides_replaced_ones_unless_asked(self):
        old, new = self.add_facts()
        current = self.client.get(reverse("cerveau:facts"))
        self.assertContains(current, "Lucía lives in Toronto.")
        self.assertNotContains(current, "Lucía lives in Montreal.")
        both = self.client.get(reverse("cerveau:facts"), {"history": "1"})
        self.assertContains(both, "Lucía lives in Montreal.")
        self.assertContains(both, "replaced")

    def test_a_fact_shows_its_sources_and_history(self):
        old, new = self.add_facts()
        page = self.client.get(reverse("cerveau:fact", args=[old.id]))
        self.assertContains(page, "Replaced by")
        self.assertContains(page, "my sister lucia moved to montreal")   # what you said
        self.assertContains(page, "How it changed")
        self.assertContains(page, reverse("cerveau:fact", args=[new.id]))
        episode = self.client.get(reverse("cerveau:episode", args=[self.past.id]))
        self.assertContains(episode, "Facts learned from it")

    def test_forgetting_a_fact(self):
        _, new = self.add_facts()
        self.post_status("fact", new.id, "forget")
        self.assertEqual(models.Fact.objects.get(pk=new.id).status, "Z")

    def test_the_lens_includes_facts(self):
        _, new = self.add_facts()
        page = self.client.get(reverse("cerveau:lens"), {"q": "lucia lives in toronto"})
        self.assertIn(f"F{new.id}", [r.memory_id for r in page.context["rows"]])

    # -- words --------------------------------------------------------------------------

    def test_tool_calls_in_words(self):
        self.assertEqual(
            describe_call("search_memory", {"query": "Pedro", "entities": ["Pedro"], "kinds": ["episode"]}),
            "Searched memory for “Pedro” · about Pedro · episodes only",
        )
        self.assertEqual(describe_call("read_memory", {"id": "M34"}), "Opened message M34")
        self.assertEqual(describe_call("read_memory", {"id": "F7"}), "Opened fact F7")
        self.assertEqual(describe_call("read_memory", {"id": "E12", "detail": "messages"}),
                         "Opened E12 · its messages")


def models_kind(name):
    from brain.types import MemoryKind
    return MemoryKind(name)

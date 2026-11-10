"""Warm start: models load at server start in the background, failures never crash."""

from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from apps.cerveau.warmup import start_warm_up, sweep_consolidation, warm_up
from brain.config import EngineConfig
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE


class Loadable(FakeLLM):
    def __init__(self, error: Exception | None = None):
        super().__init__()
        self.loaded = 0
        self.error = error

    def load(self):
        self.loaded += 1
        if self.error:
            raise self.error


class LoadableEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.loaded = 0

    def load(self):
        self.loaded += 1


def engine(llm=None, embedder=None) -> Engine:
    llm = llm or Loadable()
    return Engine(
        store=DjangoMemoryStore(), llm=llm, embedder=embedder or LoadableEmbedder(),
        profile=PROFILE, config=EngineConfig(), background_llm=llm, background_profile=PROFILE,
    )


class WarmUpTests(SimpleTestCase):
    def test_loads_both_models(self):
        e = engine()
        with self.assertLogs("cerveau", "INFO") as logs:
            warm_up(e)
        self.assertEqual((e.llm.loaded, e.embedder.loaded), (1, 1))  # type: ignore[attr-defined]
        self.assertEqual(sum("ready" in line for line in logs.output), 2)

    def test_a_failure_is_logged_not_raised(self):
        e = engine(llm=Loadable(error=ConnectionError("Ollama is not running")))
        with self.assertLogs("cerveau", "WARNING") as logs:
            warm_up(e)  # must not raise
        self.assertIn("Ollama is not running", logs.output[0])
        self.assertEqual(e.embedder.loaded, 1)  # type: ignore[attr-defined]

    def test_parts_without_load_are_skipped(self):
        warm_up(engine(llm=FakeLLM(), embedder=FakeEmbedder()))  # no error

    @override_settings(CERVEAU_WARM_START=False)
    def test_disabled_by_setting(self):
        with patch("apps.cerveau.warmup.get_engine") as get_engine:
            self.assertEqual(start_warm_up(), [])
            get_engine.assert_not_called()

    @override_settings(CERVEAU_WARM_START=True)
    def test_models_and_sweep_run_in_their_own_background_threads(self):
        e = engine()
        with (
            patch("apps.cerveau.warmup.get_engine", return_value=e),
            patch("apps.cerveau.warmup.sweep_consolidation") as sweep,
            self.assertLogs("cerveau", "INFO"),
        ):
            threads = start_warm_up()
            for thread in threads:
                thread.join(timeout=5)
        self.assertEqual(len(threads), 3)
        self.assertTrue(all(t.daemon for t in threads))
        self.assertEqual((e.llm.loaded, e.embedder.loaded), (1, 1))  # type: ignore[attr-defined]
        sweep.assert_called_once()

    def test_a_failed_sweep_is_logged_not_raised(self):
        with (
            patch("apps.cerveau.tasks.sweep", side_effect=ConnectionError("database down")),
            self.assertLogs("cerveau", "WARNING") as logs,
        ):
            sweep_consolidation()
        self.assertIn("database down", logs.output[0])

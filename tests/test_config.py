"""Config loading and validation, and building a provider from a profile."""

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from adapters.llm.factory import provider_from_profile
from adapters.llm.ollama import OllamaProvider
from brain.config import (
    ConfigError,
    EmbeddingProfile,
    EngineConfig,
    ModelProfile,
    SourceConfig,
    load_embedding_profile,
    load_engine_config,
    load_model_profile,
    with_min_score,
)

REPO = Path(__file__).resolve().parent.parent


def profile(**overrides) -> ModelProfile:
    values: dict[str, Any] = dict(
        name="m", provider="ollama", context_window=4096, output_reserve=512,
        memory_budget=2000,
    )
    values.update(overrides)
    return ModelProfile(**values)


class LoadingTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_shipped_engine_config_loads(self):
        config = load_engine_config(REPO / "configs" / "default.toml")
        self.assertEqual(config.episode_gap, timedelta(hours=2))

    def test_shipped_model_profiles_load(self):
        paths = list((REPO / "configs" / "models").glob("*.toml"))
        self.assertTrue(paths)
        for path in paths:
            load_model_profile(path)

    def test_shipped_embedding_profiles_load_and_are_named_after_their_file(self):
        paths = list((REPO / "configs" / "embeddings").glob("*.toml"))
        self.assertTrue(paths)
        for path in paths:
            self.assertEqual(load_embedding_profile(path).name, path.stem)

    def test_embedding_profile_validation(self):
        with self.assertRaisesRegex(ConfigError, "unknown provider"):
            EmbeddingProfile("x", "openai", "m", 0.5, 512)
        with self.assertRaisesRegex(ConfigError, "min_score"):
            EmbeddingProfile("x", "ollama", "m", 1.5, 512)
        with self.assertRaisesRegex(ConfigError, "max_tokens"):
            EmbeddingProfile("x", "ollama", "m", 0.5, 0)
        with self.assertRaisesRegex(ConfigError, "unknown keys"):
            load_embedding_profile(self.write(
                'name = "x"\nprovider = "ollama"\nmodel = "m"\nmin_score = 0.5\n'
                'max_tokens = 512\nthreshold = 0.4\n'
            ))

    def test_short_reply_chars_cannot_be_negative(self):
        with self.assertRaisesRegex(ConfigError, "short_reply_chars"):
            EngineConfig(short_reply_chars=-1)

    def test_with_min_score_sets_every_source_and_keeps_the_rest(self):
        config = with_min_score(EngineConfig(), 0.53)
        self.assertEqual((config.messages.min_score, config.episodes.min_score), (0.53, 0.53))
        self.assertEqual(config.episodes.max_items, EngineConfig().episodes.max_items)
        self.assertEqual(config.messages.w_lexical, EngineConfig().messages.w_lexical)

    def test_episode_gap_hours_becomes_a_timedelta(self):
        config = load_engine_config(self.write("episode_gap_hours = 0.5\n"))
        self.assertEqual(config.episode_gap, timedelta(minutes=30))

    def test_missing_keys_use_defaults(self):
        self.assertEqual(load_engine_config(self.write("")), EngineConfig())

    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "recent_windw"):
            load_engine_config(self.write("recent_windw = 5\n"))

    def test_missing_required_profile_key_is_rejected(self):
        with self.assertRaises(ConfigError):
            load_model_profile(self.write('name = "m"\nprovider = "ollama"\n'))

    def test_profile_options_table_is_loaded(self):
        loaded = load_model_profile(
            self.write(
                'name = "m"\nprovider = "ollama"\ncontext_window = 4096\n'
                "output_reserve = 512\nmemory_budget = 2000\n\n[options]\ntemperature = 0.2\n"
            )
        )
        self.assertEqual(loaded.options, {"temperature": 0.2})

    def test_shipped_config_gives_episodes_a_tag_weight(self):
        config = load_engine_config(REPO / "configs" / "default.toml")
        self.assertGreater(config.episodes.w_tag, 0)
        self.assertEqual((config.messages.w_tag, config.facts.w_tag), (0.0, 0.0))

    def test_source_table_overrides_only_the_keys_it_names(self):
        config = load_engine_config(self.write("[episodes]\nw_lexical = 0.2\n"))
        self.assertEqual(config.episodes.w_lexical, 0.2)
        self.assertEqual(config.episodes.max_items, EngineConfig().episodes.max_items)
        self.assertEqual(config.messages, EngineConfig().messages)

    def test_min_score_in_engine_config_points_to_the_embedding_profile(self):
        with self.assertRaisesRegex(ConfigError, "configs/embeddings"):
            load_engine_config(self.write("[messages]\nmin_score = 0.5\n"))
        for key in ("search_min_score", "fact_same_score", "fact_related_score", "tag_same_score",
                    "tag_match_score"):
            with self.assertRaisesRegex(ConfigError, f"{key} depends on the embedding model"):
                load_engine_config(self.write(f"{key} = 0.5\n"))

    def test_unknown_key_in_a_source_table_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "min_scor"):
            load_engine_config(self.write("[messages]\nmin_scor = 0.5\n"))

    def test_unreadable_file_raises_config_error(self):
        with self.assertRaises(ConfigError):
            load_engine_config(REPO / "configs" / "missing.toml")


class ValidationTests(unittest.TestCase):
    def test_budget_and_reserve_must_leave_room_in_the_window(self):
        with self.assertRaisesRegex(ConfigError, "must leave room"):
            profile(context_window=2000, output_reserve=500, memory_budget=1500)

    def test_sizes_must_be_positive(self):
        with self.assertRaises(ConfigError):
            profile(memory_budget=0)

    def test_engine_values_are_checked(self):
        with self.assertRaises(ConfigError):
            EngineConfig(recent_window=0)
        with self.assertRaises(ConfigError):
            EngineConfig(episode_gap=timedelta(0))
        with self.assertRaises(ConfigError):
            EngineConfig(memory_share=1.0)
        with self.assertRaises(ConfigError):
            SourceConfig(min_score=1.5)
        with self.assertRaises(ConfigError):
            EngineConfig(recent_episodes=-1)
        with self.assertRaisesRegex(ConfigError, "summary_chunk"):
            EngineConfig(recent_window=10, summary_chunk=11)
        with self.assertRaises(ConfigError):
            EngineConfig(summary_chunk=0)
        with self.assertRaisesRegex(ConfigError, "Unknown timezone"):
            EngineConfig(timezone="Mars/Olympus_Mons")
        self.assertEqual(EngineConfig(timezone="America/Santo_Domingo").timezone, "America/Santo_Domingo")


class FactoryTests(unittest.TestCase):
    def test_ollama_provider_runs_with_the_profile_limits(self):
        provider = provider_from_profile(profile(options={"temperature": 0.3}))
        assert isinstance(provider, OllamaProvider)
        self.assertEqual(provider.model, "m")
        self.assertEqual(
            provider.options, {"temperature": 0.3, "num_ctx": 4096, "num_predict": 512}
        )

    def test_thinking_is_off_unless_the_profile_turns_it_on(self):
        default = provider_from_profile(profile())
        enabled = provider_from_profile(profile(think=True))
        assert isinstance(default, OllamaProvider) and isinstance(enabled, OllamaProvider)
        self.assertIs(default.think, False)
        self.assertIs(enabled.think, True)

    def test_profile_limits_win_over_conflicting_options(self):
        provider = provider_from_profile(profile(options={"num_ctx": 999_999}))
        assert isinstance(provider, OllamaProvider)
        self.assertEqual(provider.options["num_ctx"], 4096)

    def test_extra_arguments_reach_the_provider(self):
        provider = provider_from_profile(profile(), base_url="http://gpu-box:11434")
        assert isinstance(provider, OllamaProvider)
        self.assertEqual(provider.base_url, "http://gpu-box:11434")

    def test_a_remote_profile_uses_its_server_and_the_key_local_ones_never_get_it(self):
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "secret"}):
            cloud = provider_from_profile(profile(base_url="https://ollama.com"), base_url="http://127.0.0.1:11434")
            local = provider_from_profile(profile(), base_url="http://127.0.0.1:11434")
        assert isinstance(cloud, OllamaProvider) and isinstance(local, OllamaProvider)
        self.assertEqual(cloud.base_url, "https://ollama.com")
        self.assertEqual(cloud.session.headers["Authorization"], "Bearer secret")
        self.assertNotIn("Authorization", local.session.headers)

    def test_unknown_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            provider_from_profile(profile(provider="nope"))


if __name__ == "__main__":
    unittest.main()

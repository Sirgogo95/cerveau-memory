"""evals.tag_cases: tags written into case files without disturbing the rest."""

import tempfile
import unittest
from pathlib import Path

from evals.cases import load_scenarios
from evals.tag_cases import write_tags

CASE = """{
  "name": "t",
  "episodes": [
    {
      "key": "dog",
      "messages": [["user", "We adopted a dog."]]
    },
    {
      "key": "padre",
      "tags": ["pets"],
      "messages": [["user", "Mi pap\u00e1 cumple 70."]]
    }
  ]
}
"""


class WriteTagsTests(unittest.TestCase):
    def test_adds_and_replaces_tags_after_the_key(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "t.json"
            file.write_text(CASE, encoding="utf-8")
            write_tags(file, {"dog": ["pets", "adoption"], "padre": ["family", "birthday"]})
            text = file.read_text(encoding="utf-8")
            [scenario] = load_scenarios(file)
        self.assertEqual([e.tags for e in scenario.episodes],
                         [("pets", "adoption"), ("family", "birthday")])
        self.assertIn('      "key": "dog",\n      "tags": ["pets", "adoption"],\n      "messages"', text)
        self.assertNotIn('"tags": ["pets"],', text)

    def test_an_unknown_episode_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "t.json"
            file.write_text(CASE, encoding="utf-8")
            with self.assertRaises(ValueError):
                write_tags(file, {"cat": ["pets"]})


if __name__ == "__main__":
    unittest.main()

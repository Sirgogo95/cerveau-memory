import unittest

from brain.hashing import is_stale, text_hash


class TextHashTests(unittest.TestCase):
    def test_ignores_surrounding_whitespace(self):
        self.assertEqual(text_hash("  Paris\n"), text_hash("Paris"))

    def test_differs_for_different_text(self):
        self.assertNotEqual(text_hash("Paris"), text_hash("paris"))

    def test_is_hex_sha256(self):
        self.assertEqual(len(text_hash("x")), 64)

    def test_is_stale(self):
        stored = text_hash("old summary")
        self.assertFalse(is_stale(stored, "old summary "))
        self.assertTrue(is_stale(stored, "new summary"))


if __name__ == "__main__":
    unittest.main()

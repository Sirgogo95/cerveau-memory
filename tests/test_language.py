"""Detecting the user's language, and the reply-language hint in the prompt."""

import unittest

from brain.language import detect_language, main_language
from brain.prompt import assemble_prompt
from brain.types import Role
from tests.test_engine import PROFILE, message


class DetectLanguageTests(unittest.TestCase):
    def test_clear_messages_in_each_language(self):
        self.assertEqual(detect_language("What did we talk about in September?"), "English")
        self.assertEqual(detect_language("¿Qué tengo que preparar para la boda de mi hermano?"), "Spanish")
        self.assertEqual(detect_language("Qu'est-ce que je dois préparer pour le congrès ?"), "French")
        self.assertEqual(detect_language("Dime cuándo es el viaje"), "Spanish")
        self.assertEqual(detect_language("Je pars à Québec avec ma collègue"), "French")
        # "me" is as French as it is Spanish: it mustn't tip the balance.
        self.assertEqual(detect_language("Combien va me coûter la réparation de la voiture ?"), "French")

    def test_spanish_punctuation_counts(self):
        self.assertEqual(detect_language("¿Y Pedro?"), "Spanish")

    def test_unclear_messages_get_no_answer(self):
        self.assertIsNone(detect_language("ok"))
        self.assertIsNone(detect_language("Pedro"))
        self.assertIsNone(detect_language("SELECT * FROM users;"))
        self.assertIsNone(detect_language(""))


class MainLanguageTests(unittest.TestCase):
    def test_most_clear_messages_decide(self):
        self.assertEqual(main_language(["What did I decide for the app?", "ok",
                                        "Which database should I use for the project?"]), "English")

    def test_short_messages_count_together(self):
        self.assertEqual(main_language(["is it", "the one", "you said"]), "English")

    def test_a_tie_or_nothing_clear_gives_no_answer(self):
        self.assertIsNone(main_language(["¿Qué base de datos uso para el proyecto?", "What is the best one for me?"]))
        self.assertIsNone(main_language(["ok", "hola"]))


class ReplyLanguageHintTests(unittest.TestCase):
    def test_the_hint_ends_the_last_user_message_in_the_prompt_only(self):
        recent = [message(1, "Hola"), message(2, "What is a decorator?")]
        prompt = assemble_prompt(recent, PROFILE, reply_language="English")
        self.assertEqual(prompt[-1].role, Role.USER)
        self.assertEqual(prompt[-1].content, "What is a decorator?\n\n(Reply in English.)")
        self.assertEqual(recent[-1].content, "What is a decorator?")  # stored text untouched

    def test_no_language_no_hint(self):
        prompt = assemble_prompt([message(1, "ok")], PROFILE)
        self.assertEqual(prompt[-1].content, "ok")


if __name__ == "__main__":
    unittest.main()

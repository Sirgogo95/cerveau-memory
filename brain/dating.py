"""Relative dates made absolute: "last Friday" in a fact learned on Sunday 22 October
2023 becomes "on Friday 20 October 2023".

A fact is recalled with the date it was learned, so a fact saying "last Friday" was
read as that Sunday: gemma4:e4b wrote a year in 53 of its 2,102 LoCoMo facts and left
368 relative, despite the prompt's rule, and answered 31% of the temporal questions.
The anchor is the date of the line a fact cites (or a summary's conversation), so
the rewrite is exact, in code, for any model.

Exact days ("yesterday", "last Friday", "tomorrow") get a date; spans get anchored
the way LoCoMo's answer key words them ("the week before 22 October 2023"); vague
words ("recently", "the other day") and plain weekdays ("on Friday": past or future?)
are left alone. English, Spanish, and French, each written in its own language.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta

from .periods import MONTH_NAMES

_ES_MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
              "septiembre", "octubre", "noviembre", "diciembre")
_FR_MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
              "septembre", "octobre", "novembre", "décembre")
_EN_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_ES_DAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_FR_DAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_ES_DAY_KEYS = ("lunes", "martes", "mi[eé]rcoles", "jueves", "viernes", "s[aá]bado", "domingo")

# After these words a date stands alone: "since 21 October 2023", not "since on ...".
_BARE_AFTER = re.compile(r"\b(?:since|until|till|by|before|after|from|of|desde|hasta|antes|después|depuis|jusqu'au|avant|après)\s+$", re.I)


def en(d: date) -> str:
    return f"{d.day} {MONTH_NAMES[d.month - 1]} {d.year}"


def es(d: date) -> str:
    return f"{d.day} de {_ES_MONTHS[d.month - 1]} de {d.year}"


def fr(d: date) -> str:
    return f"{d.day} {_FR_MONTHS[d.month - 1]} {d.year}"


def _month_after(d: date, months: int) -> date:
    index = d.year * 12 + d.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _last(day: date, weekday: int) -> date:
    """The most recent `weekday` strictly before `day`."""
    return day - timedelta(days=(day.weekday() - weekday - 1) % 7 + 1)


def _next(day: date, weekday: int) -> date:
    """The first `weekday` strictly after `day`."""
    return day + timedelta(days=(weekday - day.weekday() - 1) % 7 + 1)


def _saturday_of(day: date) -> date:
    """The Saturday of the weekend `day` is in, or of the coming one."""
    return day - timedelta(days=day.weekday() - 5) if day.weekday() >= 5 else day + timedelta(days=5 - day.weekday())


Rule = tuple[re.Pattern[str], Callable[[re.Match[str], date], tuple[str, str]]]


def _both(phrase: str) -> tuple[str, str]:
    """A phrase that reads the same with or without a preposition before it."""
    return phrase, phrase


def _rules() -> list[Rule]:
    """(pattern, phrase) pairs, longest first; phrase gives (with a preposition, bare)."""
    day_words = "|".join(_EN_DAYS)
    rules: list[Rule] = [
        # English
        (re.compile(r"\b(?:the )?day before yesterday\b", re.I),
         lambda m, d: (f"on {en(d - timedelta(2))}", en(d - timedelta(2)))),
        (re.compile(r"\byesterday (morning|afternoon|evening)\b", re.I),
         lambda m, d: (f"on the {m.group(1).lower()} of {en(d - timedelta(1))}", en(d - timedelta(1)))),
        (re.compile(r"\blast night\b", re.I),
         lambda m, d: (f"on the night of {en(d - timedelta(1))}", en(d - timedelta(1)))),
        (re.compile(r"\byesterday\b", re.I), lambda m, d: (f"on {en(d - timedelta(1))}", en(d - timedelta(1)))),
        (re.compile(r"\btomorrow (morning|afternoon|evening|night)\b", re.I),
         lambda m, d: (f"on the {m.group(1).lower()} of {en(d + timedelta(1))}", en(d + timedelta(1)))),
        (re.compile(r"\btomorrow\b", re.I), lambda m, d: (f"on {en(d + timedelta(1))}", en(d + timedelta(1)))),
        (re.compile(rf"\b(?:last|this past|past) ({day_words})\b", re.I),
         lambda m, d: _weekday(m, d, _last)),
        (re.compile(rf"\bnext ({day_words})\b", re.I), lambda m, d: _weekday(m, d, _next)),
        (re.compile(r"\b(?:the )?(?:last|this past|past) weekend\b", re.I),
         lambda m, d: _both(f"the weekend before {en(d)}")),
        (re.compile(r"\b(?:the )?next weekend\b", re.I), lambda m, d: _both(f"the weekend after {en(d)}")),
        (re.compile(r"\bthis weekend\b", re.I),
         lambda m, d: _both(f"the weekend of {en(_saturday_of(d))}")),
        (re.compile(r"\b(?:the )?(?:last|this past|past|previous) week\b(?! of)(?!'s)", re.I),
         lambda m, d: _both(f"the week before {en(d)}")),
        (re.compile(r"\b(?:the )?next week\b(?! of)(?!'s)", re.I), lambda m, d: _both(f"the week after {en(d)}")),
        (re.compile(r"\b(\d+|a|one|two|three|four|five|six) days? ago\b", re.I),
         lambda m, d: _ago(m, d, 1)),
        (re.compile(r"\b(\d+|a|one|two|three|four) weeks? ago\b", re.I),
         lambda m, d: _both(f"{m.group(1).lower()} week{'s' * (_count(m.group(1)) > 1)} before {en(d)}")),
        # The rest of "... ago", anchored without inventing precision: "a few days
        # before 22 October 2023", "a month before ...", "years before ...".
        (re.compile(r"\b((?:a few|a couple of|several|many|some|\d+|a|one|two|three|four|five|six|seven|eight|nine|ten) )?"
                    r"(days?|weeks?|months?|years?) ago\b", re.I),
         lambda m, d: _both(f"{m.group(1) or ''}{m.group(2).lower()} before {en(d)}")),
        (re.compile(r"\btoday\b", re.I), lambda m, d: (f"on {en(d)}", en(d))),
        (re.compile(r"\b(?:the )?(?:last|past) month\b(?! of)(?!'s)", re.I),
         lambda m, d: _month(_month_after(d, -1))),
        (re.compile(r"\bnext month\b(?!'s)", re.I), lambda m, d: _month(_month_after(d, 1))),
        (re.compile(r"\bthis month\b(?!'s)", re.I), lambda m, d: _month(d)),
        (re.compile(r"\b(?:the )?(?:last|past) year\b(?! of)(?!'s)", re.I), lambda m, d: (f"in {d.year - 1}", str(d.year - 1))),
        (re.compile(r"\bnext year\b(?!'s)", re.I), lambda m, d: (f"in {d.year + 1}", str(d.year + 1))),
        (re.compile(r"\bthis year\b(?!'s)", re.I), lambda m, d: (f"in {d.year}", str(d.year))),
        # Spanish ("mañana" also means "morning": not after "la", "esta", "cada", "por")
        (re.compile(r"\banteayer\b", re.I), lambda m, d: (f"el {es(d - timedelta(2))}", es(d - timedelta(2)))),
        (re.compile(r"\bayer\b", re.I), lambda m, d: (f"el {es(d - timedelta(1))}", es(d - timedelta(1)))),
        (re.compile(r"(?<!la )(?<!esta )(?<!cada )(?<!por )\bmañana\b(?! por)", re.I),
         lambda m, d: (f"el {es(d + timedelta(1))}", es(d + timedelta(1)))),
        (re.compile(rf"\bel ({'|'.join(_ES_DAY_KEYS)}) pasado\b", re.I),
         lambda m, d: _es_weekday(m, d)),
        (re.compile(r"\b(?:la )?semana pasada\b", re.I),
         lambda m, d: _both(f"la semana anterior al {es(d)}")),
        (re.compile(r"\b(?:la )?próxima semana\b|\b(?:la )?semana que viene\b", re.I),
         lambda m, d: _both(f"la semana posterior al {es(d)}")),
        (re.compile(r"\b(?:el )?mes pasado\b", re.I), lambda m, d: _es_month(_month_after(d, -1))),
        (re.compile(r"\b(?:el )?próximo mes\b|\b(?:el )?mes que viene\b", re.I),
         lambda m, d: _es_month(_month_after(d, 1))),
        (re.compile(r"\b(?:el )?año pasado\b", re.I), lambda m, d: (f"en {d.year - 1}", str(d.year - 1))),
        (re.compile(r"\b(?:el )?próximo año\b|\b(?:el )?año que viene\b", re.I),
         lambda m, d: (f"en {d.year + 1}", str(d.year + 1))),
        # French
        (re.compile(r"\bavant-hier\b", re.I), lambda m, d: (f"le {fr(d - timedelta(2))}", fr(d - timedelta(2)))),
        (re.compile(r"\bhier\b", re.I), lambda m, d: (f"le {fr(d - timedelta(1))}", fr(d - timedelta(1)))),
        (re.compile(r"\bdemain\b", re.I), lambda m, d: (f"le {fr(d + timedelta(1))}", fr(d + timedelta(1)))),
        (re.compile(rf"\b({'|'.join(_FR_DAYS)}) dernier\b", re.I), lambda m, d: _fr_weekday(m, d)),
        (re.compile(r"\b(?:la )?semaine (?:dernière|passée)\b", re.I),
         lambda m, d: _both(f"la semaine précédant le {fr(d)}")),
        (re.compile(r"\b(?:la )?semaine prochaine\b", re.I),
         lambda m, d: _both(f"la semaine suivant le {fr(d)}")),
        (re.compile(r"\b(?:le )?mois (?:dernier|passé)\b", re.I), lambda m, d: _fr_month(_month_after(d, -1))),
        (re.compile(r"\b(?:le )?mois prochain\b", re.I), lambda m, d: _fr_month(_month_after(d, 1))),
        (re.compile(r"\bl'année (?:dernière|passée)\b", re.I), lambda m, d: (f"en {d.year - 1}", str(d.year - 1))),
        (re.compile(r"\bl'année prochaine\b", re.I), lambda m, d: (f"en {d.year + 1}", str(d.year + 1))),
    ]
    return rules


_NUMBERS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _count(word: str) -> int:
    return int(word) if word.isdigit() else _NUMBERS[word.lower()]


def _ago(m: re.Match[str], d: date, unit: int) -> tuple[str, str]:
    day = d - timedelta(days=_count(m.group(1)) * unit)
    return f"on {en(day)}", en(day)


def _weekday(m: re.Match[str], d: date, which: Callable[[date, int], date]) -> tuple[str, str]:
    index = [w.lower() for w in _EN_DAYS].index(m.group(1).lower())
    day = which(d, index)
    return f"on {_EN_DAYS[index]} {en(day)}", f"{_EN_DAYS[index]} {en(day)}"


def _es_weekday(m: re.Match[str], d: date) -> tuple[str, str]:
    key = m.group(1).lower().replace("é", "e").replace("á", "a")
    index = [w.replace("é", "e").replace("á", "a") for w in _ES_DAYS].index(key)
    day = _last(d, index)
    return f"el {_ES_DAYS[index]} {es(day)}", f"{_ES_DAYS[index]} {es(day)}"


def _fr_weekday(m: re.Match[str], d: date) -> tuple[str, str]:
    index = _FR_DAYS.index(m.group(1).lower())
    day = _last(d, index)
    return f"le {_FR_DAYS[index]} {fr(day)}", f"{_FR_DAYS[index]} {fr(day)}"


def _month(first: date) -> tuple[str, str]:
    return f"in {MONTH_NAMES[first.month - 1]} {first.year}", f"{MONTH_NAMES[first.month - 1]} {first.year}"


def _es_month(first: date) -> tuple[str, str]:
    return f"en {_ES_MONTHS[first.month - 1]} de {first.year}", f"{_ES_MONTHS[first.month - 1]} de {first.year}"


def _fr_month(first: date) -> tuple[str, str]:
    return f"en {_FR_MONTHS[first.month - 1]} {first.year}", f"{_FR_MONTHS[first.month - 1]} {first.year}"


RULES = _rules()


def annotate_dates(text: str, day: date) -> str:
    """`text` with each relative date followed by the date it means, in brackets, the
    words kept: "a pic from last Friday [Friday 19 May 2023]". For text quoted as said
    (a message shown in the notes), where rewriting the words would misquote it. On
    LoCoMo, 38 of gemma4:e4b's 48 temporal misses with the date in its notes had it in
    a message line; annotated, temporal went 65% to 69% and 70% to 74% on two halves of
    the conversations (8 turned right, 1 and 0 wrong)."""
    for pattern, phrase in RULES:
        def mark(m: re.Match[str]) -> str:
            _, bare = phrase(m, day)
            return f"{m.group(0)} [{bare}]"
        text = pattern.sub(mark, text)
    return text


def absolute_dates(text: str, day: date) -> str:
    """`text` with its relative dates made absolute, anchored on `day` (the date it
    was said). Each rule runs once over the text; a rewritten phrase isn't matched
    again, since it holds no relative words."""
    for pattern, phrase in RULES:
        def swap(m: re.Match[str]) -> str:
            with_preposition, bare = phrase(m, day)
            new = bare if _BARE_AFTER.search(text[:m.start()]) else with_preposition
            return new[0].upper() + new[1:] if m.group(0)[0].isupper() and m.start() == 0 else new
        text = pattern.sub(swap, text)
    return text

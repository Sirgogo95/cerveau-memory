"""Time expressions in a message: "in August", "last week", "la semana pasada".

When a message asks about a period, automatic recall adds the memories dated in it,
whatever their similarity: "what did we talk about in August?" resembles nothing that
was said in August, and the model doesn't search for periods on its own (measured
2026-10-17: it described September's conversations as August's, 12 times in 12).

English, Spanish, and French, matched on normalized words (no accents or case), so
"août" and "aout" are the same. A month name only counts after a preposition ("in",
"en", "durante", "au mois de") or before a year: "May I ask..." is not a month.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo

from .text import normalize_key

MONTHS = {
    1: ("january", "enero", "janvier"),
    2: ("february", "febrero", "fevrier"),
    3: ("march", "marzo", "mars"),
    4: ("april", "abril", "avril"),
    5: ("may", "mayo", "mai"),
    6: ("june", "junio", "juin"),
    7: ("july", "julio", "juillet"),
    8: ("august", "agosto", "aout"),
    9: ("september", "septiembre", "setiembre", "septembre"),
    10: ("october", "octubre", "octobre"),
    11: ("november", "noviembre", "novembre"),
    12: ("december", "diciembre", "decembre"),
}
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December")
_MONTH_BY_WORD = {word: number for number, words in MONTHS.items() for word in words}
_MONTH_WORDS = "|".join(sorted(_MONTH_BY_WORD, key=len, reverse=True))
# "in august", "en agosto", "durante agosto", "au mois d aout", "en el mes de agosto",
# then an optional year; or a month directly followed by a year: "august 2026".
_MONTH = re.compile(
    rf"\b(?:(?:in|during|en|durante|dans|pendant|au|el)(?: el| le)?(?: mois| mes)?(?: d| de)? "
    rf"(?P<m1>{_MONTH_WORDS})(?: (?:de |of )?(?P<y1>\d{{4}}))?"
    rf"|(?P<m2>{_MONTH_WORDS}) (?:de |of )?(?P<y2>\d{{4}}))\b"
)
_RELATIVE = (
    ("yesterday", re.compile(r"\b(?:yesterday|ayer|hier)\b")),
    ("last_week", re.compile(
        r"\b(?:last week|past week|(?:la )?semana pasada|(?:la )?semaine (?:derniere|passee))\b")),
    ("this_week", re.compile(r"\b(?:this week|esta semana|cette semaine)\b")),
    ("last_month", re.compile(
        r"\b(?:last month|past month|(?:el )?mes pasado|(?:le )?mois (?:dernier|passe))\b")),
    ("this_month", re.compile(r"\b(?:this month|este mes|ce mois ci|ce mois)\b")),
    ("last_year", re.compile(
        r"\b(?:last year|(?:el )?ano pasado|(?:l )?annee (?:derniere|passee))\b")),
)


@dataclass(frozen=True, slots=True)
class Period:
    """A stretch of time a message asks about: `start` inclusive, `end` exclusive,
    both timezone-aware. `label` says it in words for the prompt."""

    start: datetime
    end: datetime
    label: str

    def contains(self, moment: datetime | None) -> bool:
        return moment is not None and self.start <= moment < self.end


def find_period(text: str, now: datetime, tz: tzinfo | None = None) -> Period | None:
    """The period `text` asks about, in the user's timezone (the machine's when `tz`
    is None), or None. A named month comes before relative words."""
    key = normalize_key(text)
    today = now.astimezone(tz).date()
    match = _MONTH.search(key)
    if match:
        month = _MONTH_BY_WORD[match.group("m1") or match.group("m2")]
        year_text = match.group("y1") or match.group("y2")
        # A month still ahead this year, without a year, is almost always about a
        # plan ("what am I doing in December?"), and a plan made in September isn't
        # dated December: no window, normal recall finds it by meaning.
        if year_text or month <= today.month:
            year = int(year_text) if year_text else today.year
            first = date(year, month, 1)
            return _period(first, _next_month(first), f"{MONTH_NAMES[month - 1]} {year}", tz)
    for name, pattern in _RELATIVE:
        if pattern.search(key):
            return _relative(name, today, tz)
    return None


def _relative(name: str, today: date, tz: tzinfo | None) -> Period:
    monday = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    if name == "yesterday":
        day = today - timedelta(days=1)
        return _period(day, today, "yesterday", tz)
    if name == "last_week":
        return _period(monday - timedelta(days=7), monday, "last week", tz)
    if name == "this_week":
        return _period(monday, today + timedelta(days=1), "this week", tz)
    if name == "last_month":
        start = (month_start - timedelta(days=1)).replace(day=1)
        return _period(start, month_start, f"last month, {MONTH_NAMES[start.month - 1]} {start.year}", tz)
    if name == "this_month":
        return _period(month_start, today + timedelta(days=1), "this month", tz)
    # last_year
    start = date(today.year - 1, 1, 1)
    return _period(start, date(today.year, 1, 1), f"last year, {start.year}", tz)


def _next_month(first: date) -> date:
    return date(first.year + (first.month == 12), first.month % 12 + 1, 1)


def _period(first: date, end: date, label: str, tz: tzinfo | None) -> Period:
    start = datetime.combine(first, time())
    stop = datetime.combine(end, time())
    if tz is not None:
        start, stop = start.replace(tzinfo=tz), stop.replace(tzinfo=tz)
    else:
        start, stop = start.astimezone(), stop.astimezone()
    last = end - timedelta(days=1)
    span = f"{first:%b} {first.day} to {last:%b} {last.day}" if last != first else f"{first:%b} {first.day}"
    return Period(start=start, end=stop, label=f"{label} ({span})")


def period_note(period: Period, found: bool) -> str:
    """The line the memory block opens with when the message asks about a period."""
    if found:
        return (f"The user asked about {period.label}. Notes dated then are included; "
                "go by the dates on the notes, not by which conversations are recent.")
    return (f"The user asked about {period.label}. No conversation took place then. If "
            "they mean what was said at that time, say you don't remember anything from "
            "then; don't describe conversations from other dates as if they were. If they "
            "mean something planned for that time, the notes below still apply.")

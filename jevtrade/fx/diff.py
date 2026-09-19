"""What changed since the last statement, found by code so the model can judge it.

A rate decision is read by comparing it with the previous one. Traders do this
by hand within minutes: pull up the July statement next to the September one,
find the sentences that moved, decide whether each move is a signal or a
rewrite. The diff itself is mechanical -- ``difflib`` does it in a millisecond
-- and the judgment is not, which is exactly the split this repo keeps making.

So the code finds the previous statement of the same family and the sentences
that differ, and the model is asked, per changed sentence, which way it cuts and
whether it is a real change or a rephrasing. Those questions ride along in round
one at no extra latency, because width is free.
"""

from __future__ import annotations

import bisect
import difflib
import re
from typing import Iterable, Sequence

from .documents import Document

# A single capital letter before a period is an initial ("Lorie K. Logan"), not
# the end of a sentence. Getting this wrong split the September FOMC vote
# paragraph into three fragments and made the diff look bigger than it was.
_ABBREVIATIONS = frozenset(
    "mr mrs ms dr jr sr st no vs etc inc corp co approx fig al ca pp vol "
    "sen rep gov messrs prof".split()
)
_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[\"“‘(\[]?[A-Z0-9])')
_INITIALS = re.compile(r"(?:[A-Za-z]\.){1,4}$")
# Fragments that survive tag-stripping on every CMS in the world.
_CHROME = re.compile(
    r"^(share|menu|home|print|last update|return to text|for media inquiries|"
    r"skip to|back to)\b",
    re.I,
)

MIN_SENTENCE = 20


def split_sentences(text: str, *, min_chars: int = MIN_SENTENCE) -> list[str]:
    """Sentences, with initials kept whole and boilerplate dropped."""
    out: list[str] = []
    for part in _SPLIT.split(re.sub(r"\s+", " ", text or "").strip()):
        if out:
            tail = out[-1].rstrip()
            last = tail.rsplit(" ", 1)[-1] if " " in tail else tail
            if _INITIALS.fullmatch(last) or last.rstrip(".").lower() in _ABBREVIATIONS:
                out[-1] = f"{tail} {part}"
                continue
        out.append(part)
    return [s.strip() for s in out if len(s.strip()) >= min_chars and not _CHROME.match(s.strip())]


# ------------------------------------------------------------ the predecessor

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_DATE_BITS = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|q[1-4]|h[12])\b",
    re.I,
)
_NUMBERS = re.compile(r"[\d–—-]+")


def title_family(title: str) -> str:
    """A title with its dates and numbers removed, so editions collapse together.

    "Minutes of the Federal Open Market Committee, July 28-29, 2026" and the
    June edition of the same thing become one family; "Federal Reserve issues
    FOMC statement" is already its own.
    """
    text = _YEAR.sub(" ", (title or "").lower())
    text = _DATE_BITS.sub(" ", text)
    text = _NUMBERS.sub(" ", text)
    text = re.sub(r"[^a-z ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Editions:
    """Documents grouped by (issuer, kind, title family), each list in time order.

    The linear scan this replaces is fine for a sixty-day window and quadratic
    over seventeen years: 2,300 documents is 5.3 million title comparisons, done
    once per document. Built once, the previous edition is a bisection.
    """

    def __init__(self, documents: Iterable[Document]) -> None:
        self.by_family: dict[tuple[str, str, str], list[Document]] = {}
        for document in documents:
            if not document.body:
                continue  # an empty body cannot be diffed against
            key = (document.issuer, document.kind, title_family(document.title))
            self.by_family.setdefault(key, []).append(document)
        self._stamps: dict[tuple[str, str, str], list[float]] = {}
        for key, group in self.by_family.items():
            group.sort(key=lambda d: d.ts)
            self._stamps[key] = [d.ts for d in group]

    def previous(self, document: Document) -> Document | None:
        """The most recent earlier document from the same issuer, kind and family."""
        key = (document.issuer, document.kind, title_family(document.title))
        group = self.by_family.get(key)
        if not group:
            return None
        i = bisect.bisect_left(self._stamps[key], document.ts)
        while i > 0:
            i -= 1
            if group[i].id != document.id:
                return group[i]
        return None


def previous_of(document: Document, documents: Iterable[Document]) -> Document | None:
    """The most recent earlier document from the same issuer, kind and family."""
    if isinstance(documents, Editions):
        return documents.previous(document)
    return Editions(documents).previous(document)


# ----------------------------------------------------------------- the diff


def changed_sentences(
    old_body: str, new_body: str, limit: int = 12
) -> list[tuple[str, str]]:
    """``(was, now)`` pairs for the slots that differ, in document order.

    An inserted sentence pairs with ``""`` and a deleted one pairs ``""`` on the
    right, so the model always sees both sides of a slot and can tell "this
    sentence is gone" from "this sentence was reworded".
    """
    old = split_sentences(old_body)
    new = split_sentences(new_body)
    pairs: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        width = max(i2 - i1, j2 - j1)
        for k in range(width):
            was = old[i1 + k] if i1 + k < i2 else ""
            now = new[j1 + k] if j1 + k < j2 else ""
            pairs.append((was, now))
    return pairs[:limit]


def diff_for(
    document: Document, documents: Sequence[Document] | Editions, *, limit: int = 12
) -> tuple[Document | None, list[tuple[str, str]]]:
    """The previous edition and what changed, or ``(None, [])`` if there is none.

    ``documents`` may be an ``Editions`` index built once for the whole corpus,
    which is what a run over seventeen years passes.
    """
    previous = previous_of(document, documents)
    if previous is None or not document.body:
        return None, []
    return previous, changed_sentences(previous.body, document.body, limit=limit)


__all__ = [
    "Editions", "changed_sentences", "diff_for", "previous_of", "split_sentences",
    "title_family",
]

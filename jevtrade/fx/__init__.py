"""Central-bank text, read by the same tree, graded on spot FX.

The listing study pointed the reader at a feed where the incumbent is fast but
illiterate. This one points it at a feed where the incumbent is *literate but
slow* -- a human desk reading a statement -- and where the machine-readable
part of the same event is already gone.

The thesis, stated so it can fail: in FX the numeric releases (NFP, CPI, the
headline rate itself) are priced within five minutes, because a number needs no
reading and every machine on the tape has it. The *text* events -- rate-decision
statements, minutes, press conferences, speeches, testimony, intervention
language -- keep moving price for fifteen to sixty minutes, because somebody has
to read them first. A model that answers thirty questions about a statement in
one 400 ms round is early relative to a fifteen-minute digestion, and that is
the only window this study claims.

What is measured here and what is assumed:

* **Measured** -- the tape. Every arm turns a document into (pair, side) and is
  graded by signed log returns on spot FX bars, against a null of the same pair
  and side at random nearby moments.
* **Measured** -- the feeds. Timestamps, archive depth and parsing are checked
  against the live sources; documents without a minute-precision timestamp are
  dropped rather than guessed at, and the count is reported.
* **Assumed** -- that the published timestamp is when the text became readable.
  For a scheduled statement that is nearly true (the page is pollable to the
  second). For a speech or a wire headline it is not, and no arm here can fix
  that; see the README's "What this does not show".
* **Not claimed** -- that any of this is executable. Entry is at the open of the
  first bar *after* the timestamp, which on 5-minute bars is up to five minutes
  late, on purpose.

The same tree is also pointed at a retail FX **wire**, where the stream is not
one issuer's scheduled text but everything a scalper reads: data prints from
every country, every central bank's speakers, intervention talk, tariffs,
geopolitics and order flow. That mode is graded on 1-minute candles rather than
ticks, because twenty thousand posts a year is a different budget, and it
carries one more caveat of its own: a wire runs behind the primary feeds, so
its post time is not the event's time. The ``pre`` column measures that gap
instead of assuming it away. Nothing is fetched from that source until its own
``robots.txt`` has been read and has said yes.

Every number is arithmetic on the model's probabilities. The model is never
asked to compute one.
"""

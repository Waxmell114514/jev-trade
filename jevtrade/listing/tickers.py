"""Candidate tickers, found by code.

Jev answers questions; it does not return strings. So the tokens an
announcement might be about are extracted here, generously, and the model is
asked about each one. Over-extraction is the point: a notice removing the ADA/TUSD pair
puts ADA on the list, and deciding that ADA is *not* being delisted is exactly
the judgment we want to measure.
"""

from __future__ import annotations

import re

# Quote, settlement and fiat symbols: nobody "buys the listing" in these.
QUOTE = frozenset(
    "USDT USDC FDUSD BUSD TUSD USD1 USDP DAI USDE USDS USD EUR TRY BRL ARS JPY "
    "GBP ZAR UAH PLN RON CZK MXN COP NGN KRW IDR VND AUD CAD CHF RUB BIDR BVND "
    "USDⓈ".split()
)
# Capitalised words that sit inside parentheses in every second notice.
STOP = frozenset(
    "UTC AM PM APR APY VIP API P2P NFT ETF KYC FAQ OTC CM UM UID ID TBA TBD "
    "GMT NA AI USDS-M COIN-M LP TGE IPO IEO DEX CEX FDV ATH ATL EST PST CET "
    "SGT HKT JST KST PDT".split()
)
_SYMBOL = r"[A-Z0-9\u4e00-\u9fff]{2,12}"
NAME_TICKER = re.compile(rf"\(({_SYMBOL})\)")
PAIR = re.compile(
    rf"(?<![A-Z0-9])({_SYMBOL})/(?:USDT|USDC|USD|BTC|ETH|BNB|FDUSD|USD1|TRY|EUR|BRL|TUSD|USDE)\b"
)
PERP = re.compile(r"(?<![A-Z0-9/])([A-Z0-9\u4e00-\u9fff]{2,12}?)(?:USDT|USDC)\b")
MULTIPLIER = re.compile(r"^(?:1000000|1000|1M)(?=[A-Z])")


def _acceptable(token: str) -> bool:
    if token in QUOTE or token in STOP:
        return False
    if token.isdigit():
        return False
    if re.fullmatch(r"[0-9\-:.]+", token):
        return False
    return True


def candidates(title: str, body: str = "", *, limit: int = 24) -> list[str]:
    """Unique tickers in order of first appearance, title before body."""
    seen: list[str] = []
    for text in (title or "", body or ""):
        found: list[tuple[int, str]] = []
        for pattern in (NAME_TICKER, PAIR, PERP):
            for match in pattern.finditer(text):
                token = MULTIPLIER.sub("", match.group(1))
                if _acceptable(token):
                    found.append((match.start(), token))
        for _, token in sorted(found):
            if token not in seen:
                seen.append(token)
    return seen[:limit]

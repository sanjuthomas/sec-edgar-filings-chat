from __future__ import annotations

import logging
import re

from app.models import CompanyRecord, ResolvedTicker

TICKER_TOKEN = re.compile(r"\b[A-Z]{2,5}\b")
COMPANY_SUFFIX = re.compile(
    r"\b(INC\.?|CORP\.?|LTD\.?|LLC|CO\.?|PLC|GROUP|HOLDINGS?|THE)\b",
    re.IGNORECASE,
)

log = logging.getLogger(__name__)


class TickerResolver:
    def __init__(self) -> None:
        self._known_tickers: list[str] = []
        self._known_companies: list[CompanyRecord] = []

    @classmethod
    def for_testing(
        cls,
        tickers: list[str],
        companies: list[CompanyRecord],
    ) -> TickerResolver:
        resolver = cls()
        resolver._known_tickers = tickers
        resolver._known_companies = companies
        return resolver

    def load_metadata(self, tickers: list[str], companies: list[CompanyRecord]) -> None:
        self._known_tickers = tickers
        self._known_companies = companies

    def resolve(self, question: str, explicit_ticker: str | None) -> ResolvedTicker:
        if explicit_ticker:
            return ResolvedTicker(ticker=explicit_ticker.strip().upper(), inferred=False)

        from_company = self._detect_company_name(question)
        if from_company:
            return ResolvedTicker(ticker=from_company, inferred=True)

        from_ticker = self._detect_ticker_symbol(question)
        if from_ticker:
            return ResolvedTicker(ticker=from_ticker, inferred=True)

        return ResolvedTicker(ticker=None, inferred=False)

    def _detect_ticker_symbol(self, question: str) -> str | None:
        ticker_set = {ticker.upper() for ticker in self._known_tickers}
        candidates = TICKER_TOKEN.findall(question)
        matches = [candidate for candidate in candidates if candidate in ticker_set]
        if not matches:
            return None
        return max(matches, key=len)

    def _detect_company_name(self, question: str) -> str | None:
        normalized_question = self._normalize_text(question)
        best: CompanyRecord | None = None
        for company in self._known_companies:
            if self._company_mentioned(normalized_question, company.company_name):
                if best is None or len(company.company_name) > len(best.company_name):
                    best = company
        return best.ticker if best else None

    def _company_mentioned(self, normalized_question: str, company_name: str) -> bool:
        normalized_company = self._normalize_company_name(company_name)
        if len(normalized_company) < 4:
            return False
        if normalized_company in normalized_question:
            return True
        words = normalized_company.split()
        if len(words) >= 2:
            return f"{words[0]} {words[1]}" in normalized_question
        return len(words[0]) >= 5 and words[0] in normalized_question

    def _normalize_company_name(self, company_name: str) -> str:
        return self._normalize_text(COMPANY_SUFFIX.sub(" ", company_name))

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()

"""Live provider checks. Run with `pytest -m network`; skipped by default in CI-like runs without network."""

import pytest

from bigboss.data.providers.base import HttpClient, ProviderError

pytestmark = pytest.mark.network


def _http(tmp_path, name, ua="BigBoss research (test@example.com)"):
    return HttpClient(tmp_path / name, 2.0, user_agent=ua, max_retries=2)


def test_sec_edgar_fundamentals(tmp_path):
    from bigboss.data.providers.sec import SecEdgarProvider, normalize_company_facts

    sec = SecEdgarProvider(_http(tmp_path, "sec"))
    try:
        facts = sec.company_facts("AAPL")
    except ProviderError as e:
        pytest.skip(f"SEC unreachable: {e}")
    s = normalize_company_facts(facts)
    assert {"revenue", "net_income", "equity", "shares"} <= set(s)
    assert (s["revenue"]["available"] > s["revenue"]["end"]).all()  # filed after period end


@pytest.mark.parametrize("provider", ["stooq", "yahoo"])
def test_price_provider(tmp_path, provider):
    from bigboss.data.providers.prices import StooqProvider, YahooChartProvider

    p = {"stooq": StooqProvider, "yahoo": YahooChartProvider}[provider](_http(tmp_path, provider, "Mozilla/5.0"))
    try:
        df = p.fetch_daily("SPY", "2024-01-01")
    except ProviderError as e:
        pytest.skip(f"{provider} unreachable from this network: {e}")
    assert len(df) > 100 and (df["close"] > 0).all()

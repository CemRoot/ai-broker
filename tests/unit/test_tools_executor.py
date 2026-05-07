import pytest

from app.core.config import Settings
from app.tools.executor import ToolDeps, ToolExecutor


class _StubT212:
    async def is_us_equity_instrument_tradeable(self, ticker: str) -> tuple[bool, str]:
        if ticker.upper().strip() == "ZZZ":
            return False, "ZZZ_US_EQ not listed"
        return True, "AAPL_US_EQ"


@pytest.mark.asyncio
async def test_tool_executor_unknown_tool():
    deps = ToolDeps(
        settings=Settings(),
        db=None,
        http_client=None,
        cerebras=None,
        groq=None,
        retriever=None,
        paper_broker=None,
        screener=None,
        t212=None,
    )
    ex = ToolExecutor(deps)
    out = await ex.execute("does_not_exist", {})
    assert "Unknown tool" in out


@pytest.mark.asyncio
async def test_check_t212_equity_instrument_tool_skipped_when_not_t212_mode():
    s = Settings()
    s.paper_execution_backend = "supabase"
    deps = ToolDeps(
        settings=s,
        db=None,
        http_client=None,
        cerebras=None,
        groq=None,
        retriever=None,
        paper_broker=None,
        screener=None,
        t212=_StubT212(),
    )
    ex = ToolExecutor(deps)
    out = await ex.execute("check_t212_equity_instrument", {"ticker": "AAPL"})
    assert "skipped" in out.lower()


@pytest.mark.asyncio
async def test_check_t212_equity_instrument_tool_ok_when_t212_mode():
    s = Settings(t212_demo_api_key="k", t212_demo_api_secret="s")
    s.paper_execution_backend = "t212"
    deps = ToolDeps(
        settings=s,
        db=None,
        http_client=None,
        cerebras=None,
        groq=None,
        retriever=None,
        paper_broker=None,
        screener=None,
        t212=_StubT212(),
    )
    ex = ToolExecutor(deps)
    ok_line = await ex.execute("check_t212_equity_instrument", {"ticker": "AAPL"})
    assert "OK" in ok_line
    no_line = await ex.execute("check_t212_equity_instrument", {"ticker": "ZZZ"})
    assert "NO" in no_line

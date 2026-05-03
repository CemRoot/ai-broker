import pytest

from app.tools.executor import ToolDeps, ToolExecutor
from app.core.config import Settings


@pytest.mark.asyncio
async def test_tool_executor_unknown_tool():
    deps = ToolDeps(
        settings=Settings(),
        db=None,
        http_client=None,
        groq=None,
        ollama=None,
        retriever=None,
        paper_broker=None,
        screener=None,
        t212=None,
    )
    ex = ToolExecutor(deps)
    out = await ex.execute("does_not_exist", {})
    assert "Unknown tool" in out


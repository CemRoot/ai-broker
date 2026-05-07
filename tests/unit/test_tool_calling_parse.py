import pytest

from app.services.llm.tool_calling import _extract_json_array, analyze_with_tools


def test_extract_json_array_no_json():
    reasoning, decisions = _extract_json_array("hello world")
    assert reasoning == "hello world"
    assert decisions == []


def test_extract_json_array_with_json():
    text = "Analysis line.\n\n[{\"ticker\":\"AAPL\",\"action\":\"HOLD\"}]"
    reasoning, decisions = _extract_json_array(text)
    assert "Analysis" in reasoning
    assert decisions == [{"ticker": "AAPL", "action": "HOLD"}]


class _DummyTools:
    async def execute(self, tool_name: str, arguments: dict | None = None) -> str:
        return ""


@pytest.mark.asyncio
async def test_analyze_with_tools_returns_safe_when_no_llm():
    result = await analyze_with_tools(
        cerebras=None,
        groq=None,
        tool_executor=_DummyTools(),
        system_prompt="",
        user_message="hello",
        tools=[],
        max_iterations=1,
    )
    assert result.decisions == []
    assert "No LLM available" in result.reasoning_text

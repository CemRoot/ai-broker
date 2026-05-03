from app.services.llm.tool_calling import _extract_json_array


def test_extract_json_array_no_json():
    reasoning, decisions = _extract_json_array("hello world")
    assert reasoning == "hello world"
    assert decisions == []


def test_extract_json_array_with_json():
    text = "Analysis line.\n\n[{\"ticker\":\"AAPL\",\"action\":\"HOLD\"}]"
    reasoning, decisions = _extract_json_array(text)
    assert "Analysis" in reasoning
    assert decisions == [{"ticker": "AAPL", "action": "HOLD"}]


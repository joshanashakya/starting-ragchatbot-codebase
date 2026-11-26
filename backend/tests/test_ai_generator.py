from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_generator import AIGenerator


def text_response(text):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def tool_response(*calls):
    blocks = [
        SimpleNamespace(type="tool_use", id=f"tu_{i}", name=name, input=inp)
        for i, (name, inp) in enumerate(calls)
    ]
    return SimpleNamespace(stop_reason="tool_use", content=blocks)


@pytest.fixture
def gen():
    with patch("ai_generator.anthropic.Anthropic") as cls:
        g = AIGenerator("key", "model-x")
        g.client = cls.return_value
        yield g


def test_plain_answer(gen):
    gen.client.messages.create.return_value = text_response("hi")
    assert gen.generate_response("q") == "hi"
    kwargs = gen.client.messages.create.call_args.kwargs
    assert kwargs["model"] == "model-x"
    assert kwargs["messages"] == [{"role": "user", "content": "q"}]
    assert kwargs["system"] == AIGenerator.SYSTEM_PROMPT
    assert "tools" not in kwargs


def test_history_is_appended_to_system_prompt(gen):
    gen.client.messages.create.return_value = text_response("hi")
    gen.generate_response("q", conversation_history="User: a\nAssistant: b")
    system = gen.client.messages.create.call_args.kwargs["system"]
    assert system.endswith("Previous conversation:\nUser: a\nAssistant: b")


def test_tools_are_passed_with_auto_choice(gen):
    gen.client.messages.create.return_value = text_response("hi")
    tools = [{"name": "t"}]
    gen.generate_response("q", tools=tools, tool_manager=MagicMock())
    kwargs = gen.client.messages.create.call_args.kwargs
    assert kwargs["tools"] == tools and kwargs["tool_choice"] == {"type": "auto"}


def test_tool_use_flow_makes_second_call_without_tools(gen):
    first = tool_response(("search_course_content", {"query": "mcp", "lesson_number": 0}))
    gen.client.messages.create.side_effect = [first, text_response("final")]
    tm = MagicMock()
    tm.execute_tool.return_value = "RESULT"

    out = gen.generate_response("q", tools=[{"name": "t"}], tool_manager=tm)

    assert out == "final"
    tm.execute_tool.assert_called_once_with("search_course_content", query="mcp", lesson_number=0)
    assert gen.client.messages.create.call_count == 2
    second = gen.client.messages.create.call_args.kwargs
    assert "tools" not in second  # only one search round is possible
    assert second["messages"][1] == {"role": "assistant", "content": first.content}
    assert second["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu_0", "content": "RESULT"}],
    }


def test_multiple_tool_calls_return_one_result_message(gen):
    gen.client.messages.create.side_effect = [
        tool_response(("a", {}), ("b", {"x": 1})),
        text_response("done"),
    ]
    tm = MagicMock()
    tm.execute_tool.side_effect = ["A", "B"]
    gen.generate_response("q", tools=[{}], tool_manager=tm)
    results = gen.client.messages.create.call_args.kwargs["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_0", "tu_1"]
    assert [r["content"] for r in results] == ["A", "B"]


def test_tool_use_without_tool_manager_returns_first_block_text(gen):
    """Documents current behaviour: no manager means the tool_use response is not handled."""
    resp = SimpleNamespace(stop_reason="tool_use", content=[SimpleNamespace(type="text", text="partial")])
    gen.client.messages.create.return_value = resp
    assert gen.generate_response("q", tools=[{}]) == "partial"
    assert gen.client.messages.create.call_count == 1

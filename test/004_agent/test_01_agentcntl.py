# Third-party imports
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

# Local imports
from cape.agent.agentcntl import AgentCntl, show_reasoning


# Current vLLM reasoning extension is shown after completion
def test_01_show_reasoning(capsys):
    msg = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "content": "final answer",
        "reasoning": "  inspected the run matrix  ",
    })
    assert show_reasoning(msg) is True
    output = capsys.readouterr().out
    assert "[reasoning]" in output
    assert "inspected the run matrix" in output
    assert "final answer" not in output


# Older reasoning-content extension remains supported
def test_02_show_legacy_reasoning(capsys):
    msg = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "content": "final answer",
        "reasoning_content": "  inspected the legacy response  ",
    })
    assert show_reasoning(msg) is True
    output = capsys.readouterr().out
    assert "inspected the legacy response" in output


# Servers that omit reasoning produce no extra output
def test_03_no_reasoning(capsys):
    msg = ChatCompletionMessage(role="assistant", content="final answer")
    assert show_reasoning(msg) is False
    assert capsys.readouterr().out == ""


# Long reasoning paragraphs use the same terminal wrapping as final answers
def test_04_reasoning_wraps(capsys):
    reasoning = " ".join(["reasoning-word"] * 12)
    msg = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "content": "final answer",
        "reasoning": reasoning,
    })
    assert show_reasoning(msg) is True
    output = capsys.readouterr().out
    assert reasoning not in output
    assert output.count("reasoning-word") == 12


def test_05_tui_sections_ignore_visibility_options(capsys):
    class FakeOpts:
        def get_ModelOpt(self, model, name):
            return 2

        def get_opt(self, name):
            return False

    call = ChatCompletionMessageToolCall.model_validate({
        "id": "call-1",
        "type": "function",
        "function": {"name": "fake_tool", "arguments": "{}"},
    })
    messages = [
        ChatCompletionMessage.model_validate({
            "role": "assistant",
            "content": None,
            "reasoning": "reasoning remains available",
            "tool_calls": [call],
        }),
        ChatCompletionMessage(role="assistant", content="finished"),
    ]

    class Completions:
        def create(self, **kw):
            message = messages.pop(0)
            return type("Response", (), {
                "choices": [type("Choice", (), {"message": message})()],
            })()

    cntl = object.__new__(AgentCntl)
    cntl.opts = FakeOpts()
    cntl.model = "test-model"
    cntl.system_prompt = "test system prompt"
    cntl.history = None
    cntl.client = type("Client", (), {
        "chat": type("Chat", (), {"completions": Completions()})(),
    })()
    cntl.tool_schemas = []
    cntl.tools = {"fake_tool": lambda: {"value": 42}}
    events = []

    cntl.run_agent(
        "test request", spinner=False,
        section_handler=lambda *args: events.append(args))

    output = capsys.readouterr().out
    assert "reasoning remains available" in output
    assert '"value": 42' in output
    starts = [event for event in events if event[0] == "start"]
    assert starts == [
        ("start", "reasoning", "[reasoning]", False),
        ("start", "tool", "[tool call] fake_tool()", False),
    ]

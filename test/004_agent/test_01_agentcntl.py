# Third-party imports
from openai.types.chat import ChatCompletionMessage

# Local imports
from cape.agent.agentcntl import show_reasoning


# A server-provided reasoning extension is shown after completion
def test_01_show_reasoning(capsys):
    msg = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "content": "final answer",
        "reasoning_content": "  inspected the run matrix  ",
    })
    assert show_reasoning(msg) is True
    output = capsys.readouterr().out
    assert "[reasoning]" in output
    assert "inspected the run matrix" in output
    assert "final answer" not in output


# Servers that omit reasoning produce no extra output
def test_02_no_reasoning(capsys):
    msg = ChatCompletionMessage(role="assistant", content="final answer")
    assert show_reasoning(msg) is False
    assert capsys.readouterr().out == ""

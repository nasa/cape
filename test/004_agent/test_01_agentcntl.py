# Third-party imports
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

# Local imports
from cape.agent.agentcntl import (
    AgentCntl,
    SYSTEM_PROMPT,
    genr8_system_prompt,
    show_reasoning,
)
from cape.agent.skills.skillbase import Skill


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

    def record_section(*args, **kwargs):
        events.append((args, kwargs))

    cntl.run_agent(
        "test request", spinner=False,
        section_handler=record_section)

    output = capsys.readouterr().out
    assert "reasoning remains available" in output
    assert '"value": 42' in output
    starts = [event for event in events if event[0][0] == "start"]
    assert starts == [
        (("start", "reasoning", "[reasoning]", False), {}),
        (("start", "tool", "[tool call] fake_tool()", False), {}),
        (("start", "response", "Agent:", True), {
            "color": "bold italic #FF9E64",
        }),
    ]


# Base prompt returned verbatim when nothing is exposed
def test_06_system_prompt_base():
    prompt = genr8_system_prompt({}, {})
    assert prompt == SYSTEM_PROMPT
    assert "PASS*" in prompt
    assert "0-based" in prompt
    # No tool named in shared doctrine unless actually exposed
    assert "cape_c`" not in prompt


# Tool-specific guidance follows the exposed tools
def test_07_system_prompt_tool_lines():
    skills = {"demo": Skill("demo", "Demo skill.", "Instructions")}
    # No tool lines when the matching tools are absent
    prompt = genr8_system_prompt(skills, {})
    assert "cape_find" not in prompt
    assert "view_subfig" not in prompt
    assert "background task" not in prompt
    assert "Available skills:" in prompt
    assert "`demo`" in prompt
    # Tools exposed -> corresponding lines appear
    tools = {
        "cape_find": lambda: None,
        "cape_report": lambda: None,
        "view_subfig": lambda: None,
    }
    prompt = genr8_system_prompt(skills, tools)
    assert "best to call `cape_find`" in prompt
    assert "view_subfig` return images" in prompt
    assert "background task" in prompt


# Dummy options for assemble_tools tests
class ToolOpts:
    def __init__(self, toolset="full", vision=True):
        self.toolset = toolset
        self.vision = vision

    def get_ModelOpt(self, model, name, vdef=None):
        if name == "ToolSet":
            return self.toolset
        if name == "Vision":
            return self.vision
        return vdef


def make_tool_cntl(toolset="full", vision=True):
    cntl = AgentCntl.__new__(AgentCntl)
    cntl.model = "test-model"
    cntl.opts = ToolOpts(toolset=toolset, vision=vision)
    cntl.assemble_tools()
    return cntl


def schema_names(cntl):
    return {schema["function"]["name"] for schema in cntl.tool_schemas}


# Vision-capable models get the image tool; text-only models do not
def test_08_assemble_tools_vision():
    cntl = make_tool_cntl()
    assert "view_subfig" in cntl.tools
    assert "view_subfig" in schema_names(cntl)
    cntl = make_tool_cntl(vision=False)
    assert "view_subfig" not in cntl.tools
    assert "view_subfig" not in schema_names(cntl)
    # Non-image tools unaffected
    assert "get_subfigs" in cntl.tools
    assert "getcwd" in cntl.tools


# The full tool set no longer includes the simple cape_c tool
def test_09_assemble_tools_full():
    cntl = make_tool_cntl()
    assert "cape_c" not in cntl.tools
    # But low-tier models keep it
    cntl = make_tool_cntl(toolset="low")
    assert "cape_c" in cntl.tools

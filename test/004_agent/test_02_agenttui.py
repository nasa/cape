"""Tests for the Textual CAPE-agent interface."""

# Standard library
import asyncio

# Third-party imports
import pytest

# Only run these tests if the optional Textual package is installed
pytest.importorskip("textual")

# Local imports
from cape import agent  # noqa: E402
from cape.agent.agenttui import AgentTuiApp  # noqa: E402
from cape.cfdx.cli import CfdxFrontDesk  # noqa: E402


def run_async(coro):
    return asyncio.run(coro)


async def wait_for(pilot, cond, n=100, dt=0.05):
    for _ in range(n):
        await pilot.pause(dt)
        if cond():
            return True
    return False


def log_text(app):
    return "\n".join(
        str(getattr(line, "text", line)) for line in app._log.lines)


class FakeAgent:
    base_url = "http://localhost:8000/v1"
    model = "test-model"

    def __init__(self):
        self.calls = []
        self.tasks = []
        self.reaps = 0

    def reap_tasks(self, **kw):
        self.reaps += 1

    def run_agent(self, message, **kw):
        self.calls.append((message, kw))
        section_handler = kw["section_handler"]
        section_handler(
            "start", "reasoning", "[reasoning]", False)
        print("A folded test thought")
        section_handler("end", "reasoning", "", True)
        section_handler(
            "start", "tool", "[tool call] fake_tool()", True)
        print("A visible test tool result")
        section_handler("end", "tool", "", True)
        section_handler(
            "start", "response", "Agent:", True,
            color="bold italic #FF9E64")
        print("Agent response from the test backend")
        section_handler("end", "response", "", True)
        return {"n_tool_calls": 2, "n_tool_fails": 1}


def test_01_agent_prompt_uses_shared_tui(tmp_path):
    async def drive():
        cntl = FakeAgent()
        app = AgentTuiApp(
            CfdxFrontDesk,
            cntl,
            histfile=str(tmp_path / "agent_history"))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.title == "CAPE Agent"
            assert "test-model" in log_text(app)
            app._input.value = "Who owns Mach 1.2?"
            await pilot.press("enter")
            assert await wait_for(pilot, lambda: not app._input.disabled)
            assert len(cntl.calls) == 1
            message, kwargs = cntl.calls[0]
            assert message == "Who owns Mach 1.2?"
            assert kwargs["spinner"] is False
            assert kwargs["capture_subprocess"] is True
            assert callable(kwargs["section_handler"])
            assert cntl.reaps == 1
            assert "Agent response from the test backend" in log_text(app)
            assert "A folded test thought" not in log_text(app)
            assert "A visible test tool result" in log_text(app)
            assert [group["collapsed"] for group in app._log._groups] == [
                True, False, False]
            response_header = app._log._groups[-1]["header"]
            assert "Agent:" in response_header.plain
            assert "#FF9E64" in str(response_header.spans)
            assert app._stats["n_user_msgs"] == 1
            assert app._stats["n_tool_calls"] == 2
            assert app._stats["n_tool_fails"] == 1
    run_async(drive())


def test_02_natural_language_does_not_complete_files(tmp_path):
    async def drive():
        app = AgentTuiApp(
            CfdxFrontDesk,
            FakeAgent(),
            histfile=str(tmp_path / "agent_history"))
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = "Who owns"
            app._input.cursor_position = len(app._input.value)
            await pilot.press("tab")
            await pilot.pause()
            assert app._input.value == "Who owns"
            assert not app._suggestions.display
    run_async(drive())


def test_03_agent_main_selects_tui(monkeypatch):
    import cape.agent.agenttui as agenttui

    called = []
    monkeypatch.setattr(agent, "_agent_tui_ok", lambda: True)
    monkeypatch.setattr(
        agenttui, "main", lambda cls=None: called.append(cls) or (0, {}))
    result = agent.main(CfdxFrontDesk)
    assert result == (0, {})
    assert called == [CfdxFrontDesk]


def test_04_shift_enter_inserts_newline_and_enter_submits(tmp_path):
    async def drive():
        cntl = FakeAgent()
        app = AgentTuiApp(
            CfdxFrontDesk,
            cntl,
            histfile=str(tmp_path / "agent_history"))
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = "first line"
            app._input.cursor_position = len(app._input.value)
            await pilot.press("shift+enter")
            await pilot.pause()
            assert app._input.value == "first line\n"
            app._input.insert("second line")
            await pilot.press("enter")
            assert await wait_for(pilot, lambda: not app._input.disabled)
            assert cntl.calls[0][0] == "first line\nsecond line"
    run_async(drive())

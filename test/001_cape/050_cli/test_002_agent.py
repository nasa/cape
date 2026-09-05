"""Tests for noninteractive and interactive CAPE agent commands."""

# Standard library
from unittest.mock import patch

# Local imports
from cape import agent
from cape.cfdx import cli


def test_run_agent_once():
    """Test the noninteractive entry point invokes one agent turn."""
    prompt = "Find DONE cases and generate the report"
    with patch("cape.agent.AgentCntl") as agentcntl:
        agentcntl.return_value.run_agent.return_value = {"n_tool_calls": 1}
        result = agent.run(prompt)
    assert result == (0, {"n_tool_calls": 1})
    agentcntl.return_value.run_agent.assert_called_once_with(prompt)
    agentcntl.return_value.main.assert_not_called()


def test_agent_runs_one_prompt():
    """Test ``--agent`` sends one prompt without starting the UI."""
    prompt = "Find DONE cases and generate the report"
    with (
            patch("cape.agent.run", return_value=(0, {})) as run,
            patch("cape.agent.main") as main):
        ierr = cli.main(["cape", "--agent", prompt])
    assert ierr == 0
    run.assert_called_once_with(prompt)
    main.assert_not_called()


def test_agentic_starts_interactive_ui():
    """Test ``--agentic`` retains the interactive behavior."""
    with (
            patch("cape.agent.run") as run,
            patch("cape.agent.main", return_value=(0, {})) as main):
        ierr = cli.main(["cape", "--agentic"])
    assert ierr == 0
    main.assert_called_once_with(cli.CfdxFrontDesk)
    run.assert_not_called()

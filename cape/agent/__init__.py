r"""
The main interface for running the interactive ``cape agentic`` conversation
or a single ``cape agent`` turn, passing user responses to an external LLM,
and processing the results. Interactive sessions use the Textual interface
when available, with the traditional readline loop as a fallback. Most of the
actual capability is implemented by
:class:`cape.agent.agentcntl.AgentCntl`.
"""

# Local imports
from .agentcntl import AgentCntl, AgentResult


# Main loop
def main(
        cls: type | None = None,
        tui: bool | None = None) -> AgentResult:
    r"""Run an interactive CAPE-agent session

    The Textual interface is selected automatically when Textual is
    installed and stdin/stdout are attached to a capable terminal. Pass
    ``tui=False`` to force the traditional readline interface.
    """
    # Prefer the shared Textual experience on a capable terminal.
    if tui is not False and _agent_tui_ok():
        from . import agenttui
        return agenttui.main(cls)
    # Fall back to the traditional readline interface.
    cntl = AgentCntl()
    return cntl.main(cls)


def _agent_tui_ok() -> bool:
    r"""Check for Textual and a suitable interactive terminal."""
    from ..promptutils import textual_terminal_ok
    return textual_terminal_ok()


def run(user_message: str) -> AgentResult:
    r"""Run one noninteractive CAPE agent turn

    :Call:
        >>> ierr, result = run(user_message)
    :Inputs:
        *user_message*: :class:`str`
            Prompt to send to the CAPE agent
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *result*: :class:`dict`
            Information about tool calls made during the turn
    """
    # Create controller
    cntl = AgentCntl()
    # Run one turn and return without entering the interactive UI
    return AgentResult(0, cntl.run_agent(user_message))

r"""
The main interface for running the interactive ``cape --agentic`` loop or a
single ``cape --agent`` turn, passing user responses to an external LLM, and
processing the results. Most of the actual capability is implemented by the
:class:`cape.agent.agentcntl.AgentCntl` class.
"""

# Local imports
from .agentcntl import AgentCntl, AgentResult


# Main loop
def main(cls: type | None = None) -> AgentResult:
    # Create controller
    cntl = AgentCntl()
    # Run the interface
    return cntl.main(cls)


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

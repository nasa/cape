r"""
:mod:`cape.agent`: Agentic interface to CAPE
==============================================

The main interface for running the ``cape --agentic`` loop, processing
user reponses, passing them to an external LLM, and processing the
results. Most of the actual capability is implemented by the
:class:`cape.agent.agentcntl.AgentCntl` class.
"""

# Local imports
from .agentcntl import AgentCntl


# Main loop
def main(cls: type | None = None):
    # Create controller
    cntl = AgentCntl()
    # Run the interface
    return cntl.main(cls)

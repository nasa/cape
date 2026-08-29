r"""
:mod:`cape.agent.agentcntl`: Controller class for CAPE agent
=============================================================

This module provdes the class :class:`AgentCntl` which serves as an
object-oriented interface to the CAPE agentic capability and interface.
"""

# Standard library
import os

# Local imports
from .options import AgentOpts
from ..errors import assert_isinstance


# Control class
class AgentCntl:
    # Attributes
    __slots__ = (
        "opts",
    )

    # Initialize
    def __init__(self, fname: str | None = "cape-agent.json"):
        # Read options
        self.read_opts(fname)

    # Read options
    def read_opts(self, fname: str | None):
        # Default options if no file
        if fname is None:
            self.opts = AgentOpts()
        # Make sure it's a string
        assert_isinstance(fname, str, "Name of CAPE-agentic JSON file")
        # Check if file exists
        if os.path.isfile(fname):
            # Read it
            self.opts = AgentOpts(fname)
        else:
            # Default
            print(f"No agents file '{fname}' found; using defaults")
            self.opts = AgentOpts()


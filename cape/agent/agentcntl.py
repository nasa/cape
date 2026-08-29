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
        "RootDir",
        "fdir",
        "fname",
        "opts",
    )

    #: Name of default JSON file
    _fjson_default = "cape-agent.json"

    # Initialize
    def __init__(self, fname: str | None = None):
        # Default file name
        fname = self._fjson_default if fname is None else fname
        # Make sure it's a string
        assert_isinstance(fname, str, "Name of CAPE-agentic JSON file")
        #: :class:`str`
        #: Root folder for this controller
        self.RootDir = os.getcwd()
        # Get actual name of root file (follows links if necessary)
        fjson = os.path.realpath(fname)
        # Absolutize
        if os.path.isabs(fjson):
            # Already absolute
            fjson_rel = os.path.relpath(fjson, self.RootDir)
        else:
            # Already relative
            fjson_rel = fjson
        #: :class:`str`
        #: JSON file name (follows links if necessary) rel. to root dir
        self.fname = os.path.basename(fjson_rel)
        #: :class:`str`
        #: Folder in which JSON file is located, relative to root dir
        self.fdir = os.path.dirname(fjson_rel)
        # Read options
        self.read_opts(fname)

    # Read options
    def read_opts(self, fname: str):
        # Check if file exists
        if os.path.isfile(fname):
            # Read it
            self.opts = AgentOpts(fname)
        else:
            # Default options if file name does not exist
            print(f"No agents file '{fname}' found; using defaults")
            self.opts = AgentOpts()


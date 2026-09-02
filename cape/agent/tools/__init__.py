r"""
This package collects the tool functions and JSON schemas exposed to the
CAPE agent from its submodules:

* :mod:`cape.agent.tools.cfdxtools`: wrappers to :mod:`cape.cfdx.cli`
* :mod:`cape.agent.tools.systools`: basic system tools
"""

# Local imports
from . import cfdxtools
from . import cntltools
from . import systools


# Combine JSON-schema defns from tool modules
# OpenAI-compatible
TOOL_SCHEMAS = (
    cfdxtools.TOOL_SCHEMAS +
    cntltools.TOOL_SCHEMAS +
    systools.TOOL_SCHEMAS)
TOOLS = dict(cfdxtools.TOOLS, **cntltools.TOOLS, **systools.TOOLS)

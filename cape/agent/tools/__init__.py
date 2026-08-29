r"""
:mod:`cape.agent.tools`: Tools exposed to the CAPE agent
=========================================================

This package collects the tool functions and JSON schemas exposed to the
CAPE agent from its submodules:

* :mod:`cape.agent.tools.cfdxtools`: wrappers to :mod:`cape.cfdx.cli`
* :mod:`cape.agent.tools.systools`: basic system tools
"""

# Local imports
from . import cfdxtools
from . import systools


# Combine JSON-schema defns from tool modules
# OpenAI-compatible
# (works with llama.cpp's /v1/chat/completions "tools" param
#  when the server is started with --jinja
#  and the model's chat template supports tool calling).
TOOL_SCHEMAS = cfdxtools.TOOL_SCHEMAS + systools.TOOL_SCHEMAS
TOOLS = dict(cfdxtools.TOOLS, **systools.TOOLS)

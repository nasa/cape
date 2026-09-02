r"""
The :mod:`cape.agent.options` provides tools to control options for the
CAPE agent and the Large Language Models that it uses.

If a model is capable of using all CAPE agentic abilities (has
sufficient ``model_len`` (context window) and is reliable enough with
accurate tool selection and usage, the simplest method is to simply
leave that model blank. In fact, it need not even be added to
*ModelList*; CAPE will just default to using all capabilities in that
case.

Here is a sample file in which Nemotron-3 is listed but has no custom
settings. CAPE agentic will expose all tools and skills to it.

.. code-block:: javascript

    {
        "ModelList": [
            "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "meta-llama/Llama-3.1-8B-Instruct",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
        ],
        "bartowski/Llama-3.2-3B-Instruct-GGUF": {
            "ToolSet": "small",
            "SkillSet": "none"
        },
        "meta-llama/Llama-3.1-8B-Instruct": {
            "ToolSet": "medium",
            "SkillSet": "low"
        }
    }

Users may also define a group of models with common settings:

.. code-block:: javascript

    {
        "ModelList": [
            "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "meta-llama/Llama-3.1-8B-Instruct",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
        ],
        "bartowski/Llama-3.2-3B-Instruct-GGUF": {
            "ToolSet": "small",
            "SkillSet": "none"
        },
        "meta-llama/Llama-3.1-8B-Instruct": {
            "Parent": "bartowski/Llama-3.2-3B-Instruct-GGUF"
        }
    }

:See Also:
    * :mod:`cape.optdict`
    * :mod:`cape.agent.options.modelopts`
"""

# Standard library

# Local imports
from .modelopts import ModelOpts
from ...optdict import OptionsDict


# Class definition
class AgentOpts(OptionsDict):
    r"""Options structure, subclass of :class:`dict`

    :Call:
        >>> opts = AgentOpts(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Options added into *opts*
    :Outputs:
        *opts*: :class:`Options`
            Options interface
    """
   # === Class attributes ===
    # Overall name
    _name = "Library of JSON Options for CAPE Agent"
    _label = "cape-agent-json"
    _description = r"""
The following JSON settings are available to the :mod:`cape` agent defined in
:mod:`cape.agent`. The primary purpose is to define which capabilities are
available to the LLM agent to use (which should be scaled to match the
capability of the LLM itself). The main use of this is to avoid asking too much
from smaller language models.

Each section is the settings for a particular model. The section/model name
should match what the OpenAPI access point reports in its ``v1/models`` page.
    """

    # Accepted options/sections
    _optlist = {
        "Model",
        "ModelList",
        "ShowToolResult",
        "ToolDir",
        "URL",
    }

    # Key defining additional *_xoptlist* components
    _xoptkey = "ModelList"

    # Aliases
    _optmap = {
        "Models": "ModelList",
    }

    # Known option types
    _opttypes = {
        "_default_": ModelOpts,
        "Model": str,
        "ModelList": str,
        "ShowToolResult": bool,
        "ToolDir": str,
        "URL": str,
    }

    # Option default list depth
    _optlistdepth = {
        "ModelList": 1,
    }

    # Defaults
    _rc = {
        "ShowToolResult": False,
        "ToolDir": "tools",
    }

    # Descriptions for methods
    _rst_descriptions = {
        "Model": "name of LLM to use; overrides model list from server",
        "ModelList": "list of models with tailored settings",
        "ShowToolResult": "display full result of each tool call",
        "ToolDir": "folder (rel. to cwd) of scripts for user-tools skill",
        "URL": "base URL of LLM server's OpenAI-compatible API",
    }

    # Section classes
    _sec_cls = {}

   # === Model Interface ===
    # Get specific option for a model
    def get_ModelOpt(self, model: str, opt: str, vdef=None):
        # Check if model is present
        if model not in self:
            # Create default instance
            opts = ModelOpts()
            # Return default value
            return opts.get_opt(opt, vdef=vdef)
        # Special case: return (immediate) parent w/o cascading
        if opt == "Parent":
            return self[model].get_opt(opt, vdef=vdef)
        # Use cascading options
        return self.get_subopt(model, opt, key="Parent", vdef=vdef)


# Add global properties
AgentOpts.add_properties(
    ("Model", "ModelList", "ShowToolResult", "ToolDir", "URL"))

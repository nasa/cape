r"""
:mod:`cape.agent.options.modelopts`: Single-LLM options for CAPE agentic
========================================================================

The :mod:`cape.agent.options` provides tools to control options for the
CAPE agent and the Large Language Models that it uses.

Here is a sample file:

.. code-block:: javascript

    {
        "ModelList": [
            "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "meta-llama/Llama-3.1-8B-Instruct",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
        ],
        "bartowski/Llama-3.2-3B-Instruct-GGUF": {
            "ToolSet": "small",
            "SkillSet": "none",
        }
    }

:See Also:
    * :mod:`cape.optdict`
    * :mod:`cape.agent.options`
"""

# Standard library

# Local imports
from ...optdict import OptionsDict


# Class definition
class ModelOpts(OptionsDict):
    r"""Options structure, subclass of :class:`dict`

    :Call:
        >>> opts = ModelOpts(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Options added into *opts*
    :Outputs:
        *opts*: :class:`Options`
            Options interface
    """
    # Attributes
    __slots__ = ()

    # Overall name
    _name = "Custom options for a particular LLM used with CAPE agentic"
    _label = "cape-agent-model-json"

    # List of options
    _optlist = (
        "ToolSet",
        "SkillSet",
        "Parent",
    )

    # Aliases
    _optmap = {
        "ToolLevel": "ToolSet",
        "SkillLevel": "SkillSet",
        "Type": "Parent",
    }

    # Types
    _opttypes = {
        "ToolSet": str,
        "SkillSet": str,
        "Parent": str,
    }

    # Allowed values
    _optvals = {
        "ToolSet": (
            "none",
            "low",
            "medium",
            "full",
        ),
        "SkillSet": (
            "none",
            "low",
            "medium",
            "full",
        ),
    }

    # Value aliases
    _optvalmap = {
        "ToolSet": {
            "off": "none",
            "lo": "low",
            "med": "medium",
            "hi": "full",
            "high": "full",
            "all": "full",
        },
        "SkillSet": {
            "off": "none",
            "lo": "low",
            "med": "medium",
            "hi": "full",
            "high": "full",
            "all": "full",
        },
    }

    # Defaults
    _rc = {
        "ToolSet": "full",
        "SkillSet": "full",
    }

    # Descriptions
    _rst_descriptions = {
        "ToolSet": "descriptive level of how many tools to expose",
        "SkillSet": "descriptive level of how many skill to expose",
        "Parent": "name of model to inherit settings from",
    }

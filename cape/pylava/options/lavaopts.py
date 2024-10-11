r"""
:mod:`cape.pylava.options.lavaopts`: Options for Superlava executable
=====================================================================
"""

# Local imports
from ...cfdx.options.execopts import ExecOpts


# Class for `nodet` inputs
class SuperlavaOpts(ExecOpts):
    r"""Class for ``superlava`` settings

    :Call:
        >>> opts = SuperlavaOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`SuperlavaOpts`
            CLI options for ``superlava``
    :Versions:
        * 2024-08-01 ``@sneuhoff``: v1.0
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "executable",
        "yamlfile",
    }

    # Types
    _opttypes = {
        "executable": str,
        "yamlfile": str,
    }

    # Defaults
    _rc = {
        "yamlfile": "run.yaml",
    }

    # Descriptions
    _rst_descriptions = {
        "executable": "path to superlava executable",
        "yamlfile": "name of LAVA YAML file to use",
    }


# Add properties
SuperlavaOpts.add_properties(SuperlavaOpts._optlist)

r"""
:mod:`cape.pylava.options`: Options interface for :mod:`cape,pylava`
====================================================================

This is the LAVA-specific implementation of the CAPE options
package, based on

    :mod:`cape.cfdx.options`
"""

# Local imports
from . import util
from .runctlopts import RunControlOpts
from .meshopts import MeshOpts
from ...cfdx import options


# Class definition
class Options(options.Options):
    r"""Options interface for :mod:`cape.pylava`

    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary of raw options
    :Outputs:
        *opts*: :class:`cape.pylava.options.Options`
            Options interface
    :Versions:
        * 2024-09-30 ``@sneuhoff``: Version 1.0
    """
   # ======================
   # Class Attributes
   # ======================
   # <
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "CartInputFile",
        "LAVASolver",
        "YAML",
        "RunYAMLFile",
    )

    # Aliases
    _optmap = {
        "CartesianInputFile": "CartInputFile",
        "LavaSolver": "LAVASolver",
        "RunInputFile": "CartInputFile",
        "RunYAML": "YAML",
        "Solver": "LAVASolver",
        "YAMLFile": "RunYAMLFile",
    }

    # Types
    _opttypes = {
        "CartInputFile": str,
        "LAVASolver": str,
        "RunYAMLFile": str,
    }

    # Permitted values
    _optvals = {
        "LAVASolver": (
            "cartesian",
            "curvilinear",
            "unstructured",
        ),
    }

    # Defaults
    _rc = {
        "LAVASolver": "curvilinear",
    }

    # Descriptions
    _rst_descriptions = {
        "RunYAMLFile": "template LAVA input file (YAML/JSON)",
    }

    # Replaced or renewed sections
    _sec_cls = {
        "RunControl": RunControlOpts,
        "Mesh": MeshOpts,
    }
   # >

   # =============
   # Configuration
   # =============
   # <
    # Initialization hook
    def init_post(self):
        r"""Initialization hook for :class:`Options`

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`Options`
                Options interface
        :Versions:
            * 2024-08-05 ``@sneuhoff``: Version 1.0
        """
        # Read the defaults
        defs = util.getPyLavaDefaults()
        # Apply the defaults
        self = util.applyDefaults(self, defs)
        # Add extra folders to path.
        self.AddPythonPath()
   # >


# Add properties
Options.add_properties(
    (
        "RunYAMLFile",
    ))
# Add methods from subsections
Options.promote_sections()

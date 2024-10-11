r"""
:mod:`cape.pylava.options.runctlopts`: LAVACURV run control options
=====================================================================

This is the ``"RunControl"`` options module specific to
:mod:`cape.pylava`. It is based primarily on

    :mod:`cape.cfdx.options.runctlopts`

and provides the class

    :class:`cape.pylava.options.runctlopts.RunControlOpts`

Among the LAVA-specific options interfaced here are the options for
the LAVA executables, defined in the ``"RunControl"`` section
"""

# Local imports
from .lavaopts import SuperlavaOpts
from ...cfdx.options import runctlopts


# Class for RunControl settings
class RunControlOpts(runctlopts.RunControlOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "superlava",
        "RunYaml",
    }

    # Types
    _opttypes = {
        "RunYaml": str,        
    }

    # Defaults
    _rc = {
        "RunYaml": "run.yaml",
    }

    # Descriptions
    _rst_descriptions = {
        "RunYaml": "Name of yaml input file to superlava",
    }

    # Additional sections
    _sec_cls = {
        "superlava": SuperlavaOpts,
    }    


# Add properties
RunControlOpts.add_properties(RunControlOpts._optlist)
# Promote sections
RunControlOpts.promote_sections()

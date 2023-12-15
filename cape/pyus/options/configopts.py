r"""

This module provides options for defining some aspects of the surface
configuration for a US3D run.  It can point to a surface configuration
file such as ``Config.xml` or ``Config.json``.

In addition to the options in
:class:`cape.cfdx.options.configopts.ConfigOpts`, this module provides
the options *Inputs* and *PostScrFile*.

:See Also:
    * :mod:`cape.cfdx.options.configopts`
    * :mod:`cape.config`
"""

# Local imports
from ...optdict import OptionsDict, INT_TYPES
from ...cfdx.options import configopts


# Class for Inputs Section
class ConfigInputOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Option types
    _opttypes = {
        "_default_": INT_TYPES,
    }

    # List depth
    _optlistdepth = {
        "_default_": 1,
    }


# Class for SurfConfig
class ConfigOpts(configopts.ConfigOpts):
    # No additional attibutes
    __slots__ = ()

    # Additional options
    _optlist = {
        "Inputs",
        "PostScrFile",
    }

    # Aliases
    _optmap = {
        "post.scr": "PostScrFile",
    }

    # Option types
    _opttypes = {
        "PostScrFile": str,
    }

    # Descriptions
    _rst_descriptions = {
        "Inputs": "dict of BC integers in each component name",
        "PostScrFile": "name of ``post.scr`` file template",
    }

    # Section map
    _sec_cls = {
        "Inputs": ConfigInputOpts,
    }

    # Get inputs for a particular component
    def get_ConfigInput(self, comp: str, **kw):
        r"""Return the input for a particular component

        :Call:
            >>> inp = opts.get_ConfigInput(comp, **kw)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *inp*: :class:`int` | :class:`list`\ [:class:`int`]
                List of BCs in this component
        :Versions:
            * 2015-10-20 ``@ddalle``: v1.0
            * 2023-03-28 ``@ddalle``: v2.0; use :mod:`cape.optdict`
        """
        return self.get_subopt("Inputs", comp, **kw)


# Add properties
ConfigOpts.add_properties(["PostScrFile"])


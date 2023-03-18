r"""
:mod:`cape.pykes.options`: Options interface for :mod:`cape,pykes`
====================================================================

This is the Kestrel-specific implementation of the CAPE options package,
based on

    :mod:`cape.cfdx.options`

"""

# Local imports
from . import util
from .meshopts import MeshOpts
from .runctlopts import RunControlOpts
from .util import applyDefaults, getPyKesDefaults
from ...cfdx import options


# Class definition
class Options(options.Options):
    r"""Options interface for :mod:`cape.pykes`

    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Additional manual options
    :Versions:
        * 2021-10-18 ``@ddalle``: Version 1.0
        * 2022-11-03 ``@ddalle``: Version 2.0; :class:`OptionsDict`
    """
   # ===================
   # Class Attributes
   # ===================
   # <
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "JobXML",
        "XML",
    }

    # Option types
    _opttypes = {
        "JobXML": str,
    }

    # Defaults
    _rc = {
        "JobXML": "kestrel.xml",
        "XML": {},
    }

    # Descriptions
    _rst_descriptions = {
        "JobXML": "Template XML file for Kestrel cases",
    }

    # New or rewritten sections
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
            * 2022-10-23 ``@ddalle``: Version 1.0
        """
        # Read the defaults.
        defs = getPyKesDefaults()
        # Apply the defaults.
        self = applyDefaults(self, defs)
        # Add to Python path
        self.AddPythonPath()
   # >

   # =============
   # XML section
   # =============
   # <
    # Get XML options for a given phase
    def select_xml_phase(self, j=None, **kw):
        r"""Get all items from the *XML* section for phase *j*

        :Call:
            >>> xmlitems = opts.select_xml_phase(j=None)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *xmlitems*: :class:`list`\ [:class:`dict`]
                List of XML element descriptors
        :Versions:
            * 2021-10-26 ``@ddalle``: Version 1.0
        """
        # Get the *XML* section
        xmlsec = self.get("XML")
        # Check if it's a list
        if not isinstance(xmlsec, list):
            return []
        # Initialize items
        xmlitems = []
        # Loop through *xmlsec*
        for elem in xmlsec:
            # Check if it's a dict
            if not isinstance(elem, dict):
                continue
            # Get value
            v = elem.get("value")
            # Set value to phase *j* and save it
            xmlitems.append(dict(elem, value=util.getel(v, j)))
        # Output
        return xmlitems
   # >


# Upgrade any local sections
Options.promote_sections()


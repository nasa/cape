r"""
:mod:`cape.cfdx.options.comp2triopts`: CLI options for ``comp2tri``
=======================================================================

This module interfaces options for a Cart3D tool called ``comp2tri``.

There are a variety of input flags to this utility, and options are
controlled via the :class:`Comp2TriOpts`.
"""

# Ipmort options-specific utilities
from ...optdict import BOOL_TYPES, INT_TYPES
from .execopts import ExecOpts


# Resource limits class
class Comp2TriOpts(ExecOpts):
    r"""Class for Cart3D ``comp2tri`` command-line settings

    :Call:
        >>> opts = Comp2Tri(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of ``comp2tri`` command-line options
    :Outputs:
        *opts*: :class:`Comp2TriOpts`
            ``comp2tri`` utility options interface
    :Versions:
        * 2025-02-28 ``@ddalle``: v1.0
    """
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "CLI options for ``comp2tri``"

    # List of accepted options
    _optlist = {
        "ascii",
        "config",
        "dp",
        "gmp2comp",
        "gmpTagOffset",
        "i",
        "inflate",
        "keepComps",
        "makeGMPtags",
        "o",
        "run",
        "trix",
        "v",
    }

    # Aliases
    _optmap = {
        "gmptagoffset": "gmptagoffset",
        "ifile": "i",
        "ifiles": "i",
        "keepcomps": "keepComps",
        "maketags": "makeGMPtags",
        "ofile": "o",
        "output": "o",
        "tagoffset": "gmpTagOffset",
    }

    # Types
    _opttypes = {
        "ascii": BOOL_TYPES,
        "config": BOOL_TYPES,
        "dp": BOOL_TYPES,
        "gmp2cmp": BOOL_TYPES,
        "gmpTagOffset": INT_TYPES,
        "i": str,
        "inflate": BOOL_TYPES,
        "keepComps": BOOL_TYPES,
        "makeGMPtags": BOOL_TYPES,
        "o": str,
        "trix": BOOL_TYPES,
        "v": BOOL_TYPES
    }

    # List options
    _optlistdepth = {
        "i": 2,
    }

    # Defaults
    _rc = {
        "dp": False,
        "o": "Components.tri",
        "v": False,
    }

    # Descriptions
    _rst_descriptions = {
        "ascii": "output file will be ASCII (if not trix)",
        "config": "write ``Config.xml`` using component tags",
        "dp": "use double-precision vert-coordinates",
        "gmp2comp": "copy GMPtags to IntersectComponents",
        "gmpTagOffset": "renumber GMPtags by NxOffset for file *N*",
        "i": "input file or list thereof",
        "inflate": "inflate geometry to break degeneracies",
        "keepComps": "preserve 'intersect' component tags",
        "makeGMPtags": "create GMPtags from volume indexes",
        "o": "output filename",
        "trix": "output file will be eXtended-TRI (trix) format",
        "v": "verbose mode",
    }


# Add all properties
Comp2TriOpts.add_properties(Comp2TriOpts._optlist, prefix="comp2tri_")


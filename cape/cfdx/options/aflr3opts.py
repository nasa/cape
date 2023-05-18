r"""
:mod:`cape.cfdx.options.aflr3opts`: AFLR3 mesh generation options
====================================================================

This module provides a class to access command-line options to the AFLR3
mesh-generation program. It is specified in the ``"RunControl"`` section
for modules that utilize the solver, which includes FUN3D.

The options in this module are among the command-line options to AFLR3.
Other AFLR3 options that do not have specific methods defined in the
:class:`AFLR3Opts` options class can be accessed using two generic
functions:

    * :func:`AFLR3Opts.get_aflr3_flags`: options with ``-blr 1.2`` format
    * :func:`AFLR3Opts.get_aflr3_keys`: options with ``cdfs=7.5`` format
"""

# Local imports
from ...optdict import BOOL_TYPES, FLOAT_TYPES, INT_TYPES
from ...optdict.optitem import setel
from .util import ExecOpts


# Resource limits class
class AFLR3Opts(ExecOpts):
    r"""Class for AFLR3 command-line settings

    :Call:
        >>> opts = AFLR3Opts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of AFLR3 command-line options
    :Outputs:
        *opts*: :class:`AFLR3Opts`
            AFLR3 options interface
    :Versions:
        * 2016-04-04 ``@ddalle``: Version 1.0 (:class:`aflr3`)
        * 2022-10-14 ``@ddalle``: Version 2.0
    """
    __slots__ = tuple()

    _optlist = {
        "BCFile",
        "angblisimx",
        "angqbf",
        "blc",
        "blds",
        "bli",
        "blr",
        "cdfr",
        "cdfs",
        "flags",
        "grow",
        "i",
        "keys",
        "mdf",
        "mdsblf",
        "nqual",
        "o",
        "run",
    }

    _opttypes = {
        "BCFile": str,
        "angblisimx": FLOAT_TYPES,
        "angqbf": FLOAT_TYPES,
        "blc": BOOL_TYPES,
        "blds": FLOAT_TYPES,
        "bli": INT_TYPES,
        "blr": FLOAT_TYPES,
        "cdfr": FLOAT_TYPES,
        "cdfs": FLOAT_TYPES + INT_TYPES,
        "flags": dict,
        "grow": FLOAT_TYPES,
        "i": str,
        "keys": dict,
        "nqual": INT_TYPES,
        "o": str,
    }

    _optvals = {
        "mdf": (1, 2),
        "mdsblf": (0, 1, 2),
    }

    _rc = {
        "flags": {},
        "keys": {},
        "mdf": 2,
        "mdsblf": 1,
        "nqual": 0,
    }

    # Descriptions
    _rst_descriptions = {
        "BCFile": "AFLR3 boundary condition file",
        "angblisimx": "AFLR3 max angle b/w BL intersecting faces",
        "angqbf": "AFLR3 max angle on surface triangles",
        "blc": "AFLR3 prism layer option",
        "blds": "AFLR3 initial boundary-layer spacing",
        "bli": "number of AFLR3 prism layers",
        "blr": "AFLR3 boundary layer stretching ratio",
        "cdfr": "AFLR3 max geometric growth rate",
        "cdfs": "AFLR3 geometric growth exclusion zone size",
        "flags": "AFLR3 options using ``-flag val`` format",
        "grow": "AFLR3 off-body growth rate",
        "i": "input file for AFLR3",
        "keys": "AFLR3 options using ``key=val`` format",
        "mdf": "AFLR3 volume grid distribution flag",
        "mdsblf": "AFLR3 BL spacing thickness factor option",
        "nqual": "number of AFLR3 mesh quality passes",
        "o": "output file for AFLR3",
        "run": "whether or not to run AFLR3",
    }

    # Get single key from AFLR3 *keys* key
    def get_aflr3_key(self, k, j=0, i=None, vdef=None):
        r"""Get AFLR3 AFLR3 option that uses *key=val* format

        :Call:
            >>> v = opts.get_aflr3_key(k, j=0, vdef=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *k*: :class:`str`
                Name of key to query
            *j*: {``0``} | ``None`` | :class:`int`
                Phase number
            *vdef*: {``None``} | :class:`any`
                Manually specified default
        :Outputs:
            *v*: :class:`any`
                Value of AFLR3 key *k*
        :Versions:
            * 2019-05-11 ``@ddalle``: Version 1.0
            * 2022-10-14 ``@ddalle``: Version 1.1; use :mod:`optdict`
            * 2022-10-29 ``@ddalle``: Version 2.0
                - add *i*
                - use :func:`get_subopt`
        """
        return self.get_subopt("keys", k, i=i, j=j, vdef=vdef)

    # Set single key from AFLR3 *keys* section
    def set_aflr3_key(self, k, v, j=None):
        r"""Get AFLR3 AFLR3 option that uses *key=val* format

        :Call:
            >>> opts.get_aflr3_key(k, v, j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *k*: :class:`str`
                Name of key to query
            *v*: :class:`any`
                Value of AFLR3 key *k*
            *j*: {``None``} | :class:`int`
                Phase number
        :Versions:
            * 2019-05-11 ``@ddalle``: Version 1.0
            * 2022-10-14 ``@ddalle``: Version 1.1; use :mod:`optdict`
        """
        # Get or create "keys" key
        d = self.setdefault("keys", {})
        # Set value
        d[k] = setel(d.get(k), v, j=j)


# Add all properties
AFLR3Opts.add_properties(AFLR3Opts._optlist, prefix="aflr3_")


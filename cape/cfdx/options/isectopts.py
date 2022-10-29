r"""
:mod:`cape.cfdx.options.isectopts`: CLI options for ``intersect``
=======================================================================

This module interfaces options for two Cart3D triangulation tools called
``intersect`` and ``verify``. This allows the user to intersect
overlapping triangulated bodies and create a single water-tight body.

The ``verify`` executable
tests for a well-defined water-tight triangulation. Intersections,
missing triangles, holes, etc. all cause ``verify`` to fail.

The main option for both ``intersect`` and ``verify`` is simply whether
or not to use them. The user can specify the following in the
``"RunControl"`` section of the JSON file.

    .. code-block:: javascript

        "RunControl": {
            "intersect": True,
            "verfiy": False
        }

There are a variety of input flags to those utilities, which are
interfaced via the :class:`IntersectOpts` and
:class:`VerifyOpts` classes defined here.
"""

# Ipmort options-specific utilities
from ...optdict import BOOL_TYPES, FLOAT_TYPES, INT_TYPES
from .util import ExecOpts


# Resource limits class
class IntersectOpts(ExecOpts):
    r"""Class for Cart3D ``intersect`` command-line settings

    :Call:
        >>> opts = intersect(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of ``intersect`` command-line options
    :Outputs:
        *opts*: :class:`InterseectOpts`
            Intersect utility options interface
    :Versions:
        * 2016-04-05 ``@ddalle``: Version 1.0
        * 2022-10-29 ``@ddalle``: Version 2.0; :class:`OptionsDict`
    """
    # List of accepted options
    _optlist = {
        "T",
        "ascii",
        "cutout",
        "fast",
        "i",
        "intersections",
        "o",
        "overlap",
        "rm",
        "smalltri",
        "triged",
        "v",
    }

    # Types
    _opttypes = {
        "T": BOOL_TYPES,
        "ascii": BOOL_TYPES,
        "cutout": INT_TYPES,
        "fast": BOOL_TYPES,
        "i": str,
        "intersections": BOOL_TYPES,
        "o": str,
        "overlap": INT_TYPES,
        "rm": BOOL_TYPES,
        "smalltri": FLOAT_TYPES,
        "triged":  BOOL_TYPES,
        "v": BOOL_TYPES
    }

    # Defaults
    _rc = {
        "T": False,
        "fast": False,
        "i": "Components.tri",
        "intersections": False,
        "o": "Components.i.tri",
        "rm": False,
        "smalltri": 1e-4,
        "triged": True,
        "v": False,
    }

    # Descriptions
    _rst_descriptions = {
        "T": "option to also write Tecplot file ``Components.i.plt``",
        "ascii": "flag that input file is ASCII",
        "cutout": "number of component to subtract",
        "fast": "also write unformatted FAST file ``Components.i.fast``",
        "i": "input file to ``intersect``",
        "intersections": "option to write intersections to ``intersect.dat``",
        "o": "output file for ``intersect``",
        "overlap": "perform boolean intersection of this comp number",
        "rm": "option to remove small triangles from results",
        "smalltri": "cutoff size for small triangles with *rm*",
        "triged": "option to use CGT ``triged`` to clean output file",
        "v": "verbose mode",
    }


# Add all properties
IntersectOpts.add_properties(IntersectOpts._opt.ist, prefix="intersect_")


# Resource limits class
class VerifyOpts(ExecOpts):
    r"""Options for command-line options to Cart3D ``verify`` tool

    :Call:
        >>> opts = VerifyOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of ``verify`` command-line options
    :Outputs:
        *opts*: :class:`VerifyOpts`
            Options interface for ``verify``` options
    :Versions:
        * 2016-04-05 ``@ddalle``: Version 1.0 (``verify``)
        * 2022-10-29 ``@ddalle``: Version 2.0; :class:`OptionsDict`
    """
    # List of accepted options
    _optlist = {
        "i",
        "ascii",
    }

    # Types
    _opttypes = {
        "i": str,
        "ascii": BOOL_TYPES,
    }

    # Defaults
    _rc = {
        "ascii": True,
    }

    # Descriptions
    _rst_descriptions = {
        "i": "input file for ``verify``",
        "ascii": "option for ASCII input file to ``verify``",
    }


# Add all properties
VerifyOpts.add_properties(VerifyOpts._opt.ist, prefix="verify_")


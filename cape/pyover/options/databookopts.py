r"""
:mod:`cape.pyover.options.databookopts`: OVERFLOW databook options
===================================================================

This module contains the interface for *DataBook* options that are
specific to OVERFLOW and :mod:`cape.pyover`.

The OVERFLOW-specific options for these classes are limited, but a few
methods are modified in order to change default data book component
types and the columns of data available for each.  In particular, may
special options for ``usurp`` or ``mixsur`` are specified, which is
needed to extract a surface triangulation from an OVERFLOW solution.
"""

# Local imports
from ...cfdx.options import databookopts


# OVERFLOW-specific options
class DBTriqFMOpts(databookopts.DBTriqFMOpts):
    # No attributes
    __slots__ = ()

    # Additional properties
    _optlist = {
        "QIn",
        "QOut",
        "QSurf",
        "XIn",
        "XOut",
        "XSurf",
        "fomo",
        "mixsur",
        "splitmq",
        "usurp",
    }

    # Aliases
    _optmap = {
        "overint": "mixsur",
        "q.in": "QIn",
        "q.out": "QOut",
        "q.surf": "QSurf",
        "x.in": "XIn",
        "x.out": "XOut",
        "x.surf": "XSurf",
    }

    # Types
    _opttypes = {
        "QIn": str,
        "QOut": str,
        "QSurf": str,
        "XIn": str,
        "XOut": str,
        "XSurf": str,
        "fomo": str,
        "mixsur": str,
        "splitmq": str,
        "usurp": str,
    }

    # Defaults
    _rc = {
        "QIn": "q.pyover.p3d",
        "QSurf": "q.pyover.surf",
        "XIn": "x.pyover.p3d",
        "XSurf": "x.pyover.surf",
        "mixsur": "mixsur.i",
        "splitmq": "splitmq.i",
        "usurp": "",
    }

    # Descriptions
    _rst_descriptions = {
        "QIn": "input ``q`` file",
        "QOut": "preprocessed ``q`` file for a databook component",
        "QSurf": "preprocessed ``q.srf`` file name",
        "XIn": "input ``x`` file",
        "XOut": "preprocessed ``x`` file for a databook component",
        "XSurf": "preprocessed ``x.srf`` file name",
        "fomo": r"""path to ``mixsur`` output files

        If each of the following files is found, there is no need to run
        ``mixsur``, and files are linked instead.

            * ``grid.i.tri``
            * ``grid.bnd``
            * ``grid.ib``
            * ``grid.ibi``
            * ``grid.map``
            * ``grid.nsf``
            * ``grid.ptv``
            * ``mixsur.fmp``""",
        "mixsur": "input file for ``mixsur``, ``overint``, or ``usurp``",
        "splitmq": "input file for ``splitmq``",
        "usurp": "input file for ``usurp``",
    }


class DBLineLoadOpts(databookopts.DBLineLoadOpts):
    # No attributes
    __slots__ = ()

    # Additional properties
    _optlist = {
        "QIn",
        "QOut",
        "QSurf",
        "XIn",
        "XOut",
        "XSurf",
        "fomo",
        "mixsur",
        "splitmq",
        "usurp",
    }

    # Aliases
    _optmap = {
        "overint": "mixsur",
        "q.in": "QIn",
        "q.out": "QOut",
        "q.surf": "QSurf",
        "x.in": "XIn",
        "x.out": "XOut",
        "x.surf": "XSurf",
    }

    # Types
    _opttypes = {
        "QIn": str,
        "QOut": str,
        "QSurf": str,
        "XIn": str,
        "XOut": str,
        "XSurf": str,
        "fomo": str,
        "mixsur": str,
        "splitmq": str,
        "usurp": str,
    }

    # Defaults
    _rc = {
        "QIn": "q.pyover.p3d",
        "QSurf": "q.pyover.surf",
        "XIn": "x.pyover.p3d",
        "XSurf": "x.pyover.surf",
        "mixsur": "mixsur.i",
        "splitmq": "splitmq.i",
        "usurp": "",
    }

    # Descriptions
    _rst_descriptions = {
        "QIn": "input ``q`` file",
        "QOut": "preprocessed ``q`` file for a databook component",
        "QSurf": "preprocessed ``q.srf`` file name",
        "XIn": "input ``x`` file",
        "XOut": "preprocessed ``x`` file for a databook component",
        "XSurf": "preprocessed ``x.srf`` file name",
        "fomo": r"""path to ``mixsur`` output files

        If each of the following files is found, there is no need to run
        ``mixsur``, and files are linked instead.

            * ``grid.i.tri``
            * ``grid.bnd``
            * ``grid.ib``
            * ``grid.ibi``
            * ``grid.map``
            * ``grid.nsf``
            * ``grid.ptv``
            * ``mixsur.fmp``""",
        "mixsur": "input file for ``mixsur``, ``overint``, or ``usurp``",
        "splitmq": "input file for ``splitmq``",
        "usurp": "input file for ``usurp``",
    }


# Class for DataBook options
class DataBookOpts(databookopts.DataBookOpts):
    # No attributes
    __slots__ = ()

    # Additional properties
    _optlist = {
        "QIn",
        "QOut",
        "QSurf",
        "XIn",
        "XOut",
        "XSurf",
        "fomo",
        "mixsur",
        "splitmq",
        "usurp",
    }

    # Aliases
    _optmap = {
        "overint": "mixsur",
        "q.in": "QIn",
        "q.out": "QOut",
        "q.surf": "QSurf",
        "x.in": "XIn",
        "x.out": "XOut",
        "x.surf": "XSurf",
    }

    # Types
    _opttypes = {
        "QIn": str,
        "QOut": str,
        "QSurf": str,
        "XIn": str,
        "XOut": str,
        "XSurf": str,
        "fomo": str,
        "mixsur": str,
        "splitmq": str,
        "usurp": str,
    }

    # Descriptions
    _rst_descriptions = {
        "QIn": "input ``q`` file",
        "QOut": "preprocessed ``q`` file for a databook component",
        "QSurf": "preprocessed ``q.srf`` file name",
        "XIn": "input ``x`` file",
        "XOut": "preprocessed ``x`` file for a databook component",
        "XSurf": "preprocessed ``x.srf`` file name",
        "fomo": r"""path to ``mixsur`` output files

        If each of the following files is found, there is no need to run
        ``mixsur``, and files are linked instead.

            * ``grid.i.tri``
            * ``grid.bnd``
            * ``grid.ib``
            * ``grid.ibi``
            * ``grid.map``
            * ``grid.nsf``
            * ``grid.ptv``
            * ``mixsur.fmp``""",
        "mixsur": "input file for ``mixsur``, ``overint``, or ``usurp``",
        "splitmq": "input file for ``splitmq``",
        "usurp": "input file for ``usurp``",
    }

    # Section map
    _sec_cls_optmap = {
        "FM": databookopts.DBFMOpts,
        "IterPoint": databookopts.DBIterPointOpts,
        "LineLoad": DBLineLoadOpts,
        "PyFunc": databookopts.DBPyFuncOpts,
        "TriqFM": DBTriqFMOpts,
        "TriqPoint": databookopts.DBTriqPointOpts,
    }



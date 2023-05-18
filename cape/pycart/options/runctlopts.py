"""
:mod:`cape.pycart.options.rctlopts`: Cart3D run control options
======================================================================

Options interface for case control for pycart runs of Cart3D. This is
mostly a fork of

    :mod:`cape.cfdx.options.runctlopts`

but with several special methods, including command-line options to
``autoInputs``, ``cubes``, ``flowCart``, and ``adjointCart``.

:Classes:
    * :class:`cape.pycart.options.runControl.RunControl`
    * :class:`cape.pycart.options.runControl.Adaptation`
    * :class:`cape.pycart.options.runControl.flowCart`
    * :class:`cape.pycart.options.runControl.adjointCart`
    * :class:`cape.pycart.options.runControl.autoInputs`
    * :class:`cape.pycart.options.runControl.cubes`
    * :class:`cape.pycart.options.Archive.Archive`
    * :class:`cape.cfdx.options.runControl.Environ`
    * :class:`cape.cfdx.options.ulimit.ulimit`
:See Also:
    * :mod:`cape.cfdx.options.runControl`
    * :mod:`cape.cfdx.options.ulimit`
    * :mod:`cape.pycart.options.Archive`
"""


# Local imports
from .archiveopts import ArchiveOpts
from ...cfdx.options import runctlopts
from ...cfdx.options.util import ExecOpts, OptionsDict
from ...optdict import BOOL_TYPES, FLOAT_TYPES, INT_TYPES


# Class for flowCart inputs
class FlowCartOpts(ExecOpts):
    r"""Class for ``flowCart`` settings

    :Call:
        >>> opts = FlowCartOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`FlowCartOpts`
            CLI options for ``flowCart`` or ``mpi_flowCart``
    :Versions:
        * 2014-12-17 ``@ddalle``: Version 1.0 (``flowCart``)
        * 2022-11-01 ``@ddalle``: Version 2.0; use ``optdict``
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "RKScheme",
        "binaryIO",
        "buffLim",
        "cfl",
        "cflmin",
        "checkptTD",
        "clic",
        "dt",
        "fc_clean",
        "fc_stats",
        "first_order",
        "fmg",
        "it_avg",
        "it_fc",
        "it_start",
        "it_sub",
        "limiter",
        "mg_fc",
        "mpi_fc",
        "nOrders",
        "pmg",
        "robust_mode",
        "tecO",
        "tm",
        "unsteady",
        "vizTD",
        "y_is_spanwise",
    }

    # Aliases
    _optmap = {
        "RKscheme": "RKScheme",
    }

    # Types
    _opttypes = {
        "RKScheme": (str, list),
        "buffLim": BOOL_TYPES,
        "binaryIO": BOOL_TYPES,
        "cfl": FLOAT_TYPES,
        "cflmin": FLOAT_TYPES,
        "checkptTD": BOOL_TYPES,
        "clic": BOOL_TYPES,
        "dt": FLOAT_TYPES,
        "fc_clean": BOOL_TYPES,
        "fc_stats": INT_TYPES,
        "first_order": BOOL_TYPES,
        "fmg": BOOL_TYPES,
        "it_avg": INT_TYPES,
        "it_fc": INT_TYPES,
        "it_start": INT_TYPES,
        "it_sub": INT_TYPES,
        "limiter": INT_TYPES,
        "mg_fc": INT_TYPES,
        "mpi_fc": BOOL_TYPES + INT_TYPES,
        "nOrders": INT_TYPES,
        "pmg": BOOL_TYPES,
        "robust_mode": BOOL_TYPES,
        "tecO": BOOL_TYPES,
        "tm": BOOL_TYPES,
        "unsteady": BOOL_TYPES,
        "vizTD": BOOL_TYPES,
        "y_is_spanwise": BOOL_TYPES,
    }

    # Defaults
    _rc = {
        "binaryIO": True,
        "buffLim": False,
        "cfl": 1.1,
        "cflmin": 0.8,
        "clic": True,
        "dt": 0.1,
        "fc_clean": False,
        "fc_stats": 0,
        "first_order": False,
        "fmg": True,
        "it_avg": 0,
        "it_fc": 200,
        "it_start": 100,
        "it_sub": 10,
        "limiter": 2,
        "mg_fc": 3,
        "mpi_fc": False,
        "nOrders": 12,
        "pmg": False,
        "robust_mode": False,
        "tecO": True,
        "tm": False,
        "unsteady": False,
        "y_is_spanwise": True,
    }

    # Descriptions
    _rst_descriptions = {
        "RKScheme": "the Runge-Kutta scheme for a phase",
        "binaryIO": "whether ``flowCart`` is set for binary I/O",
        "buffLim": "whether ``flowCart`` will use buffer limits",
        "cfl": "nominal CFL number for ``flowCart``",
        "cflmin": "min CFL number for ``flowCart``",
        "checkptTD": "steps between unsteady ``flowCart`` checkpoints",
        "clic": "whether to write ``Components.i.triq``",
        "dt": "nondimensional physical time step",
        "first_order": "whether ``flowCart`` should be run first-order",
        "fc_clean": "whether to run relaxation step before time-accurate step",
        "fc_stats": "number of iters for iterative or time averaging",
        "fmg": "whether to run ``flowCart`` w/ full multigrid",
        "pmg": "whether to run ``flowCart`` w/ poly multigrid",
        "it_avg": "number of ``flowCart`` iters b/w ``.triq`` outputs",
        "it_fc": "number of ``flowCart`` iterations",
        "it_start": "number of ``flowCart`` iters b4 ``.triq`` outputs",
        "it_sub": "number of subiters for each ``flowCart`` time step",
        "limiter": "limiter for ``flowCart``",
        "mg_fc": "multigrid levels for ``flowCart``",
        "mpi_fc": "whether or not to run ``mpi_flowCart``",
        "nOrders": "convergence drop orders of magnitude for early exit",
        "robust_mode": "whether ``flowCart`` should be run in robust mode",
        "tecO": "whether ``flowCart`` dumps Tecplot triangulations",
        "tm": "whether ``flowCart`` is set for cut cell gradient",
        "unsteady": "whether to run time-accurate ``flowCart``",
        "vizTD": "steps between ``flowCart`` visualization outputs",
        "y_is_spanwise": "whether *y* is spanwise axis for ``flowCart``",
    }


# Add properties
FlowCartOpts.add_properties(FlowCartOpts._optlist)


# Class for flowCart settings
class AdjointCartOpts(ExecOpts):
    r"""Class for ``adjointCart`` settings

    :Call:
        >>> opts = AdjointCartOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`AdjointCartOpts`
            CLI options for ``adjointCart``
    :Versions:
        * 2014-12-17 ``@ddalle``: Version 1.0 (``adjointCart``)
        * 2022-11-02 ``@ddalle``: Version 2.0; use ``optdict``
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "adj_first_order",
        "it_ad",
        "mg_ad",
    }

    # Types
    _opttypes = {
        "adj_first_order": BOOL_TYPES,
        "it_ad": INT_TYPES,
        "mg_ad": INT_TYPES,
    }

    # Defaults
    _rc = {
        "adj_first_order": False,
        "it_ad": 120,
        "mg_ad": 3,
    }

    # Descriptions
    _rst_descriptions = {
        "whether to run ``adjointCart`` first-order",
        "number of iterations for ``adjointCart``",
        "multigrid levels for ``adjointCart``",
    }


# Add properties
AdjointCartOpts.add_properties(AdjointCartOpts._optlist)


# Class for mesh adaptation settings
class AdaptationOpts(OptionsDict):
    r"""Class for Cart3D mesh adaptation settings

    This mostly affects the file ``aero.csh``.

    :Call:
        >>> opts = AdaptationOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`AdaptationOpts`
            Cart3D mesh adaptation settings for pycart
    :Versions:
        * 2014-12-17 ``@ddalle``: Version 1.0 (``Adaptation``)
        * 2022-11-02 ``@ddalle``: Version 2.0; use ``optdict``
    """
    # Additional attributes
    __slots__ = ()

    # Accepted options
    _optlist = {
        "apc",
        "buf",
        "etol",
        "final_mesh_xref",
        "jumpstart",
        "max_nCells",
        "mesh_growth",
        "n_adapt_cycles",
        "ws_it",
    }

    # Types
    _opttypes = {
        "apc": set,
        "buf": INT_TYPES,
        "etol": FLOAT_TYPES,
        "final_mesh_xref": INT_TYPES,
        "jumpstart": BOOL_TYPES,
        "max_nCells": INT_TYPES,
        "mesh_growth": FLOAT_TYPES,
        "n_adapt_cycles": INT_TYPES,
        "ws_it": INT_TYPES,
    }

    # Allowed values
    _optvals = {
        "apc": ("a", "p"),
    }

    # List parameters
    _optlistdepth = {
        "apc": 1,
        "mesh_growth": 1,
        "ws_it": 1,
    }

    # Defaults
    _rc = {
        "apc": "a",
        "buf": 1,
        "etol": 1e-6,
        "final_mesh_xref": 0,
        "jumpstart": False,
        "max_nCells": 5e6,
        "mesh_growth": 1.5,
        "n_adapt_cycles": 0,
        "ws_it": 50,
    }

    # Descriptions
    _rst_descriptions = {
        "buf": "number of buffer layers",
        "apc": "adaptation cycle type (adapt/propagate)",
        "etol": "target output error tolerance",
        "final_mesh_xref": "num. of additional adapts using final error map",
        "jumpstart": "whether to create meshes b4 running ``aero.csh``",
        "max_nCells": "maximum cell count",
        "mesh_growth": "mesh growth ratio between cycles of ``aero.csh``",
        "n_adapt_cycles": "number of Cart3D adaptation cycles in phase",
        "ws_it": "number of ``flowCart`` iters for ``aero.csh`` cycles",
    }


AdaptationOpts.add_property("buf", name="abuff")
AdaptationOpts.add_properties(AdaptationOpts._optlist)


# Class for autoInputs
class AutoInputsOpts(ExecOpts):
    r"""Class for cart3D ``autoInputs`` settings

    :Call:
        >>> opts = AdjointCartOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`AutoInputsOpts`
            CLI options for ``autoInputs``
    :Versions:
        * 2014-12-17 ``@ddalle``: Version 1.0 (``autoInputs``)
        * 2022-11-03 ``@ddalle``: Version 2.0; use ``optdict``
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "maxR",
        "nDiv",
        "r",
        "run",
    }

    # Types
    _opttypes = {
        "maxR": INT_TYPES,
        "nDiv": INT_TYPES,
        "r": FLOAT_TYPES + INT_TYPES,
    }

    # Defaults
    _rc = {
        "maxR": 10,
        "nDiv": 4,
        "r": 30.0,
    }

    # Descriptions
    _rst_descriptions = {
        "maxR": "maximum number of cell refinements",
        "nDiv": "number of divisions in background mesh",
        "r": "nominal ``autoInputs`` mesh radius",
    }


# Add properties
AutoInputsOpts.add_properties(AutoInputsOpts._optlist, prefix="autoInputs_")


# Class for cubes
class CubesOpts(ExecOpts):
    r"""Class for Cart3D ``cubes`` settings

    :Call:
        >>> opts = CubesOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`CubesOpts`
            CLI options for ``cubes``
    :Versions:
        * 2014-12-17 ``@ddalle``: Version 1.0 (``cubes``)
        * 2022-11-03 ``@ddalle``: Version 2.0; use ``optdict``
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "a",
        "b",
        "maxR",
        "pre",
        "reorder",
        "sf",
    }

    # Aliases
    _optmap = {
        "cubes_a": "a",
        "cubes_b": "b",
    }

    # Types
    _opttypes = {
        "a": INT_TYPES + FLOAT_TYPES,
        "b": INT_TYPES,
        "maxR": INT_TYPES,
        "pre": str,
        "reorder": BOOL_TYPES,
        "sf": INT_TYPES,
    }

    # Defaults
    _rc = {
        "a": 10.0,
        "b": 2,
        "maxR": 11,
        "pre": "preSpec.c3d.cntl",
        "reorder": True,
        "sf": 0,
    }

    # Descriptions
    _rst_descriptions = {
        "a": "angle threshold [deg] for geom refinement",
        "b": "number of layers of buffer cells",
        "maxR": "maximum number of refinements in ``cubes`` mesh",
        "reorder": "whether to reorder output mesh",
        "sf": "additional levels at sharp edges",
    }


# Property lists
_CUBES_PREFIX = ("a", "b", "run")
_CUBES_PROPS = ("maxR", "reorder", "sf")
# Add properties
CubesOpts.add_property("pre", name="preSpecCntl")
CubesOpts.add_properties(_CUBES_PREFIX, prefix="cubes_")
CubesOpts.add_properties(_CUBES_PROPS)


# Class for flowCart settings
class RunControlOpts(runctlopts.RunControlOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "Adaptation",
        "Adaptive",
        "Archive",
        "adjointCart",
        "autoInputs",
        "cubes",
        "flowCart",
    }

    # Option types
    _opttypes = {
        "Adaptive": BOOL_TYPES,
    }

    # Defaults
    _rc = {
        "Adaptive": False,
    }

    # Descriptions
    _rst_descriptions = {
        "Adaptive": "whether or not to use ``aero.csh`` in phase",
    }

    # Additional sections
    _sec_cls = {
        "Adaptation": AdaptationOpts,
        "Archive": ArchiveOpts,
        "adjointCart": AdjointCartOpts,
        "autoInputs": AutoInputsOpts,
        "cubes": CubesOpts,
        "flowCart": FlowCartOpts,
    }


# Add properties
RunControlOpts.add_property("Adaptive")
# Promote subsections
RunControlOpts.promote_sections()

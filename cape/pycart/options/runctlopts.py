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
from .util import rc0, odict, isArray, getel
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
        "nDiv",
        "r",
    }

    # Types
    _opttypes = {
        "nDiv": INT_TYPES,
        "r": FLOAT_TYPES + INT_TYPES,
    }

    # Defaults
    _rc = {
        "nDiv": 4,
        "r": 30.0,
    }

    # Descriptions
    _rst_descriptions = {
        "nDiv": "number of divisions in background mesh",
        "r": "nominal ``autoInputs`` mesh radius",
    }
    

# Add properties
AutoInputsOpts.add_properties(AutoInputsOpts._optlist)


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
        "reorder"
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
_CUBES_PREFIX = ("a", "b")
_CUBES_PROPS = ("maxR", "reorder", "sf")
# Add properties
CubesOpts.add_property("pre", name="preSpecCntl")
CubesOpts.add_properties(_CUBES_PREFIX, prefix="cubes_")
CubesOpts.add_properties(_CUBES_PROPS)


# Class for flowCart settings
class RunControl(runControl.RunControl):
    """Dictionary-based interface for options for running Cart3D"""
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._flowCart()
        self._adjointCart()
        self._Adaptation()
        self._autoInputs()
        self._cubes()
        self._Environ()
        self._ulimit()
        self._Archive()
    
   # ============ 
   # Initializers
   # ============
   # <
   
    # Initialization and confirmation for flowCart options
    def _flowCart(self):
        """Initialize `flowCart` optiosn if necessary"""
        if 'flowCart' not in self:
            # Empty/default
            self['flowCart'] = flowCart()
        elif type(self['flowCart']).__name__ == 'dict':
            # Convert to special class
            self['flowCart'] = flowCart(**self['flowCart'])
    
    # Initialization and confirmation for adjointCart options
    def _adjointCart(self):
        """Initialize `adjointCart` optiosn if necessary"""
        if 'adjointCart' not in self:
            # Empty/default
            self['adjointcart'] = adjointCart()
        elif type(self['adjointCart']).__name__ == 'dict':
            # Convert to special class
            self['adjointCart'] = adjointCart(**self['adjointCart'])
    
    # Initialization and confirmation for autoInputs options
    def _Adaptation(self):
        """Initialize adaptive options if necessary"""
        # Check for missing entirely.
        if 'Adaptation' not in self:
            # Empty/default
            self['Adaptation'] = Adaptation()
        elif type(self['Adaptation']).__name__ == 'dict':
            # Convert to special class.
            self['Adaptation'] = Adaptation(**self['Adaptation'])
    
    # Initialization and confirmation for autoInputs options
    def _autoInputs(self):
        """Initialize ``autoInputs`` options if necessary"""
        # Get the value and type
        v = self.get('autoInputs')
        t = type(v).__name__
        # Check inputs
        if t == 'autoInputs':
            # Already initialized
            return
        elif v is None:
            # Empty/default
            self['autoInputs'] = autoInputs()
        elif t == 'dict':
            # Convert to special class
            self['autoInputs'] = autoInputs(**v)
        else:
            # Initialize
            self['autoInputs'] = autoInputs()
            # Set a flag
            if v:
                self['autoInputs']['run'] = True
            else:
                self['autoInputs']['run'] = False
    
    # Initialization and confirmation for cubes options
    def _cubes(self):
        """Initialize ``cubes`` options if necessary"""
        # Get the value and type
        v = self.get('cubes')
        t = type(v).__name__
        # Check inputs
        if t == 'cubes':
            # Already initialized
            return
        elif v is None:
            # Empty/default
            self['cubes'] = cubes()
        elif t == 'dict':
            # Convert to special class
            self['cubes'] = cubes(**v)
        else:
            # Initialize
            self['cubes'] = cubes()
            # Set a flag
            if v:
                self['cubes']['run'] = True
            else:
                self['cubes']['run'] = False
            
    # Environment variable interface
    def _Environ(self):
        """Initialize environment variables if necessary"""
        if 'Environ' not in self:
            # Empty/default
            self['Environ'] = Environ()
        elif type(self['Environ']).__name__ == 'dict':
            # Convert to special class
            self['Environ'] = Environ(**self['Environ'])
            
    # Environment variable interface
    def _ulimit(self):
        """Initialize environment variables if necessary"""
        if 'ulimit' not in self:
            # Empty/default
            self['ulimit'] = ulimit()
        elif type(self['ulimit']).__name__ == 'dict':
            # Convert to special class
            self['ulimit'] = ulimit(**self['ulimit'])
            
   # >
    
   # ============== 
   # Local settings
   # ==============
   # <
    # Get aero.csh status
    def get_Adaptive(self, i=None):
        """Return whether or not to use `aero.csh`
        
        :Call:
            >>> ac = opts.get_Adaptive(i=None)
        :Inputs:
            *opts*: :class:`cape.pycart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *ac*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use `aero.csh`
        :Versions:
            * 2014-10-03 ``@ddalle``: Version 1.0
            * 2015-11-13 ``@ddalle``: ``use_aero_csh`` -> ``Adaptive``
        """
        return self.get_key('Adaptive', i)
        
    # Set aero.csh status
    def set_Adaptive(self, ac=rc0('Adaptive'), i=None):
        """Return whether or not to use `aero.csh`
        
        :Call:
            >>> opts.set_Adaptive(ac, i=None)
        :Inputs:
            *opts*: :class:`cape.pycart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *ac*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use `aero.csh`
        :Versions:
            * 2014-10-03 ``@ddalle``: Version 1.0
            * 2015-11-13 ``@ddalle``: ``use_aero_csh`` -> ``Adaptive``
        """
        self.set_key('Adaptive', ac, i)
   # >
    
   # ===================
   # flowCart parameters
   # ===================
   # <
    
    # Get flowCart order
    def get_first_order(self, i=None):
        self._flowCart()
        return self['flowCart'].get_first_order(i)
        
    # Set flowCart order
    def set_first_order(self, fo=rc0('first_order'), i=None):
        self._flowCart()
        self['flowCart'].set_first_order(fo, i)
    
    # Get flowCart robust mode
    def get_robust_mode(self, i=None):
        self._flowCart()
        return self['flowCart'].get_robust_mode(i)
        
    # Set flowCart robust mode
    def set_robust_mode(self, rm=rc0('robust_mode'), i=None):
        self._flowCart()
        self['flowCart'].set_robust_mode(rm, i)
    
    # Number of iterations
    def get_it_fc(self, i=None):
        self._flowCart()
        return self['flowCart'].get_it_fc(i)
        
    # Set flowCart iteration count
    def set_it_fc(self, it_fc=rc0('it_fc'), i=None):
        self._flowCart()
        self['flowCart'].set_it_fc(it_fc, i)
    
    # Averaging interval
    def get_it_avg(self, i=None):
        self._flowCart()
        return self['flowCart'].get_it_avg(i)
        
    # Set flowCart averaging interval
    def set_it_avg(self, it_avg=rc0('it_avg'), i=None):
        self._flowCart()
        self['flowCart'].set_it_avg(it_avg, i)
        
    # Get the number of subiterations
    def get_it_sub(self, i=None):
        self._flowCart()
        return self['flowCart'].get_it_sub(i)
        
    # Set the number of subiterations
    def set_it_sub(self, it_sub=rc0('it_sub'), i=None):
        self._flowCart()
        self['flowCart'].set_it_sub(it_sub, i)
        
    # Get the number of startup iterations
    def get_it_start(self, i=None):
        self._flowCart()
        return self['flowCart'].get_it_start(i)
        
    # Set the number of subiterations
    def set_it_start(self, it_start=rc0('it_start'), i=None):
        self._flowCart()
        self['flowCart'].set_it_start(it_start, i)
        
    # Setting for ``Components.i.triq``
    def get_clic(self, i=None):
        self._flowCart()
        return self['flowCart'].get_clic(i)
        
    # Setting for ``Components.i.triq``
    def set_clic(self, clic=rc0('clic'), i=None):
        self._flowCart()
        self['flowCart'].set_clic(clic, i)
        
    # Get number of orders for early termination
    def get_nOrders(self, i=None):
        self._flowCart()
        return self['flowCart'].get_nOrders(i)
        
    # Set number of orders for early termination
    def set_nOrders(self, nOrders=rc0('nOrders'), i=None):
        self._flowCart()
        self['flowCart'].set_nOrders(nOrders, i)
        
    # Get flowCart iteration count
    def get_mg_fc(self, i=None):
        self._flowCart()
        return self['flowCart'].get_mg_fc(i)
        
    # Set flowCart iteration count
    def set_mg_fc(self, mg_fc=rc0('mg_fc'), i=None):
        self._flowCart()
        self['flowCart'].set_mg_fc(mg_fc, i)
        
    # Get flowCart full multigrid setting
    def get_fmg(self, i=None):
        self._flowCart()
        return self['flowCart'].get_fmg(i)
        
    # Set flowCart multigrid
    def set_fmg(self, fmg=rc0('fmg'), i=None):
        self._flowCart()
        self['flowCart'].set_fmg(fmg, i)
        
    # Get flowCart ploy multigrid setting
    def get_pmg(self, i=None):
        self._flowCart()
        return self['flowCart'].get_pmg(i)
        
    # Set flowCart multigrid
    def set_pmg(self, pmg=rc0('pmg'), i=None):
        self._flowCart()
        self['flowCart'].set_pmg(pmg, i)
        
    # Get unsteady status
    def get_unsteady(self, i=None):
        self._flowCart()
        return self['flowCart'].get_unsteady(i)
        
    # Set unsteady status
    def set_unsteady(self, td_fc=rc0('unsteady'), i=None):
        self._flowCart()
        self['flowCart'].set_unsteady(td_fc, i)
        
    # Get the nominal CFL number
    def get_cfl(self, i=None):
        self._flowCart()
        return self['flowCart'].get_cfl(i)
        
    # Set the nominal CFL number
    def set_cfl(self, cfl=rc0('cfl'), i=None):
        self._flowCart()
        self['flowCart'].set_cfl(cfl, i)
        
    # Get the minimum CFL number
    def get_cflmin(self, i=None):
        self._flowCart()
        return self['flowCart'].get_cflmin(i)
    
    # Set the minimum CFL number
    def set_cflmin(self, cflmin=rc0('cflmin'), i=None):
        self._flowCart()
        self['flowCart'].set_cflmin(cflmin, i)
        
    # Get the nondimensional physical time step
    def get_dt(self, i=None):
        self._flowCart()
        return self['flowCart'].get_dt(i)
        
    # Set the nondimensional physical time step
    def set_dt(self, dt=rc0('dt'), i=None):
        self._flowCart()
        self['flowCart'].set_dt(dt, i)
        
    # Get cut-cell gradient flag
    def get_tm(self, i=None):
        self._flowCart()
        return self['flowCart'].get_tm(i)
        
    # Set cut-cell gradient flag
    def set_tm(self, tm=rc0('tm'), i=None):
        self._flowCart()
        self['flowCart'].set_tm(tm, i)
        
    # Get buffer limiter switch
    def get_buffLim(self, i=None):
        self._flowCart()
        return self['flowCart'].get_buffLim(i)
        
    # Set buffer limiter switch.
    def set_buffLim(self, buffLim=rc0('buffLim'), i=None):
        self._flowCart()
        self['flowCart'].set_buffLim(buffLim, i)
        
    # Get the number of time steps between checkpoints
    def get_checkptTD(self, i=None):
        self._flowCart()
        return self['flowCart'].get_checkptTD(i)
        
    # Set the number of time steps between checkpoints
    def set_checkptTD(self, checkptTD=rc0('checkptTD'), i=None):
        self._flowCart()
        self['flowCart'].set_checkptTD(checkptTD, i)
        
    # Get the number of time steps between visualization outputs
    def get_vizTD(self, i=None):
        self._flowCart()
        return self['flowCart'].get_vizTD(i)
        
    # Set the number of time steps visualization outputs
    def set_vizTD(self, vizTD=rc0('vizTD'), i=None):
        self._flowCart()
        self['flowCart'].set_vizTD(vizTD, i)
        
    # Get the relaxation step command
    def get_fc_clean(self, i=None):
        self._flowCart()
        return self['flowCart'].get_fc_clean(i)
        
    # Set the relaxation step command
    def set_fc_clean(self, fc_clean=rc0('fc_clean'), i=None):
        self._flowCart()
        self['flowCart'].set_fc_clean(fc_clean, i)
        
    # Get the number of iterations to average over
    def get_fc_stats(self, i=None):
        self._flowCart()
        return self['flowCart'].get_fc_stats(i)
    
    # Set the number of iterations to average over
    def set_fc_stats(self, nstats=rc0('fc_stats'), i=None):
        self._flowCart()
        self['flowCart'].set_fc_stats(nstats, i)
        
    # Get the limiter
    def get_limiter(self, i=None):
        self._flowCart()
        return self['flowCart'].get_limiter(i)
    
    # Set the limiter
    def set_limiter(self, limiter=rc0('limiter'), i=None):
        self._flowCart()
        self['flowCart'].set_limiter(limiter, i)
        
    # Get the y_is_spanwise status
    def get_y_is_spanwise(self, i=None):
        self._flowCart()
        return self['flowCart'].get_y_is_spanwise(i)
        
    # Set the y_is_spanwise status
    def set_y_is_spanwise(self, y_is_spanwise=rc0('y_is_spanwise'), i=None):
        self._flowCart()
        self['flowCart'].set_y_is_spanwise(y_is_spanwise, i)
        
    # Get the binary I/O status
    def get_binaryIO(self, i=None):
        self._flowCart()
        return self['flowCart'].get_binaryIO(i)
        
    # Set the binary I/O status
    def set_binaryIO(self, binaryIO=rc0('binaryIO'), i=None):
        self._flowCart()
        self['flowCart'].set_binaryIO(binaryIO, i)
        
    # Get the Tecplot output status
    def get_tecO(self, i=None):
        self._flowCart()
        return self['flowCart'].get_tecO(i)
        
    # Set the Tecplot output status
    def set_tecO(self, tecO=rc0('tecO'), i=None):
        self._flowCart()
        self['flowCart'].set_tecO(tecO, i)
        
    # Get the current Runge-Kutta scheme
    def get_RKScheme(self, i=None):
        self._flowCart()
        return self['flowCart'].get_RKScheme(i)
        
    # Set the Runge-Kutta scheme
    def set_RKScheme(self, RK=rc0('RKScheme'), i=None):
        self._flowCart()
        self['flowCart'].set_RKScheme(RK, i)
        
    # Copy over the documentation.
    for k in ['it_fc', 'it_sub', 'it_avg', 'it_start', 'dt', 'clic',
            'unsteady', 'first_order', 'robust_mode', 'RKScheme',
            'tm', 'mg_fc', 'cfl', 'cflmin', 'limiter', 'fmg', 'pmg', 
            'checkptTD', 'vizTD', 'fc_clean', 'fc_stats',
            'nOrders', 'buffLim', 'y_is_spanwise',
            'binaryIO', 'tecO']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(flowCart,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(flowCart,'set_'+k).__doc__
   # >
   
   # ====================
   # adjointCart settings
   # ====================
   # <
    
    # Number of iterations
    def get_it_ad(self, i=None):
        self._adjointCart()
        return self['adjointCart'].get_it_ad(i)
        
    # Set adjointCart iteration count
    def set_it_ad(self, it_ad=rc0('it_ad'), i=None):
        self._adjointCart()
        self['adjointCart'].set_it_ad(it_ad, i)
    
    # Get adjointCart iteration count
    def get_mg_ad(self, i=None):
        self._adjointCart()
        return self['adjointCart'].get_mg_ad(i)
        
    # Set adjointCart iteration count
    def set_mg_ad(self, mg_ad=rc0('mg_ad'), i=None):
        self._adjointCart()
        self['adjointCart'].set_mg_ad(mg_ad, i)
        
    # First-order adjoint
    def get_adj_first_order(self, i=None):
        self._adjointCart()
        return self['adjointCart'].get_adj_first_order(i)
        
    # First-order adjoint
    def set_adj_first_order(self, adj=rc0('adj_first_order'), i=None):
        self._adjointCart()
        self['adjointCart'].set_adj_first_order(adj, i)
        
    # Copy over the documentation.
    for k in ['it_ad', 'mg_ad', 'adj_first_order']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(adjointCart,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(adjointCart,'set_'+k).__doc__
   # >
    
   # ===================
   # Adaptation settings
   # ===================
   # <
    
    # Get number of adapt cycles
    def get_n_adapt_cycles(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_n_adapt_cycles(i)
        
    # Set number of adapt cycles
    def set_n_adapt_cycles(self, nAdapt=rc0('n_adapt_cycles'), i=None):
        self._Adaptation()
        self['Adaptation'].set_n_adapt_cycles(nAdapt, i)
        
    # Get jumpstart status
    def get_jumpstart(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_jumpstart(i)
        
    # Jumpstart status
    def set_jumpstart(self, js=rc0('jumpstart'), i=None):
        self._Adaptation()
        self['Adaptation'].set_jumpstart(js, i)
    
    # Get error tolerance
    def get_etol(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_etol(i)
        
    # Set error tolerance
    def set_etol(self, etol=rc0('etol'), i=None):
        self._Adaptation()
        self['Adaptation'].set_etol(etol, i)
    
    # Get maximum cell count
    def get_max_nCells(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_max_nCells(i)
        
    # Set maximum cell count
    def set_max_nCells(self, etol=rc0('max_nCells'), i=None):
        self._Adaptation()
        self['Adaptation'].set_max_nCells(etol, i)
    
    # Get flowCart iterations on refined meshes
    def get_ws_it(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_ws_it(i)
        
    # Set flowCart iterations on refined meshes
    def set_ws_it(self, ws_it=rc0('ws_it'), i=None):
        self._Adaptation()
        self['Adaptation'].set_ws_it(ws_it, i)
        
    # Get mesh growth ratio
    def get_mesh_growth(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_mesh_growth(i)
        
    # Set mesh growth ratio
    def set_mesh_growth(self, mesh_growth=rc0('mesh_growth'), i=None):
        self._Adaptation()
        self['Adaptation'].set_mesh_growth(mesh_growth, i)
        
    # Get mesh refinement cycle type
    def get_apc(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_apc(i)
        
    # Set mesh refinement cycle type
    def set_apc(self, apc=rc0('apc'), i=None):
        self._Adaptation()
        self['Adaptation'].set_apc(apc, i)
        
    # Get number of buffer layers
    def get_abuff(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_abuff(i)
        
    # Set number of buffer layers
    def set_abuff(self, buf=rc0('buf'), i=None):
        self._Adaptation()
        self['Adaptation'].set_abuff(abuff, i)
    
    # Get number of additional adaptations on final error map
    def get_final_mesh_xref(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_final_mesh_xref(i)
    
    # Set number of additional adaptations on final error map
    def set_final_mesh_xref(self, xref=rc0('final_mesh_xref'), i=None):
        self._Adaptation()
        self['Adaptation'].set_final_mesh_xref(xref, i)
        
    # Copy over the documentation.
    for k in ['n_adapt_cycles', 'etol', 'max_nCells', 'ws_it',
            'mesh_growth', 'apc', 'abuff', 'final_mesh_xref']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Adaptation,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(Adaptation,'set_'+k).__doc__
   # >
    
   # ==========
   # autoInputs
   # ==========
   # <
    
    # Get flag for autoInputs use
    def get_autoInputs(self, j=0):
        """Return whether or not to run ``autoInputs``
        
        :Call:
            >>> q = opts.get_autoInputs(j=0)
        :Inputs:
            *opts*: :class:`cape.pycart.options.Options`
                Options interface
            *j*: :class:`int`
                Phase number
        :Outputs:
            *q*: :class:`bool`
                Whether or not there are nontrivial ``autoInputs`` settings
        :Versions:
            * 2016-04-06 ``@ddalle``: Version 1.0
        """
        # Initialize if necessary
        self._autoInputs()
        # Get the value
        v = self.get('autoInputs')
        # Get the flag
        q = v.get('run')
        # Check.
        if q is None:
            # Check for nontrivial entries
            return len(v.keys()) > 0
        else:
            # Return the 'run' flag
            return getel(q,j) == True
        
    # Get the nominal mesh radius
    def get_r(self, i=None):
        self._autoInputs()
        return self['autoInputs'].get_r(i)
        
    # Set the nominal mesh radius
    def set_r(self, r=rc0('r'), i=None):
        self._autoInputs()
        self['autoInputs'].set_r(r, i)
        
    # Get the background mesh divisions
    def get_nDiv(self, i=None):
        self._autoInputs()
        return self['autoInputs'].get_nDiv(i)
    
    # Set the background mesh divisions
    def set_nDiv(self, nDiv=rc0('nDiv'), i=None):
        self._autoInputs()
        self['autoInputs'].set_nDiv(nDiv, i)
        
    # Copy over the documentation.
    for k in ['r', 'nDiv']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(autoInputs,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(autoInputs,'set_'+k).__doc__
   # >
        
   # =====
   # cubes
   # =====
   # <
    
    # Get flag for autoInputs use
    def get_cubes(self, j=0):
        """Return whether or not to run ``cubes``
        
        :Call:
            >>> q = opts.get_cubes(j=0)
        :Inputs:
            *opts*: :class:`cape.pycart.options.Options`
                Options interface
            *j*: :class:`int`
                Phase number
        :Outputs:
            *q*: :class:`bool`
                Whether or not there are nontrivial ``cubes`` settings
        :Versions:
            * 2016-04-06 ``@ddalle``: Version 1.0
        """
        # Initialize if necessary
        self._cubes()
        # Get the value
        v = self.get('cubes')
        # Get the flag
        q = v.get('run')
        # Check.
        if q is None:
            # Check for nontrivial entries
            return len(v.keys()) > 0
        else:
            # Return the 'run' flag
            return getel(q,j) == True
    
    # Get the number of refinements
    def get_maxR(self, i=None):
        self._cubes()
        return self['cubes'].get_maxR(i)
        
    # Set the number of refinements
    def set_maxR(self, maxR=rc0('maxR'), i=None):
        self._cubes()
        self['cubes'].set_maxR(maxR, i)
        
    # Get the 'cubes_a' parameter
    def get_cubes_a(self, i=None):
        self._cubes()
        return self['cubes'].get_cubes_a(i)
        
    # Set the 'cubes_a' parameter
    def set_cubes_a(self, cubes_a=rc0('cubes_a'), i=None):
        self._cubes()
        self['cubes'].set_cubes_a(cubes_a, i)
        
    # Get the 'cubes_b' parameter
    def get_cubes_b(self, i=None):
        self._cubes()
        return self['cubes'].get_cubes_b(i)
        
    # Set the 'cubes_a' parameter
    def set_cubes_b(self, cubes_b=rc0('cubes_b'), i=None):
        self._cubes()
        self['cubes'].set_cubes_b(cubes_b, i)
        
    # Get the mesh reordering status
    def get_reorder(self, i=None):
        self._cubes()
        return self['cubes'].get_reorder(i)
        
    # Set the mesh reordering status
    def set_reorder(self, reorder=rc0('reorder'), i=None):
        self._cubes()
        self['cubes'].set_reorder(reorder, i)
        
    # Get the additional refinements around sharp edges
    def get_sf(self, i=None):
        self._cubes()
        return self['cubes'].get_sf(i)
        
    # Set the additional refinements around sharp edges
    def set_sf(self, sf=rc0('sf'), i=None):
        self._cubes()
        self['cubes'].set_sf(sf, i)
    
    # Get preSpec file
    def get_preSpecCntl(self):
        self._cubes()
        return self['cubes'].get_preSpecCntl()
        
    # Set preSpec file
    def set_preSpecCntl(self, fpre=rc0('pre')):
        self._cubes()
        self['cubes'].set_preSpecCntl(fpre)


    # Copy over the documentation.
    for k in ['maxR', 'cubes_a', 'cubes_b', 'reorder', 'sf', 'preSpecCntl']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(cubes,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(cubes,'set_'+k).__doc__
   # >
    
   # =================
   # Folder management
   # =================
   # <
    
    # Initialization method for folder management optoins
    def _Archive(self):
        self.init_section(ArchiveOpts, "Archive")
    
    # Get the number of check point files to keep around
    def get_nCheckPoint(self, i=None):
        self._Archive()
        return self['Archive'].get_nCheckPoint(i)
        
    # Set the number of check point files to keep around
    def set_nCheckPoint(self, nchk=rc0('nCheckPoint'), i=None):
        self._Archive()
        self['Archive'].set_nCheckPoint(nchk, i)
        
    # Get the archive status for adaptation folders
    def get_TarAdapt(self):
        self._Archive()
        return self['Archive'].get_TarAdapt()
        
    # Get the archive status for adaptation folders
    def set_TarAdapt(self, fmt=rc0('TarAdapt')):
        self._Archive()
        self['Archive'].set_TarAdapt(fmt)
        
    # Get the archive format for visualization files
    def get_TarViz(self):
        self._Archive()
        return self['Archive'].get_TarViz()
        
    # Set the archive format for visualization files
    def set_TarViz(self, fmt=rc0('TarViz')):
        self._Archive()
        self['Archive'].set_TarViz(fmt)
        
    # Copy over the documentation.
    for k in ['nCheckPoint', 'TarViz', 'TarAdapt']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(ArchiveOpts,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(ArchiveOpts,'set_'+k).__doc__
   # >
# class RunControl


"""
Cart3D and pyCart settings module: :mod:`pyCart.options`
========================================================

This module provides tools to read, access, modify, and write settings for
:mod:`pyCart`.  The class is based off of the built-int :class:`dict` class, so
its default behavior, such as ``opts['InputCntl']`` or 
``opts.get('InputCntl')`` are also present.  In addition, many convenience
methods, such as ``opts.set_it_fc(n)``, which sets the number of
:file:`flowCart` iterations,  are provided.

In addition, this module controls default values of each pyCart
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    *. Values directly specified in the input file, :file:`pyCart.json`
    
    *. Values specified in the default control file,
       :file:`$PYCART/settings/pyCart.default.json`
    
    *. Hard-coded defaults from this module
"""

# Import options-specific utilities (loads :mod:`os`, too)
from util import *

# Import template module
import cape.options

# Import modules for controlling specific parts of Cart3D
from .runControl  import RunControl
from .Mesh        import Mesh
from .pbs         import PBS
from .Config      import Config
from .Functional  import Functional
from .DataBook    import DataBook
from .Report      import Report


# Class definition
class Options(cape.options.Options):
    """
    Options structure, subclass of :class:`dict`
    
    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary to be transformed into :class:`pyCart.options.Options`
    :Versions:
        * 2014.07.28 ``@ddalle``: First version
    """
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Initialization method with optional JSON input"""
        # Check for an input file.
        if fname:
            # Read the JSON file
            d = loadJSONFile(fname)
            # Loop through the keys.
            for k in d:
                kw[k] = d[k]
        # Read the defaults.
        defs = getPyCartDefaults()
        
        # Apply the defaults.
        kw = applyDefaults(kw, defs)
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._RunControl()
        self._Mesh()
        self._PBS()
        self._Config()
        self._Functional()
        self._DataBook()
        self._Report()
        # Add extra folders to path.
        self.AddPythonPath()
    
    # ============
    # Initializers
    # ============
   # <
    
    # Initialization and confirmation for RunControl options
    def _RunControl(self):
        """Initialize `RunControl` options if necessary"""
        # Check for missing entirely.
        if 'RunControl' not in self:
            # Empty/default
            self['RunControl'] = RunControl()
        elif type(self['RunControl']).__name__ == 'dict':
            # Convert to special class.
            self['RunControl'] = RunControl(**self['RunControl'])
    
    # Initialization and confirmation for Adaptation options
    def _Mesh(self):
        """Initialize mesh options if necessary"""
        # Check status
        if 'Mesh' not in self:
            # Missing entirely
            self['Mesh'] = Mesh()
        elif type(self['Mesh']).__name__ == 'dict':
            # Convert to special class.
            self['Mesh'] = Mesh(**self['Mesh'])
            
    # Initialization and confirmation for PBS options
    def _PBS(self):
        """Initialize PBS options if necessary"""
        # Check status.
        if 'PBS' not in self:
            # Missing entirely
            self['PBS'] = PBS()
        elif type(self['PBS']).__name__ == 'dict':
            # Add prefix to all the keys.
            tmp = {}
            for k in self['PBS']:
                tmp["PBS_"+k] = self['PBS'][k]
            # Convert to special class.
            self['PBS'] = PBS(**tmp)
            
    # Initialization method for databook
    def _DataBook(self):
        """Initialize data book options if necessary"""
        # Check status.
        if 'DataBook' not in self:
            # Missing entirely.
            self['DataBook'] = DataBook()
        elif type(self['DataBook']).__name__ == 'dict':
            # Convert to special class
            self['DataBook'] = DataBook(**self['DataBook'])
            
    # Initialization method for automated report
    def _Report(self):
        """Initialize report options if necessary"""
        # Check status.
        if 'Report' not in self:
            # Missing entirely.
            self['Report'] = Report()
        elif type(self['Report']).__name__ == 'dict':
            # Convert to special class
            self['Report'] = Report(**self['Report'])
            
    # Initialization and confirmation for PBS options
    def _Config(self):
        """Initialize configuration options if necessary"""
        # Check status.
        if 'Config' not in self:
            # Missing entirely
            self['Config'] = Config()
        elif type(self['Config']).__name__ == 'dict':
            # Add prefix to all the keys.
            tmp = {}
            for k in self['Config']:
                # Check for "File"
                if k == 'File':
                    # Add prefix.
                    tmp["Config"+k] = self['Config'][k]
                else:
                    # Use the key as is.
                    tmp[k] = self['Config'][k]
            # Convert to special class.
            self['Config'] = Config(**tmp)
            
    # Initialization method for Cart3D output functional
    def _Functional(self):
        """Initialize Cart3D output functional if neccessary"""
        # Check status.
        if 'Functional' not in self:
            # Missing entirely.
            self['Functional'] = Functional()
        elif type(self['Functional']).__name__ == 'dict':
            # Convert to (barely) special class.
            self['Functional'] = Functional(**self['Functional'])
   # >
    
    # ==============
    # Global Options
    # ==============
   # <
    
    # Method to get the input file
    def get_InputCntl(self):
        """Return the name of the master :file:`input.cntl` file
        
        :Call:
            >>> fname = opts.get_InputCntl()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fname*: :class:`str`
                Name of Cart3D input control template file
        :Versions:
            * 2014-09-30 ``@ddalle``: First version
        """
        return self.get('InputCntl', rc0('InputCntl'))
        
    # Method to set the input file
    def set_InputCntl(self, fname):
        """Set the name of the master :file:`input.cntl` file
        
        :Call:
            >>> opts.set_InputCntl(fname)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of Cart3D input control template file
        :Versions:
            * 2014-09-30 ``@ddalle``: First version
        """
        self['InputCntl'] = fname
    
    # Method to get the aero shell file
    def get_AeroCsh(self):
        """Return the name of the master :file:`aero.csh` file
        
        :Call:
            >>> fname = opts.get_AeroCsh()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fname*: :class:`str`
                Name of Cart3D aero shell template file
        :Versions:
            * 2014-09-30 ``@ddalle``: First version
        """
        return self.get('AeroCsh', rc0('AeroCsh'))
        
    # Method to set the input file
    def set_AeroCsh(self, fname):
        """Set the name of the master :file:`aero.csh` file
        
        :Call:
            >>> opts.set_AeroCsh(fname)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of Cart3D asero shell template file
        :Versions:
            * 2014-09-30 ``@ddalle``: First version
        """
        self['AeroCsh'] = fname
    
    # Method to determine if groups have common meshes.
    def get_GroupMesh(self):
        """Determine whether or not groups have common meshes
        
        :Call:
            >>> qGM = opts.get_GroupMesh()
        :Inputs:
            *opts* :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
        """
        # Safely get the trajectory.
        x = self.get('Trajectory', {})
        return x.get('GroupMesh', rc0('GroupMesh'))
        
    # Method to specify that meshes do or do not use the same mesh
    def set_GroupMesh(self, qGM=rc0('GroupMesh')):
        """Specify that groups do or do not use common meshes
        
        :Call:
            >>> opts.get_GroupMesh(qGM)
        :Inputs:
            *opts* :class:`pyCart.options.Options`
                Options interface
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
        """
        self['Trajectory']['GroupMesh'] = qGM
   # >
    
    # =====================
    # RunControl parameters
    # =====================
   # <
        
    # Get aero.csh status
    def get_Adaptive(self, i=None):
        self._RunControl()
        return self['RunControl'].get_Adaptive(i)
        
    # Set aero.csh status
    def set_Adaptive(self, ac=rc0('Adaptive'), i=None):
        self._RunControl()
        self['RunControl'].set_Adaptive(ac, i)
    
    # Get flowCart order
    def get_first_order(self, i=None):
        self._RunControl()
        return self['RunControl'].get_first_order(i)
        
    # Set flowCart order
    def set_first_order(self, fo=rc0('first_order'), i=None):
        self._RunControl()
        self['RunControl'].set_first_order(fo, i)
    
    # Get flowCart robust mode
    def get_robust_mode(self, i=None):
        self._RunControl()
        return self['RunControl'].get_robust_mode(i)
        
    # Set flowCart robust mode
    def set_robust_mode(self, rm=rc0('robust_mode'), i=None):
        self._RunControl()
        self['RunControl'].set_robust_mode(rm, i)
    
    # Number of iterations
    def get_it_fc(self, i=None):
        self._RunControl()
        return self['RunControl'].get_it_fc(i)
        
    # Set RunControl iteration count
    def set_it_fc(self, it_fc=rc0('it_fc'), i=None):
        self._RunControl()
        self['RunControl'].set_it_fc(it_fc, i)
    
    # Startup iteration interval
    def get_it_start(self, i=None):
        self._RunControl()
        return self['RunControl'].get_it_start(i)
        
    # Set startup iteration interval
    def set_it_start(self, it_start=rc0('it_start'), i=None):
        self._RunControl()
        self['RunControl'].set_it_start(it_start, i)
    
    # Averaging write interval
    def get_it_avg(self, i=None):
        self._RunControl()
        return self['RunControl'].get_it_avg(i)
        
    # Set RunControl write averaging interval
    def set_it_avg(self, it_avg=rc0('it_avg'), i=None):
        self._RunControl()
        self['RunControl'].set_it_avg(it_avg, i)
    
    # Subiterations
    def get_it_sub(self, i=None):
        self._RunControl()
        return self['RunControl'].get_it_sub(i)
        
    # Subiterations
    def set_it_sub(self, it_sub=rc0('it_sub'), i=None):
        self._RunControl()
        self['RunControl'].set_it_sub(it_sub, i)
        
    # Setting for ``Components.i.triq``
    def get_clic(self, i=None):
        self._RunControl()
        return self['RunControl'].get_clic(i)
        
    # Setting for ``Components.i.triq``
    def set_clic(self, clic=rc0('clic'), i=None):
        self._RunControl()
        self['RunControl'].set_clic(clic, i)
        
    # Get number of orders for early termination
    def get_nOrders(self, i=None):
        self._RunControl()
        return self['RunControl'].get_nOrders(i)
        
    # Set number of orders for early termination
    def set_nOrders(self, nOrders=rc0('nOrders'), i=None):
        self._RunControl()
        self['RunControl'].set_nOrders(nOrders, i)
        
    # Get flowCart iteration count
    def get_mg_fc(self, i=None):
        self._RunControl()
        return self['RunControl'].get_mg_fc(i)
        
    # Set flowCart iteration count
    def set_mg_fc(self, mg_fc=rc0('mg_fc'), i=None):
        self._RunControl()
        self['RunControl'].set_mg_fc(mg_fc, i)
        
    # Get flowCart full multigrid setting
    def get_fmg(self, i=None):
        self._RunControl()
        return self['RunControl'].get_fmg(i)
        
    # Set flowCart multigrid
    def set_fmg(self, fmg=rc0('fmg'), i=None):
        self._RunControl()
        self['RunControl'].set_fmg(fmg, i)
        
    # Get flowCart ploy multigrid setting
    def get_pmg(self, i=None):
        self._RunControl()
        return self['RunControl'].get_pmg(i)
        
    # Set flowCart multigrid
    def set_pmg(self, pmg=rc0('pmg'), i=None):
        self._RunControl()
        self['RunControl'].set_pmg(pmg, i)
        
    # Get unsteady status
    def get_unsteady(self, i=None):
        self._RunControl()
        return self['RunControl'].get_unsteady(i)
        
    # Set unsteady status
    def set_unsteady(self, td_fc=rc0('unsteady'), i=None):
        self._RunControl()
        self['RunControl'].set_unsteady(td_fc, i)
        
    # Get the nominal CFL number
    def get_cfl(self, i=None):
        self._RunControl()
        return self['RunControl'].get_cfl(i)
        
    # Set the nominal CFL number
    def set_cfl(self, cfl=rc0('cfl'), i=None):
        self._RunControl()
        self['RunControl'].set_cfl(cfl, i)
        
    # Get the minimum CFL number
    def get_cflmin(self, i=None):
        self._RunControl()
        return self['RunControl'].get_cflmin(i)
    
    # Set the minimum CFL number
    def set_cflmin(self, cflmin=rc0('cflmin'), i=None):
        self._RunControl()
        self['RunControl'].set_cflmin(cflmin, i)
        
    # Get the nondimensional physical time step
    def get_dt(self, i=None):
        self._RunControl()
        return self['RunControl'].get_dt(i)
        
    # Set the nondimensional physical time step
    def set_dt(self, dt=rc0('dt'), i=None):
        self._RunControl()
        self['RunControl'].set_dt(dt, i)
        
    # Get cut-cell gradient flag
    def get_tm(self, i=None):
        self._RunControl()
        return self['RunControl'].get_tm(i)
        
    # Set cut-cell gradient flag
    def set_tm(self, tm=rc0('tm'), i=None):
        self._RunControl()
        self['RunControl'].set_tm(tm, i)
        
    # Get buffer limiter switch
    def get_buffLim(self, i=None):
        self._RunControl()
        return self['RunControl'].get_buffLim(i)
        
    # Set buffer limiter switch.
    def set_buffLim(self, buffLim=rc0('buffLim'), i=None):
        self._RunControl()
        self['RunControl'].set_buffLim(buffLim, i)
        
    # Get the number of time steps between checkpoints
    def get_checkptTD(self, i=None):
        self._RunControl()
        return self['RunControl'].get_checkptTD(i)
        
    # Set the number of time steps between checkpoints
    def set_checkptTD(self, checkptTD=rc0('checkptTD'), i=None):
        self._RunControl()
        self['RunControl'].set_checkptTD(checkptTD, i)
        
    # Get the number of time steps between visualization outputs
    def get_vizTD(self, i=None):
        self._RunControl()
        return self['RunControl'].get_vizTD(i)
        
    # Set the number of time steps visualization outputs
    def set_vizTD(self, vizTD=rc0('vizTD'), i=None):
        self._RunControl()
        self['RunControl'].set_vizTD(vizTD, i)
        
    # Get the relaxation step command
    def get_fc_clean(self, i=None):
        self._RunControl()
        return self['RunControl'].get_fc_clean(i)
        
    # Set the relaxation step command
    def set_fc_clean(self, fc_clean=rc0('fc_clean'), i=None):
        self._RunControl()
        self['RunControl'].set_fc_clean(fc_clean, i)
        
    # Get the number of iterations to average over
    def get_fc_stats(self, i=None):
        self._RunControl()
        return self['RunControl'].get_fc_stats(i)
    
    # Set the number of iterations to average over
    def set_fc_stats(self, nstats=rc0('fc_stats'), i=None):
        self._RunControl()
        self['RunControl'].set_fc_stats(nstats, i)
        
    # Get the limiter
    def get_limiter(self, i=None):
        self._RunControl()
        return self['RunControl'].get_limiter(i)
    
    # Set the limiter
    def set_limiter(self, limiter=rc0('limiter'), i=None):
        self._RunControl()
        self['RunControl'].set_limiter(limiter, i)
        
    # Get the y_is_spanwise status
    def get_y_is_spanwise(self, i=None):
        self._RunControl()
        return self['RunControl'].get_y_is_spanwise(i)
        
    # Set the y_is_spanwise status
    def set_y_is_spanwise(self, y_is_spanwise=rc0('y_is_spanwise'), i=None):
        self._RunControl()
        self['RunControl'].set_y_is_spanwise(y_is_spanwise, i)
        
    # Get the binary I/O status
    def get_binaryIO(self, i=None):
        self._RunControl()
        return self['RunControl'].get_binaryIO(i)
        
    # Set the binary I/O status
    def set_binaryIO(self, binaryIO=rc0('binaryIO'), i=None):
        self._RunControl()
        self['RunControl'].set_binaryIO(binaryIO, i)
        
    # Get the Tecplot output status
    def get_tecO(self, i=None):
        self._RunControl()
        return self['RunControl'].get_tecO(i)
        
    # Set the Tecplot output status
    def set_tecO(self, tecO=rc0('tecO'), i=None):
        self._RunControl()
        self['RunControl'].set_tecO(tecO, i)
        
    # Get the current Runge-Kutta scheme
    def get_RKScheme(self, i=None):
        self._RunControl()
        return self['RunControl'].get_RKScheme(i)
        
    # Set the Runge-Kutta scheme
    def set_RKScheme(self, RK=rc0('RKScheme'), i=None):
        self._RunControl()
        self['RunControl'].set_RKScheme(RK, i)
        
    # Copy over the documentation.
    for k in ['Adaptive', 'first_order', 'robust_mode', 'unsteady', 
            'tm', 'dt', 'checkptTD',
            'vizTD', 'fc_clean', 'fc_stats', 'RKScheme',
            'nOrders', 'buffLim', 'it_avg', 'it_sub', 'clic',
            'it_fc', 'mg_fc', 'cfl', 'cflmin', 'limiter', 'tecO', 'fmg', 'pmg',
            'y_is_spanwise', 'binaryIO']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(RunControl,'set_'+k).__doc__
   # >
    
    
    # ====================
    # adjointCart settings
    # ====================
   # <
    
    # Number of iterations
    def get_it_ad(self, i=None):
        self._RunControl()
        return self['RunControl'].get_it_ad(i)
        
    # Set adjointCart iteration count
    def set_it_ad(self, it_ad=rc0('it_ad'), i=None):
        self._RunControl()
        self['RunControl'].set_it_ad(it_ad, i)
    
    # Get adjointCart iteration count
    def get_mg_ad(self, i=None):
        self._RunControl()
        return self['RunControl'].get_mg_ad(i)
        
    # Set adjointCart iteration count
    def set_mg_ad(self, mg_ad=rc0('mg_ad'), i=None):
        self._RunControl()
        self['RunControl'].set_mg_ad(mg_ad, i)
        
    # First-order adjoint
    def get_adj_first_order(self, i=None):
        self._RunControl()
        return self['RunControl'].get_adj_first_order(i)
        
    # First-order adjoint
    def set_adj_first_order(self, adj=rc0('adj_first_order'), i=None):
        self._RunControl()
        self['RunControl'].set_adj_first_order(adj, i)
        
    # Copy over the documentation.
    for k in ['it_ad', 'mg_ad', 'adj_first_order']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(RunControl,'set_'+k).__doc__
   # >
    
    # ====================
    # autoInputs and cubes
    # ====================
   # <
   
    # Get intersect flag
    def get_autoInputs(self, j=0):
        self._RunControl()
        return self['RunControl'].get_autoInputs(j)
   
    # Get intersect flag
    def get_cubes(self, j=0):
        self._RunControl()
        return self['RunControl'].get_cubes(j)
    
    # Copy documentation
    for k in ['autoInputs', 'cubes']:
        # Get the documentation for the "get"
        eval('get_'+k).__doc__ = getattr(runControl.RunControl,'get_'+k).__doc__
   
    # Get the nominal mesh radius
    def get_r(self, i=None):
        self._RunControl()
        return self['RunControl'].get_r(i)
        
    # Set the nominal mesh radius
    def set_r(self, r=rc0('r'), i=None):
        self._RunControl()
        self['RunControl'].set_r(r, i)
        
    # Get the number of background mesh divisions.
    def get_nDiv(self, i=None):
        self._RunControl()
        return self['RunControl'].get_nDiv(i)
        
    # Set the number of background mesh divisions.
    def set_nDiv(self, nDiv=rc0('nDiv'), i=None):
        self._RunControl()
        self['RunControl'].set_nDiv(nDiv, i)
    
    # Get the number of refinements
    def get_maxR(self, i=None):
        self._RunControl()
        return self['RunControl'].get_maxR(i)
        
    # Set the number of refinements
    def set_maxR(self, maxR=rc0('maxR'), i=None):
        self._RunControl()
        self['RunControl'].set_maxR(maxR, i)
        
    # Get the 'cubes_a' parameter
    def get_cubes_a(self, i=None):
        self._RunControl()
        return self['RunControl'].get_cubes_a(i)
        
    # Set the 'cubes_a' parameter
    def set_cubes_a(self, cubes_a=rc0('cubes_a'), i=None):
        self._RunControl()
        self['RunControl'].set_cubes_a(cubes_a, i)
        
    # Get the 'cubes_b' parameter
    def get_cubes_b(self, i=None):
        self._RunControl()
        return self['RunControl'].get_cubes_b(i)
        
    # Set the 'cubes_b' parameter
    def set_cubes_b(self, cubes_b=rc0('cubes_b'), i=None):
        self._RunControl()
        self['RunControl'].set_cubes_b(cubes_b, i)
        
    # Get the mesh reordering status
    def get_reorder(self, i=None):
        self._RunControl()
        return self['RunControl'].get_reorder(i)
        
    # Set the mesh reordering status
    def set_reorder(self, reorder=rc0('reorder'), i=None):
        self._RunControl()
        self['RunControl'].set_reorder(reorder, i)
        
    # Get the number of extra refinements around sharp edges
    def get_sf(self, i=None):
        self._RunControl()
        return self['RunControl'].get_sf(i)
        
    # Seth the number of extra refinements around sharp edges
    def set_sf(self, sf=rc0('sf'), i=None):
        self._RunControl()
        self['RunControl'].set_sf(sf, i)
    
    # Get preSpec file
    def get_preSpecCntl(self):
        self._RunControl()
        return self['RunControl'].get_preSpecCntl()
        
    # Set preSpec file
    def set_preSpecCntl(self, fpre=rc0('pre')):
        self._RunControl()
        self['RunControl'].set_preSpecCntl(fpre)
        
    # Copy over the documentation.
    for k in ['r', 'nDiv', 
            'preSpecCntl', 'maxR', 'cubes_a', 'cubes_b', 'reorder', 'sf']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(RunControl,'set_'+k).__doc__
   # >
        
    # ================
    # multigrid levels
    # ================
   # <
    
    # Method to get the number of multigrid levels
    def get_mg(self, i=None):
        """Return the number of multigrid levels
        
        :Call:
            >>> mg = opts.get_mg(i=None) 
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *mg*: :class:`int`
                Maximum of *mg_fc* and *mg_ad*
        :Versions:
            * 2014.08.01 ``@ddalle``: First version
        """
        # Get the two values.
        mg_fc = self.get_mg_fc(i)
        mg_ad = self.get_mg_ad(i)
        # Check for valid settings.
        if mg_fc and mg_ad:
            # Handle lists...
            if type(mg_fc).__name__ == "list":
                mg_fc = mg_fc[0]
            if type(mg_ad).__name__ == "list":
                mg_ad = mg_ad[0]
            # Both are defined, use maximum.
            return max(mg_fc, mg_ad)
        elif mg_fc:
            # Only one valid nonzero setting; use it.
            return mg_fc
        elif mg_ad:
            # Only one valid nonzero setting; use it.
            return mg_ad
        else:
            # Both either invalid or zero.  Return 0.
            return 0
    
    # Method to set both multigrid levels
    def set_mg(self, mg=rc0('mg_fc'), i=None):
        """Set number of multigrid levels for `flowCart` and `adjointCart`
        
        :Call:
            >>> opts.set_mg(mg, i=None) 
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mg*: :class:`int`
                Multigrid levels for both adjointCart and flowCart
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014.08.01 ``@ddalle``: First version
        """
        self.set_mg_fc(mg, i)
        self.set_mg_ad(mg, i)
   # >
        
    # ===================
    # Adaptation settings
    # ===================
   # <
    
    # Get number of adapt cycles
    def get_n_adapt_cycles(self, i=None):
        self._RunControl()
        return self['RunControl'].get_n_adapt_cycles(i)
        
    # Set number of adapt cycles
    def set_n_adapt_cycles(self, nAdapt=rc0('n_adapt_cycles'), i=None):
        self._RunControl()
        self['RunControl'].set_n_adapt_cycles(nAdapt, i)
        
    # Get jumpstart status
    def get_jumpstart(self, i=None):
        self._RunControl()
        return self['RunControl'].get_jumpstart(i)
        
    # Jumpstart status
    def set_jumpstart(self, js=rc0('jumpstart'), i=None):
        self._RunControl()
        self['RunControl'].set_jumpstart(js, i)
    
    # Get error tolerance
    def get_etol(self, i=None):
        self._RunControl()
        return self['RunControl'].get_etol(i)
        
    # Set error tolerance
    def set_etol(self, etol=rc0('etol'), i=None):
        self._RunControl()
        self['RunControl'].set_etol(etol, i)
    
    # Get maximum cell count
    def get_max_nCells(self, i=None):
        self._RunControl()
        return self['RunControl'].get_max_nCells(i)
        
    # Set maximum cell count
    def set_max_nCells(self, etol=rc0('max_nCells'), i=None):
        self._RunControl()
        self['RunControl'].set_max_nCells(etol, i)
    
    # Get flowCart iterations on refined meshes
    def get_ws_it(self, i=None):
        self._RunControl()
        return self['RunControl'].get_ws_it(i)
        
    # Set flowCart iterations on refined meshes
    def set_ws_it(self, ws_it=rc0('ws_it'), i=None):
        self._RunControl()
        self['RunControl'].set_ws_it(ws_it, i)
        
    # Get mesh growth ratio
    def get_mesh_growth(self, i=None):
        self._RunControl()
        return self['RunControl'].get_mesh_growth(i)
        
    # Set mesh growth ratio
    def set_mesh_growth(self, mesh_growth=rc0('mesh_growth'), i=None):
        self._RunControl()
        self['RunControl'].set_mesh_growth(mesh_growth, i)
        
    # Get mesh refinement cycle type
    def get_apc(self, i=None):
        self._RunControl()
        return self['RunControl'].get_apc(i)
        
    # Set mesh refinement cycle type
    def set_apc(self, apc=rc0('apc'), i=None):
        self._RunControl()
        self['RunControl'].set_apc(apc, i)
        
    # Get number of buffer layers
    def get_abuff(self, i=None):
        self._RunControl()
        return self['RunControl'].get_abuff(i)
        
    # Set number of buffer layers
    def set_abuff(self, buf=rc0('buf'), i=None):
        self._RunControl()
        self['RunControl'].set_abuff(abuff, i)
    
    # Get number of additional adaptations on final error map
    def get_final_mesh_xref(self, i=None):
        self._RunControl()
        return self['RunControl'].get_final_mesh_xref(i)
    
    # Set number of additional adaptations on final error map
    def set_final_mesh_xref(self, xref=rc0('final_mesh_xref'), i=None):
        self._RunControl()
        self['RunControl'].set_final_mesh_xref(xref, i)
        
    # Copy over the documentation.
    for k in ['n_adapt_cycles', 'jumpstart', 'etol', 'max_nCells', 'ws_it',
            'mesh_growth', 'apc', 'abuff', 'final_mesh_xref']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(RunControl,'set_'+k).__doc__
   # >
   
        
    # =================
    # Output functional
    # =================
   # <
    
    # Get the optForces
    def get_optForces(self):
        self._Functional()
        return self['Functional'].get_optForces()
    get_optForces.__doc__ = Functional.get_optForces.__doc__
    
    # Get the optSensors
    def get_optSensors(self):
        self._Functional()
        return self['Functional'].get_optSensors()
    get_optSensors.__doc__ = Functional.get_optSensors.__doc__
    
    # Get the optMoments
    def get_optMoments(self):
        self._Functional()
        return self['Functional'].get_optMoments()
    get_optMoments.__doc__ = Functional.get_optMoments.__doc__
   # >
    
    # ========================
    # mesh creation parameters
    # ========================
   # <
    
    # Get BBoxes
    def get_BBox(self):
        self._Mesh()
        return self['Mesh'].get_BBox()
        
    # Set BBoxes
    def set_BBox(self, BBox=rc0('BBox')):
        self._Mesh()
        self['Mesh'].set_BBox(BBox)
    
    # Get XLevs
    def get_XLev(self):
        self._Mesh()
        return self['Mesh'].get_XLev()
        
    # Set XLevs
    def set_XLev(self, XLev=rc0('XLev')):
        self._Mesh()
        self['Mesh'].set_XLev(XLev)
    
    # Get mesh2d status
    def get_mesh2d(self, i=None):
        self._Mesh()
        return self['Mesh'].get_mesh2d(i)
        
    # Set error tolerance
    def set_mesh2d(self, mesh2d=rc0('mesh2d'), i=None):
        self._Mesh()
        self['Mesh'].set_mesh2d(mesh2d, i)
        
    # ``cubes`` input file
    def get_inputC3d(self):
        self._Mesh()
        return self['Mesh'].get_inputC3d()
    
    # ``cubes`` input file
    def set_inputC3d(self, fc3d=rc0('inputC3d')):
        self._Mesh()
        self['Mesh'].set_inputC3d(fc3d)
        
    # Copy over the documentation.
    for k in ['BBox', 'XLev', 'mesh2d', 'inputC3d']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Mesh,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(Mesh,'set_'+k).__doc__
   # >
   
        
    # ============
    # PBS settings
    # ============
   # <
   # >
    
    
    # =================
    # Folder management
    # =================
   # <
        
    # Get the number of check point files to keep around
    def get_nCheckPoint(self, i=None):
        self._RunControl()
        return self['RunControl'].get_nCheckPoint(i)
        
    # Set the number of check point files to keep around
    def set_nCheckPoint(self, nchk=rc0('nCheckPoint'), i=None):
        self._RunControl()
        self['RunControl'].set_nCheckPoint(nchk, i)
        
    # Get the archive status for adaptation folders
    def get_TarAdapt(self):
        self._RunControl()
        return self['RunControl'].get_TarAdapt()
        
    # Get the archive status for adaptation folders
    def set_TarAdapt(self, fmt=rc0('TarAdapt')):
        self._RunControl()
        self['RunControl'].set_TarAdapt(fmt)
        
    # Get the archive format for visualization files
    def get_TarViz(self):
        self._RunControl()
        return self['RunControl'].get_TarViz()
        
    # Set the archive format for visualization files
    def set_TarViz(self, fmt=rc0('TarViz')):
        self._RunControl()
        self['RunControl'].set_TarViz(fmt)
        
    # Copy over the documentation.
    for k in ['nCheckPoint', 'TarViz', 'TarAdapt']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(RunControl,'set_'+k).__doc__
   # >
   
    
    # =============
    # Configuration
    # =============
   # <
        
    # Get list of components to request forces for
    def get_ClicForces(self, i=None):
        self._Config()
        return self['Config'].get_ClicForces(i)
        
    # Set list of components to request forces for
    def set_ClicForces(self, comp="entire", i=None):
        self._Config()
        self['Config'].set_ClicForces(comp, i)
        
    # Add a component to get force history of
    def add_ClicForce(self, comp="entire"):
        self._Config()
        self['Config'].add_ClicForce(comp)
        
    # Get list of cut planes
    def get_Xslices(self, i=None):
        self._Config()
        return self['Config'].get_Xslices(i)
        
    # Set list of cut planes
    def set_Xslices(self, x, i=None):
        self._Config()
        self['Config'].set_Xslices(x, i)
        
    # Add a cut plane
    def add_Xslice(self, x):
        self._Config()
        self['Config'].add_Xslice(x)
        
    # Get list of cut planes
    def get_Yslices(self, i=None):
        self._Config()
        return self['Config'].get_Yslices(i)
        
    # Set list of cut planes
    def set_Yslices(self, y, i=None):
        self._Config()
        self['Config'].set_Yslices(y, i)
        
    # Add a cut plane
    def add_Yslice(self, y):
        self._Config()
        self['Config'].add_Yslice(y)
        
    # Get list of cut planes
    def get_Zslices(self, i=None):
        self._Config()
        return self['Config'].get_Zslices(i)
        
    # Set list of cut planes
    def set_Zslices(self, z, i=None):
        self._Config()
        self['Config'].set_Zslices(z, i)
        
    # Add a cut plane
    def add_Zslice(self, z):
        self._Config()
        self['Config'].add_Zslice(z)
    
    # Get list of sensors
    def get_LineSensors(self, name=None):
        self._Config()
        return self['Config'].get_LineSensors(name)
        
    # Set list of sensors
    def set_LineSensors(self, LS={}, name=None, X=[]):
        self._Config()
        self['Config'].set_LineSensors(LS=LS, name=name, X=X)
    
    # Set line sensors
    def add_LineSensor(self, name, X):
        self._Config()
        self['Config'].add_LineSensor(name, X)
    
    # Get list of sensors
    def get_PointSensors(self, name=None):
        self._Config()
        return self['Config'].get_PointSensors(name)
        
    # Set list of sensors
    def set_PointSensors(self, PS={}, name=None, X=[]):
        self._Config()
        self['Config'].set_PointSensors(PS=PS, name=name, X=X)
    
    # Set line sensors
    def add_PointSensor(self, name, X):
        self._Config()
        self['Config'].add_PointSensor(name, X)
        
    # Copy over the documentation.
    for k in ['ClicForce', 'Xslice', 'Yslice', 'Zslice',
            'PointSensor', 'LineSensor']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k+'s').__doc__ = getattr(Config,'get_'+k+'s').__doc__
        eval('set_'+k+'s').__doc__ = getattr(Config,'set_'+k+'s').__doc__
        eval('add_'+k).__doc__ = getattr(Config,'add_'+k).__doc__
   # >
   
    
    # ========
    # Plotting
    # ========
   # <
   # >
   
    
    # =========
    # Data book
    # =========
   # <
        
    # Get Mach number option
    def get_ComponentMach(self, comp):
        self._DataBook()
        return self['DataBook'].get_ComponentMach(comp)
        
    # Get Gamma option
    def get_ComponentGamma(self, comp):
        self._DataBook()
        return self['DataBook'].get_ComponentGamma(comp)
        
    # Get Reynolds number option
    def get_ComponentReynoldsNumber(self, comp):
        self._DataBook()
        return self['DataBook'].get_ComponentReynoldsNumber(comp)
    
    # Copy over the documentation.
    for k in [
        'ComponentGamma', 'ComponentMach', 'ComponentReynoldsNumber'
    ]:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(DataBook,'get_'+k).__doc__
   # >
   
    
    # =======
    # Reports
    # =======
   # <
    
    # Copy over the documentation
    for k in []:
        # Get the documentation from the submodule
        eval('get_'+k).__doc__ = getattr(Report,'get_'+k).__doc__
   # >
    
# class Options



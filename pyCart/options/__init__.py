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

# Import options-specific utilities
from util import *

# Import modules for controlling specific parts of Cart3D
from .flowCart    import flowCart
from .adjointCart import adjointCart
from .Adaptation  import Adaptation

    

# Class definition
class Options(odict):
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
            # Read the input file.
            lines = open(fname).readlines()
            # Strip comments and join list into a single string.
            lines = stripComments(lines, '#')
            lines = stripComments(lines, '//')
            # Get the equivalent dictionary.
            d = json.loads(lines)
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
        self._flowCart()
        self._adjointCart()
        self._Adaptation()
        
    
    
    # Initialization and confirmation for flowCart options
    def _flowCart(self):
        """Initialize `flowCart` options if necessary"""
        # Check for missing entirely.
        if 'flowCart' not in self:
            # Empty/default
            self['flowCart'] = flowCart()
        elif type(self['flowCart']).__name__ == 'dict':
            # Convert to special class.
            self['flowCart'] = flowCart(**self['flowCart'])
            
    # Initialization and confirmation for adjointCart options
    def _adjointCart(self):
        """Initialize `adjointCart` options if necessary"""
        # Check for missing entirely.
        if 'adjointCart' not in self:
            # Empty/default
            self['adjointCart'] = adjointCart()
        elif type(self['adjointCart']).__name__ == 'dict':
            # Convert to special class.
            self['adjointCart'] = adjointCart(**self['adjointCart'])
    
    # Initialization and confirmation for Adaptation options
    def _Adaptation(self):
        """Initialize adaptation options if necessary"""
        # Check status
        if 'Adaptation' not in self:
            # Missing entirely
            self['Adaptation'] = Adaptation()
        elif type(self['Adaptation']).__name__ == 'dict':
            # Convert to special class.
            self['Adaptation'] = Adaptation(**self['Adaptation'])
    
            
    # Method to get the input file
    
    
    
    # ===================
    # flowCart parameters
    # ===================
    
    # Number of iterations
    def get_it_fc(self, i=None):
        self._flowCart()
        return self['flowCart'].get_it_fc(i)
        
    # Set flowCart iteration count
    def set_it_fc(self, it_fc=rc0('it_fc'), i=None):
        self._flowCart()
        self['flowCart'].set_it_fc(it_fc, i)
        
    # Get flowCart iteration count
    def get_mg_fc(self, i=None):
        self._flowCart()
        return self['flowCart'].get_mg_fc(i)
        
    # Set flowCart iteration count
    def set_mg_fc(self, mg_fc=rc0('mg_fc'), i=None):
        self._flowCart()
        self['flowCart'].set_mg_fc(mg_fc, i)
        
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
        return self.get_cflmin(i)
    
    # Set the minimum CFL number
    def set_cflmin(self, cflmin=rc0('cflmin'), i=None):
        self._flowCart()
        self['flowCart'].set_cflmin(cflmin, i)
        
    # Get the limiter
    def get_limiter(self, i=None):
        self._flowCart()
        return self['flowCart'].get_limiter(i)
    
    # Set the limiter
    def set_limiter(self, limiter=rc0('limiter'), i=None):
        self._flowCart()
        self['flowCart'].set_limiter(limiter, i)
        
    # Get the y_is_spanwise status
    def get_y_is_spanwise(i=None):
        self._flowCart()
        return self['flowCart'].get_y_is_spanwise(i)
        
    # Set the y_is_spanwise status
    def set_y_is_spanwise(y_is_spanwise=rc0('y_is_spanwise'), i=None):
        self._flowCart()
        self['flowCart'].set_y_is_spanwise(y_is_spanwise, i)
        
    # Get the binary I/O status
    def get_binaryIO(i=None):
        self._flowCart()
        return self['flowCart'].get_binaryIO(i)
        
    # Set the binary I/O status
    def set_binaryIO(binaryIO=rc0('binaryIO'), i=None):
        self._flowCart()
        self['flowCart'].set_binaryIO(binaryIO, i)
        
    # Get the number of threads for flowCart
    def get_OMP_NUM_THREADS(i=None):
        self._flowCart()
        return self['flowCart'].get_OMP_NUM_THREADS(i)
        
    # Set the number of threads for flowCart
    def set_OMP_NUM_THREADS(nThreads=rc0('OMP_NUM_THREADS'), i=None):
        self._flowCart()
        self['flowCart'].set_OMP_NUM_THREADS(nThreads, i)
        
    # Copy over the documentation.
    for k in ['it_fc', 'mg_fc', 'cfl', 'cflmin', 'limiter',
            'y_is_spanwise', 'binaryIO', 'OMP_NUM_THREADS']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(flowCart,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(flowCart,'set_'+k).__doc__
        
    # Aliases for the above
    get_nThreads = get_OMP_NUM_THREADS
    set_nThreads = set_OMP_NUM_THREADS
    
    
    # ====================
    # adjointCart settings
    # ====================
    
    # Number of iterations
    def get_it_ad(self, i=None):
        self._adjointCart()
        return self['adjointCart'].get_it_ad(i)
        
    # Set flowCart iteration count
    def set_it_ad(self, it_ad=rc0('it_ad'), i=None):
        self._adjointCart()
        self['adjointCart'].set_it_ad(it_ad, i)
    
    # Get flowCart iteration count
    def get_mg_ad(self, i=None):
        self._adjointCart()
        return self['adjointCart'].get_mg_ad(i)
        
    # Set flowCart iteration count
    def set_mg_ad(self, mg_ad=rc0('mg_ad'), i=None):
        self._adjointCart()
        self['adjointCart'].set_mg_ad(mg_ad, i)
        
    # Copy over the documentation.
    for k in ['it_ad', 'mg_ad']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(adjointCart,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(adjointCart,'set_'+k).__doc__
    
    # ================
    # multigrid levels
    # ================
        
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
        mg_fc = self.get_mg_fc(mg, i)
        mg_ad = self.get_mg_ad(mg_fc, i)
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
        
        
    # ===================
    # Adaptation settings
    # ===================
    
    # Get number of adapt cycles
    def get_n_adapt_cycles(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_n_adapt_cycles(i)
        
    # Set number of adapt cycles
    def set_n_adapt_cycles(self, nAdapt=rc0('n_adapt_cycles'), i=None):
        self._Adaptation()
        self['Adaptation'].set_n_adapt_cycles(nAdapt, i)
    
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
        
    # Copy over the documentation.
    for k in ['n_adapt_cycles', 'etol', 'max_nCells', 'ws_it',
            'mesh_growth', 'apc']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Adaptation,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(Adaptation,'set_'+k).__doc__
            
    
    # ========================
    # mesh creation parameters
    # ========================
    
    # Tri file
    def get_TriFile(self):
        """Get the input :file:`*.tri` file(s)"""
        # Get the mesh settings safely.
        opts_mesh = self.get('Mesh',{})
        # Get the .tri file safely
        TriFile = opts_mesh.get('TriFile', rc['TriFile'])
        return TriFile
    def set_TriFile(self, TriFile=rc['TriFile']):
        """Set the input :file:`*.tri` file(s)"""
        # Ensure the  'Mesh' key exists.
        self.setdefault('Mesh', {})
        # Apply the setting
        self['Mesh']['TriFile'] = TriFile
    
    # Mesh radius
    def get_r(self, r=8):
        """Get the value for the `autoInputs` mesh radius"""
        # Get to the autoInputs section safely.
        opts_ai = self.get('Mesh', {}).get('autoInputs', {})
        # Get the mesh radius value
        return opts_ai.get('r', r)
    def set_r(self, r=8):
        """Set the value for `autoInputs` mesh radius"""
        # Make sure the 'Mesh' and 'autoInputs' keys exist.
        self.setdefault('Mesh', {})
        self['Mesh'].setdefault('autoInptus', {})
        # Apply the setting.
        self['Mesh']['autoInputs'] = r
        
    # Refinement levels
    def get_maxR(self, maxR=10):
        """Get the value for `cubes` maximum refinement levels"""
        # Get to the `cubes` section safely.
        opts_cubes = self.get('Mesh',{}).get('cubes',{})
        # Get the refinement count.
        return opts_cubes.get('maxR')
    def set_maxR(self, maxR=10):
        """Set the value for `cubes` maximum refinement levels"""
        # Make sure 'Mesh' and 'cubes' keys exist.
        self.setdefault('Mesh', {})
        self['Mesh'].setdefault('cubes', {})
        # Apply the setting.
        self['Mesh']['cubes']['maxR'] = maxR
        
        
    

    



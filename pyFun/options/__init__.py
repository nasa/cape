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
from .util import *

# Import template module
import cape.options

# Import modules for controlling specific parts of Cart3D
from .pbs         import PBS
from .DataBook    import DataBook
from .Report      import Report
from .runControl  import RunControl
from .fun3dnml    import Fun3DNml
from .Mesh        import Mesh
from .Config      import Config
from .Functional  import Functional

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
            # Get the equivalent dictionary.
            d = loadJSONFile(fname)
            # Loop through the keys.
            for k in d:
                kw[k] = d[k]
        # Read the defaults.
        defs = getPyFunDefaults()
        
        # Apply the defaults.
        kw = applyDefaults(kw, defs)
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._PBS()
        self._DataBook()
        self._Report()
        self._RunControl()
        self._Mesh()
        self._Config()
        self._Functional()
        # Pre/post-processings PBS settings
        self._BatchPBS()
        self._PostPBS()
        # Add extra folders to path.
        self.AddPythonPath()
    
   # ============
   # Initializers
   # ============
   # <
    
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
            
    # Initialization method for overall run control
    def _RunControl(self):
        """Initialize report options if necessary"""
        # Check status.
        if 'RunControl' not in self:
            # Missing entirely.
            self['RunControl'] = Report()
        elif type(self['RunControl']).__name__ == 'dict':
            # Convert to special class
            self['RunControl'] = RunControl(**self['RunControl'])
    
    # Initialization method for namelist variables
    def _Fun3D(self):
        """Initialize namelist options"""
        # Check status.
        if 'Fun3D' not in self:
            # Missing entirely.
            self['Fun3D'] = Fun3DNml()
        elif type(self['Fun3D']).__name__ == 'dict':
            # Convert to special class
            self['Fun3D'] = Fun3DNml(**self['Fun3D'])
            
    # Initialization for dual namelist variables
    def _DualFun3D(self):
        """Initialize ``dual`` namelist options"""
        # Check status
        if 'DualFun3D' not in self:
            # Missing entirely
            self['DualFun3D'] = Fun3DNml()
        elif type(self['DualFun3D']).__name__ == 'dict':
            # Convert to special class
            self['DualFun3D'] = Fun3DNml(**self['DualFun3D'])
    
    # Initialization method for mesh settings
    def _Mesh(self):
        """Initialize mesh options"""
        # Check status.
        if 'Mesh' not in self:
            # Missing entirely.
            self['Mesh'] = Mesh()
        elif type(self['Mesh']).__name__ == 'dict':
            # Convert to special class
            self['Mesh'] = Mesh(**self['Mesh'])
            
    # Initialization method for adaptive function definition
    def _Functional(self):
        """Initialize report options if necessary"""
        # Check status.
        if 'Functional' not in self:
            # Missing entirely.
            self['Functional'] = Functional()
        elif type(self['Functional']).__name__ == 'dict':
            # Convert to special class
            self['Functional'] = Functional(**self['Functional'])
    
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
   # >
    
   # ==============
   # Global Options
   # ==============
   # <
    
    # Method to get the namelist template
    def get_Namelist(self, j=None):
        """Return the name of the master :file:`fun3d.nml` file
        
        :Call:
            >>> fname = opts.get_Namelist(j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *j*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *fname*: :class:`str`
                Name of FUN3D namelist template file
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        return self.get_key('Namelist', j)
        
    # Method to set the namelist template
    def set_Namelist(self, fname):
        """Set the name of the master :file:`fun3d.nml` file
        
        :Call:
            >>> opts.set_Namelist(fname)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of FUN3D namelist template file
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        self['Namelist'] = fname
    
    # Get template file
    def get_RubberDataFile(self, j=None):
        """Get the ``rubber.data`` file name
        
        :Call:
            >>> fname = opts.get_RubberFile(j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *fname*: :class:`str`
                Name of file template
        :Versions:
            * 2016-04-27 ``@ddalle``: First version
        """
        return self.get_key('RubberData', j)
    
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
    
   # ===================
   # Overall run control
   # ===================
   # <
    
    # Keep restart files
    def get_KeepRestarts(self, i=None):
        self._RunControl()
        return self['RunControl'].get_KeepRestarts(i)
        
    # Keep restart files
    def set_KeepRestarts(self, qr=rc0("KeepRestarts"), i=None):
        self._RunControl()
        self['RunControl'].set_KeepRestarts(qr, i)
        
    # Adaptation setting
    def get_Adaptive(self, i=None):
        self._RunControl()
        return self['RunControl'].get_Adaptive(i)
    
    # Adaptation setting
    def set_Adaptive(self, ac=rc0('Adaptive'), i=None):
        self._RunControl()            
        self['RunControl'].set_Adaptive(ac, i)
        
    # Dual setting
    def get_Dual(self, i=None):
        self._RunControl()
        return self['RunControl'].get_Dual(i)
        
    # Dual setting
    def set_Dual(self, d=rc0('Dual'), i=None):
        self._Dual()
        self['RunControl'].set_dual(d, i)
        
    # Adaptive phase setting
    def get_AdaptPhase(self, i=None):
        self._RunControl()
        return self['RunControl'].get_AdaptPhase(i)
        
    # Adaptive phase setting
    def set_AdaptPhase(self, qa=rc0('AdaptPhase'), i=None):
        self._RunControl()
        self['RunControl'].set_AdaptPhase(qa, i)
        
    # Dual phase setting
    def get_DualPhase(self, i=None):
        self._RunControl()
        return self['RunControl'].get_DualPhase(i)
        
    # Adaptive phase setting
    def set_DualPhase(self, qd=rc0('DualPhase'), i=None):
        self._RunControl()
        self['RunControl'].set_DualPhase(qd, i)
        
    # Iterations for adjoint
    def get_nIterAdjoint(self, j=None):
        self._RunControl()
        return self['RunControl'].get_nIterAdjoint(j)
        
    # Iterations for adjoint
    def set_nIterAdjoint(self, n=rc0('nIterAdjoint'), j=None):
        self._RunControl()
        self['RunControl'].set_nIterAdjoint(n, j)
        
    # Copy documentation     
    for k in ['Adaptive', 'Dual', 'AdaptPhase', 'DualPhase', 'nIterAdjoint']:
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(RunControl,'set_'+k).__doc__
   
    # Adaptation number
    def get_AdaptationNumber(self, i):
        self._RunControl()
        return self['RunControl'].get_AdaptationNumber(i)
        
    # Copy documentation
    for k in ['AdaptationNumber']:
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
   
   # >
   
   # =================
   # Namelist settings
   # =================
   # <
    
    # Project settings
    def get_project(self, i=None):
        self._Fun3D()
        return self['Fun3D'].get_project(i)
        
    # Grid settings
    def get_raw_grid(self, i=None):
        self._Fun3D()
        return self['Fun3D'].get_raw_grid(i)
        
    # Project rootname
    def get_project_rootname(self, i=None):
        self._Fun3D()
        return self['Fun3D'].get_project_rootname(i)
        
    # Grid format
    def get_grid_format(self, i=None):
        self._Fun3D()
        return self['Fun3D'].get_grid_format(i)
        
    # Generic value
    def get_namelist_var(self, sec, key, i=None):
        self._Fun3D()
        return self['Fun3D'].get_namelist_var(sec, key, i)
        
    # Copy documentation
    for k in ['project', 'project_rootname', 'raw_grid', 'grid_format',
    'namelist_var']:
        eval('get_'+k).__doc__ = getattr(Fun3DNml,'get_'+k).__doc__
        
    # Set generic value
    def set_namelist_var(self, sec, key, val, i=None):
        self._Fun3D()
        return self['Fun3D'].set_namelist_var(sec, key, val, i)
        
    # Copy documentation
    for k in ['namelist_var']:
        eval('set_'+k).__doc__ = getattr(Fun3DNml,'set_'+k).__doc__
        
    
    # Downselect
    def select_namelist(self, i=None):
        self._Fun3D()
        return self['Fun3D'].select_namelist(i)
    select_namelist.__doc__ = Fun3DNml.select_namelist.__doc__
    # Downselect dual namelist
    def select_dual_namelist(self, i=None):
        self._DualFun3D()
        return self['DualFun3D'].select_namelist(i)
    select_dual_namelist.__doc__ = Fun3DNml.select_namelist.__doc__
    
    # Get value from dual namelist
    def get_dual_namelist_var(self, sec, key, i=None):
        """Get namelist variable from ``"DualFun3D"`` section
        
        :Call:
            >>> v = opts.get_dual_namelist_var(sec, key, i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *sec*: :class:`str`
                Name of namelist section
            *key*: :class:`str`
                Name of variable in namelist section
            *i*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *v*: :class:`any`
                Value of variable for phase *i*
        :Versions:
            * 2016-05-02 ``@ddalle``: First version
        """
        # Initialize sections as appropriate
        self._DualFun3D()
        # Options for dual namelist
        dopts = self['DualFun3D']
        # Get the value from the dual section if appropriate
        if sec not in dopts or key not in dopts[sec]:
            # Fall back to the main value
            return self.get_namelist_var(sec, key, i)
        else:
            # Return the main value
            return dopts.get_namelist_var(sec, key, i)
   # >
   
    
   # =============
   # Mesh settings
   # =============
   # <
        
    # BCfile
    def get_MapBCFile(self, i=None):
        self._Mesh()
        return self['Mesh'].get_MapBCFile(i)
        
    # Freeze file
    def get_FreezeFile(self, i=None):
        self._Mesh()
        return self['Mesh'].get_FreezeFile(i)
        
    # faux_input file
    def get_FauxFile(self, i=None):
        self._Mesh()
        return self['Mesh'].get_FauxFile(i)
        
    # Freeze component list
    def get_FreezeComponents(self):
        self._Mesh()
        return self['Mesh'].get_FreezeComponents()
        
    # Faux geometry
    def get_Faux(self, comp=None):
        self._Mesh()
        return self['Mesh'].get_Faux(comp)
        
    # Copy documentation
    for k in ['MapBCFile', 'FreezeFile', 'FauxFile',
        'FreezeComponents', 'Faux'
    ]:
        eval('get_'+k).__doc__ = getattr(Mesh,'get_'+k).__doc__
   # >
    
    
   # =============
   # Configuration
   # =============
   # <
    
    # Get components
    def get_ConfigInput(self, comp):
        self._Config()
        return self['Config'].get_ConfigInput(comp)
        
    # Set components
    def set_ConfigInput(self, comp, inp):
        self._Config()
        self['Config'].set_ConfigInput(comp, inp)
        
    # Boundary point groups
    def get_BoundaryPointGroups(self):
        self._Config()
        return self['Config'].get_BoundaryPointGroups()
        
    # Boundary point groups
    def set_BoundaryPointGroups(self, BP):
        self._Config()
        self['Config'].set_BoundaryPointGroups(BP)
        
    # Boundary points
    def get_BoundaryPoints(self, name=None):
        self._Config()
        return self['Config'].get_BoundaryPoints(name)
        
    # Boundary points
    def set_BoundaryPoints(self, PS, name=None):
        self._Config()
        self['Config'].set_BoundaryPoints(PS, name=name)
        
    # Copy over the documentation.
    for k in [
            'ConfigInput',
            'BoundaryPointGroups', 'BoundaryPoints'
    ]:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Config,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(Config,'set_'+k).__doc__
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
        
        
    ## Copy over the documentation.
    #for k in []:
    #    # Get the documentation for the "get" and "set" functions
    #    eval('get_'+k).__doc__ = getattr(Management,'get_'+k).__doc__
    #    eval('set_'+k).__doc__ = getattr(Management,'set_'+k).__doc__
   # >
   
    
   # =============
   # Configuration
   # =============
   # <
        
    
        
    ## Copy over the documentation.
    #for k in ['ClicForce', 'Xslice', 'Yslice', 'Zslice',
    #        'PointSensor', 'LineSensor']:
    #    # Get the documentation for the "get" and "set" functions
    #    eval('get_'+k+'s').__doc__ = getattr(Config,'get_'+k+'s').__doc__
    #    eval('set_'+k+'s').__doc__ = getattr(Config,'set_'+k+'s').__doc__
    #    eval('add_'+k).__doc__ = getattr(Config,'add_'+k).__doc__
   # >
   
    
   # =========
   # Functions
   # =========
   # <
    
    # Function to get adaptive coefficients
    def get_AdaptCoeffs(self):
        self._Functional()
        return self['Functional'].get_AdaptCoeffs()
        
    # Function to get objective functions
    def get_AdaptFuncs(self):
        self._Functional()
        return self['Functional'].get_AdaptFuncs()
        
    # Function to get objective functions
    def get_OptFuncs(self):
        self._Functional()
        return self['Functional'].get_OptFuncs()
    
    # Function to get constraint functions
    def get_ConstraintFuncs(self):
        self._Functional()
        return self['Functional'].get_ConstraintFuncs()
        
    # Get coefficients for a function
    def get_FuncCoeffs(self, fn):
        self._Functional()
        return self['Functional'].get_FuncCoeffs(fn)
        
    # Get function type
    def get_FuncType(self, fn):
        self._Functional()
        return self['Functional'].get_FuncType(fn)
    
    # Function to get functional coeff type
    def get_FuncCoeffType(self, coeff):
        self._Functional()
        return self['Functional'].get_FuncCoeffType(coeff)
    
    # Weight
    def get_FuncCoeffWeight(self, coeff):
        self._Functional()
        return self['Functional'].get_FuncCoeffWeight(coeff)
    
    # Target value
    def get_FuncCoeffTarget(self, coeff):
        self._Functional()
        return self['Functional'].get_FuncCoeffTarget(coeff)
    
    # Exponent
    def get_FuncCoeffPower(self, coeff):
        self._Functional()
        return self['Functional'].get_FuncCoeffPower(coeff)
    
    # Component
    def get_FuncCoeffCompID(self, coeff):
        self._Functional()
        return self['Functional'].get_FuncCoeffCompID(coeff)
        
    # Copy documentation
    for k in ['AdaptCoeffs', 'AdaptFuncs', 'OptFuncs', 'ConstraintFuncs',
        'FuncCoeffs', 'FuncType',
        'FuncCoeffWeight', 'FuncCoeffTarget', 'FuncCoeffPower',
        'FuncCoeffCompID'
    ]:
        eval('get_'+k).__doc__ = getattr(Functional,'get_'+k).__doc__
        
   # >
    
   # =========
   # Data book
   # =========
   # <
    
    # Copy over the documentation.
    for k in []:
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



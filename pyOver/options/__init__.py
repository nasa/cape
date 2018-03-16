"""
OVERFLOW and pyOver settings module: :mod:`pyOver.options`
==========================================================

This module provides tools to read, access, modify, and write settings for
:mod:`pyOver`.  The class is based off of the built-int :class:`dict` class.

In addition, this module controls default values of each pyOver
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    *. Values directly specified in the input file, :file:`pyOver.json`
    
    *. Values specified in the default control file,
       :file:`$PYOVER/settings/pyOver.default.json`
    
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
from .overnml     import OverNml
from .gridSystem  import GridSystemNml
from .Mesh        import Mesh
from .Config      import Config

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
        * 2014-07-28 ``@ddalle``: First version
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
        defs = getPyOverDefaults()
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
        self._Overflow()
        self._Grids()
        self._Mesh()
        self._Config()
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
        """Initialize run control options if necessary"""
        # Check status.
        if 'RunControl' not in self:
            # Missing entirely.
            self['RunControl'] = RunControl()
        elif type(self['RunControl']).__name__ == 'dict':
            # Convert to special class
            self['RunControl'] = RunControl(**self['RunControl'])
            
    # Initialization method for namelist interface
    def _Overflow(self):
        """Initialize OVERFLOW namelist options if necessary"""
        # Check status.
        if 'Overflow' not in self:
            # Missing entirely.
            self['Overflow'] = OverNml()
        elif type(self['Overflow']).__name__ == 'dict':
            # Convert to special class
            self['Overflow'] = OverNml(**self['Overflow'])
            
    # Initialization method for namelist interface
    def _Grids(self):
        """Initialize OVERFLOW grid system namelist options if necessary"""
        # Check status.
        if 'Grids' not in self:
            # Missing entirely.
            self['Grids'] = GridSystemNml()
        elif type(self['Grids']).__name__ == 'dict':
            # Convert to special class
            self['Grids'] = GridSystemNml(**self['Grids'])
    
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
    
    # Get the namelist file name
    def get_OverNamelist(self, j=None):
        """Return the name of the master :file:`over.namelist` file
        
        :Call:
            >>> fname = opts.get_OverNamelist()
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
        :Outputs:
            *fname*: :class:`str`
                Name of OVERFLOW namelist template file
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.get_key('OverNamelist', j)
        
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
    # Get project root name
    def get_Prefix(self, j=None):
        self._RunControl()
        return self['RunControl'].get_Prefix(j)
        
    # Get OVERFLOW aux inputs
    def get_overrun_aux(self, j=None):
        self._RunControl()
        return self['RunControl'].get_overrun_aux(j)
        
    # Get OVERFLOW function name
    def get_overrun_cmd(self, j=None):
        self._RunControl()
        return self['RunControl'].get_overrun_cmd(j)

    # Dictionary of additional commands
    def get_overrun_kw(self, j=None):
        self._RunControl()
        return self['RunControl'].get_overrun_kw(j)

    # Number of OpenMP threads
    def get_overrun_nthreads(self, j=None):
        self._RunControl()
        return self['RunControl'].get_overrun_nthreads(j)
        
    # Get OVERFLOW arguments
    def get_overrun_args(self, j=None):
        self._RunControl()
        return self['RunControl'].get_overrun_args(j)
        
    # Copy documentation
    for k in ['Prefix',
        'overrun_aux', 'overrun_cmd', 'overrun_kw',
        'overrun_nthreads', 'overrun_args']:
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
   # >
   
    # =================
    # Namelist settings
    # =================
   # <
    
    # GLOBAL section
    def get_GLOBAL(self, i=None):
        self._Overflow()
        return self['Overflow'].get_GLOBAL(i)
    
    # FLOINP section
    def get_FLOINP(self, i=None):
        self._Overflow()
        return self['Overflow'].get_FLOINP(i)
        
    # Generic value
    def get_namelist_var(self, sec, key, i=None):
        self._Overflow()
        return self['Overflow'].get_namelist_var(sec, key, i)
        
    # Copy documentation
    for k in ['GLOBAL', 'FLOINP', 'namelist_var']:
        eval('get_'+k).__doc__ = getattr(OverNml,'get_'+k).__doc__
        
    # Downselect
    def select_namelist(self, i=None):
        self._Overflow()
        return self['Overflow'].select_namelist(i)
    select_namelist.__doc__ = OverNml.select_namelist.__doc__
   # >
   
    
    # ==================== 
    # Grid system settings
    # ====================
   #<
    
    # Grid system dictionary
    def get_GridByName(self, grdnam, i=None):
        self._Grids()
        return self['Grids'].get_GridByName(grdnam, i)
    
    # Get default grid options
    def get_ALL(self, i=None):
        self._Grids()
        return self['Grids'].get_ALL(i)
        
    # Get grid value (raw)
    def get_grid_var(self, grdnam, sec, key, i=None):
        self._Grids()
        return self['Grids'].get_grid_var(grdnam, sec, key, i=None)
    
    # Get grid value with default
    def get_GridKey(self, grdnam, sec, key, i=None):
        self._Grids()
        return self['Grids'].get_GridKey(grdnam, sec, key, i=None)
        
    # Copy documentation
    for k in ['ALL', 'GridByName', 'grid_var', 'GridKey']:
        eval('get_'+k).__doc__ = getattr(GridSystemNml,'get_'+k).__doc__
    
   #>
   
    # =============
    # Mesh settings
    # =============
   # <
    
    # File names
    def get_MeshFiles(self, config=None):
        self._Mesh()
        return self['Mesh'].get_MeshFiles(config)
        
    # Files to copy
    def get_MeshCopyFiles(self, config=None):
        self._Mesh()
        return self['Mesh'].get_MeshCopyFiles(config)
        
    # Files to link
    def get_MeshLinkFiles(self, config=None):
        self._Mesh()
        return self['Mesh'].get_MeshLinkFiles(config)
        
    # Configuration directory
    def get_ConfigDir(self, config=None):
        self._Mesh()
        return self['Mesh'].get_ConfigDir(config)
        
    # Mesh type
    def get_MeshType(self, config=None):
        self._Mesh()
        return self['Mesh'].get_MeshType(config)
        
    # Copy documentation
    for k in ['MeshFiles', 'MeshCopyFiles', 'MeshLinkFiles',
    'ConfigDir', 'MeshType']:
        eval('get_'+k).__doc__ = getattr(Mesh,'get_'+k).__doc__
   # >
    
    
    # =============
    # Configuration
    # =============
   #<
        
    # Copy over the documentation.
    for k in []:
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
    # Data book
    # =========
   # <
   
    # Input file for ``mixsur``
    def get_DataBook_mixsur(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_mixsur(comp)
   
    # Input file for ``usurp``
    def get_DataBook_usurp(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_usurp(comp)
        
    # Input file for ``splitmq``
    def get_DataBook_splitmq(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_splitmq(comp)
        
    # MIXSUR output file folder
    def get_DataBook_fomo(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_fomo(comp)
        
    # Input *q* file
    def get_DataBook_QIn(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_QIn(comp)
        
    # Output *q* file
    def get_DataBook_QOut(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_QOut(comp)
        
    # Surface *q* file
    def get_DataBook_QSurf(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_QSurf(comp)
        
    # Input *x* file
    def get_DataBook_XIn(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_XIn(comp)
    
    # Output *x* file
    def get_DataBook_XOut(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_XOut(comp)
    
    # Output *x* file
    def get_DataBook_XSurf(self, comp):
        self._DataBook()
        return self["DataBook"].get_DataBook_XSurf(comp)
    
    # Copy over the documentation.
    for k in [
        "DataBook_mixsur", "DataBook_splitmq", "DataBook_fomo",
        "DataBook_usurp",
        "DataBook_QIn", "DataBook_QOut", "DataBook_QSurf",
        "DataBook_XIn", "DataBook_XOut", "DataBook_XSurf"
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



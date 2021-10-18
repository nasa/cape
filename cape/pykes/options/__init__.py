"""

This module provides tools to read, access, modify, and write settings for
:mod:`cape.pyfun`.  The class is based off of the built-in :class:`dict` class, so
its default behavior, such as ``opts['Namelist']`` or 
``opts.get('Namelist')`` are also present.  In addition, many convenience
methods, such as ``opts.get_project_rootname()``, are also provided.

In addition, this module controls default values of each pyFun
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    1. Values directly specified in the input file, :file:`pyFun.json`
    
    2. Values specified in the default control file,
       ``$PYFUN/settings/pyFun.default.json``
    
    3. Hard-coded defaults from this module

"""

# Import options-specific utilities (loads :mod:`os`, too)
from .util import *

# Import template module
from ...cfdx import options

# Import modules for controlling specific parts of Cart3D
from ...cfdx.options.pbs         import PBS
from ...cfdx.options.DataBook    import DataBook
from ...cfdx.options.Report      import Report
from ...cfdx.options.runControl  import RunControl
from ...cfdx.options.Mesh        import Mesh
from ...cfdx.options.Config      import Config


# Class definition
class Options(options.Options):
    r"""Options interface for :mod:`cape.pykes`
    
    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Additional manual options
    :Versions:
        * 2021-10-18 ``@ddalle``: Version 1.0
    """
    # Initialization method
    def __init__(self, fname=None, **kw):
        r"""Initialization method"""
        # Check for an input file.
        if fname:
            # Get the equivalent dictionary.
            d = loadJSONFile(fname)
            # Loop through the keys.
            for k in d:
                kw[k] = d[k]
        # Read the defaults.
        defs = getPyKesDefaults()
        
        # Apply the defaults.
        kw = applyDefaults(kw, defs)
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._PBS()
        self._Slurm()
        self._DataBook()
        self._Report()
        self._RunControl()
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
        """Initialize report options if necessary"""
        # Check status.
        if 'RunControl' not in self:
            # Missing entirely.
            self['RunControl'] = Report()
        elif type(self['RunControl']).__name__ == 'dict':
            # Convert to special class
            self['RunControl'] = RunControl(**self['RunControl'])
    
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
    
    # Method to get the namelist template
    def get_JobXML(self, j=None):
        r"""Return the name of the main Kestrel XML input file
        
        :Call:
            >>> fname = opts.get_JobXML(j=None)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: :class:`int` or ``None``
                Phase index
        :Outputs:
            *fname*: :class:`str`
                Name of Kestrel XML template file
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        return self.get_key("JobXML", j)
        
    # Method to set the namelist template
    def set_JobXML(self, fname):
        r"""Set the name of the main Kestrel XML input file
        
        :Call:
            >>> opts.set_JobXML(fname)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *fname*: :class:`str`
                Name of Kestrel XML template file
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        self["JobXML"] = fname
    
    # Method to determine if groups have common meshes.
    def get_GroupMesh(self):
        """Determine whether or not groups have common meshes
        
        :Call:
            >>> qGM = opts.get_GroupMesh()
        :Inputs:
            *opts* :class:`Options`
                Options interface
        :Outputs:
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: Version 1.0
        """
        # Safely get the trajectory.
        x = self.get('RunMatrix', {})
        return x.get('GroupMesh', rc0('GroupMesh'))
        
    # Method to specify that meshes do or do not use the same mesh
    def set_GroupMesh(self, qGM=rc0('GroupMesh')):
        """Specify that groups do or do not use common meshes
        
        :Call:
            >>> opts.get_GroupMesh(qGM)
        :Inputs:
            *opts* :class:`Options`
                Options interface
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: Version 1.0
        """
        self['RunMatrix']['GroupMesh'] = qGM
   # >
    
   # ===================
   # Overall run control
   # ===================
   # <
        
    # Copy documentation     
    for k in []:
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(RunControl,'set_'+k).__doc__
   
   # >
   
   # =================
   # JobXML settings
   # =================
   # <
   # >
   
    
   # =============
   # Mesh settings
   # =============
   # <
        
    # Copy documentation
    for k in []:
        eval('get_'+k).__doc__ = getattr(Mesh,'get_'+k).__doc__
   # >
    
    
   # =============
   # Configuration
   # =============
   # <
        
    # Copy over the documentation.
    for k in []:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Config,'get_'+k).__doc__
        
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



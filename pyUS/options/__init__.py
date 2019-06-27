"""

This module provides tools to read, access, modify, and write settings for
:mod:`pyFun`.  The class is based off of the built-in :class:`dict` class, so
its default behavior, such as ``opts['InputInp']`` or 
``opts.get('InputInp')`` are also present.  In addition, many convenience
methods, such as ``opts.get_CFDSOLVER_ires()``, are also provided.

In addition, this module controls default values of each pyUS
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    1. Values directly specified in the input file, :file:`pyUS.json`
    
    2. Values specified in the default control file,
       ``$PYFUN/settings/pyUS.default.json``
    
    3. Hard-coded defaults from this module

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
from .inputInp    import InputInpOpts
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
        defs = getPyUSDefaults()
        
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
    
    # Initialization method for ``input.inp`` variables
    def _US3D(self):
        """Initialize namelist options"""
        # Check status.
        if 'US3D' not in self:
            # Missing entirely.
            self['US3D'] = InputInpOpts()
        elif type(self['US3D']).__name__ == 'dict':
            # Convert to special class
            self['US3D'] = InputInpOpts(**self['US3D'])
    
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
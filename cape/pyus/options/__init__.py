"""

This module provides tools to read, access, modify, and write settings for
:mod:`cape.pyfun`.  The class is based off of the built-in :class:`dict` class, so
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
import cape.cfdx.options

# Import modules for controlling specific parts of Cart3D
from .pbs         import PBS
from .DataBook    import DataBook
from .Report      import Report
from .runControl  import RunControl
from .inputInp    import InputInpOpts
from .Mesh        import Mesh
from .Config      import Config


# Class definition
class Options(cape.cfdx.options.Options):
    r"""Options class based on :class:`dict`

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
        self._US3D()
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

   # ==============
   # Global Options
   # ==============
   # <
    # Method to get the ``input.inp`` template file name
    def get_InputInp(self, j=None):
        """Return the name of the master :file:`fun3d.nml` file

        :Call:
            >>> fname = opts.get_InputInp(j=None)
        :Inputs:
            *opts*: :class:`pyUS.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *fname*: :class:`str`
                Name of US3D ``input.inp`` template file
        :Versions:
            * 2019-06-27 ``@ddalle``: First version
        """
        return self.get_key('InputInp', j)

    # Method to set the ``input.inp`` template file name
    def set_InputInp(self, fname):
        """Set the name of the master :file:`fun3d.nml` file

        :Call:
            >>> opts.set_InputInp(fname)
        :Inputs:
            *opts*: :class:`pyUS.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of US3D ``input.inp`` template file
        :Versions:
            * 2019-06-27 ``@ddalle``: First version
        """
        self['InputInp'] = fname

    # Method to determine if groups have common meshes.
    def get_GroupMesh(self):
        """Determine whether or not groups have common meshes

        :Call:
            >>> qGM = opts.get_GroupMesh()
        :Inputs:
            *opts* :class:`pyUS.options.Options`
                Options interface
        :Outputs:
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
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
            *opts* :class:`pyCart.options.Options`
                Options interface
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
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

   # >

   # =============
   # us3d-prepar
   # =============
   # <
    # Option to run
    def get_us3d_prepar_run(self, i=0):
        self._RunControl()
        return self["RunControl"].get_us3d_prepar_run(i=i)

    # Option to run
    def set_us3d_prepar_run(self, run=rc0("us3d_prepar_run"), i=0):
        self._RunControl()
        self["RunControl"].set_us3d_prepar_run(run, i=i)

    # Option for input file
    def get_us3d_prepar_grid(self, i=None):
        self._RunControl()
        return self["RunControl"].get_us3d_prepar_grid(i=i)

    # Option for input file
    def set_us3d_prepar_grid(self, grid=rc0("us3d_prepar_grid"), i=None):
        self._RunControl()
        self["RunControl"].set_us3d_prepar_grid(grid, i=i)

    # Option for conn file
    def get_us3d_prepar_conn(self, i=None):
        self._RunControl()
        return self["RunControl"].get_us3d_prepar_conn(i=i)

    # Option for conn file
    def set_us3d_prepar_conn(self, conn=rc0("us3d_prepar_conn"), i=None):
        self._RunControl()
        self["RunControl"].set_us3d_prepar_conn(conn, i=i)

    # Option for grid file
    def get_us3d_prepar_output(self, i=None):
        self._RunControl()
        return self["RunControl"].get_us3d_prepar_output(i=i)

    # Option for grid file
    def set_us3d_prepar_output(self, fout=rc0("us3d_prepar_output"), i=None):
        self._RunControl()
        self["RunControl"].set_us3d_prepar_output(fout, i=i)

    # Copy documentation     
    for k in ["run", "grid", "conn", "output"]:
        n1 = "get_us3d_prepar_" + k
        n2 = "set_us3d_prepar_" + k
        eval(n1).__doc__ = getattr(RunControl, n1).__doc__
        eval(n2).__doc__ = getattr(RunControl, n2).__doc__
   # >

   # ===================
   # input.inp settings
   # ===================
   # <

    # CFD_SOLVER section
    def get_CFDSOLVER(self, j=None):
        self._US3D()
        return self['US3D'].get_CFDSOLVER(j)

    # One CFD_SOLVER setting
    def get_CFDSOLVER_key(self, k, j=None):
        self._US3D()
        return self['US3D'].get_CFDSOLVER_key(k, j)

    # Project rootname
    def get_section(self, sec, j=None):
        self._US3D()
        return self['US3D'].get_section(sec, j)

    # Generic value
    def get_InputInp_key(self, sec, key, j=None):
        self._US3D()
        return self['US3D'].get_InputInp_key(sec, key, j)

    # Copy documentation
    for k in ['CFDSOLVER', 'CFDSOLVER_key', 'section', 'InputInp_key']:
        eval('get_'+k).__doc__ = getattr(InputInpOpts,'get_'+k).__doc__

    # Set generic value
    def set_InputInp_key(self, sec, key, val, j=None):
        self._US3D()
        return self['US3D'].set_InputInp_key(sec, key, val, j)

    # Copy documentation
    for k in ['InputInp_key']:
        eval('set_'+k).__doc__ = getattr(InputInpOpts,'set_'+k).__doc__

    # Downselect
    def select_InputInp(self, j=0):
        self._US3D()
        return self['US3D'].select_InputInp(j)
    select_InputInp.__doc__ = InputInpOpts.select_InputInp.__doc__
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
    # Get components
    def get_ConfigInput(self, comp):
        self._Config()
        return self['Config'].get_ConfigInput(comp)

    # Set components
    def set_ConfigInput(self, comp, inp):
        self._Config()
        self['Config'].set_ConfigInput(comp, inp)

    # Copy over the documentation.
    for k in [
            'ConfigInput'
    ]:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Config,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(Config,'set_'+k).__doc__

    # post.scr file
    def get_PostScrFile(self, j=None):
        self._Config()
        return self['Config'].get_PostScrFile(j)

    # Copy over the documentation.
    for k in [
            'PostScrFile'
    ]:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Config,'get_'+k).__doc__


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

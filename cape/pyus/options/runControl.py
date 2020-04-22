"""
:mod:`cape.pyus.options.runControl.RunControl`: Run control options
=====================================================================

Options interface for aspects of running a case of US3D.  The settings are
read from the ``"RunControl"`` of a JSON file, and the contents of this section
are written to :file:`case.json` within each run folder.

The methods of :class:`cape.cfdx.options.runControl.RunControl` are also present.
These control options such as whether to submit as a PBS job, whether or not to
use MPI, etc.

This contains options that determine how long the solver is ran (primarily 
via the ``"PhaseSequence"`` and ``"PhaseIters"`` options), what basic mode it
is run in (such as a submitted or local job or serial or MPI job), and options
for command-line options to the FUN3D binaries.  There is also an
``"Archive"`` section that can be used for copying files and cleaning up after
one or more cases have been completed.

This module primarily provides a class :class:`pyFun.options.RunControl`. Many
of the options that are common to all solvers are inherited from
:class:`cape.cfdx.options.runControl.RunControl`. This class also has an interface
for environment variables and ``ulimit`` parameters.

In particular, all of the commands available to the classes listed below are
also available to :class:`pyFun.options.runControl.RunControl`.

:Classes:
    * :class:`pyUS.options.runControl.RunControl`
    * :class:`pyUS.options.runControl.nodet`
    * :class:`pyUS.options.Archive.Archive`
    * :class:`cape.cfdx.options.aflr3.aflr3`
    * :class:`cape.cfdx.options.runControl.Environ`
    * :class:`cape.cfdx.options.ulimit.ulimit`
:See Also:
    * :mod:`cape.cfdx.options.runControl`
    * :mod:`cape.cfdx.options.ulimit`
    * :mod:`cape.cfdx.options.intersect`
    * :mod:`cape.pyus.options.Archive`
"""

# Import options-specific utilities
from .util import rc0, getel, odict

# Import template module
import cape.cfdx.options.runControl

# Submodules
from .Archive import Archive

# Class for inputs to the US3D executable
class US3D(odict):
    pass


# Class for inputs to the ``us3d-prepar`` executable
class US3DPrepar(odict):
    # Option to run ``us3d-prepar``
    def get_us3d_prepar_run(self, i=0):
        r"""Get option to run or not run ``us3d-prepar``

        :Call:
            >>> run = opts.get_us3d_prepar_run(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *i*: {``0``} | :class:`int` | ``None``
                Phase number
        :Outputs:
            *run*: {``True``} | ``False``
                Option to run or not run ``us3d-prepar``
        :Versions:
            * 2020-04-22 ``@ddalle``: First version
        """
        return self.get_key("run", i, rck="us3d_prepar_run")

    # Option to run ``us3d-prepar``
    def set_us3d_prepar_run(self, run=rc0("us3d_prepar_run"), i=0):
        r"""Set option to run or not run ``us3d-prepar``

        :Call:
            >>> opts.set_us3d_prepar_run(run=True, i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *run*: {``True``} | ``False``
                Option to run or not run ``us3d-prepar``
            *i*: {``0``} | :class:`int` | ``None``
                Phase number
        :Versions:
            * 2020-04-22 ``@ddalle``: First version
        """
        self.set_key("run", run, i)

    # Name of grid input file
    def get_us3d_prepar_grid(self, i=None):
        r"""Get name of input grid to ``us3d-prepar``

        :Call:
            >>> grid = opts.get_us3d_prepar_grid(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *grid*: :class:`str`
                Fluent-format input mesh file name
        :Versions:
            * 2020-04-21 ``@ddalle``: First version
        """
        return self.get_key("grid", i, rck="us3d_prepar_grid")

    # Name of grid input file
    def set_us3d_prepar_grid(self, grid=rc0("us3d_prepar_grid"), i=None):
        r"""Set name of input grid to ``us3d-prepar``

        :Call:
            >>> opts.set_us3d_prepar_grid(grid, i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *grid*: :class:`str`
                Fluent-format input mesh file name
            *i*: :class:`int`
                Phase number
        :Versions:
            * 2020-04-21 ``@ddalle``: First version
        """
        self.set_key("grid", grid, i)

    # Name of grid input file
    def get_us3d_prepar_conn(self, i=None):
        r"""Get name of connectivity file made by ``us3d-prepar``

        :Call:
            >>> conn = opts.get_us3d_prepar_conn(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *conn*: :class:`str`
                Name of HDF5 file for US3D mesh connectivity
        :Versions:
            * 2020-04-22 ``@ddalle``: First version
        """
        return self.get_key("conn", i, rck="us3d_prepar_conn")

    # Name of grid input file
    def set_us3d_prepar_conn(self, conn=rc0("us3d_prepar_conn"), i=None):
        r"""Set name of connectivity file made by ``us3d-prepar``

        :Call:
            >>> opts.set_us3d_prepar_conn(conn, i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *conn*: :class:`str`
                Name of HDF5 file for US3D mesh connectivity
            *i*: :class:`int`
                Phase number
        :Versions:
            * 2020-04-22 ``@ddalle``: First version
        """
        self.set_key("conn", conn, i)

    # Name of grid input file
    def get_us3d_prepar_out(self, i=None):
        r"""Get name of mesh file converted by ``us3d-prepar``

        :Call:
            >>> out = opts.get_us3d_prepar_out(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *out*: :class:`str`
                Name of HDF5 mesh file for US3D
        :Versions:
            * 2020-04-22 ``@ddalle``: First version
        """
        return self.get_key("out", i, rck="us3d_prepar_out")

    # Name of grid input file
    def set_us3d_prepar_out(self, out=rc0("us3d_prepar_out"), i=None):
        r"""Get name of mesh file converted by ``us3d-prepar``

        :Call:
            >>> out = opts.get_us3d_prepar_out(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *out*: :class:`str`
                Name of HDF5 mesh file for US3D
            *i*: :class:`int`
                Phase number
        :Versions:
            * 2020-04-22 ``@ddalle``: First version
        """
        self.set_key("out", out, i)


# Class for Report settings
class RunControl(cape.cfdx.options.runControl.RunControl):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = RunControl(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._Environ()
        self._ulimit()
        self._US3DPrepar()
        self._Archive()
        
   # ============ 
   # Initializers
   # ============
   # <
    # Initialization and confirmation for ``us3d-prepar`` options
    def _US3DPrepar(self):
        """Initialize ``us3d-prepar`` options if necessary"""
        # Get object
        opts = self.get("us3d-prepar", {})
        # Check types
        if isinstance(opts, US3DPrepar):
            # Nothing to do
            pass
        elif isinstance(opts, dict):
            # Convert to special class
            self["us3d-prepar"] = US3DPrepar(**opts)
        elif opts:
            # Create empty class with defaults
            self["us3d-prepar"] = US3DPrepar(run=True)
        else:
            # Create empty class, ``us3d-prepar`` turned off
            self["us3d-prepar"] = US3DPrepar(run=False)

    # Initialization and confirmation for options to US3D executable


    # Initialization method for folder management options
    def _Archive(self):
        """Initialize folder management options if necessary"""
        # Get object
        opts = self.get("Archive", {})
        # Check type
        if isinstance(opts, Archive):
            # Nothing to do
            pass
        elif isinstance(opts, dict):
            # Convert
            self["Archive"] = Archive(**opts)
        else:
            # Bad type
            raise TypeError('"Archive" option has bad type')
   # >
   
   # ============== 
   # Local settings
   # ==============
   # <
   # >

   # =============
   # us3d-prepar
   # =============
   # <
    # Option to run
    def get_us3d_prepar_run(self, i=0):
        self._US3DPrepar()
        return self["us3d-prepar"].get_us3d_prepar_run(i=i)

    # Option to run
    def set_us3d_prepar_run(self, run=rc0("us3d_prepar_run"), i=0):
        self._US3DPrepar()
        self["us3d-prepar"].set_us3d_prepar_run(run, i=i)

    # Option for input file
    def get_us3d_prepar_grid(self, i=None):
        self._US3DPrepar()
        return self["us3d-prepar"].get_us3d_prepar_grid(i=i)

    # Option for input file
    def set_us3d_prepar_grid(self, grid=rc0("us3d_prepar_grid"), i=None):
        self._US3DPrepar()
        self["us3d-prepar"].set_us3d_prepar_grid(grid, i=i)

    # Option for conn file
    def get_us3d_prepar_conn(self, i=None):
        self._US3DPrepar()
        return self["us3d-prepar"].get_us3d_prepar_conn(i=i)

    # Option for conn file
    def set_us3d_prepar_conn(self, conn=rc0("us3d_prepar_conn"), i=None):
        self._US3DPrepar()
        self["us3d-prepar"].set_us3d_prepar_conn(conn, i=i)

    # Option for grid file
    def get_us3d_prepar_out(self, i=None):
        self._US3DPrepar()
        return self["us3d-prepar"].get_us3d_prepar_out(i=i)

    # Option for grid file
    def set_us3d_prepar_out(self, out=rc0("us3d_prepar_out"), i=None):
        self._US3DPrepar()
        self["us3d-prepar"].set_us3d_prepar_out(out, i=i)

    # Copy documentation     
    for k in ["run", "grid", "conn", "out"]:
        n1 = "get_us3d_prepar_" + k
        n2 = "set_us3d_prepar_" + k
        eval(n1).__doc__ = getattr(US3DPrepar, n1).__doc__
        eval(n2).__doc__ = getattr(US3DPrepar, n2).__doc__
   # >
    
   # =================
   # Folder management
   # =================
   # <

   # >
# class RunControl



r"""

Options interface for aspects of running a case of US3D. The settings
are read from the ``"RunControl"`` of a JSON file, and the contents of
this section are written to ``case.json`` within each run folder.

The methods of :class:`cape.cfdx.options.runctlopts.RunControlOpts` are
also present. These control options such as whether to submit as a PBS
job, whether or not to use MPI, etc.

:See Also:
    * :mod:`cape.cfdx.options.runctlopts`
    * :mod:`cape.cfdx.options.ulimitopts`
    * :mod:`cape.pyus.options.archiveopts`
"""

# Local imports
from ...cfdx.options import runctlopts
from ...cfdx.options.util import ExecOpts
from .archiveopts import ArchiveOpts


# Class for inputs to the US3D executable
class US3DRunOpts(ExecOpts):
    # Option for input file
    def get_us3d_input(self, i=None):
        r"""Get name of input file for US3D

        :Call:
            >>> finp = opts.get_us3d_input(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *i*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *finp*: :class:`str`
                Name of input file (usually ``"input.inp"``)
        :Versions:
            * 2020-04-29 ``@ddalle``: First version
        """
        return self.get_key("input", i, rck="us3d_input")

    # Option to set input file
    def set_us3d_input(self, finp=rc0("us3d_input"), i=None):
        r"""Set name of input file for US3D

        :Call:
            >>> opts.get_us3d_input(finp, i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *finp*: :class:`str`
                Name of input file (usually ``"input.inp"``)
            *i*: {``None``} | :class:`int`
                Phase number
        :Versions:
            * 2020-04-29 ``@ddalle``: First version
        """
        self.set_key("input", finp, i)

    # Get grid name for US3D
    def get_us3d_grid(self, i=None):
        r"""Get name of grid file for US3D

        :Call:
            >>> grid = opts.get_us3d_grid(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *i*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *grid*: :class:`str`
                Name of grid file (usually ``"grid.h5"``)
        :Versions:
            * 2020-04-29 ``@ddalle``: First version
        """
        return self.get_key("input", i, rck="us3d_input")

    # Set grid name for US3D
    def set_us3d_grid(self, grid=rc0("us3d_grid"), i=None):
        r"""Set name of grid file for US3D

        :Call:
            >>> opts.set_us3d_grid(grid, i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *grid*: :class:`str`
                Name of grid file (usually ``"grid.h5"``)
            *i*: {``None``} | :class:`int`
                Phase number
        :Versions:
            * 2020-04-29 ``@ddalle``: First version
        """
        self.set_key("grid", grid, i)

    # Get gas to use for US3D
    def get_us3d_gas(self, i=None):
        r"""Get name of gas model to use for US3D

        :Call:
            >>> gas = opts.get_us3d_gas(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *i*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *gas*: ``None`` | :class:`str`
                Name of gas model
        :Versions:
            * 2020-04-29 ``@ddalle``: First version
        """
        return self.get_key("gas", i, rck="us3d_gas")

    # Set gas model for US3D
    def set_us3d_gas(self, gas=rc0("us3d_gas"), i=None):
        r"""Set name of gas model to use for US3D

        :Call:
            >>> opts.set_us3d_gas(gas, i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *gas*: ``None`` | :class:`str`
                Name of gas model
            *i*: {``None``} | :class:`int`
                Phase number
        :Versions:
            * 2020-04-29 ``@ddalle``: First version
        """
        self.set_key("gas", gas, i)


# Class for inputs to the ``us3d-prepar`` executable
class US3DPreparOpts(ExecOpts):
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

    # Name of grid output file
    def get_us3d_prepar_output(self, i=None):
        r"""Get name of mesh file converted by ``us3d-prepar``

        :Call:
            >>> fout = opts.get_us3d_prepar_output(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *fout*: :class:`str`
                Name of HDF5 mesh file for US3D
        :Versions:
            * 2020-04-22 ``@ddalle``: First version
        """
        return self.get_key("output", i, rck="us3d_prepar_output")

    # Name of grid output file
    def set_us3d_prepar_output(self, fout=rc0("us3d_prepar_output"), i=None):
        r"""Get name of mesh file converted by ``us3d-prepar``

        :Call:
            >>> fout = opts.get_us3d_prepar_output(i=None)
        :Inputs:
            *opts*: :class:`cape.pyus.options.Options`
                Options interface
            *fout*: :class:`str`
                Name of HDF5 mesh file for US3D
            *i*: :class:`int`
                Phase number
        :Versions:
            * 2020-04-22 ``@ddalle``: First version
        """
        self.set_key("output", fout, i)


# Class for Report settings
class RunControlOpts(runctlopts.RunControlOpts):
    # No additional atributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "us3d-prepar",
    }

    # Section map
    _sec_cls = {
        "Archive": ArchiveOpts,
        "us3d": US3DRunOpts,
        "us3d-prepar": US3DPreparOpts,
    }

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
    def get_us3d_prepar_output(self, i=None):
        self._US3DPrepar()
        return self["us3d-prepar"].get_us3d_prepar_output(i=i)

    # Option for grid file
    def set_us3d_prepar_output(self, fout=rc0("us3d_prepar_output"), i=None):
        self._US3DPrepar()
        self["us3d-prepar"].set_us3d_prepar_output(fout, i=i)

    # Copy documentation
    for k in ["run", "grid", "conn", "output"]:
        n1 = "get_us3d_prepar_" + k
        n2 = "set_us3d_prepar_" + k
        eval(n1).__doc__ = getattr(US3DPrepar, n1).__doc__
        eval(n2).__doc__ = getattr(US3DPrepar, n2).__doc__
   # >

   # =================
   # us3d
   # =================
   # <
    # Option for input file
    def get_us3d_input(self, i=None):
        self._US3D()
        return self["us3d"].get_us3d_input(i)

    # Option for input file
    def set_us3d_input(self, finp=rc0("us3d_input"), i=None):
        self._US3D()
        self["us3d"].set_us3d_input(finp, i=i)

    # Option for grid file
    def get_us3d_grid(self, i=None):
        self._US3D()
        return self["us3d"].get_us3d_grid(i)

    # Option for grid file
    def set_us3d_grid(self, grid=rc0("us3d_grid"), i=None):
        self._US3D()
        self["us3d"].set_us3d_grid(grid, i=i)

    # Option for gas file
    def get_us3d_gas(self, i=None):
        self._US3D()
        return self["us3d"].get_us3d_gas(i)

    # Option for gas file
    def set_us3d_gas(self, gas=rc0("us3d_gas"), i=None):
        self._US3D()
        self["us3d"].set_us3d_gas(gas, i=i)

    # Copy documentation
    for k in ["input", "grid", "gas"]:
        n1 = "get_us3d_" + k
        n2 = "set_us3d_" + k
        eval(n1).__doc__ = getattr(US3D, n1).__doc__
        eval(n2).__doc__ = getattr(US3D, n2).__doc__
   # >
# class RunControl



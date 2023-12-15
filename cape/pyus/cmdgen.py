#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyus.cmd`: Create commands for US3D executables
=============================================================

This module creates system commands as lists of strings for executable
binaries or scripts for FUN3D.  The main US3D executables are

    * ``us3d-prepar``: Convert Fluent grid to proper format
    * ``us3d-genbc``: Generate boundary condition inputs from grid
    * ``us3d``: Execute US3D flow solver

Commands are created in the form of a list of strings.  This is the
format used in the built-in module :mod:`subprocess` and also with
:func:`cape.bin.calli`. As a very simple example, the system command
``"ls -lh"`` becomes the list ``["ls", "-lh"]``.

These commands also include prefixes such as ``mpiexec`` if necessary.
The decision to use ``mpiexec`` or not is based on the keyword input
``"MPI"``.  For example, two versions of the command returned by
:func:`us3d` could be

    .. code-block:: python

        ["mpiexec", "-np", "240", "us3d"]
        ["mpiexec", "-np", "1", "us3d-prepar" "--grid", "pyus.cas"]

:See also:
    * :mod:`cape.cfdx.cmd`
    * :mod:`cape.cfdx.bin`
    * :mod:`cape.pyus.bin`
    * :mod:`cape.pyus.options.runControl`

"""

# Local imports
from ..cfdx.cmdgen import isolate_subsection, append_cmd_if
from .options import Options
from .options import runControl


# Handle to defaults
rc0 = runControl.rc0


# Function to execute ``us3d-prepar``
def us3d_prepar(opts=None, j=0, **kw):
    r"""Interface to US3D executable ``us3d-prepar``

    :Call:
        >>> cmdi = us3d_prepar(opts, j=0, **kw)
    :Inputs:
        *opts*: :class:`cape.pyus.options.Options`
            Global or "RunControl" pyUS options
        *j*: {``0``} | :class:`int`
            Phase number
        *grid*: {``"pyus.cas"``} | :class:`str`
            Name of input Fluent mesh
        *output*: {``None``} | :class:`str`
            Name of output HDF5 grid (``"grid.h5"``)
        *conn*: {``None``} | :class:`str`
            Name of output connectivity file (``"conn.h5"``)
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2020-04-20 ``@ddalle``: v1.0
        * 2023-08-21 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate options
    rc = isolate_subsection(opts, Options, ("RunControl",))
    opts = rc["us3d-prepar"]
    # Get values for run configuration
    n_mpi  = rc.get_MPI(j)
    mpicmd = rc.get_mpicmd(j)
    # Get values for command line
    grid = opts.get_us3d_prepar_grid(j)
    conn = opts.get_us3d_prepar_conn(j)
    fout = opts.get_us3d_prepar_output(j)
    # Process keyword overrides
    n_mpi  = kw.get('MPI',    n_mpi)
    mpicmd = kw.get('mpicmd', mpicmd)
    # Process keyword overrides
    grid = kw.get("grid", grid)
    fout = kw.get("output", fout)
    conn = kw.get("conn", conn)
    # Form the initial command
    cmdi = []
    # Append MPI commands
    append_cmd_if(cmdi, n_mpi, [mpicmd, "-np", "1"])
    # Append name of command
    cmdi.append("us3d-prepar")
    # Process known options
    append_cmd_if(cmdi, grid, ["--grid", grid])
    append_cmd_if(cmdi, fout, ["--output", fout])
    append_cmd_if(cmdi, conn, ["--conn", conn])
    # Output
    return cmdi


# Function to execute ``us3d-genbc``
def us3d_genbc(opts=None, j=0, **kw):
    r"""Interface to US3D executable ``us3d-genbc``

    :Call:
        >>> cmdi = us3d_genbc(opts, i=0)
        >>> cmdi = us3d_genbc(**kw)
    :Inputs:
        *opts*: :class:`cape.pyus.options.Options`
            Global or "RunControl" pyUS options
        *j*: {``0``} | :class:`int`
            Phase number
        *grid*: {``"grid.h5"``} | :class:`str`
            Name of prepared US3D grid
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2020-04-27 ``@ddalle``: v1.0
        * 2023-08-21 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate options
    rc = isolate_subsection(opts, Options, ("RunControl",))
    # Get subsection
    opts = rc["us3d-prepar"]
    # Get values for command line
    grid = opts.get_us3d_prepar_output(j)
    # Process keyword overrides
    grid = kw.get("grid", grid)
    # Form the initial command
    cmdi = ["us3d-genbc"]
    # Process known options
    append_cmd_if(cmdi, grid, ["--grid", grid])
    # Output
    return cmdi


# Function to create ``us3d`` command
def us3d(opts=None, j=0, **kw):
    r"""Interface to US3D executable ``us3d``

    :Call:
        >>> cmdi = us3d(opts, j=0, **kw)
    :Inputs:
        *opts*: :class:`cape.pyus.options.Options`
            Global or "RunControl" pyUS options
        *j*: {``0``} | :class:`int`
            Phase number
        *input*: {``"input.inp"``} | :class:`str`
            Name of US3D input file
        *grid*: {``"grid.h5"``} | :class:`str`
            Name of input HDF5 mesh
        *gas*: {``None``} | :class:`str`
            Name of gas file to use
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2020-04-29 ``@ddalle``: v1.0
    """
    # Isolate options
    rc = isolate_subsection(opts, Options, ("RunControl",))
    # Get subsection
    cli_us3d = rc["us3d"]
    # Get values for run configuration
    n_mpi = rc.get_MPI(j)
    mpicmd = rc.get_mpicmd(j)
    nProc = rc.get_nProc(j)
    # Form the initial command
    cmdi = []
    # Append MPI flags if appropriate
    append_cmd_if(cmdi, n_mpi, [mpicmd, "-np", str(nProc)])
    # Append executable
    cmdi.append("us3d")
    # Specific options
    finp = cli_us3d.get_us3d_input(j)
    grid = cli_us3d.get_us3d_grid(j)
    gas = cli_us3d.get_us3d_gas(j)
    # Append these options
    append_cmd_if(cmdi, finp, ["--input", finp])
    append_cmd_if(cmdi, grid, ["--grid", grid])
    append_cmd_if(cmdi, gas, ["--gas", gas])
    for k in ("input", "grid", "gas"):
        cli_us3d.pop(k, None)
    # Loop through command-line inputs
    for k in cli_us3d.items():
        # Get value
        v = cli_us3d.get_option(k, j=j)
        # Create command for this option
        cmdk = [f"--{k}"]
        # Append extra portion for *v*
        if v and v is not True:
            cmdk.append(str(v))
        # Append command
        append_cmd_if(cmdi, v, cmdk)
    # Output
    return cmdi

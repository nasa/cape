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

# Options for the binaries
from .options import runControl, getel


# Handle to defaults
rc0 = runControl.rc0


# Function to create ``nodet`` or ``nodet_mpi`` command
def nodet(opts=None, i=0, **kw):
    r"""Interface to FUN3D binary ``nodet`` or ``nodet_mpi``
    
    :Call:
        >>> cmdi = cmd.nodet(opts, i=0)
        >>> cmdi = cmd.nodet(**kw)
    :Inputs:
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options interface or "RunControl" interface
        *i*: :class:`int`
            Phase number
        *animation_freq*: :class:`int`
            Output frequency
    :Outputs:
        *cmdi*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-11-24 ``@ddalle``: First version
    """
    # Check for options input
    if opts is not None:
        # Get values for run configuration
        n_mpi  = opts.get_MPI(i)
        nProc  = opts.get_nProc(i)
        mpicmd = opts.get_mpicmd(i)
        # Get dictionary of command-line inputs
        if "nodet" in opts:
            # pyFun.options.runControl.RunControl instance
            cli_nodet = opts["nodet"]
        elif "RunControl" in opts and "nodet" in opts["RunControl"]:
            # pyFun.options.Options instance
            cli_nodet = opts["RunControl"]["nodet"]
        else:
            # No command-line arguments
            cli_nodet = {}
    else:
        # Get values from keyword arguments
        n_mpi  = kw.get("MPI", False)
        nProc  = kw.get("nProc", 1)
        mpicmd = kw.get("mpicmd", "mpiexec")
        # Form other command-line argument dictionary
        cli_nodet = kw
        # Remove above options
        if "MPI"    in cli_nodet: del cli_nodet["MPI"]
        if "nProc"  in cli_nodet: del cli_nodet["nProc"]
        if "mpicmd" in cli_nodet: del cli_nodet["mpicmd"]
    # Form the initial command.
    if n_mpi:
        # Use the ``nodet_mpi`` command
        cmdi = [mpicmd, '-np', str(nProc), 'nodet_mpi']
    else:
        # Use the serial ``nodet`` command
        cmdi = ['nodet']
    # Check for "--adapt" flag
    if kw.get("adapt"):
        # That should be the only command-line argument
        cmdi.append("--adapt")
        return cmdi
    # Loop through command-line inputs
    for k in cli_nodet:
        # Get the value
        v = cli_nodet[k]
        # Check the type
        if v == True:
            # Just an option with no value
            cmdi.append('--'+k)
        elif v == False or v is None:
            # Do not use.
            pass
        else:
            # Select option for this phase
            vi = getel(v, i)
            # Append the option and value
            cmdi.append('--'+k)
            cmdi.append(str(vi))
    # Output
    return cmdi


# Function to execute ``us3d-prepar``
def us3d_prepar(opts=None, i=0, **kw):
    r"""Interface to US3D executable ``us3d-prepar``
    
    :Call:
        >>> cmdi = cmd.us3d_prepar(opts, i=0)
        >>> cmdi = cmd.us3d_prepar(**kw)
    :Inputs:
        *opts*: :class:`cape.pyus.options.Options`
            Global or "RunControl" pyUS options
        *i*: :class:`int`
            Phase number
        *grid*: {``"pyus.case"``} | :class:`str`
            Name of input Fluent mesh
        *output*: {``None``} | :class:`str`
            Name of output HDF5 grid (``"grid.h5"``)
        *conn*: {``None``} | :class:`str`
            Name of output connectivity file (``"conn.h5"``)
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2020-04-20 ``@ddalle``: First version
    """
    # Check for options input
    if opts is not None:
        # Downselect to "RunControl" section if necessary
        if "RunControl" in opts:
            # Get subsection
            rc = opts["RunControl"]
        else:
            # Use whole thing
            rc = opts
        # Downselect to "us3d-prepar" section if necessary
        if "us3d-prepar" in rc:
            # Get subsection
            opts = rc["us3d-prepar"]
        else:
            # Use whole thing
            opts = rc
        # Check if we have a valid "RunControl" instance
        if isinstance(rc, runControl.RunControl):
            # Get values for run configuration
            n_mpi  = rc.get_MPI(i)
            mpicmd = rc.get_mpicmd(i)
        else:
            # Use defaults
            n_mpi  = rc.get("MPI", rc0('MPI'))
            mpicmd = rc.get("mpicmd", rc0('mpicmd'))
        # Check if we have a valid "us3d-prepar" instance
        if isinstance(opts, runControl.US3DPrepar):
            # Get values for command line
            grid = opts.get_us3d_prepar_grid(i)
            conn = opts.get_us3d_prepar_conn(i)
            fout = opts.get_us3d_prepar_output(i)
        else:
            # Defaults
            grid = opts.get("grid", rc0("us3d_prepar_grid"))
            conn = opts.get("conn", rc0("us3d_prepar_conn"))
            fout = opts.get("output", rc0("us3d_prepar_output"))
    else:
        # Use defaults
        n_mpi  = rc0('MPI')
        mpicmd = rc0('mpicmd')
        grid = rc0("us3d_prepar_grid")
        conn = rc0("us3d_prepar_conn")
        fout = rc0("us3d_prepar_output")
    # Process keyword overrides
    n_mpi  = kw.get('MPI',    n_mpi)
    mpicmd = kw.get('mpicmd', mpicmd)
    # Process keyword overrides
    grid = kw.get("grid", grid)
    fout = kw.get("output", fout)
    conn = kw.get("conn", conn)
    # Form the initial command
    if n_mpi:
        # Use MPI
        cmdi = [mpicmd, "-np", "1", "us3d-prepar"]
    else:
        # Serial
        cmdi = ["us3d-prepar"]
    # Process known options
    if grid:
        cmdi.extend(["--grid", grid])
    if fout:
        cmdi.extend(["--output", fout])
    if conn:
        cmdi.extend(["--conn", conn])
    # Output
    return cmdi


# Function to execute ``us3d-genbc``
def us3d_genbc(opts=None, i=0, **kw):
    r"""Interface to US3D executable ``us3d-genbc``
    
    :Call:
        >>> cmdi = cmd.us3d_genbc(opts, i=0)
        >>> cmdi = cmd.us3d_genbc(**kw)
    :Inputs:
        *opts*: :class:`cape.pyus.options.Options`
            Global or "RunControl" pyUS options
        *i*: :class:`int`
            Phase number
        *grid*: {``"grid.h5"``} | :class:`str`
            Name of prepared US3D grid
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 20120-04-27 ``@ddalle``: First version
    """
    # Check for options input
    if opts is not None:
        # Downselect to "RunControl" section if necessary
        if "RunControl" in opts:
            # Get subsection
            rc = opts["RunControl"]
        else:
            # Use whole thing
            rc = opts
        # Downselect to "us3d-prepar" section if necessary
        if "us3d-prepar" in rc:
            # Get subsection
            opts = rc["us3d-prepar"]
        else:
            # Use whole thing
            opts = rc
        # Check if we have a valid "us3d-prepar" instance
        if isinstance(opts, runControl.US3DPrepar):
            # Get values for command line
            grid = opts.get_us3d_prepar_output(i)
        else:
            # Defaults
            grid = opts.get("output", rc0("us3d_prepar_output"))
    else:
        # Use defaults
        grid = rc0("us3d_prepar_output")
    # Process keyword overrides
    grid = kw.get("grid", grid)
    # Form the initial command
    cmdi = ["us3d-genbc"]
    # Process known options
    if grid:
        cmdi.extend(["--grid", grid])
    # Output
    return cmdi


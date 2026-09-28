r"""
:mod:`cape.pyvul.tcshrc`: VULCAN-CFD tcsh startup files
=========================================================

The VULCAN-CFD ``vulcan`` executable is a ``tcsh`` script that calls
further ``tcsh`` scripts and shell aliases (``run_vulcan_int``,
``apart_vulcan``, ``pprep_vulcan``, ...) that are normally defined only
when the VULCAN environment module is loaded into an interactive
shell.  Because ``LOADEDMODULES`` is exported to subprocesses, a child
``tcsh`` cannot recreate the aliases by loading the module again (the
load is a no-op), so the aliases must be provided through a shell
startup file.

CAPE accomplishes this portably:

1. Each case folder contains ``vulcan.tcshrc``, a ``tcsh`` startup
   file that defines the VULCAN aliases.  It is written by
   :func:`cape.pyvul.cntl.Cntl.PrepareCase` and rewritten by
   :class:`cape.pyvul.casecntl.CaseRunner` before each run.
2. The user's ``~/.tcshrc`` sources the file named by the
   ``CAPE_TCSHRC`` environment variable, if present:

   .. code-block:: csh

       if ( $?CAPE_TCSHRC ) then
           source "$CAPE_TCSHRC"
       endif

   which :func:`update_user_tcshrc` appends to ``~/.tcshrc`` (or
   creates the file) during :func:`cape.pyvul.cntl.PrepareCase`.
3. :meth:`cape.pyvul.casecntl.CaseRunner.run_vulcan` exports
   ``CAPE_TCSHRC`` (along with ``CAPE_NUM_CPUS``, ``CAPE_VULCAN_CMND``,
   and ``VULCAN_ROOT``) in the environment of the ``vulcan`` call, so
   that every ``tcsh`` subprocess picks up the case's aliases.

The aliases in :data:`VULCAN_ALIASES` are hard coded.  When
the supported VULCAN version changes, update this table.
"""

# Standard library
import os
import re
import shutil
from typing import Optional

# Name of the case-local tcsh startup file
TCSHRC_NAME = "vulcan.tcshrc"

# Environment variable that points ``~/.tcshrc`` at the case file
CAPE_TCSHRC_VAR = "CAPE_TCSHRC"

# Root path
VULCAN_PATH = os.environ.get("vulcanpath", "")

# Snippet added to the user's ``~/.tcshrc``
TCSHRC_SNIPPET = (
    'if ( $?%s ) then\n' % CAPE_TCSHRC_VAR +
    '    source "$%s"\n' % CAPE_TCSHRC_VAR +
    'endif\n')

# Detect the snippet (or an equivalent) in a ``~/.tcshrc`` file
REGEX_TCSHRC_SNIPPET = re.compile(r"if\s*\(\s*\$\?CAPE")

#: :class:`tuple`\\ [\\ :class:`tuple`\\ (:class:`str`, :class:`str`)\\ ]
#: VULCAN tcsh aliases, hard coded from the ``vulcan/2026-09-21``
#: modulefile on NAS Aitken.  Paths are built from
#: :data:`VULCAN_PATH`; the remaining variables are those that
#: ``Scripts/vulcan.tcsh`` defines when an alias is used
#: (``$num_cpus``, ``$vulcan_cmnd``, ``$ofn``).
VULCAN_ALIASES = (
    ('run_vulcan_int', 'mpirun -n $num_cpus $vulcan_cmnd'),
    ('run_vulcan_btc', 'mpirun -n $num_cpus $vulcan_cmnd >> $ofn'),
    ('cvulcan', f'{VULCAN_PATH}'),
    ('cvulcancom', f'{VULCAN_PATH}/Common_blocks'),
    ('cvulcandat', f'{VULCAN_PATH}/Data_base'),
    ('cvulcandoc', f'{VULCAN_PATH}/Doc_manual'),
    ('cvulcanexe', f'{VULCAN_PATH}/Executable'),
    ('cvulcanmak', f'{VULCAN_PATH}/Make_file'),
    ('cvulcansam', f'{VULCAN_PATH}/Sample_cases'),
    ('cvulcanscr', f'{VULCAN_PATH}/Scripts'),
    ('cvulcansrc', f'{VULCAN_PATH}/Source_code'),
    ('cvulcantst', f'{VULCAN_PATH}/Test'),
    ('cvulcanutl', f'{VULCAN_PATH}/Utilities'),
    ('cvulcanval', f'{VULCAN_PATH}/Validate'),
    ('grid_split', f'{VULCAN_PATH}/Scripts/grid_split'),
    ('grid_split_tinf', f'{VULCAN_PATH}/Scripts/grid_split_tinf'),
    ('install_vulcan', f'{VULCAN_PATH}/Scripts/install_vulcan'),
    ('vulcan', f'{VULCAN_PATH}/Scripts/vulcan'),
    ('apart_vulcan', f'{VULCAN_PATH}/Scripts/apart_vulcan.tcsh'),
    ('compile_vulcan', f'{VULCAN_PATH}/Scripts/vulcan.compile.tcsh'),
    ('pprep_vulcan', f'{VULCAN_PATH}/Scripts/pprep_vulcan.tcsh'),
    ('tar_vulcan', f'{VULCAN_PATH}/Scripts/vulcan.tar.tcsh'),
    ('test_vulcan', f'{VULCAN_PATH}/Scripts/vulcan.test.tcsh'),
    ('validate_vulcan', f'{VULCAN_PATH}/Scripts/vulcan.validate.tcsh'),
    ('vulvi', f'{VULCAN_PATH}/Scripts/vulsrcvi.tcsh'),
    ('vulcanled', f'{VULCAN_PATH}/Scripts/vulcan.led.tcsh'),
    (
        'profile_split',
        f'{VULCAN_PATH}/Utilities/Load_Balance_codes/SGLD/scripts'
        '/profile_split.py'),
    (
        'restart_split',
        f'{VULCAN_PATH}/Utilities/Load_Balance_codes/SGLD/src'
        '/restart_split'),
    (
        'restart_merge',
        f'{VULCAN_PATH}/Utilities/Load_Balance_codes/SGLD/src'
        '/restart_merge'),
    (
        'plot3d_merge',
        f'{VULCAN_PATH}/Utilities/Load_Balance_codes/SGLD/src'
        '/plot3d_merge'),
    ('atmos76', f'{VULCAN_PATH}/Utilities/Dbase_codes/atmos76'),
    ('conv_chem', f'{VULCAN_PATH}/Utilities/Dbase_codes/conv_chem'),
    ('ls_fit', f'{VULCAN_PATH}/Utilities/Dbase_codes/ls_fit'),
    ('mix_fit', f'{VULCAN_PATH}/Utilities/Dbase_codes/mix_fit'),
    ('mw_coef', f'{VULCAN_PATH}/Utilities/Dbase_codes/mw_coef'),
    (
        'grid_plot3d',
        f'{VULCAN_PATH}/Utilities/Grid_codes/Plot3d/grid_plot3d'),
    ('gridgent', f'{VULCAN_PATH}/Utilities/Grid_codes/Gridgen/gridgent'),
    ('gridprot', f'{VULCAN_PATH}/Utilities/Grid_codes/Gridpro/gridprot'),
    ('v2knmapt', f'{VULCAN_PATH}/Utilities/Grid_codes/V2K/v2knmapt'),
    ('gpro2nmf', f'{VULCAN_PATH}/Utilities/Grid_codes/Gridpro/gpro2nmf'),
    (
        'time_merge',
        f'{VULCAN_PATH}/Utilities/Post_Process_codes/Time_files/time_merge'),
    (
        'perf_ext',
        f'{VULCAN_PATH}/Utilities/Post_Process_codes/Perform/perf_ext'),
    (
        'fv_flux',
        f'{VULCAN_PATH}/Utilities/Post_Process_codes/Perform/FV_files'
        '/fv_flux'),
    ('gci_ext', f'{VULCAN_PATH}/Utilities/Post_Process_codes/GCI/gci_ext'),
    ('propatch', f'{VULCAN_PATH}/Utilities/Profile_codes/propatch'),
    ('vulcan_prof', f'{VULCAN_PATH}/Utilities/Profile_codes/vulcan_prof'),
    (
        'vulcan_prof_mrg',
        f'{VULCAN_PATH}/Utilities/Profile_codes/vulcan_prof_mrg'),
    ('vulcan_rest', f'{VULCAN_PATH}/Utilities/Restart_codes/vulcan_rest'),
    ('fluct_ext', f'{VULCAN_PATH}/Utilities/LES_tools/fluct_ext'),
    ('patcher', f'{VULCAN_PATH}/Executable/VULCAN_pchutl'),
    ('VULCAN-CFD', f'{VULCAN_PATH}/Utilities/Tcltk/GUI/VULCAN-CFD'),
    ('vulcanig', f'{VULCAN_PATH}/Utilities/Tcltk/GUI/vulcan_input_gui'),
    ('blprops', f'{VULCAN_PATH}/Utilities/Tecplot_tools/blprops/blprops'),
    (
        'lam_sub-blks',
        f'{VULCAN_PATH}/Utilities/Tecplot_tools/blprops/lam_sub-blks'),
    (
        'composite',
        f'{VULCAN_PATH}/Utilities/Tecplot_tools/composite/composite'),
    (
        'makecompinp',
        f'{VULCAN_PATH}/Utilities/Tecplot_tools/composite'
        '/makecompositeinput.csh'),
    (
        'massflow3d',
        f'{VULCAN_PATH}/Utilities/Tecplot_tools/massflow3d/massflow3d.csh'),
    (
        'merge1dzones',
        f'{VULCAN_PATH}/Utilities/Tecplot_tools/merge1dzones.csh'),
    (
        'nozinflow2d',
        f'{VULCAN_PATH}/Utilities/Tecplot_tools/nozinflow2d/nozinflow2d'),
    (
        'nozinflow3d',
        f'{VULCAN_PATH}/Utilities/Tecplot_tools/nozinflow3d/nozinflow3d'),
    ('surf1d', f'{VULCAN_PATH}/Utilities/Tecplot_tools/surf1d/surf1d.csh'),
    ('tecinfo', f'{VULCAN_PATH}/Utilities/Tecplot_tools/tecinfo.csh'),
    (
        'tectopprf',
        f'{VULCAN_PATH}/Utilities/Tecplot_tools/tectopprf/tectopprf'),
    ('vpp', f'{VULCAN_PATH}/Utilities/Tecplot_tools/vpp/vpp.csh'),
    ('vtls', f'{VULCAN_PATH}/Utilities/Tecplot_tools/vtls.csh'),
    ('vuln', f'{VULCAN_PATH}/Utilities/Tecplot_tools/vuln.csh'),
)


# Generate the text of a case-local ``vulcan.tcshrc`` file
def vulcan_tcshrc_text() -> str:
    r"""Create contents of a ``vulcan.tcshrc`` startup file

    :Call:
        >>> txt = vulcan_tcshrc_text()
    :Outputs:
        *txt*: :class:`str`
            Text of the ``tcsh`` startup file
    """
    # Header
    txt = (
        "#!/bin/tcsh\n"
        "#\n"
        "# Written by CAPE (cape.pyvul); do not edit.\n"
        "# Defines the VULCAN-CFD tcsh aliases for the tcsh\n"
        "# subprocesses launched by the 'vulcan' command.\n"
        "#\n"
        "# Aliases from the 'vulcan/2026-09-21' modulefile\n"
        "# on NAS Aitken (see cape.pyvul.tcshrc.VULCAN_ALIASES).\n"
        "#\n"
    )
    # Write the aliases
    for name, value in VULCAN_ALIASES:
        # Protect any single quotes from the tcsh alias quoting
        value = value.replace("'", r"'\''")
        txt += f"alias {name} '{value}'\n"
    # Output
    return txt


# Write the case-local ``vulcan.tcshrc`` file
def write_vulcan_tcshrc(fname: str | None = TCSHRC_NAME) -> str:
    r"""Write ``vulcan.tcshrc`` into a case folder

    :Call:
        >>> fname = write_vulcan_tcshrc(fname=None)
    :Inputs:
        *fname*: ``None`` | :class:`str`
            Name of file to write
    :Outputs:
        *fname*: :class:`str`
            Full name of the file written
    """
    # Apply defaults
    fname = TCSHRC_NAME if (fname is None) else fname
    # Generate and write
    with open(fname, "w") as f:
        f.write(vulcan_tcshrc_text())
    # Output
    return fname


# Get the VULCAN install root
def get_vulcan_root(environ: Optional[dict] = None) -> Optional[str]:
    r"""Get the VULCAN install root folder

    This is the folder that contains ``Executable``, ``Scripts``,
    ``Data_base``, and so on.

    :Call:
        >>> vroot = get_vulcan_root(environ=None)
    :Inputs:
        *environ*: {``None``} | :class:`dict`
            Environment to search, defaults to :class:`os.environ`
    :Outputs:
        *vroot*: :class:`str` | ``None``
            Path of the VULCAN root folder, or ``None`` if unknown
    """
    # Select environment
    env = os.environ if environ is None else environ
    # Check environment variables set by the module
    for key in ("VULCAN_ROOT", "vulcanpath", "version_path"):
        vroot = env.get(key)
        if vroot:
            if key == "version_path":
                vroot = os.path.join(vroot, "Vulcan")
            return vroot
    # Fall back to the ``vulcan`` script on the path
    exe = shutil.which("vulcan")
    if exe:
        fdir = os.path.dirname(os.path.realpath(exe))
        if os.path.basename(fdir) == "Scripts":
            return os.path.dirname(fdir)
    # Nothing found
    return None


# Build the environment for a VULCAN call
def get_vulcan_env(
        dirname: str = ".",
        nproc: Optional[int] = None,
        environ: Optional[dict] = None) -> dict:
    r"""Create the environment for running VULCAN-CFD

    :Call:
        >>> env = get_vulcan_env(dirname=".", nproc=None, environ=None)
    :Inputs:
        *dirname*: :class:`str`
            Case folder containing ``vulcan.tcshrc``
        *nproc*: {``None``} | :class:`int`
            Number of CPUs to request, written to ``CAPE_NUM_CPUS``
        *environ*: {``None``} | :class:`dict`
            Base environment, defaults to a copy of
            :class:`os.environ`
    :Outputs:
        *env*: :class:`dict`
            Environment for :func:`cape.cfdx.cmdrun.callf`
    """
    # Copy the base environment
    env = dict(os.environ if environ is None else environ)
    # Point ``~/.tcshrc`` at the case's aliases
    env[CAPE_TCSHRC_VAR] = os.path.join(os.path.abspath(dirname), TCSHRC_NAME)
    # Number of CPUs for this run
    if nproc is not None:
        env["CAPE_NUM_CPUS"] = str(int(nproc))
    # VULCAN locations
    vroot = get_vulcan_root(environ=environ)
    if vroot:
        env["VULCAN_ROOT"] = vroot
        env["CAPE_VULCAN_CMND"] = os.path.join(vroot, "Executable", "vulcan")
    # Output
    return env


# Install the ``CAPE_TCSHRC`` source line in ``~/.tcshrc``
def update_user_tcshrc(fname: Optional[str] = None) -> bool:
    r"""Ensure the user's ``~/.tcshrc`` sources ``$CAPE_TCSHRC``

    The following lines are appended to (or used to create) the
    startup file unless an equivalent ``if ( $?CAPE`` line is already
    present:

    .. code-block:: csh

        if ( $?CAPE_TCSHRC ) then
            source "$CAPE_TCSHRC"
        endif

    :Call:
        >>> didit = update_user_tcshrc(fname=None)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of startup file, defaults to ``~/.tcshrc``
    :Outputs:
        *didit*: :class:`bool`
            ``True`` if the file was created or modified
    """
    # Default file name
    if fname is None:
        fname = os.path.join(os.path.expanduser("~"), ".tcshrc")
    # Read any existing contents
    txt = ""
    if os.path.isfile(fname):
        with open(fname) as f:
            txt = f.read()
    # Check for an existing hook
    if REGEX_TCSHRC_SNIPPET.search(txt):
        return False
    # Append the snippet
    if txt and not txt.endswith("\n"):
        txt += "\n"
    with open(fname, "w") as f:
        f.write(txt + "\n" + TCSHRC_SNIPPET)
    # Report the modification
    return True

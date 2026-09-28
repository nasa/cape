r"""
:mod:`cape.pyvul.cntl`: VULCAN-CFD control module
===================================================

This module provides tools to setup and run VULCAN-CFD in CAPE.

    .. code-block:: pycon

        >>> import cape.pyvul.cntl
        >>> cntl = cape.pyvul.cntl.Cntl("pyVul.json")
        >>> cntl
        <cape.pyvul.Cntl(nCase=892)>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'


An instance of this :class:`cape.pyvul.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and the appropriate input files (such as
``cntl.VulcanInp``), and possibly others.

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*               :class:`cape.pyvul.runmatrix.RunMatrix`
    *cntl.opts*            :class:`cape.pyvul.options.Options`
    *cntl.tri*             :class:`cape.trifile.Tri`
    *cntl.inp*             :class:`cape.pyvul.inpfile.VulcanInpFile`
    ====================   =============================================

Finally, the :class:`cape.pyfun.cntl.Cntl` class is subclassed from the
:class:`cape.cfdx.cntl.Cntl` class, so any methods available to the CAPE
class are also available here.

"""

# Standard library
import os
import re
import shutil
from math import cos, radians, sin, sqrt
from typing import Optional

# Third-party modules
import numpy as np

# Local imports
from . import options
from . import casecntl
from . import tcshrc
from .inpfile import VulcanInpFile
from ..cfdx import cntl
from ..gruvoc.umesh import Umesh
from ..gruvoc.ugridfile import get_ugrid_mode, get_ugrid_mode_fname

# Regular expression to parse a slice
REGEX_SLICE = re.compile(r"(?P<a>[0-9]+)([-:](?P<b>[0-9]+))?")
REGEX_IS_SLICE = re.compile(r"[0-9]+([,:-][0-9]+)*")

# FUN3D ``mapbc`` wall boundary condition numbers
# (from ``cape.pyfun.cntl``)
BCS_WALL = (4000, 4100, 4110)
BCS_INVISCID_WALL = (3000,)
BCS_SYMMETRY = (5000, 5051, 5052)
BCS_FARFIELD = (5050,)

# Map FUN3D ``mapbc`` BC numbers to VULCAN-CFD BC group types
# (the middle column is ignored by VULCAN itself, but it follows FUN3D
# conventions in CAPE run matrices)
BC_NUM_TYPE_MAP = {
    1000: 'FIX_IN',  # Freestream
    2000: 'EXTRAP',  # Outflow
    3000: 'IWALL',   # Inviscid (slip) wall
    4000: 'AWALL',   # No-slip (adiabatic) wall
    4100: 'AWALL',   # No-slip (adiabatic) wall
    4110: 'AWALL',   # No-slip (adiabatic) wall
    5000: 'SYMM',    # Symmetry plane
    5050: 'CHAR_REF',  # Farfield (external state = reference state)
    5051: 'SYMM',    # Symmetry plane (weak)
    5052: 'SYMM',    # Symmetry plane (strong)
}

# Fallback map for BC families not listed above (by 1000s digit)
BC_FAMILY_TYPE_MAP = {
    1: 'FIX_IN',   # 1xxx inflows
    2: 'EXTRAP',   # 2xxx outflows
    3: 'IWALL',    # 3xxx slip walls
    4: 'AWALL',    # 4xxx no-slip walls
    5: 'SYMM',     # 5xxx symmetry/farfield
}


def MapbcBcToVulcanType(bc):
    r"""Convert a FUN3D ``mapbc`` BC number to a VULCAN BC TYPE

    :Call:
        >>> btype = MapbcBcToVulcanType(bc)
    :Inputs:
        *bc*: :class:`int`
            BC number from the middle column of a ``.mapbc`` file
    :Outputs:
        *btype*: :class:`str` | ``None``
            VULCAN BC group TYPE, or ``None`` if no mapping exists
    """
    # Exact match
    bc = int(bc)
    if bc in BC_NUM_TYPE_MAP:
        return BC_NUM_TYPE_MAP[bc]
    # Fall back to the BC family (1000s digit)
    return BC_FAMILY_TYPE_MAP.get(bc // 1000)


# Truncate a ``.mapbc`` surface name to VULCAN's group-name limit
def VulcanBCGroupName(name: str) -> str:
    r"""Truncate a surface name to VULCAN's 12-character group limit

    :Call:
        >>> vname = VulcanBCGroupName(name)
    :Inputs:
        *name*: :class:`str`
            Surface name from the ``.mapbc`` file
    :Outputs:
        *vname*: :class:`str`
            VULCAN BC group name (12 characters max)
    :Versions:
        * 2026-09-26 ``@ddalle``: v1.0
    """
    return name[:12]


# Class to read input files
class Cntl(cntl.Cntl):
    r"""Class for handling global options and setup for VULCAN-CFD

    This class is intended to handle all settings used to describe a
    group of VULCAN cases.  For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha
    sweep for each single control variable setting.

    The settings are read from a JSON file, which is robust and simple
    to read, but has the disadvantage that there is no support for
    comments. Hopefully the various names are descriptive enough not to
    require explanation.

    :Call:
        >>> cntl = Cntl(fname="pyVul.json")
    :Inputs:
        *fname*: :class:`str`
            Name of pyFun input file
    :Outputs:
        *cntl*: :class:`cape.pyfun.cntl.Cntl`
            Instance of the pyFun control class
    :Data members:
        *cntl.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cntl.x*: :class:`pyFun.runmatrix.RunMatrix`
            Values and definitions for variables in the run matrix
        *cntl.RootDir*: :class:`str`
            Absolute path to the root directory
    """
  # === Class attributes ===
    # Names
    _name = "pyvul"
    _solver = "vulcan"
    # Hooks to py{x} specific modules
    # Hooks to py{x} specific classes
    _case_cls = casecntl.CaseRunner
    _opts_cls = options.Options
    # Other settings
    _fjson_default = "pyVul.json"
    _warnmode_default = cntl.DEFAULT_WARNMODE
    _file_opts = [
        ".VulcanInpFile",
        ".Mesh.MapBCFile",
    ]
    _zombie_files = [
        "*.out",
        "*.flow",
        "*.ugrid",
    ]

  # === Init config ===
    def init_post(self):
        self.ReadVulcanInpFile()

  # === Main Input File ===
    # Read the namelist
    def ReadVulcanInpFile(self, j: int = 0, q: bool = True):
        r"""Read the :file:`fun3d.nml` file

        :Call:
            >>> cntl.ReadVulcanInpFile(j=0, q=True)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of the pyFun control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not to read to *Namelist*, else *Namelist0*
        """
        # Namelist file
        finp = self.opts.get_VulcanInpFile(j)
        # Check for empty value
        if finp is None:
            return
        # Check for absolute path
        if not os.path.isabs(finp):
            # Use path relative to JSON root
            finp = os.path.join(self.RootDir, finp)
        # Read the file
        inp = VulcanInpFile(finp)
        # Save it.
        if q:
            # Read to main slot for modification
            self.inp = inp
        else:
            # Template for reading original parameters
            self.inp0 = inp

    # Get namelist var
    def GetNamelistVar(self, sec, key, j=0):
        r"""Get a namelist variable's value

        The JSON file overrides the value from the namelist file

        :Call:
            >>> val = cntl.GetNamelistVar(sec, key, j=0)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *sec*: :class:`str`
                Name of namelist section
            *key*: :class:`str`
                Variable to read
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *val*::class:`int`|:class:`float`|:class:`str`|:class:`list`
                Value
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
        """
        # Get the namelist value.
        nml = getattr(self, "Namelist", None)
        nval = None if nml is None else nml.get_opt(sec, key)
        # Check for options value.
        if nval is None:
            # No namelist file value
            return self.opts.get_namelist_var(sec, key, j)
        elif 'Fun3D' not in self.opts:
            # No namelist in options
            return nval
        elif sec not in self.opts['Fun3D']:
            # No corresponding options section
            return nval
        elif key not in self.opts['Fun3D'][sec]:
            # Value not specified in the options namelist
            return nval
        else:
            # Default to the options
            return self.opts.get_namelist_var(sec, key, j)

    # Get the project rootname
    def GetProjectRootName(self, j: int = 0) -> str:
        r"""Get the project root name

        Unlike FUN3D, VULCAN-CFD has no project rootname setting in
        the input file, so this is a CAPE-only option given by the
        ``"ProjectRootname"`` setting of the ``"RunControl"`` section,
        which defaults to ``"vulcan"``.

        :Call:
            >>> name = cntl.GetProjectRootName(j=0)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of global pyVul settings object
            *j*: :class:`int`
                Phase number
        :Outputs:
            *name*: :class:`str`
                Project root name
        :Versions:
            * 2026-09-25 ``@ddalle``: v1.0; RunControl-based
        """
        # Get the options value; default "vulcan"
        name = self.opts.get_ProjectRootname(j)
        # Output
        if name is None:
            return "vulcan"
        return name

    # Get the grid format
    def GetGridFormat(self, j=0):
        r"""Get the grid format

        The JSON file overrides the value from the namelist file

        :Call:
            >>> fmt = cntl.GetGridFormat(j=0)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *fmt*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
        """
        return self.GetNamelistVar('raw_grid', 'grid_format', j)

  # === Case ===
    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self, v: bool = False) -> bool:
        return False

    # Check for a failure
    @cntl.run_rootdir
    def CheckError(self, i: int) -> bool:
        r"""Check if a case has a failure

        :Call:
            >>> q = cntl.CheckError(i)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                VULCAN control interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`FAIL` file in it
        """
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the FAIL file.
        q = os.path.isfile(os.path.join(frun, 'FAIL'))
        # Check for manual marker
        q = q or self.x.ERROR[i]
        # Output
        return q

  # === Mesh ===
    # Function to check if the mesh for case *i* is prepared
    def CheckMesh(self, i):
        r"""Check if the mesh for case *i* is prepared

        :Call:
            >>> q = cntl.CheckMesh(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Index of the case to check
        :Outputs:
            *q*: :class:`bool`
                Whether or not the mesh for case *i* is prepared
        """
        # Check input
        if not type(i).__name__.startswith("int"):
            raise TypeError("Case index must be an integer")
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get the group name.
        fgrp = self.x.GetGroupFolderNames(i)
        frun = self.x.GetFolderNames(i)
        # Go safely to root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Check for the group folder.
        if not os.path.isdir(fgrp):
            os.chdir(fpwd)
            return False
        # Extract options
        opts = self.opts
        # Enter the group folder.
        os.chdir(fgrp)
        # Check for individual-folder mesh settings
        if not opts.get_GroupMesh():
            # Check for the case folder.
            if not os.path.isdir(frun):
                # No case folder; no mesh
                os.chdir(fpwd)
                return False
            # Enter the folder.
            os.chdir(frun)
        # Check for mesh files
        q = self.CheckMeshFiles()
        # Return to original folder.
        os.chdir(fpwd)
        # Output
        return q

    # Check mesh files
    def CheckMeshFiles(self, v=False):
        r"""Check for the mesh files in the present folder

        :Call:
            >>> q = cntl.CheckMeshFiles(v=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *v*: ``True`` | {``False``}
                Verbose flag
        :Outputs:
            *q*: :class:`bool`
                Whether or not the present folder has the required mesh files
        """
        # Initialize status
        q = True
        # Get list of mesh file names
        fmesh = self.GetProcessedMeshFileNames()
        # Check for presence
        for f in fmesh:
            # Check for the file
            q = q and os.path.isfile(f)
            # Verbose option
            if v and not q:
                print("    Missing mesh file '%s'" % fmesh)
        # If running AFLR3, check for tri file
        if q and self.opts.get_aflr3():
            # Project name
            fproj = self.GetProjectRootName(0)
            # Check for mesh files
            if os.path.isfile('%s.ugrid' % fproj):
                # Have a volume mesh
                q = True
            elif os.path.isfile('%s.b8.ugrid' % fproj):
                # Have a binary volume mesh
                q = True
            elif os.path.isfile('%s.lb8.ugrid' % fproj):
                # Have a little-endian volume mesh
                q = True
            elif os.path.isfile('%s.r8.ugrid' % fproj):
                # Fortran unformatted
                q = True
            elif os.path.isfile('%s.surf' % fproj):
                # AFLR3 input file
                q = True
            elif self.opts.get_intersect():
                # Check for both required inputs
                q = os.path.isfile('%s.tri' % fproj)
                q = q and os.path.isfile('%s.c.tri' % fproj)
                # Verbose flag
                if v and not q:
                    print(
                        "    Missing TRI file for INTERSECT: '%s' or '%s'"
                        % ('%s.tri' % fproj, '%s.c.tri' % fproj))
            else:
                # No surface or mesh files
                q = False
                # Verbosity option
                if v:
                    print(
                        "    Missing mesh file '%s.{%s,%s,%s,%s,%s}'"
                        % (fproj, "ugrid", "b8.ugrid", "lb8.ugrid", "r8.ugrid",
                            "surf"))
        # Output
        return q

  # === Preparation ===
   # --- General Case ---
    # Prepare the mesh for case *i* (if necessary)
    @cntl.run_rootdir
    def PrepareMesh(self, i: int):
        r"""Prepare the mesh for case *i* if necessary

        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        """
       # ---------
       # Case info
       # ---------
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get the case name
        frun = self.x.GetFullFolderNames(i)
        # Get the name of the group
        fgrp = self.x.GetGroupFolderNames(i)
        # Create case folder
        self.make_case_folder(i)
       # ------------------
       # Folder preparation
       # ------------------
        # Check for groups with common meshes.
        if self.opts.get_GroupMesh():
            # Get the group index.
            j = self.x.GetGroupIndex(i)
            # Status update
            print("  Group name: '%s' (index %i)" % (fgrp, j))
            # Enter the group folder.
            os.chdir(fgrp)
        else:
            # Status update
            print("  Case name: '%s' (index %i)" % (frun, i))
            # Enter the case folder.
            os.chdir(frun)
       # ----------
       # Copy files
       # ----------
        # Copy/link basic files
        self.copy_files(i)
        self.link_files(i)
        # Prepare warmstart files, if any
        warmstart = self.PrepareMeshWarmStart(i)
        # Finish if case was warm-started
        if warmstart:
            return
        # Option to linke instead of copying
        linkopt = self.opts.get_LinkMesh()
        # Get the names of the raw input files and target files
        finp = self.GetInputMeshFileNames()
        fmsh = self.GetProcessedMeshFileNames()
        # Loop through those files
        for finpj, fmshj in zip(finp, fmsh):
            # Original and final file names
            f0 = os.path.join(self.RootDir, finpj)
            f1 = fmshj
            # Copy fhe file.
            if os.path.isfile(f0) and not os.path.isfile(f1):
                if linkopt:
                    os.symlink(f0, f1)
                else:
                    shutil.copyfile(f0, f1)
       # ------------------
       # BC tag prep
       # ------------------
        # Renumber grid boundary tags to match ``.mapbc`` row order
        self.PrepareGridBCTags()
       # ------------------
       # Triangulation prep
       # ------------------
        # Prepare surface triangulation for AFLR3 if appropriate
        self.PrepareMeshTri(i)

    # Prepare a case
    @cntl.run_rootdir
    def PrepareCase(self, i: int):
        r"""Prepare a case for running if it is not already prepared

        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Index of case to prepare/analyze
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2026-09-28 ``@ddalle``: v1.1; ``vulcan.tcshrc`` &
              ``~/.tcshrc`` hook for tcsh subprocesses
        """
        # Ensure ``~/.tcshrc`` sources ``$CAPE_TCSHRC`` so that the
        # tcsh subprocesses launched by ``vulcan`` get their aliases,
        # even for cases that are already prepared
        tcshrc.update_user_tcshrc()
        # Ensure case index is set
        self.opts.setx_i(i)
        # Read MapBC file fresh
        self.ReadMapBC()
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if already prepared.
        if n is not None:
            return
        # Case function
        self.CaseFunction(i)
        # Prepare the mesh (and create folders if necessary).
        self.PrepareMesh(i)
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Create folder, just in case
        self.make_case_folder(i)
        # Enter the run directory
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Different processes for GroupMesh and CaseMesh
        if self.opts.get_GroupMesh():
            # Required file names
            fmsh = self.GetProcessedMeshFileNames()
            # Copy the required files.
            for fname in fmsh:
                # Link to the present folder
                fto = fname
                # Source path
                fsrc = os.path.join(os.path.abspath('..'), fname)
                # Check for the file
                if os.path.isfile(fto):
                    os.remove(fto)
                # Create the link.
                if os.path.isfile(fsrc):
                    os.symlink(fsrc, fto)
        # Get function for setting boundary conditions, etc.
        keys = self.x.GetKeysByType('CaseFunction')
        # Get the list of functions.
        funcs = [self.x.defns[key]['Function'] for key in keys]
        # Reread namelist
        self.ReadVulcanInpFile()
        # Loop through the functions.
        for (key, func) in zip(keys, funcs):
            # Form args and kwargs
            a = (self, self.x[key][i])
            kw = dict(i=i)
            # Apply it
            self.exec_modfunction(func, a, kw, name="RunMatrixCaseFunction")
        # Write the cntl.nml file(s).
        self.PrepareVulcanInp(i)
        # Write a JSON file with
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Write the case-local ``vulcan.tcshrc`` file with the VULCAN
        # aliases (it is rewritten when the case runs, when the VULCAN
        # environment is guaranteed to be loaded)
        tcshrc.write_vulcan_tcshrc(os.getcwd())
        # Ensure ``~/.tcshrc`` sources ``$CAPE_TCSHRC``
        tcshrc.update_user_tcshrc()

   # --- Grid BC tags ---
    # Source mesh file name -> case mesh file name (``.lb8.ugrid``)
    def process_mesh_filename(
            self,
            fname: str,
            fproj: Optional[str] = None) -> str:
        r"""Write ASCII ``.ugrid`` sources as little-endian binary

        VULCAN-CFD reads AFLR3 grid files whose format is specified by
        the file name, and the little-endian ``.lb8.ugrid`` format is
        much faster to read and write than ASCII ``.ugrid`` on current
        computers. This method converts the processed file name of a
        plain ``.ugrid`` mesh to the corresponding ``.lb8.ugrid`` name.
        The file is (re)written in the appropriate format by
        :func:`PrepareGridBCTags`.

        :Call:
            >>> fname2 = cntl.process_mesh_filename(fname)
            >>> fname2 = cntl.process_mesh_filename(fname, fproj)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of VULCAN-CFD control class
            *fname*: :class:`str`
                Name of source mesh file
            *fproj*: ``None`` | :class:`str`
                Project root name override
        :Outputs:
            *fname2*: :class:`str`
                Case mesh file name
        """
        # Defer to generic CFD method
        fname = super().process_mesh_filename(fname, fproj)
        # Only ASCII .ugrid files are converted
        if fname.endswith('.ugrid') and not fname.endswith((
                ".b4.ugrid", ".b8.ugrid", ".r4.ugrid", ".r8.ugrid",
                ".lb4.ugrid", ".lb8.ugrid", ".lr4.ugrid", ".lr8.ugrid")):
            fname = fname[:-len('.ugrid')] + '.lb8.ugrid'
        return fname

    # Renumber grid boundary tags to match the ``.mapbc`` file order
    def PrepareGridBCTags(self):
        r"""Renumber grid boundary tags to ``.mapbc`` row order 1...N

        VULCAN-CFD assigns the *k*\ th ``BC GROUPS`` entry in the input
        file to grid boundary tag *k*, so grid tags must be the
        sequential integers 1...N. FUN3D-style ``.ugrid`` files often
        have non-sequential boundary tags, so each boundary tag is
        renumbered to the (1-based) row number of its surface in the
        ``.mapbc`` file. The grid file(s) in the current folder are
        rewritten in the format implied by their name (e.g. binary
        ``.lb8.ugrid``).

        Grids whose tags are already the sequential integers 1...N and
        whose file format already matches their name are left
        untouched, which makes this operation idempotent. Symbolic
        links are replaced by actual files so that the source mesh
        files are never modified.

        :Call:
            >>> cntl.PrepareGridBCTags()
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of VULCAN-CFD control class
        """
        # Reread the source mapbc file so raw tag numbering is used
        # even if *self.MapBC* was already prepared
        self.ReadMapBC()
        # Check for mapbc interface
        mapbc = getattr(self, "MapBC", None)
        if mapbc is None:
            return
        # Map of raw tag value -> mapbc row number (1-based)
        remap = {int(tag): k + 1 for k, tag in enumerate(mapbc.CompID)}
        # Process each mesh file in the present folder
        for fname in self.GetProcessedMeshFileNames():
            # Only ugrid files have AFLR3-style boundary tags
            if not fname.endswith('.ugrid'):
                continue
            # Check for the file
            if not os.path.isfile(fname):
                continue
            # Detected format of contents vs format implied by name
            cmode = get_ugrid_mode(fname)
            tmode = get_ugrid_mode_fname(fname)
            # Read the mesh (through links, if any)
            mesh = Umesh(fname)
            # Whether any tags were renumbered
            renum = False
            # Loop through boundary tag slots
            for attr in ("tri_ids", "quad_ids"):
                # Get tags
                tags = getattr(mesh, attr, None)
                if tags is None:
                    continue
                tags = np.asarray(tags)
                if tags.size == 0:
                    continue
                # Check if already sequential
                utags = np.unique(tags)
                if utags.size == len(mapbc.Names) and np.array_equal(
                        utags, np.arange(1, utags.size + 1)):
                    continue
                # Renumber the tags
                try:
                    newtags = np.array(
                        [remap[int(tag)] for tag in tags],
                        dtype=tags.dtype)
                except KeyError as err:
                    raise ValueError(
                        "Grid '%s' has boundary tag %s not found in"
                        " mapbc file" % (fname, err))
                setattr(mesh, attr, newtags)
                renum = True
                # Status update
                print(
                    "  Renumbered %i boundary tags of '%s' to"
                    " mapbc row order" % (utags.size, fname))
            # Skip if format already matches name and tags unchanged
            if (cmode.fmt == tmode.fmt) and not renum:
                continue
            # Replace symbolic links with actual files
            if os.path.islink(fname):
                os.remove(fname)
            # Rewrite the grid in the format implied by its name
            mesh.write(fname, fmt=tmode.fmt)

    # Renumber the mapbc component IDs to the (sequential) row numbers
    def PrepareMapBCTags(self):
        r"""Renumber ``mapbc`` tag numbers to row order 1...N

        This makes the ``.mapbc`` file written to each case folder
        consistent with the renumbered grid boundary tags; see
        :func:`PrepareGridBCTags`. The numbering is only applied if a
        ``.mapbc`` file and at least one grid file are available. It
        should be called *after* :func:`PrepareMapBC`, which looks up
        ``.mapbc`` rows by their original tag values.

        :Call:
            >>> cntl.PrepareMapBCTags()
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of VULCAN-CFD control class
        """
        # Check for mapbc interface
        mapbc = getattr(self, "MapBC", None)
        if mapbc is None:
            return
        # Check for a grid file to renumber along with
        fmsh = self.GetProcessedMeshFileNames()
        if not any(fname.endswith('ugrid') for fname in fmsh):
            return
        # Renumber to the row numbers
        mapbc.CompID = np.arange(1, len(mapbc.CompID) + 1)

   # --- Input File ---
    # Function to prepare "vulcan.inp" files
    @cntl.run_rootdir
    def PrepareVulcanInp(self, i: int):
        r"""Prepare and write ``vulcan.inp`` for case *i*

        :Call:
            >>> cntl.PrepareVulcanInp(i)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of VULCAN-CFD control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2026-09-25 ``@ddalle``: v1.0
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Reread the template input file
        self.ReadVulcanInpFile()
        # Reread the mapbc file fresh
        self.ReadMapBC()
        # Get the case folder name
        frun = self.x.GetFullFolderNames(i)
        # File name
        fout = os.path.join(frun, '%s.mapbc' % self.GetProjectRootName(0))
        # Customize mapbc file
        self.PrepareMapBC()
        # Renumber mapbc tags to match the (renumbered) case grid
        self.PrepareMapBCTags()
        # Reset "BC GROUPS" based on the actual mapbc contents
        self.PrepareVulcanBoundaryConditions()
        # Point to the actual case grid file
        self.PrepareVulcanInpGrid()
        # Set up the component force & moment tracking
        self.PrepareVulcanInpConfig()
        # Set the flight conditions (incl. the ``FIX_IN`` state)
        self.PrepareVulcanInpFlightConditions(i)
        # Write the BC file
        self.MapBC.Write(fout)
        # Make folder if necessary
        self.make_case_folder(i)
        # Apply any input file functions
        self.VulcanInpFunction(i)
        # Phase-end iteration checkpoints
        phb = self.GetPhaseBreaks()
        # Loop through input sequence
        for k, j in enumerate(self.opts.get_PhaseSequence()):
            # Iterations to run during this phase; VULCAN runs NITSF
            # additional iterations (added to the restart counter) so
            # later phases use the difference of consecutive break
            # points rather than the cumulative total
            nitsf = None if k >= len(phb) else \
                phb[k] - (phb[k-1] if k else 0)
            # Phase-specific settings: iteration count & restart read
            self.PrepareVulcanInpPhase(
                j=j, nitsf=nitsf, restart=(k > 0))
            # Name of output file
            fout = os.path.join(frun, 'vulcan.%02i.inp' % j)
            # Write the input file
            self.inp.write(fout)

    # Set the grid file name in the input file
    def PrepareVulcanInpGrid(self):
        r"""Point the ``UNS GRID``/``STR GRID`` line to the case grid

        VULCAN reads the grid file name from the line following the
        ``UNS GRID`` keyword; CAPE copies (or links) the user-specified
        mesh into the run folder with the project root name, so the
        template's grid file name is replaced by that file name.

        :Call:
            >>> cntl.PrepareVulcanInpGrid()
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of VULCAN-CFD control class
        :Versions:
            * 2026-09-26 ``@ddalle``: v1.0
        """
        # Check for input file
        inp = getattr(self, "inp", None)
        if inp is None:
            return
        # Only relevant for files with a grid keyword
        if ("UNS GRID" not in inp) and ("STR GRID" not in inp):
            return
        # Get the case mesh file names
        fmsh = self.GetProcessedMeshFileNames()
        if not fmsh:
            return
        # Set the grid file name (relative to the case folder)
        inp.set_gridfile('./' + fmsh[0])

    # Set phase-specific region spec entries
    def PrepareVulcanInpPhase(
            self,
            j: int = 0,
            nitsf: Optional[int] = None,
            restart: bool = False):
        r"""Set phase-specific iteration and restart controls

        For each elliptic region, the ``NITSF`` iteration count is set
        to the number of iterations to perform during phase *j* (VULCAN
        adds it to the iteration counter read from the restart file),
        and the ``REG-RES`` column of the linear-solver row is used to
        control whether restart files are read (``N`` for the first
        phase, ``Y`` for later phases).

        :Call:
            >>> cntl.PrepareVulcanInpPhase(j=0, nitsf=None, restart=False)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of VULCAN-CFD control class
            *j*: ``None`` | :class:`int`
                Phase number
            *nitsf*: ``None`` | :class:`int`
                Iteration count for phase *j*
            *restart*: {``False``} | ``True``
                Whether this phase reads restart files
        """
        # Check for input file
        inp = getattr(self, "inp", None)
        if inp is None:
            return
        regs = inp.regions
        # Loop through the regions
        for reg in regs.values():
            # Iteration count on the FMG row
            fmg = None
            for key in reg:
                if key.startswith('FMG'):
                    fmg = reg[key]
                    break
            if (fmg is not None) and (nitsf is not None):
                fmg['NITSF'] = nitsf
            # Restart toggle on the solver-scheme row
            scheme = reg.get('SCHEME')
            if (scheme is not None) and ('REG-RES' in scheme.columns):
                scheme['REG-RES'] = 'Y' if restart else 'N'

    # Apply customizations to ``.mapbc`` file
    def PrepareMapBC(self):
        r"""Customize MapBC file baed on ``"MapBC"`` section

        :Call:
            >>> cntl.PrepareMapBC()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2025-04-27 ``@ddalle``: v1.0
            * 2025-05-22 ``@ddalle``: v1.1; use MapBC IDs
        """
        # Get "MapBC" options
        bcopts = self.opts.get("MapBC", {})
        # Check for mapbc
        mapbc = getattr(self, "MapBC", None)
        # Exit if none
        if mapbc is None or bcopts is None:
            return
        # Loop through
        for surf, bc in bcopts.items():
            # Get component IDs (based on grid)
            compids = self.GetConfigBody(surf)
            # Set the BC for each
            for compid in compids:
                # Set it
                mapbc.SetBC(compid, bc)

    # Set up ``BC OBJECTS`` for the force & moment components
    def PrepareVulcanInpConfig(self):
        r"""Add ``BC OBJECTS`` entries for the requested components

        This is the VULCAN-CFD analogue of
        :func:`cape.pyfun.cntl.Cntl.PrepareNamelistConfig`: each
        component listed in the ``"Components"`` option of the
        ``"Config"`` section gets a ``BC OBJECTS`` entry whose members
        are the names of the ``.mapbc`` faces in that component's
        branch of the configuration tree. VULCAN defines families by
        name rather than by number, so no renumbering is required.

        :Call:
            >>> cntl.PrepareVulcanInpConfig()
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of VULCAN-CFD control class
        :Versions:
            * 2026-09-25 ``@ddalle``: v1.0
            * 2026-09-26 ``@ddalle``: v1.1; drop stale objects, count
        """
        # Get the components
        comps = self.opts.get_ConfigComponents()
        # Exit if no components
        if comps is None:
            return
        comps = list(np.array(comps).flatten())
        if len(comps) == 0:
            return
        # Check for input file
        inp = getattr(self, "inp", None)
        if inp is None:
            return
        # Read the configuration tree if necessary
        self.ReadConfig()
        # Object interface
        bco = inp.bcobjects
        # Map of mapbc surface name -> VULCAN group name (if built)
        bcmap = getattr(self, "_vulcan_bc_map", {})
        # Loop through components
        for comp in comps:
            # Get the family member names
            members = self.GetConfigFamilyNames(comp)
            # Warn & skip if the component is not in the mesh
            if len(members) == 0:
                print(
                    "     Component '%s' has no matches in mapbc file"
                    % comp)
                continue
            # Map member names to the VULCAN group names
            members = [bcmap.get(mm, VulcanBCGroupName(mm)) for mm in members]
            # Set the members of this BC object
            bco.set_members(comp, members)
        # Drop objects whose members are not all current BC groups
        bcg = inp.bcgroups
        dropped = []
        for name in list(bco.keys()):
            members = bco[name]
            if any(mm not in bcg for mm in members):
                dropped.append((name, [mm for mm in members if mm not in bcg]))
                dict.__delitem__(bco, name)
                bco._rawlines.pop(name, None)
                bco._dirty.discard(name)
        for name, missing in dropped:
            print(
                "     Dropping BC object '%s': member(s) %s not in"
                " 'BC GROUPS'" % (name, ', '.join(missing)))
        # Update the object count
        inp["BCOBJECTS"] = float(len(bco))

    # Reset the BC groups table from the mapbc file
    def PrepareVulcanBoundaryConditions(self):
        r"""Reset ``BC GROUPS`` based on the current ``.mapbc`` contents

        The BC table of the template file may be stale (e.g. from a
        different project that reused the same ``vulcan.inp``), so the
        block is rebuilt from scratch: one group per unique ``.mapbc``
        surface name, in file order. A template group with a matching
        name keeps its TYPE, OPTION, ``BL_delta``, and constant-state
        line; stale template groups are dropped and new names get
        their TYPE from :func:`MapbcBcToVulcanType` with a
        ``PHYSICAL`` option.

        Names longer than VULCAN's 12-character limit are truncated
        (with a de-duplication suffix if needed); the mapping from
        ``.mapbc`` surface names to VULCAN group names is saved as
        *cntl._vulcan_bc_map* for use by
        :func:`PrepareVulcanInpConfig`.

        :Call:
            >>> cntl.PrepareVulcanBoundaryConditions()
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of VULCAN-CFD control class
        :Versions:
            * 2026-09-25 ``@ddalle``: v1.0
            * 2026-09-26 ``@ddalle``: v1.1; truncate names, set count
        """
        # Check for input file
        inp = getattr(self, "inp", None)
        if inp is None:
            return
        # Check for mapbc interface
        mapbc = getattr(self, "MapBC", None)
        if mapbc is None:
            print(
                "  Warning: no mapbc file found; keeping template"
                " 'BC GROUPS' as-is")
            return
        bcg = inp.bcgroups
        # Save the template groups by name
        template = {kk: vv for kk, vv in bcg.items()}
        # Get unique names in file order
        names = []
        for nn in mapbc.Names:
            if nn not in names:
                names.append(nn)
        # Start over from an empty block
        dict.clear(bcg)
        # Map of mapbc surface name -> VULCAN group name
        bcmap = {}
        used = set()
        # Rebuild one group per name
        for nn in names:
            # VULCAN group names are limited to 12 characters
            gg = VulcanBCGroupName(nn)
            if gg != nn:
                print(
                    "  Warning: BC group name '%s' exceeds VULCAN's"
                    " 12-character limit; truncated to '%s'" % (nn, gg))
            # De-duplicate truncated names
            if gg in used:
                base = gg
                ii = 1
                while True:
                    suf = "-%i" % ii
                    gg = base[:12-len(suf)] + suf
                    if gg not in used:
                        print(
                            "  Warning: duplicate BC group name after"
                            " truncation; renamed to '%s'" % gg)
                        break
                    ii += 1
            used.add(gg)
            bcmap[nn] = gg
            # Template group with this name wins
            if nn in template:
                grp = template[nn]
                # Renaming a truncated group requires a rebuild
                if gg != nn:
                    grp.name = gg
                    grp._dirty = True
                dict.__setitem__(bcg, gg, grp)
                continue
            # Otherwise derive the TYPE from the BC number
            kk = mapbc.Names.index(nn)
            typ = MapbcBcToVulcanType(mapbc.BCs[kk])
            if typ is None:
                raise ValueError(
                    "Cannot map mapbc BC number %i for surface '%s' to"
                    " a VULCAN BC type" % (mapbc.BCs[kk], nn))
            bcg.add_group(gg, typ, options=['PHYSICAL'])
        # Save the name map for the BC OBJECTS setup
        self._vulcan_bc_map = bcmap
        # Update the group count
        inp["BCGROUPS"] = float(len(bcg))

    # Prepare freestream conditions
    def PrepareVulcanInpFlightConditions(self, i: int):
        r"""Set VULCAN input file flight conditions

        :Call:
            >>> cntl.PrepareVulcanInpFlightConditions(i)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of CAPE control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2026-09-25 ``@ddalle``: v1.0
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Check for input file
        inp = getattr(self, "inp", None)
        if inp is None:
            return
        # Get properties
        M = self.x.GetMach(i)
        a = self.x.GetAlpha(i)
        b = self.x.GetBeta(i)
        p = self.x.GetPressure(i, units="Pa")
        T = self.x.GetTemperature(i, units="K")
        # Mach number
        if M is not None:
            inp.set_mach(M)
        # Angle of attack
        if a is not None:
            inp.set_alpha(a)
        # Angle of sideslip
        if b is not None:
            inp.set_beta(b)
        # Static pressure
        if p is not None:
            inp.set_pressure(p)
        # Static temperature
        if T is not None:
            inp.set_temperature(T)
        # Apply the freestream state to the farfield BC groups
        self.PrepareVulcanInpFarfield(i)

    # Apply freestream state to farfield boundary conditions
    def PrepareVulcanInpFarfield(self, i: int):
        r"""Set the constant-state data of farfield ``FIX_IN`` groups

        The groups to update come from the ``"FarfieldComponents"``
        option of the ``"Config"`` section; if that list is empty,
        every ``FIX_IN`` group is used. The velocity components follow
        the VULCAN body axes (``x`` forward, ``y`` right wing, ``z``
        down) with
        ::

            u = V cos(alpha) cos(beta)
            v = V sin(beta)
            w = V sin(alpha)

        :Call:
            >>> cntl.PrepareVulcanInpFarfield(i)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of CAPE control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2026-09-25 ``@ddalle``: v1.0
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Check for input file & BC groups
        inp = getattr(self, "inp", None)
        if inp is None:
            return
        bcg = inp.bcgroups
        if len(bcg) == 0:
            return
        # Get the list of farfield components
        comps = self.opts.get_FarfieldComponents()
        comps = [] if comps is None else list(comps)
        # Fall back to all ``FIX_IN`` groups
        if len(comps) == 0:
            comps = bcg.find_fixin()
        # Quit if there is nothing to update
        if len(comps) == 0:
            return
        # Get freestream properties
        M = self.x.GetMach(i)
        a = self.x.GetAlpha(i)
        b = self.x.GetBeta(i)
        p = self.x.GetPressure(i, units="Pa")
        T = self.x.GetTemperature(i, units="K")
        rho = self.x.GetDensity(i, units="kg/m^3")
        V = self.x.GetVelocity(i, units="m/s")
        # Gas properties from the input file
        Rg = inp.get("GAS CONSTANT")
        if Rg is None:
            Rg = 287.052
        gam = inp.get("GAMMA")
        if gam is None:
            gam = 1.4
        # Derive the density if necessary
        if rho is None and (p is not None) and (T is not None):
            rho = p / (Rg * T)
        # Derive the velocity if necessary
        if V is None and (M is not None) and (T is not None):
            V = M * sqrt(gam * Rg * T)
        # Velocity components along VULCAN body axes
        uu = vv = ww = None
        if V is not None:
            aa = 0.0 if a is None else a
            bb = 0.0 if b is None else b
            uu = V * cos(radians(aa)) * cos(radians(bb))
            vv = V * sin(radians(bb))
            ww = V * sin(radians(aa))
        # Turbulence reference values from the input file
        tint = inp.get("TURB. INTENSITY")
        vrat = inp.get("TURB. VISC. RATIO")
        # Freestream state values
        state = {
            "density": rho,
            "uvel": uu,
            "vvel": vv,
            "wvel": ww,
            "temperature": T,
            "tint": tint,
            "vrat": vrat,
        }
        # Loop through the requested components
        for comp in comps:
            # Check for the group
            if comp not in bcg:
                raise ValueError(
                    "Farfield component '%s' not in 'BC GROUPS' of '%s'"
                    % (comp, inp.fname))
            # Skip groups without a constant-state line
            if not bcg[comp].get_state():
                print(
                    "  Warning: BC group '%s' has no constant-state"
                    " data line; skipping" % comp)
                continue
            # Apply the freestream state
            bcg[comp].set_state(state)

    # Call function to apply namelist settings for case *i*
    def VulcanInpFunction(self, i: int):
        r"""Apply a function at the end of :func:`PrepareNamelist(i)`

        This is allows the user to modify settings at a later point than
        is done using :func:`CaseFunction`

        This calls the function(s) in the global ``"NamelistFunction"``
        option from the JSON file. These functions must take *cntl* as
        an input and the case number *i*. The function(s) are usually
        from a module imported via the ``"Modules"`` option. See the
        following example:

            .. code-block:: javascript

                "Modules": ["testmod"],
                "NamelistFunction": ["testmod.nmlfunc"]

        This leads pyFun to call ``testmod.nmlfunc(cntl, i)`` near the
        end of :func:`PrepareNamelist` for each case *i* in the run
        matrix.

        :Call:
            >>> cntl.NamelistFunction(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Overall control interface
            *i*: :class:`int`
                Case number
        :Versions:
            * 2017-06-07 ``@ddalle``: v1.0
            * 2022-04-13 ``@ddalle``: v2.0; exec_modfunction()
        :See also:
            * :func:`cape.cfdx.cntl.Cntl.CaseFunction`
            * :func:`cape.pyfun.cntl.Cntl.PrepareCase`
            * :func:`cape.pyfun.cntl.Cntl.PrepareNamelist`
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get input functions
        lfunc = self.opts.get("VulcanInpFunction", [])
        # Ensure list
        lfunc = list(np.array(lfunc).flatten())
        # Loop through functions
        for func in lfunc:
            # Form args and kwargs
            a = (self, i)
            kw = dict()
            # Apply it
            self.exec_modfunction(func, a, kw, name="VulcanInpFunction")

  # === Surface IDs ===
    # Get surface ID numbers
    def CompID2SurfID(self, compID):
        r"""Convert triangulation component ID to surface index

        This relies on an XML configuration file and a FUN3D ``mapbc``
        file

        :Call:
            >>> surfID = cntl.CompID2SurfID(compID)
            >>> surfID = cntl.CompID2SurfID(face)
            >>> surfID = cntl.CompID2SurfID(comps)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *compID*: :class:`int`
                Surface boundary ID as used in surface mesh
            *face*: :class:`str`
                Name of face
            *comps*: :class:`list` (:class:`int` | :class:`str`)
                List of component IDs or face names
        :Outputs:
            *surfID*: :class:`list`\ [:class:`int`]
                List of corresponding indices of surface in MapBC
        :Versions:
            * 2016-04-27 ``@ddalle``: v1.0
        """
        # Make sure the triangulation is present.
        try:
            self.tri
        except Exception:
            self.ReadTri()
        # Get list from tri Config
        compIDs = self.tri.config.GetCompID(compID)
        # Initialize output list
        surfID = []
        # Loop through components
        for comp in compIDs:
            # Get the surface ID
            surfID.append(self.MapBC.GetSurfID(comp))
        # Output
        return surfID

    # Convert string to MapBC surfID
    def EvalSurfID(self, comp):
        r"""Convert a component name to a MapBC surface index (1-based)

        This function also works if the input, *comp*, is an integer
        (returns the same integer) or an integer string such as ``"1"``.
        Before looking up an index by name, the function attempts to
        return ``int(comp)``.

        :Call:
            >>> surfID = cntl.EvalSurfID(comp)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *comp*: :class:`str` | :class:`int`
                Component name or surface index (1-based)
        :Outputs:
            *surfID*: :class:`int`
                Surface index (1-based) according to *cntl.MapBC*
        :Versions:
            * 2017-02-23 ``@ddalle``: v1.0
        """
        # Try to convert input to an integer directly
        try:
            return int(comp)
        except Exception:
            pass
        # Check for MapBC interface
        try:
            self.MapBC
        except AttributeError:
            raise AttributeError("Interface to FUN3D 'mapbc' file not found")
        # Read from MapBC
        return self.MapBC.GetSurfID(comp)

    # Get string describing which components are in config
    def GetConfigBody(self, comp: str, warn: bool = False) -> list:
        r"""Convert face name to list of MapBC indices

        Determine which component indices are in a named component based
        on the MapBC file, which is always numbered 1,2,...,N.  Output
        the format as a nice string, such as ``"4-10,13,15-18"``.

        If possible, this is read from the ``"Inputs"`` subsection of
        the ``"Config"`` section of the master JSON file.  Otherwise,
        it is read from the ``"mapbc"`` and configuration files.

        :Call:
            >>> cids = cntl.GetConfigInput(comp, warn=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *comp*: :class:`str`
                Name of component to process
            *warn*: ``True`` | {``False``}
                Whether or not to print warnings if not raising errors
        :Outputs:
            *cids*: :class:`list`\ [:class:`str`]
                List of MapBC inds (1-based) in *comp*
        :Versions:
            * 2025-05-16 ``@ddalle``: v1.0 (from GetConfigInput())
        """
        # Initialize
        surf = []
        # Get names of all child components, including *comp*
        family = self.config.GetFamily(comp)
        # Loop through components
        for face in family:
            # Check if present
            if face not in self.MapBC.Names:
                continue
            # Get the surf from MapBC
            surfID = self.MapBC.GetSurfIndex(face, check=True, warn=False) + 1
            # If one was found, append it
            if surfID is not None:
                surf.append(surfID)
        # Check for empty
        if warn and (len(surf) == 0):
            print(f"     Component '{comp}' has no matches in mapbc file")
            return []
        # Sort the surface IDs to prepare RangeString
        surf.sort()
        return surf

    # Get mapbc names of the faces in a config component
    def GetConfigFamilyNames(self, comp: str) -> list:
        r"""List ``.mapbc`` names of the faces in a named component

        This is the name-based analogue of :func:`GetConfigBody`:
        VULCAN-CFD identifies families by name rather than number.

        :Call:
            >>> names = cntl.GetConfigFamilyNames(comp)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                CAPE main control instance
            *comp*: :class:`str`
                Name of component to process
        :Outputs:
            *names*: :class:`list`\ [:class:`str`]
                Face names of *comp* and its children that appear in
                the ``.mapbc`` file; just ``[comp]`` when there is no
                configuration tree
        :Versions:
            * 2026-09-25 ``@ddalle``: v1.0
        """
        # Check for configuration tree
        config = getattr(self, "config", None)
        if config is None:
            # Fall back to the name itself
            return [comp]
        # Get names of all child components, including *comp*
        family = config.GetFamily(comp)
        # Filter to the faces in the mapbc file, if present
        mapbc = getattr(self, "MapBC", None)
        if mapbc is not None:
            family = [ff for ff in family if ff in mapbc.Names]
        # Output
        return family

  # === Case Modification ===
    # Get case-specific number of iterations for a phase run
    def get_phase_niter(self, i: int, j: int) -> int:
        pass

    # Function to apply namelist settings to a case
    def ApplyCase(self, i: int, nPhase=None, **kw):
        r"""Apply settings from *cntl.opts* to an individual case

        This rewrites each run namelist file and the :file:`case.json`
        file in the specified directories.

        :Call:
            >>> cntl.ApplyCase(i, nPhase=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                FUN3D control interface
            *i*: :class:`int`
                Case number
            *nPhase*: {``None``} | positive :class:`int`
                Last phase number (default determined by *PhaseSequence*)
        :Versions:
            * 2016-03-31 ``@ddalle``: v1.0
        """
        # Ignore cases marked PASS
        if self.x.PASS[i] or self.x.ERROR[i]:
            return
        # Case function
        self.CaseFunction(i)
        # Read ``case.json``.
        rc = self.read_case_json(i)
        # Get present options
        rco = self.opts["RunControl"]
        # Exit if none
        if rc is None:
            return
        # Get the number of phases in ``case.json``
        nSeqC = rc.get_nSeq()
        # Get number of phases from present options
        nSeqO = self.opts.get_nSeq()
        # Check for input
        if nPhase is None:
            # Default: inherit from pyOver.json
            nPhase = nSeqO
        else:
            # Use maximum
            nPhase = max(nSeqC, int(nPhase))
        # Present number of iterations
        nIter = rc.get_PhaseIters(nSeqC)
        # Get nominal phase breaks
        PhaseIters = self.GetPhaseBreaks()
        # Loop through the additional phases
        for j in range(nSeqC, nPhase):
            # Append the new phase
            rc["PhaseSequence"].append(j)
            # Get iterations for this phase
            if j >= nSeqO:
                # Add *nIter* iterations to last phase iter
                nj = self.opts.get_nIter(j)
            else:
                # Process number of *additional* iterations expected
                nj = PhaseIters[j] - PhaseIters[j-1]
            # Set the iteration count
            nIter += nj
            rc.set_PhaseIters(nIter, j)
            # Status update
            print("  Adding phase %s (to %s iterations)" % (j, nIter))
        # Copy other sections
        for k in rco:
            # Don't copy phase and iterations
            if k in ["PhaseIters", "PhaseSequence"]:
                continue
            # Otherwise, overwrite
            rc[k] = rco[k]
        # Write it
        self.WriteCaseJSON(i, rc=rc)
        # Write the conditions to a simple JSON file
        self.WriteConditionsJSON(i)
        # (Re)Prepare mesh in case needed
        print("  Checking mesh preparations")
        self.PrepareMesh(i)
        # Rewriting phases
        print("  Writing input namelists 0 to %s" % (nPhase-1))
        self.PrepareVulcanInp(i)
        # Write PBS scripts
        nPBS = self.opts.get_nPBS()
        print("  Writing PBS scripts 0 to %s" % (nPBS-1))
        self.WritePBS(i)

  # === Case Interface ===
    # Read a namelist from a case folder
    def ReadCaseVulcanInp(self, i: int, j: int | None = None) -> VulcanInpFile:
        r"""Read inputs from case *i*, phase *j* if possible

        :Call:
            >>> inp = cntl.ReadCaseVulcanInp(i, rc=None, j=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of CAPE control class
            *i*: :class:`int`
                Run index
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
        :Outputs:
            *inp*: ``None`` | :class:`VulcanInpFile`
                Namelist interface is possible
        """
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # If no case, abort
        if runner is None:
            return
        # Read the namelist
        return runner.read_vulcan_inp(j=j)


r"""
:mod:`cape.cfdx.triqfm`: Module for reading patch loads from one case
=======================================================================



"""


# Standard library
import os

# Third-party
import numpy as np

# Local imports
from .. import pltfile
from .cntlbase import CntlBase
from ..trifile import Tri, Triq
from ..dkit.rdb import DataKit


# Constants
DEG = np.pi / 180.0


# Main class for TriqFM cases
class CaseTriqFM(DataKit):
   # --- Config ---
    def __init__(self, comp: str, ftriq: str, cntl: CntlBase, i: int):
        # Save the run matrix controller
        self.cntl = cntl
        # List of columns
        self.cols = []
        # Save the component name
        self.comp = comp
        # Save the name of the TriQ file
        self.ftriq = ftriq
        # Save case index
        self.i = i
        # Save list of patches
        self.patches = cntl.opts.get_DataBookOpt(comp, "Patches")
        # Initialize other slots
        self.compmap = None
        self.tri = None
        self.triq = None
        # Analyze
        self.get_triq_forces()
        # Process output file
        self.write_triq()

   # --- Raw data ---
    # Read the surface solution data from this case
    def read_triq(self) -> Triq:
        r"""Read the surface solution data from a ``.triq`` file

        :Call:
            >>> triq = db.read_triq()
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
        :Outputs:
            *triq*: :class:`cape.trifile.Triq`
                Triangulated surface data (also stored in *db.triq*)
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Check if already read
        if self.triq is not None:
            return self.triq
        # Check if file exists
        if not os.path.isfile(self.ftriq):
            return
        # Get configuration file
        fcfg = self.get_configfile()
        # Read it
        self.triq = Triq(self.ftriq, c=fcfg)
        # Return it for convenience
        return self.triq

    # Find the configuration file
    def get_configfile(self) -> str:
        r"""Find configuration file for ``.triq`` file

        :Call:
            >>> fcfg = db.get_configfile()
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
        :Outputs:
            *fcfg*: :class:`str`
                Name of XML/JSON/MIXSUR config file
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Get options handle
        opts = self.cntl.opts
        # Get component-specific config file
        fcfg = opts.get_DataBookOpt(self.comp, "ConfigFile")
        fcfg = fcfg if fcfg else opts.get_ConfigFile()
        # Absolutize
        return self.cntl.abspath(fcfg)

   # --- Mapping data ---
    # Read the map file for this component
    def read_tri_map(self) -> Tri:
        r"""Read the surface triangulation mapping file

        This file defines the patches for the TriqFM output.

        :Call:
            >>> triq = db.read_triq()
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
        :Outputs:
            *triq*: :class:`cape.trifile.Triq`
                Triangulated surface data (also stored in *db.triq*)
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Check if already read
        if self.tri is not None:
            return
        # Get options handle
        opts = self.cntl.opts
        # Get the name of the tri file and configuration
        ftri = opts.get_DataBookOpt(self.comp, "MapTri")
        fcfg = opts.get_DataBookOpt(self.comp, "ConfigFile")
        # Absolutize as necessary
        ftri = self.cntl.abspath(ftri)
        fcfg = fcfg if fcfg is None else self.cntl.abspath(fcfg)
        # Read
        if os.path.isfile(ftri):
            self.tri = Tri(ftri, c=fcfg)

   # --- Map components ---
    # Map the components
    def map_tri_compid(self):
        r"""Perform any component ID mapping if necessary

        :Call:
            >>> triq = db.map_tri_compid()
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
        :Attributes:
            *DBF.compmap*: :class:`dict`
                Map of component numbers altered during the mapping
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Check if already processed
        if self.compmap is not None:
            return
        # Ensure tri is present
        self.read_tri_map()
        # Check for a tri file
        if self.tri is None:
            self.compmap = {}
            return
        # Get options handle
        opts = self.cntl.opts
        # Get map file name for STDOUT
        ftri = opts.get_DataBookOpt(self.comp, "MapTri")
        print(f"    Mapping component IDs using '{os.path.basename(ftri)}'")
        # Get tolerances
        kw = {
            "AbsTol": opts.get_DataBookOpt(self.comp, "AbsTol"),
            "CompTol": opts.get_DataBookOpt(self.comp, "CompTol"),
            "RelTol": opts.get_DataBookOpt(self.comp, "RelTol"),
            "AbsProjTol": opts.get_DataBookOpt(self.comp, "AbsProjTol"),
            "CompProjTol": opts.get_DataBookOpt(self.comp, "CompProjTol"),
            "RelProjTol": opts.get_DataBookOpt(self.comp, "RelProjTol"),
            "compID": opts.get_DataBookOpt(self.comp, "ConfigCompID"),
        }
        # Pop empty options
        for k, v in dict(kw).items():
            if v is None:
                kw.pop(k)
        # Map the component IDs
        self.compmap = self.triq.MapTriCompID(self.tri, **kw)

    # Get componnent ID (w/i map) for a patch
    def get_patch_compid(self, patch: str) -> str:
        r"""Get the comp name/number in the tri-map for one patch

        :Call:
            >>> compid = db.get_patch_compid(patch)
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
            *patch*: :class:`str`
                Name of patch
        :Outputs:
            *compid*: :class:`str` | :class:`int`
                Name or number of component defining *patch*
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Get map for MapTri components
        patchmap = self.cntl.opts.get_DataBookOpt(self.comp, "PatchMap")
        patchmap = {} if patchmap is None else patchmap
        # Default to name of patch
        return patchmap.get(patch, patch)

    # Get the component numbers of the mapped patches
    def get_patch_compids(self) -> list:
        r"""Get the list of component IDs mapped from the template *tri*

        :Call:
            >>> compids = db.get_patch_compids()
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
        :Outputs:
            *compids*: :class:`list`\ [:class:`int`]
                List of component IDs that came from the mapping file
        :Versions:
            * 2017-03-30 ``@ddalle``: v1.0
        """
        # Initialize list of Component IDs
        compids = []
        # Loop through the patches
        for patch in self.patches:
            # Get the component for this patch
            compid = self.get_patch_compid(patch)
            # Check if it's a string
            if isinstance(compid, str):
                # Get the component ID from the *triq*
                try:
                    # Get the value from *triq.config* or *triq.Conf*
                    comp = self.triq.GetCompID(compid)
                except Exception:
                    # Unknown component
                    raise ValueError(
                        f"Could not determine ID for patch '{patch}'")
                # Check if it's a list
                if isinstance(comp, (list, np.ndarray)):
                    # Check length
                    if len(comp) != 1:
                        raise ValueError(
                            f"Got multiple IDs for patch '{patch}'")
                    # Get the one element
                    compids.append(comp[0])
                else:
                    # Append the integer
                    compids.append(comp)
            else:
                # If it was specified numerically, check the *compmap*
                # If the mapping had to renumber the component, it will be
                # in this dictionary; otherwise use the compID as is.
                compids.append(self.compmap.get(compid, compid))
        # Output
        return compids

   # --- Run matrix ---
    def get_conditions(self) -> dict:
        r"""Get the freestream conditions needed for forces

        :Call:
            >>> xi = db.get_conditions()
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
        :Outputs:
            *xi*: :class:`dict`
                Conditions, incl. Mach (*mach*), Reynolds number (*Re*)
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
            * 2025-08-13 ``@ddalle``: v1.1 (dex)
        """
        # Get run matrix
        x = self.cntl.x
        i = self.i
        # Attempt to get Mach number
        try:
            # Use the trajectory
            mach = x.GetMach(i)
        except Exception:
            # No Mach number specified in run matrix
            raise ValueError(
                ("Could not determine freestream Mach number\n") +
                ("TriqFM component '%s'" % self.comp))
        # Attempt to get Reynolds number (not needed if inviscid)
        try:
            # Use the trajectory
            Rey = x.GetReynoldsNumber(i)
        except Exception:
            # Assume it's not needed
            Rey = 1.0
        # Ratio of specific heats
        gam = x.GetGamma(i)
        # Dynamic pressure
        q = x.GetDynamicPressure(i)
        # Output
        return {"mach": mach, "Re": Rey, "gam": gam, "q": q}

   # --- Force & moment integration ---
    # Get all patches
    def get_triq_forces(self):
        r"""Get the forces, moments, and other states on each patch

        :Call:
            >>> db.get_triq_forces()
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0 (``TriqFMDataBook``)
            * 2025-08-13 ``@ddalle``: v2.0 (dex)
        """
        # Initialize dictionary of composite forces
        fm = {}
        # Loop through patches
        for j, patch in enumerate(self.patches):
            # Calculate forces
            fmj = self.get_triq_forces_patch(patch)
            # Save values
            for k, v in fmj.items():
                self.save_col(f"{patch}.{k}", v)
            # Initialize or accumulate
            if j == 0:
                fm.update(fmj)
                continue
            # Combine
            for k, v in fmj.items():
                # Accumulate the value
                if k.endswith("_min"):
                    fm[k] = min(fm[k], v)
                elif k.endswith("_max"):
                    fm[k] = max(fm[k], v)
                else:
                    fm[k] += v
        # Save totals
        for k, v in fm.items():
            self.save_col(k, v)

    # Calculate forces for one patch
    def get_triq_forces_patch(self, patch: str):
        r"""Get the forces and moments on a patch

        :Call:
            >>> fm = db.get_triq_forces_patch(patch, **kw)
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
            *patch*: :class:`str`
                Name of patch
        :Outputs:
            *fm*: :class:`dict`\ [:class:`float`]
                Dictionary of force & moment coefficients
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
            * 2025-08-13 ``@ddalle``: v2.0 (dex)
        """
        # Options handle
        opts = self.cntl.opts
        # Set inputs for TriqForces
        kwfm = self.get_conditions()
        # Apply remaining options
        kwfm["Aref"] = opts.get_RefArea(self.comp)
        kwfm["Lref"] = opts.get_RefLength(self.comp)
        kwfm["bref"] = opts.get_RefSpan(self.comp)
        kwfm["MRP"]  = np.array(opts.get_RefPoint(self.comp))
        kwfm["incm"] = opts.get_DataBookMomentum(self.comp)
        kwfm["gauge"] = opts.get_DataBookGauge(self.comp)
        # Get surface data
        triq = self.read_triq()
        # Apply mapping
        self.map_tri_compid()
        # Get component for this patch
        patchID = self.get_patch_compid(patch)
        # Calculate forces
        fm = triq.GetTriForces(patchID, **kwfm)
        # Apply transformations
        self.apply_transformations(fm)
        # Output
        return fm

   # --- Transformations ---
    # Apply sequence of transformations
    def apply_transformations(self, fm: dict):
        r"""Get the forces and moments on a patch

        :Call:
            >>> db.apply_transformations(fm)
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
            *fm*: :class:`dict`\ [:class:`float`]
                Force & moment coefficients, transformed in-place
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
            * 2025-08-13 ``@ddalle``: v2.0 (dex)
        """
        # Get list of transformations
        trans = self.cntl.opts.get_DataBookOpt(self.comp, "Transformations")
        # Check if any
        if (trans is None) or not len(trans):
            return
        # Loop through them
        for topts in trans:
            fm = self.transform_fm(fm, topts)
        # Output
        return fm

    # Transform force or moment reference frame
    def transform_fm(self, fm: dict, topts: dict):
        r"""Apply transformations to one patch load

        Available transformations and their parameters are listed below.

            * "Euler321": "psi", "theta", "phi"
            * "ScaleCoeffs": "CA", "CY", "CN"

        :Call:
            >>> db.transform_fm(fm, topts)
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
            *fm*: :class:`dict`\ [:class:`float`]
                Force & moment coefficients, transformed in-place
            *topts*: :class:`dict`
                Dictionary of options for the transformation
            *x*: :class:`cape.runmatrix.RunMatrix`
                The run matrix used for this analysis
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0 (``TriqFMDataBook``)
            * 2025-08-13 ``@ddalle``: v2.0 (dex)
        """
        # Get run matrix aand case
        x = self.cntl.x
        i = self.i
        # Get the transformation type.
        ttype = topts.get("Type", "")
        # Check it.
        if ttype in ["Euler321", "Euler123"]:
            # Get the angle variable names.
            # Use same as default in case it's obvious what they should be.
            kph = topts.get('phi', 0.0)
            kth = topts.get('theta', 0.0)
            kps = topts.get('psi', 0.0)
            # Extract roll
            if not isinstance(kph, str):
                # Fixed value
                phi = kph*DEG
            elif kph.startswith('-'):
                # Negative roll angle.
                phi = -x[kph[1:]][i]*DEG
            else:
                # Positive roll
                phi = x[kph][i]*DEG
            # Extract pitch
            if not isinstance(kth, str):
                # Fixed value
                theta = kth*DEG
            elif kth.startswith('-'):
                # Negative pitch
                theta = -x[kth[1:]][i]*DEG
            else:
                # Positive pitch
                theta = x[kth][i]*DEG
            # Extract yaw
            if not isinstance(kps, str):
                # Fixed value
                psi = kps*DEG
            elif kps.startswith('-'):
                # Negative yaw
                psi = -x[kps[1:]][i]*DEG
            else:
                # Positive pitch
                psi = x[kps][i]*DEG
            # Sines and cosines
            cph = np.cos(phi)
            cth = np.cos(theta)
            cps = np.cos(psi)
            sph = np.sin(phi)
            sth = np.sin(theta)
            sps = np.sin(psi)
            # Make the matrices.
            # Roll matrix
            R1 = np.array([[1, 0, 0], [0, cph, -sph], [0, sph, cph]])
            # Pitch matrix
            R2 = np.array([[cth, 0, -sth], [0, 1, 0], [sth, 0, cth]])
            # Yaw matrix
            R3 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])
            # Combined transformation matrix.
            # Remember, these are applied backwards in order to undo the
            # original Euler transformation that got the component here.
            if ttype == "Euler321":
                R = np.dot(R1, np.dot(R2, R3))
            elif ttype == "Euler123":
                R = np.dot(R3, np.dot(R2, R1))
            # Area transformations
            if "Ay" in fm:
                # Assemble area vector
                Ac = np.array([fm["Ax"], fm["Ay"], fm["Az"]])
                # Transform
                Ab = np.dot(R, Ac)
                # Reset
                fm["Ax"] = Ab[0]
                fm["Ay"] = Ab[1]
                fm["Az"] = Ab[2]
            # Force transformations
            # Loop through suffixes
            for s in ["", "p", "vac", "v", "m"]:
                # Construct force coefficient names
                cx = "CA" + s
                cy = "CY" + s
                cz = "CN" + s
                # Check if the coefficient is present
                if cy in fm:
                    # Assemble forces
                    Fc = np.array([fm[cx], fm[cy], fm[cz]])
                    # Transform
                    Fb = np.dot(R, Fc)
                    # Reset
                    fm[cx] = Fb[0]
                    fm[cy] = Fb[1]
                    fm[cz] = Fb[2]
                # Construct moment coefficient names
                cx = "CLL" + s
                cy = "CLM" + s
                cz = "CLN" + s
                # Check if the coefficient is present
                if cy in fm:
                    # Assemble moment vector
                    Mc = np.array([fm[cx], fm[cy], fm[cz]])
                    # Transform
                    Mb = np.dot(R, Mc)
                    # Reset
                    fm[cx] = Mb[0]
                    fm[cy] = Mb[1]
                    fm[cz] = Mb[2]
        elif ttype in ["ScaleCoeffs"]:
            # Loop through coefficients.
            for c in topts:
                # Get the value.
                k = topts[c]
                # Check if it's a number
                if not isinstance(k, (float, int)):
                    # Assume they meant to flip it.
                    k = -1.0
                # Loop through suffixes
                for s in ["", "p", "vac", "v", "m"]:
                    # Construct overall name
                    cc = c + s
                    # Check if it's present
                    if cc in fm:
                        fm[cc] = k*fm[cc]
        else:
            raise IOError(
                "Transformation type '%s' is not recognized." % ttype)
        # Output for clarity
        return fm

   # --- Surface Output ---
    # Function to write TRIQ file if requested
    def write_triq(self, **kw):
        r"""Write mapped solution as TRIQ or Tecplot file with zones

        :Call:
            >>> db.write_triq(**kw)
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
        :Versions:
            * 2017-03-30 ``@ddalle``: v1.0 (``TriqFMDataBook``)
            * 2025-08-13 ``@ddalle``: v2.0 (dex)
        """
        # Get options handle
        opts = self.cntl.opts
        # Get option for output
        if not opts.get_DataBookOpt(self.comp, "OutputSurface"):
            return
        # Get the output file type
        fmt = opts.get_DataBookOpt(self.comp, "OutputFormat")
        # Check the option
        if fmt is None:
            # Nothing more to do
            return
        # Get case index
        i = self.i
        # Get path to databook
        dbdir = opts.get_DataBookFolder()
        dbdir = self.cntl.abspath(dbdir)
        # Get case name for *i*
        frun = self.cntl.x.GetFullFolderNames(i)
        # Full path to output dir for surface data
        surfdir = os.path.join("triqfm", frun)
        # Create subfolder(s) as needed
        curdir = dbdir
        for part in surfdir.split(os.sep):
            # Accumulate path
            curdir = os.path.join(curdir, part)
            # Create if necessary
            if not os.path.isdir(curdir):
                os.mkdir(curdir)
        # Name of file
        fpre = opts.get_DataBookPrefix(self.comp)
        # Convert the file as needed
        if fmt.lower() in ["tri", "triq"]:
            # Down select the mapped patches
            triq = self.select_triq()
            # Get format
            triqfmt = opts.get_DataBookOpts(self.comp, "TriqFormat")
            # Write the TRIQ in this format
            fname = os.path.join(curdir, f"{fpre}.triq")
            triq.Write(fname, fmt=triqfmt)
        elif fmt.lower() == "dat":
            # Create Tecplot PLT interface
            pltq = self.triq2plt(self.triq, **kw)
            # Write ASCII file
            fname = os.path.join(curdir, f"{fpre}.dat")
            pltq.WriteDat(fname)
            # Delete it
            del pltq
        elif fmt.lower() == "plt":
            # Create Tecplot PLT interface
            pltq = self.triq2plt(self.triq, **kw)
            # Write binary file
            fname = os.path.join(curdir, f"{fpre}.plt")
            pltq.Write(fname)
            # Delete it
            del pltq

    # Select the relevant components of the mapped TRIQ file
    def select_triq(self) -> Triq:
        r"""Select the components of *triq* that are mapped patches

        :Call:
            >>> triq = db.select_triq()
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
        :Outputs:
            *triq*: :class:`cape.trifile.Triq`
                Interface to annotated surface triangulation
        :Versions:
            * 2017-03-30 ``@ddalle``: v1.0
        """
        # Get component IDs
        compIDs = self.get_patch_compids()
        # Downselect
        triq = self.triq.GetSubTri(compIDs)
        # Output
        return triq

    # Convert the TRIQ file
    def triq2plt(self, triq: Triq, **kw) -> pltfile.Plt:
        r"""Convert an annotated tri (TRIQ) interface to Tecplot (PLT)

        :Call:
            >>> plt = DBF.Triq2Plt(triq, **kw)
        :Inputs:
            *db*: :class:`CaseTriqFM`
                Interface to TriqFM data for one component of one case
            *triq*: :class:`cape.trifile.Triq`
                Interface to annotated surface triangulation
        :Outputs:
            *plt*: :class:`cape.pltfile.Plt`
                Binary Tecplot interface
        :Versions:
            * 2017-03-30 ``@ddalle``: v1.0
        """
        # Get component IDs
        compIDs = self.get_patch_compids()
        # Get freestream conditions
        if 'i' in kw:
            # Get freestream conditions
            kwfm = self.get_conditions()
            # Set those conditions
            for k in kwfm:
                kw.setdefault(k, kwfm[k])
        # Perform conversion
        pltq = pltfile.Plt(triq=triq, CompIDs=compIDs, **kw)
        # Output
        return pltq

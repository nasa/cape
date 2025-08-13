r"""
:mod:`cape.cfdx.triqfm`: Module for reading patch loads from one case
=======================================================================



"""


# Standard library
import os

# Third-party
import numpy as np

# Local imports
from .cntlbase import CntlBase
from ..trifile import Tri, Triq
from ..dkit.rdb import DataKit


# Main class for TriqFM cases
class CaseTriqFM(DataKit):
   # --- Config ---
    __slots__ = (
        "cntl",
        "comp",
        "compmap",
        "ftriq",
        "i",
        "patches",
        "tri",
        "triq",
    )

    def __init__(self, comp: str, ftriq: str, cntl: CntlBase, i: int):
        # Save the run matrix controller
        self.cntl = cntl
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
            return
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
        fcfg = self.cntl.abspath(fcfg)
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

   # --- Run matrix ---
    def get_conditions(self) -> dict:
        r"""Get the freestream conditions needed for forces

        :Call:
            >>> xi = DBF.GetConditions(i)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *i*: :class:`int`
                Case index
        :Outputs:
            *xi*: :class:`dict`
                Dictionary of Mach number (*mach*), Reynolds number (*Re*)
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
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
    def get_triq_forces_patch(self, patch: str):
        r"""Get the forces and moments on a patch

        :Call:
            >>> fm = DBF.GetTriqForces(patch, i, **kw)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *patch*: :class:`str`
                Name of patch
            *i*: :class:`int`
                Case index
        :Outputs:
            *fm*: :class:`dict`\ [:class:`float`]
                Dictionary of force & moment coefficients
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
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
        FM = self.ApplyTransformations(i, FM)
        # Get dimensional forces if requested
        FM = self.GetDimensionalForces(patch, i, FM)
        # Get additional states
        FM = self.GetStateVars(patch, FM)
        # Output
        return fm

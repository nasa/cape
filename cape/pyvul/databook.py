r"""
:mod:`cape.pyvul.databook`: Post-processing for VULCAN-CFD data
=================================================================

This module contains functions for reading and processing forces,
moments, and other statistics from cases in a run matrix.

Data book modules are also invoked during update and reporting
command-line calls.

    .. code-block:: console

        $ pyvul --aero

The available components mirror those described on the template data
book module, :mod:`cape.cfdx.databook`.  However, some data book types
may not be implemented for all CFD solvers.

VULCAN-CFD writes an iterative force & moment history file for each
boundary condition group in the :file:`BC_files` folder of each run
folder, using the ASCII Tecplot format
:file:`BC_files/{bc}.ifam_his_{N}.tec`, where *N* is the region number
(usually 1).  Wall boundaries report pressure forces (``PA<x, y, z>``),
viscous forces (``S<x, y, z>``), and moments (``M<x, y, z>``), while
inflow and outflow boundaries report total forces and flow rates.

:See Also:
    * :mod:`cape.cfdx.databook`
    * :mod:`cape.cfdx.casedata`
    * :mod:`cape.pyfun.databook`
"""

# Standard library modules
import glob
import os
from typing import Optional

# Local imports
from ..cfdx import casedata
from ..cfdx import databook
from ..dkit import tsvfile
from ..cfdx.casecntl import CaseRunner


# Column names for FM history files, ``BC_files/{bc}.ifam_his_1.tec``
COLNAMES_FM = {
    "Cycle": databook.CASE_COL_ITRAW,
    "CFL": "CFL",
    "Bleed Flow Rate [kg/s]": "mdot",
    "Mass Flow Rate [kg/s]": "mdot",
    "PA<sub>x</sub> [N]": "Fxp",
    "PA<sub>y</sub> [N]": "Fyp",
    "PA<sub>z</sub> [N]": "Fzp",
    "S<sub>x</sub> [N]": "Fxv",
    "S<sub>y</sub> [N]": "Fyv",
    "S<sub>z</sub> [N]": "Fzv",
    "F<sub>x</sub> [N]": "Fx",
    "F<sub>y</sub> [N]": "Fy",
    "F<sub>z</sub> [N]": "Fz",
    "M<sub>x</sub> [N m]": "Mx",
    "M<sub>y</sub> [N m]": "My",
    "M<sub>z</sub> [N m]": "Mz",
    "M<sub>x</sub> [N-m]": "Mx",
    "M<sub>y</sub> [N-m]": "My",
    "M<sub>z</sub> [N-m]": "Mz",
    "Qdot [W]": "Qdot",
    "H<sub>o</sub> Flow Rate [W]": "H0dot",
    "log<sub>10</sub>(L2(Res))": "L2Resid",
    "Mdot Error (%)": "mdot_err",
}

# Pressure & viscous pairs that sum to a total force
FPV_TRIPLES = (
    ("Fxp", "Fxv", "Fx"),
    ("Fyp", "Fyv", "Fy"),
    ("Fzp", "Fzv", "Fz"),
)


# Force/moment history
class CaseFM(casedata.CaseFM):
    r"""Iterative force & moment history for one case, one component

    This class reads the VULCAN-CFD boundary-condition force and moment
    history files, :file:`BC_files/{bc}.ifam_his_{N}.tec` where *bc* is
    the component (BC group) name and *N* is the region number.  Forces
    and moments are kept in their native dimensional form, with columns
    like ``"Fx"`` for the total (pressure + viscous) *x*-force,
    ``"Fxp"`` for the pressure contribution, ``"Fxv"`` for the viscous
    contribution, and ``"Mx"`` for the moment.

    :Call:
        >>> fm = CaseFM(comp, runner=None, **kw)
    :Inputs:
        *comp*: :class:`str`
            Name of component to process
        *runner*: ``None`` | :class:`CaseRunner`
            Case runner interface for current case
    :Outputs:
        *fm*: :class:`cape.pyvul.databook.CaseFM`
            Instance of the force and moment class
    """
    # Class attributes
    _base_cols = (
        "i",
        "solver_iter",
        "mdot",
        "Fx",
        "Fy",
        "Fz",
        "Fxp",
        "Fyp",
        "Fzp",
        "Fxv",
        "Fyv",
        "Fzv",
        "Mx",
        "My",
        "Mz",
        "Qdot",
    )
    # Minimal list of "coeffs"
    _base_coeffs = (
        "mdot",
        "Fx",
        "Fy",
        "Fz",
        "Fxp",
        "Fyp",
        "Fzp",
        "Fxv",
        "Fyv",
        "Fzv",
        "Mx",
        "My",
        "Mz",
        "Qdot",
    )

    # Initialization method
    def __init__(
            self, comp: str,
            runner: Optional[CaseRunner] = None, **kw):
        r"""Initialization method"""
        # Use parent initializer
        databook.CaseFM.__init__(self, comp, **kw)
        # Save the case runner
        self.runner = runner

    # Get list of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = fm.get_filelist()
        :Inputs:
            *fm*: :class:`cape.pyvul.databook.CaseFM`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        """
        # Pattern for this component's history file(s), any region
        fglob = os.path.join(
            "BC_files", f"{self.comp}.ifam_his_*.tec")
        # Find and sort matching files
        return sorted(glob.glob(fglob))

    # Read a data file
    def readfile(self, fname: str) -> tsvfile.TSVTecDatFile:
        r"""Read a Tecplot iterative history file

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`cape.pyvul.databook.CaseFM`
                Component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`cape.dkit.tsvfile.TSVTecDatFile`
                Data read from *fname*
        """
        # Read the Tecplot file
        db = tsvfile.TSVTecDatFile(fname, Translators=COLNAMES_FM)
        # The VULCAN cycle number continues across restarts, so use it
        # directly as the CAPE iteration number
        db.save_col(databook.CASE_COL_ITERS, db[databook.CASE_COL_ITRAW])
        # Add pressure & viscous forces to get totals
        for colp, colv, col in FPV_TRIPLES:
            # Get individual contributions
            vp = db.get(colp)
            vv = db.get(colv)
            # Save sum if both are present
            if (vp is not None) and (vv is not None):
                db.save_col(col, vp + vv)
        # Output
        return db

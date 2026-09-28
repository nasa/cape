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

# Force columns and corresponding coefficients
FCOL_PAIRS = (
    ("Fx", "CA"),
    ("Fy", "CY"),
    ("Fz", "CN"),
    ("Fxp", "CAp"),
    ("Fyp", "CYp"),
    ("Fzp", "CNp"),
    ("Fxv", "CAv"),
    ("Fyv", "CYv"),
    ("Fzv", "CNv"),
)

# Moment columns and corresponding coefficients
MCOL_PAIRS = (
    ("Mx", "CLL"),
    ("My", "CLM"),
    ("Mz", "CLN"),
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
        # Normalize forces & moments using freestream dynamic pressure
        self.normalize_fm()

    # Get list of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        This returns the name of the single force & moment history file
        for *fm.comp*; VULCAN-CFD limits BC group names to 12
        characters and writes the history of each BC group to
        :file:`BC_files/{bc}.ifam_his_{N}.tec` where *N* is the region
        number (usually 1).

        :Call:
            >>> filelist = fm.get_filelist()
        :Inputs:
            *fm*: :class:`cape.pyvul.databook.CaseFM`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        """
        # VULCAN limits BC group names to 12 characters
        comp = self.comp[:12]
        # Expected name of this component's history file (region 1)
        fname = os.path.join("BC_files", f"{comp}.ifam_his_1.tec")
        # Check for the file
        if os.path.isfile(fname):
            return [fname]
        # Fall back to lower-case file name
        return [os.path.join("BC_files", f"{comp.lower()}.ifam_his_1.tec")]

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

    # Normalize a force & moment history using run matrix
    def normalize_fm(self):
        r"""Normalize a force & moment history using run matrix

        This reads the run matrix control instance using the case
        runner and computes force and moment coefficients such as
        ``"CA"`` and ``"CLM"`` from the dimensional histories using
        the freestream dynamic pressure and reference scales.

        :Call:
            >>> fm.normalize_fm()
        :Inputs:
            *fm*: :class:`cape.pyvul.databook.CaseFM`
                Component iterative history instance
        """
        # Check if normalized through the latest iteration
        if ("CA" in self.cols) and (self["CA"].size == self["i"].size):
            return
        # Case controller
        runner = self.runner
        # Exit if not present
        if runner is None:
            return
        # Read run matrix control
        cntl = runner.read_cntl()
        # Cannot normalize without it
        if cntl is None:
            return
        # Get case index
        i = runner.get_case_index()
        # Check for valid case index
        if i is None:
            return
        # Get dynamic pressure (forces are in Newtons)
        q = cntl.x.GetDynamicPressure(i, units="Pa")
        # Get reference scales
        aref = cntl.opts.get_RefArea(self.comp)
        lref = cntl.opts.get_RefLength(self.comp)
        # Check for usable values
        if (q is None) or (aref is None) or (lref is None):
            return
        # Normalize
        self.normalize_by_value(q, aref, lref)

    # Normalize a force & moment history using reference values
    def normalize_by_value(self, q: float, aref: float, lref: float):
        r"""Normalize a force & moment history using reference values

        :Call:
            >>> fm.normalize_by_value(q, aref, lref)
        :Inputs:
            *fm*: :class:`cape.pyvul.databook.CaseFM`
                Component iterative history instance
            *q*: :class:`float`
                Freestream dynamic pressure [Pa]
            *aref*: :class:`float`
                Reference area [m^2]
            *lref*: :class:`float`
                Reference length [m]
        """
        # Denominators
        qA = q*aref
        qAL = qA*lref
        # Loop through force components
        for fcol, ccol in FCOL_PAIRS:
            if (fcol in self) and (self[fcol].size > 0):
                self.save_coeff(ccol, self[fcol]/qA)
        # Loop through moment components
        for mcol, ccol in MCOL_PAIRS:
            if (mcol in self) and (self[mcol].size > 0):
                self.save_coeff(ccol, self[mcol]/qAL)

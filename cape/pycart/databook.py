r"""
:mod:`cape.pycart.databook`: pyCart data book module
====================================================

Databook module for :mod:`cape.pycart`

This module contains functions for reading and processing forces,
moments, and other statistics from cases in a trajectory.

    .. code-block:: python

        # Read Cart3D control instance
        cntl = cape.pycart.cntl.Cntl("pyCart.json")
        # Read the data book
        cntl.ReadDataBook()
        # Get a handle
        db = cntl.DataBook

Data book modules are also invoked during update and reporting
command-line calls.

    .. code-block:: console

        $ pycart --aero
        $ pycart --ll
        $ pycart --report

:See Also:
    * :mod:`cape.cfdx.databook`
    * :mod:`cape.cfdx.lineload`
    * :mod:`cape.cfdx.pointsensor`
    * :mod:`cape.pycart.lineload`
    * :mod:`cape.cfdx.options.databookopts`
    * :mod:`cape.pycart.options.databookopts`
"""

# Standard library
import os

# Third party
import numpy as np

# Local imports
from . import util
from ..dkit import tsvfile
from ..cfdx import databook


# Radian -> degree conversion
deg = np.pi / 180.0


# Alternate names for iterative history files
COLNAMES_FM = {
    "cycle": "i",
    "Fx": "CA",
    "Fy": "CY",
    "Fz": "CN",
    "Mx": "CLL",
    "My": "CLM",
    "Mz": "CLN",
}

COLNAMES_HIST = {
    "mgCycle": "i",
    "col1": "i",
    "CPUtime/proc": "CPUtime",
    "col2": "CPUtime",
    "maxResidual(rho)": "maxResid",
    "col3": "maxResid",
    "globalL1Residual(rho)": "L1Resid",
    "col4": "L1Resid",
}


# Individual component force and moment
class CaseFM(databook.CaseFM):
    r"""Cart3D iterative force & moment class

    This class contains methods for reading data about an the history of
    an individual component for a single casecntl.  It reads the file
    ``{comp}.dat`` where *comp* is the name of the component. From this
    file it determines which coefficients are recorded automatically.
    If some of the comment lines from the Cart3D output file have been
    deleted, it guesses at the column definitions based on the number of
    columns.

    :Call:
        >>> fm = CaseFM(comp)
    :Inputs:
        *comp*: :class:`str`
            Name of component to process
    :Outputs:
        *fm*: :class:`cape.pycart.databook.CaseFM`
            Instance of the force and moment class
        *fm.coeffs*: :class:`list`\ [:class:`str`]
            List of coefficients
    """
    # Get list of files (single file) to read
    def get_filelist(self) -> list:
        r"""Get ordered list of files to read to build iterative history

        :Call:
            >>> filelist = h.get_filelist()
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
        """
        # Get the working folder(s)
        fdira = util.GetAdaptFolder()
        fdirb = util.GetWorkingFolder()
        # De-None the adapt folder
        fdira = '.' if fdira is None else fdira
        # Replace "." -> ""
        fdira = "" if fdira == '.' else fdira
        fdirb = "" if fdirb == '.' else fdirb
        # Expected name of the component history file(s)
        fnamea = os.path.join(fdira, f"{self.comp}.dat")
        fnameb = os.path.join(fdirb, f"{self.comp}.dat")
        # Check if the non-adaptive file is newer than the adaptive one
        if (
                os.path.isfile(fnameb) and
                os.path.isfile(fnamea) and
                os.path.getmtime(fnameb) > os.path.getmtime(fnamea)
        ):
            # Use both files
            return [fnamea, fnameb]
        else:
            # Use only most recent
            return [fnameb]

    # Read one iterative history file
    def readfile(self, fname: str) -> dict:
        r"""Read cart3D ``{COMP}.dat`` file

        :Call:
            >>> data = h.readfile(fname)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *data*: :class:`tsvfile.TSVSimple`
                Data to add to or append to keys of *h*
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
        """
        return tsvfile.TSVFile(fname, Translators=COLNAMES_FM)


# Aerodynamic history class
class CaseResid(databook.CaseResid):
    r"""Iterative history class

    This class provides an interface to residuals, CPU time, and similar
    data for a given run directory

    :Call:
        >>> hist = CaseResid()
    :Outputs:
        *hist*: :class:`cape.pycart.databook.CaseResid`
            Instance of the run history class
    """
    # Default coefficient
    _default_resid = "L1Resid"

    # Get list of files (single file) to read
    def get_filelist(self) -> list:
        r"""Get ordered list of files to read to build iterative history

        :Call:
            >>> filelist = h.get_filelist()
        :Inputs:
            *h*: :class:`CaseResid`
                Single-case iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Get the working folder
        fdir = util.GetWorkingFolder()
        # Replace "." -> ""
        fdir = "" if fdir == '.' else fdir
        # Expected name of the component history file
        fname = os.path.join(fdir, "history.dat")
        # For Cart3D, only read the most recent file
        return [fname]

    # Read one iterative history file
    def readfile(self, fname: str) -> dict:
        r"""Read cart3D ``history.dat`` file

        :Call:
            >>> data = h.readfile(fname)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *data*: :class:`tsvfile.TSVSimple`
                Data to add to or append to keys of *h*
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        return tsvfile.TSVFile(fname, Translators=COLNAMES_HIST)


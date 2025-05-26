# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.databook`: Kestrel data book module
=====================================================

This module provides Kestrel-specific interfaces to the various CFD
outputs tracked by the :mod:`cape` package.

"""

# Standard library
import os
import re

# Third-party imports

# Local imports
from . import casecntl
from ..dkit import tsvfile
from ..cfdx import databook as cdbook


# Kestrel output column names
COLNAMES_KESTREL_RESID = {
    "ITER": "i",
    "TIME": "t",
    "SUBIT": "subiters",
    "SWEEPSP": "sweeps",
    "AOA": "alpha",
    "BETA": "beta",
    "LRES(Linf)": "L_inf",
    "LRES(L2)": "L2",
    "QRES(total)": "QRES",
    "SRES(total)": "SRES",
    "Y+": "yplus",
    "URES(mass)": "URES_mass",
    "URES(xmom)": "URES_xmom",
    "URES(ymom)": "URES_ymom",
    "URES(zmom)": "URES_zmom",
    "URES(energy)": "URES_energy",
    "URES(turb1)": "URES_turb1",
    "URES(turb2)": "URES_turb2",
    "SRES(mass)": "SRES_mass",
    "SRES(xmom)": "SRES_xmom",
    "SRES(ymom)": "SRES_ymom",
    "SRES(zmom)": "SRES_zmom",
    "SRES(energy)": "SRES_energy",
    "SRES(turb1)": "SRES_turb1",
    "SRES(turb2)": "SRES_turb2",
}

COLNAMES_KESTREL_FM = {
    "ITER": "i",
    "TIME": "t",
    "AOA": "alpha",
    "BETA": "beta",
    "AOAT": "aoap",
    "CLOCKANG": "phip",
    "CAXIAL": "CA",
    "CNORMAL": "CN",
    "CLIFT": "CL",
    "CDRAG": "CD",
    "CSIDE": "CY",
    "CPITCH": "CLM",
    "CROLL": "CLL",
    "CYAW": "CLN",
    "Y+": "yplus",
}

COLNAMES_KESTREL_PROP = {
    "ITER": "i",
    "TIME": "t",
    "Mass Flow": "mdot",
    "Density(area)": "rho_area",
    "VelX(area)": "u_area",
    "VelY(area)": "v_area",
    "VelZ(area)": "w_area",
    "P(area)": "p_area",
    "T(area)": "T_area",
    "Ptotal(area)": "p0_area",
    "Ttotal(area)": "T0_area",
    "VelNorm(area)": "vn_area",
    "WallheatFlux(area)": "qdot_area",
    "Density(mass)": "rho_mass",
    "VelX(mass)": "u_mass",
    "VelY(mass)": "v_mass",
    "VelZ(mass)": "w_mass",
    "P(mass)": "p_mass",
    "T(mass)": "T_mass",
    "Ptotal(mass)": "p0_mass",
    "Ttotal(mass)": "T0_mass",
    "VelNorm(mass)": "vn_mass",
    "WallheatFlux(mass)": "qdot_mass",
    "Y+": "yplus",
}


class DataBookPropComp(cdbook.DataBookPropComp):
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.pyfun.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid(self.proj)


class DataBookPyFunc(cdbook.DataBookPyFunc):
    pass


# Target databook class
class DataBookTarget(cdbook.DataBookTarget):
    pass


# Databook for one component
class DBFM(cdbook.DBFM):
    # Read case FM history
    def ReadCase(self, comp):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> FM = DB.ReadCaseFM(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *FM*: :class:`cape.pyfun.databook.CaseFM`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseFM(self.proj, comp)

    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.pyfun.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid(self.proj)


class DataBookTimeSeries(cdbook.DataBookTimeSeries):
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.pyfun.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid(self.proj)


# Iterative property history
class CaseProp(cdbook.CaseFM):
    r"""Iterative property history

    :Call:
        >>> prop = CaseProp(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file relative to ``outputs/`` folder
    :Outputs:
        *prop*: :class:`CaseProp`
            Iterative history of properties in *fname*
    :Versions:
        * 2022-01-28 ``@ddalle``: v1.0
        * 2024-05-20 ``@ddalle``: v2.0; min code; CAPE 1.2 conventions
    """
    # List of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = prop.get_filelist()
        :Inputs:
            *prop*: :class:`CaseProp`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        :Versions:
            * 2024-05-20 ``@ddalle``: v1.0
        """
        # Work folder
        workdir = os.path.join("outputs", self.comp)
        # Name of (single) file
        return [os.path.join(workdir, "props.dat")]

    # Read a raw data file
    def readfile(self, fname: str) -> dict:
        r"""Read a Tecplot iterative history file

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`tsvfile.TSVTecDatFile`
                Data read from *fname*
        :Versions:
            * 2024-05-20 ``@ddalle``: v1.0
        """
        # Read the Tecplot file
        db = tsvfile.TSVTecDatFile(fname, Translators=COLNAMES_KESTREL_PROP)
        # Output
        return db

    # Read case generic-property history
    def ReadCase(self, comp):
        r"""Read a :class:`CaseProp` object

        :Call:
            >>> fm = db.ReadCase(comp)
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *fm*: :class:`CaseFM`
                Force and moment history
        :Versions:
            * 2022-04-08 ``@ddalle``: v1.0
        """
        # Read CaseResid object from PWD
        return CaseProp(comp)


# Iterative F&M history
class CaseFM(CaseProp):
    r"""Iterative force & moment history for one component, one case

    :Call:
        >>> fm = CaseFM(comp=None)
    :Inputs:
        *comp*: :class:`str`
            Name of component
    :Outputs:
        *fm*: :class:`CaseFM`
            One-case iterative history
    :Versions:
        * 2021-11-08 ``@ddalle``: v1.0
        * 2024-05-20 ``@ddalle``: v2.0; min code; CAPE 1.2 conventions
    """
    # List of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = fm.get_filelist()
        :Inputs:
            *prop*: :class:`CaseFM`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        :Versions:
            * 2024-05-20 ``@ddalle``: v1.0
        """
        # Work folder
        workdir = os.path.join("outputs", self.comp)
        # Name of (single) file
        return [os.path.join(workdir, "coeff.dat")]

    # Read a raw data file
    def readfile(self, fname: str) -> dict:
        r"""Read a Tecplot iterative history file

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`tsvfile.TSVTecDatFile`
                Data read from *fname*
        :Versions:
            * 2024-05-20 ``@ddalle``: v1.0
        """
        # Read the Tecplot file
        db = tsvfile.TSVTecDatFile(fname, Translators=COLNAMES_KESTREL_FM)
        # Output
        return db


# Iterative residual history
class CaseResid(cdbook.CaseResid):
    r"""Iterative residual history for one component, one case

    :Call:
        >>> hist = CaseResid(comp=None)
    :Inputs:
        *comp*: {``None``} | :class:`str`
            Name of component
    :Outputs:
        *hist*: :class:`CaseResid`
            One-case iterative history
    :Versions:
        * 2021-11-08 ``@ddalle``: v1.0
        * 2024-05-21 ``@ddalle``: v2.0; code reduction for CAPE 1.2
    """
    # Initialization method
    def __init__(self, comp: str = "body1"):
        r"""Initialization method

        :Versions:
            * 2021-11-08 ``@ddalle``: v1.0
        """
        # Initialize attributes
        self.comp = comp
        # Call parent method
        cdbook.CaseResid.__init__(self)

    # List of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = fm.get_filelist()
        :Inputs:
            *prop*: :class:`CaseFM`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        :Versions:
            * 2024-05-20 ``@ddalle``: v1.0
        """
        # Work folder
        workdir = os.path.join("outputs", "BodyTracking", self.comp)
        # Name of (single) file
        return [os.path.join(workdir, "cfd.core.dat")]

    # Read a raw data file
    def readfile(self, fname: str) -> dict:
        r"""Read a Tecplot iterative history file

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`tsvfile.TSVTecDatFile`
                Data read from *fname*
        :Versions:
            * 2024-05-20 ``@ddalle``: v1.0
        """
        # Read the Tecplot file
        db = tsvfile.TSVTecDatFile(fname, Translators=COLNAMES_KESTREL_RESID)
        # Output
        return db


# Iterative residual history
class CaseTurbResid(CaseResid):
    r"""Iterative turbulence model residual history

    :Call:
        >>> hist = CaseTurbResid(comp=None)
    :Inputs:
        *comp*: {``None``} | :class:`str`
            Name of component
    :Outputs:
        *hist*: :class:`CaseResid`
            One-case iterative history
    :Versions:
        * 2024-05-21 ``@ddalle``: v1.0
    """
    # List of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = fm.get_filelist()
        :Inputs:
            *prop*: :class:`CaseFM`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        :Versions:
            * 2024-05-20 ``@ddalle``: v1.0
        """
        # Work folder
        workdir = os.path.join("outputs", "BodyTracking", self.comp)
        # Name of (single) file
        return [os.path.join(workdir, "cfd.turb.dat")]

    # Read a raw data file
    def readfile(self, fname: str) -> dict:
        r"""Read a Tecplot iterative history file

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`tsvfile.TSVTecDatFile`
                Data read from *fname*
        :Versions:
            * 2024-05-20 ``@ddalle``: v1.0
        """
        # Read the Tecplot file
        db = tsvfile.TSVTecDatFile(fname, Translators=COLNAMES_KESTREL_RESID)
        # Output
        return db


# Aerodynamic history class
class DataBook(cdbook.DataBook):
    r"""Primary databook class for Kestrel

    :Call:
        >>> db = DataBook(x, opts)
    :Inputs:
        *x*: :class:`RunMatrix`
            Current run matrix
        *opts*: :class:`Options`
            Global CAPE options instance
    :Outputs:
        *db*: :class:`DataBook`
            Databook instance
    :Versions:
        * 21-11-08 ``@ddalle``: v1.0
    """
    _fm_cls = DBFM
    _ts_cls = DataBookTimeSeries
    _prop_cls = DataBookPropComp
    _pyfunc_cls = DataBookPyFunc
  # ===========
  # Readers
  # ===========
  # <

    # Local version of data book
    def _DataBook(self, targ):
        self.Targets[targ] = DataBook(
            self.x, self.opts, RootDir=self.RootDir, targ=targ)

    # Local version of target
    def _DataBookTarget(self, targ):
        self.Targets[targ] = DataBookTarget(targ, self.x, self.opts, self.RootDir)
  # >

  # ========
  # Case I/O
  # ========
  # <
    # Current iteration status
    def GetCurrentIter(self):
        r"""Determine iteration number of current folder

        :Call:
            >>> n = db.GetCurrentIter()
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
        :Outputs:
            *n*: :class:`int` | ``None``
                Iteration number
        :Versions:
            * 2021-11-08 ``@ddalle``: v1.0
        """
        try:
            return casecntl.get_current_iter()
        except Exception:
            return None

  # >


# Normalize a column name
def normalize_colname(colname):
    r"""Normalize a Kestrel column name, removing special chars

    :Call:
        >>> col = normalize_colname(colname)
    :Inputs:
        *colname*: :class:`str`
            Raw column name from Kestrel output file
    :Outputs:
        *col*: :class:`str`
            Normalized column name
    :Versions:
        * 2021-11-08 ``@ddalle``: v1.0
    """
    # Special substitutions
    col = colname.replace("+", "plus")
    col = col.replace("[", "_")
    col = col.replace("]", "")
    # Eliminate some chars
    col = re.sub("[({]", "_", col)
    col = re.sub("[)} ]", "", col)
    col = re.sub("[-/.]", "_", col)
    # Output
    return col


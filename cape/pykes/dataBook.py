# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.dataBook`: Kestrel data book module
=====================================================

This module provides Kestrel-specific interfaces to the various CFD
outputs tracked by the :mod:`cape` package.

"""

# Standard library
import os
import re

# Third-party imports
import numpy as np

# Local imports
from . import case
from ..cfdx import dataBook as cdbook


# Kestrel output column names
COLNAMES_KESTREL_STATE = {
    "ITER": "i",
    "TIME": "t",
    "SUBIT": "subiters",
    "SWEEPSP": "sweeps",
    "AOA": "alpha",
    "BETA": "beta",
}
COLNAMES_KESTREL_COEFF = {
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
COLNAMES_TURB = {
    "i": "i_turb",
    "t": "t_turb",
    "sweeps": "sweeps_turb",
    "URES_total": "URES_total_turb",
    "SRES_total": "SRES_total_turb",
}


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
        * 21-11-08 ``@ddalle``: Version 1.0
    """
  # ===========
  # Readers
  # ===========
  # <
    # Initialize a DBComp object
    def ReadDBComp(self, comp, check=False, lock=False):
        r"""Initialize data book for one component

        :Call:
            >>> db.ReadDBComp(comp, check=False, lock=False)
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
            *comp*: :class:`str`
                Name of component
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Read the data book
        self[comp] = DBComp(comp, self.x, self.opts,
            targ=self.targ, check=check, lock=lock)

    # Local version of data book
    def _DataBook(self, targ):
        self.Targets[targ] = DataBook(
            self.x, self.opts, RootDir=self.RootDir, targ=targ)

    # Local version of target
    def _DBTarget(self, targ):
        self.Targets[targ] = DBTarget(targ, self.x, self.opts, self.RootDir)
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
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        try:
            return case.get_current_iter()
        except Exception:
            return None

    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
        :Outputs:
            *H*: :class:`CaseResid`
                Residual history
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Read CaseResid object from PWD
        return CaseResid()

    # Read case FM history
    def ReadCaseFM(self, comp):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> fm = db.ReadCaseFM(comp)
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *fm*: :class:`CaseFM`
                Force and moment history
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Read CaseResid object from PWD
        return CaseFM(comp)
  # >


# Target databook class
class DBTarget(cdbook.DBTarget):
    pass


# Databook for one component
class DBComp(cdbook.DBComp):
    pass


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
        * 2022-01-28 ``@ddalle``: Version 1.0
    """
   # --- __dunder__ ---
    def __init__(self, fname, **kw):
        r"""Initialization method

        :Versions:
            * 2022-01-28 ``@ddalle``: Version 1.0
        """
        # Save a component name
        self.comp = fname.split("/")[0]
        # Generate full path
        fdat = os.path.join("outputs", fname.replace("/", os.sep))
        # Check for file
        if not os.path.isfile(fdat):
            return
        # Read file
        self.read_dat(fdat)

   # --- Read ---
    def read_dat(self, fdat):
        r"""Read a data file in expected Kestrel format

        :Call:
            >>> prop.read_dat(fdat)
        :Inputs:
            *prop*: :class:`CaseProp`
                Iterative property history
            *fdat*: :class:`str`
                Name of file to read
        :Versions:
            * 2022-01-28 ``@ddalle``: Version 1.0
        """
        # Figure out headers
        nhdr, cols, coeffs, inds = self.read_colnames(fdat)
        # Save entries
        self._hdr = nhdr
        self.cols = cols
        self.coeffs = coeffs
        self.inds = inds
        # Read it
        A = np.loadtxt(fdat, skiprows=nhdr, usecols=tuple(inds))
        # Save the values
        for j, col in zip(inds, cols):
            self.__dict__[col] = A[:,j]

   # --- Header ---
    def read_colnames(self, fname):
        r"""Determine column names

        :Call:
            >>> nhdr, cols, coeffs, inds = fm.read_colnames(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Case force/moment history
            *fname*: :class:`str`
                Name of file to process
        :Outputs:
            *nhdr*: :class:`int`
                Number of header rows to skip
            *cols*: :class:`list`\ [:class:`str`]
                List of column names
            *coeffs*: :class:`list`\ [:class:`str`]
                List of coefficient names
            *inds*: :class:`list`\ [:class:`int`]
                List of column indices for each entry of *cols*
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
                - forked from :class:`cape.pykes.dataBook.CaseFM`
        """
        # Initialize variables and read flag
        keys = []
        flag = 0
        # Number of header lines
        nhdr = 0
        # Open the file
        with open(fname) as fp:
            # Loop through lines
            while nhdr < 100:
                # Strip whitespace from the line.
                l = fp.readline().strip()
                # Check the line
                if flag == 0:
                    # Count line
                    nhdr += 1
                    # Check for "variables"
                    if not l.lower().startswith('variables'):
                        continue
                    # Set the flag
                    flag = 1
                    # Split on '=' sign
                    L = l.split('=')
                    # Check for first variable
                    if len(L) < 2:
                        continue
                    # Split variables on as things between quotes
                    vals = re.findall('"\w[^"]*"', L[1])
                    # Append to the list
                    keys += [v.strip('"') for v in vals]
                elif flag == 1:
                    # Count line
                    nhdr += 1
                    # Reading more lines of variables
                    if not l.startswith('"'):
                        # Done with variables; read extra headers
                        flag = 2
                        continue
                    # Split variables on as things between quotes
                    vals = re.findall('"\w[^"]*"', l)
                    # Append to the list.
                    keys += [v.strip('"') for v in vals]
                else:
                    # Check if it starts with an integer
                    try:
                        # If it's an integer, stop reading lines.
                        float(l.split()[0])
                        break
                    except Exception:
                        # Line starts with something else; continue
                        nhdr += 1
                        continue
        # Initialize column indices and their meanings.
        inds = []
        cols = []
        coeffs = []
        # Map common Kestrel column names
        for j, key in enumerate(keys):
            # See if it's a state column
            xcol = COLNAMES_KESTREL_STATE.get(key)
            # If found, save
            if xcol is not None:
                inds.append(j)
                cols.append(xcol)
                continue
            # Get coefficient name
            ycol = COLNAMES_KESTREL_COEFF.get(key, key)
            # Normalize
            ycol = normalize_colname(ycol)
            # Save coefficient
            inds.append(j)
            cols.append(ycol)
            coeffs.append(ycol)
        # Output
        return nhdr, cols, coeffs, inds


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
        * 2021-11-08 ``@ddalle``: Version 1.0
    """
    # Initialization method
    def __init__(self, comp=None):
        r"""Initialization method

        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Save inputs
        self.comp = comp
        # File name to read
        fdat = self.create_fname_coeff_dat()
        # Initialize attributes
        self.init_data()
        # Check for empty input
        if not comp:
            return
        # Check if file exists
        if not os.path.isfile(fdat):
            return
        # Read file
        self.read_coeff_dat()

   # --- Data ---
    def init_data(self):
        r"""Initialize standard force/moment attributes

        :Call:
            >>> fm.init_data()
        :Inputs:
            *fm*: :class:`CaseFM`
                Case force/moment history
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Make all entries empty
        self.i = np.zeros(0)
        self.CA = np.zeros(0)
        self.CY = np.zeros(0)
        self.CN = np.zeros(0)
        self.CLL = np.zeros(0)
        self.CLM = np.zeros(0)
        self.CLN = np.zeros(0)
        # Save a default list of columns and components.
        self.coeffs = ['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']
        self.cols = ['i'] + self.coeffs

    def read_coeff_dat(self, fdat=None):
        r"""Read ``coeff.dat`` from expected data file

        :Call:
            >>> fm.read_coeff_dat(fdat=None)
        :Inputs:
            *fm*: :class:`CaseFM`
                Case force/moment history
            *fdat*: {``None``} | :class:`str`
                Optional specific file name
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Default file name
        if fdat is None:
            fdat = self.genr8_fname_coeff_dat()
        # Check if found
        if fdat is None or not os.path.isfile(fdat):
            return
        # Figure out headers
        nhdr, cols, coeffs, inds = self.read_colnames(fdat)
        # Save entries
        self._hdr = nhdr
        self.cols = cols
        self.coeffs = coeffs
        self.inds = inds
        # Read it
        A = np.loadtxt(fdat, skiprows=nhdr, usecols=tuple(inds))
        # Save the values
        for j, col in zip(inds, cols):
            self.__dict__[col] = A[:,j]

   # --- Files ---
    def create_fname_coeff_dat(self, comp=None):
        r"""Generate full file name for ``coeff.dat``

        :Call:
            >>> fdat = fm.create_fname_coeff_dat(comp=None)
        :Inputs:
            *fm*: :class:`CaseFM`
                Case force/moment history
            *comp*: {*fm.comp*} | :class:`str`
                Name of component
        :Outputs:
            *fdat*: :class:`str`
                Name of file to read
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Get file name
        self.fdat = self.genr8_fname_coeff_dat(comp)
        # Output
        return self.fdat

    def genr8_fname_coeff_dat(self, comp=None):
        r"""Generate full file name for ``coeff.dat``

        :Call:
            >>> fdat = fm.genr8_fname_coeff_dat(comp=None)
        :Inputs:
            *fm*: :class:`CaseFM`
                Case force/moment history
            *comp*: {*fm.comp*} | :class:`str`
                Name of component
        :Outputs:
            *fdat*: :class:`str`
                Name of file to read
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Default comp
        if comp is None:
            comp = self.comp
        # Check if found
        if comp is None:
            return
        # Assemble file name
        return os.path.join("outputs", "BodyTracking", comp, "coeff.dat")


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
        * 2021-11-08 ``@ddalle``: Version 1.0
    """
    # Initialization method
    def __init__(self, comp=None):
        r"""Initialization method

        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Initialize attributes
        self.comp = None
        self.fcore = None
        self.fturb = None
        # Generate file names
        fcore, fturb = self.create_fnames(comp=comp)
        # Initialize data
        self.init_data()
        # Read data files
        self.read_core_dat(fcore)
        self.read_turb_dat(fturb)

   # --- Data ---
    def init_data(self):
        r"""Initialize standard force/moment attributes

        :Call:
            >>> fm.init_data()
        :Inputs:
            *fm*: :class:`CaseFM`
                Case force/moment history
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Make all entries empty
        self.i = np.zeros(0)
        # Save a default list of columns and components.
        self.coeffs = []
        self.cols = ['i'] + self.coeffs

    def read_core_dat(self, fdat=None):
        r"""Read ``cfd.core.dat`` from expected data file

        :Call:
            >>> hist.read_core_dat(fdat=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Case residual history
            *fdat*: {``None``} | :class:`str`
                Optional specific file name
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Default file name
        if fdat is None:
            fdat, _ = self.genr8_fnames()
        # Check if found
        if fdat is None or not os.path.isfile(fdat):
            return
        # Figure out headers
        nhdr, cols, coeffs, inds = self.read_colnames(fdat)
        # Read it
        A = np.loadtxt(fdat, skiprows=nhdr, usecols=tuple(inds))
        # Save the values
        for j, col in zip(inds, cols):
            # Check if *coeff*
            if col in coeffs:
                self.save_coeff(col, A[:,j])
            else:
                self.save_col(col, A[:,j])

    def read_turb_dat(self, fdat=None):
        r"""Read ``cfd.turb.dat`` from expected data file

        :Call:
            >>> hist.read_turb_dat(fdat=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Case residual history
            *fdat*: {``None``} | :class:`str`
                Optional specific file name
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Default file name
        if fdat is None:
            _, fdat = self.genr8_fnames()
        # Check if found
        if fdat is None or not os.path.isfile(fdat):
            return
        # Figure out headers
        nhdr, cols, coeffs, inds = self.read_colnames(fdat)
        # Read it
        A = np.loadtxt(fdat, skiprows=nhdr, usecols=tuple(inds))
        # Save the values
        for j, col in zip(inds, cols):
            # Check for repeated (from cfd.core.dat) colum nnames
            col1 = COLNAMES_TURB.get(col, col)
            # Check if *coeff*
            if col in coeffs:
                self.save_coeff(col1, A[:,j])
            else:
                self.save_col(col1, A[:,j])

    def save_coeff(self, col, v):
        r"""Save data to coefficient attribute called *col*

        :Call:
            >>> hist.save_coeff(col, v)
        :Inputs:
            *hist*: :class:`CaseResid`
                Case residual history
            *col*: :class:`str`
                Name of column
            *v*: :class:`np.ndarray`
                Values to save/append
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Add to *coeff* list
        if col not in self.coeffs:
            self.coeffs.append(col)
        # Do other *col* and data saving
        self.save_col(col, v)
        
    def save_col(self, col, v):
        r"""Save data to attribute called *col*

        :Call:
            >>> hist.save_col(col, v)
        :Inputs:
            *hist*: :class:`CaseResid`
                Case residual history
            *col*: :class:`str`
                Name of column
            *v*: :class:`np.ndarray`
                Values to save/append
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Add to column list
        if col not in self.cols:
            self.cols.append(col)
        # Check if already present
        v0 = self.__dict__.get(col)
        # If present, try to stack it
        if v0 is not None:
            v = np.hstack((v0, v))
        # Save data
        self.__dict__[col] = v

   # --- Header ---
    def read_colnames(self, fname):
        r"""Determine column names

        :Call:
            >>> nhdr, cols, coeffs, inds = hist.read_colnames(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Case force/moment history
            *fname*: :class:`str`
                Name of file to process
        :Outputs:
            *nhdr*: :class:`int`
                Number of header rows to skip
            *cols*: :class:`list`\ [:class:`str`]
                List of column names
            *coeffs*: :class:`list`\ [:class:`str`]
                List of coefficient names
            *inds*: :class:`list`\ [:class:`int`]
                List of column indices for each entry of *cols*
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Initialize variables and read flag
        keys = []
        flag = 0
        # Number of header lines
        nhdr = 0
        # Open the file
        with open(fname) as fp:
            # Loop through lines
            while nhdr < 100:
                # Strip whitespace from the line.
                l = fp.readline().strip()
                # Check the line
                if flag == 0:
                    # Count line
                    nhdr += 1
                    # Check for "variables"
                    if not l.lower().startswith('variables'):
                        continue
                    # Set the flag
                    flag = 1
                    # Split on '=' sign
                    L = l.split('=')
                    # Check for first variable
                    if len(L) < 2:
                        continue
                    # Split variables on as things between quotes
                    vals = re.findall('"[\w ]+"', L[1])
                    # Append to the list
                    keys += [v.strip('"') for v in vals]
                elif flag == 1:
                    # Count line
                    nhdr += 1
                    # Reading more lines of variables
                    if not l.startswith('"'):
                        # Done with variables; read extra headers
                        flag = 2
                        continue
                    # Split variables on as things between quotes
                    vals = re.findall('"[^"]+"', l)
                    # Append to the list
                    keys += [v.strip('"') for v in vals]
                else:
                    # Check if it starts with an integer
                    try:
                        # If it's an integer, stop reading lines
                        float(l.split()[0])
                        break
                    except Exception:
                        # Line starts with something else; continue
                        nhdr += 1
                        continue
        # Initialize column indices and their meanings.
        inds = []
        cols = []
        coeffs = []
        # Map common Kestrel column names
        for j, key in enumerate(keys):
            # See if it's a state column
            xcol = COLNAMES_KESTREL_STATE.get(key)
            # If found, save
            if xcol is not None:
                inds.append(j)
                cols.append(xcol)
                continue
            # Get coefficient name
            ycol = COLNAMES_KESTREL_COEFF.get(key)
            # Normalize if not found
            if ycol is None:
                ycol = normalize_colname(key)
            # Save coefficient
            inds.append(j)
            cols.append(ycol)
            coeffs.append(ycol)
        # Output
        return nhdr, cols, coeffs, inds

   # --- Files ---
    def make_comp(self, comp=None):
        r"""Figure out a "component" name to use

        :Call:
            >>> comp = hist.make_comp(comp=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Case residual history
            *comp*: {``None``} | :class:`str`
                Directly-specified component name
        :Outputs:
            *comp*: :class:`str` | ``None``
                Component name, usually first folder found
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Use input directly
        if comp is not None:
            self.comp = comp
            return comp
        # Check for preset
        if self.comp is not None:
            return self.comp
        # Folder where components are expected to reside
        fbody = os.path.join("outputs", "BodyTracking")
        # Check for such a folder
        if not os.path.isdir(fbody):
            return
        # Look for available components
        fcomps = os.listdir(fbody)
        # Check if any found
        for comp in fcomps:
            if os.path.isdir(os.path.join(fbody, comp)):
                # Found one; save and return it
                self.comp = comp
                return comp

    def create_fnames(self, comp=None):
        r"""Generate file names for ``cfd.{core,turb}.dat``

        :Call:
            >>> fcore, fturb = hist.create_fnames(comp=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Case residual history
            *comp*: {*fm.comp*} | :class:`str`
                Name of component
        :Outputs:
            *fcore*: :class:`str`
                Path to ``cfd.core.dat`` relative to case folder
            *fturb*: :class:`str`
                Path to ``cfd.turb.dat`` relative to case folder
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Get file name
        self.fcore, self.fturb = self.genr8_fnames(comp)
        # Output
        return self.fcore, self.fturb

    def genr8_fnames(self, comp=None):
        r"""Generate file names for ``cfd.{core,turb}.dat``

        :Call:
            >>> fcore, ftrub = hist.genr8_fnames(comp=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Case residual history
            *comp*: {*fm.comp*} | :class:`str`
                Name of component
        :Outputs:
            *fcore*: :class:`str`
                Path to ``cfd.core.dat`` relative to case folder
            *fturb*: :class:`str`
                Path to ``cfd.turb.dat`` relative to case folder
        :Versions:
            * 2021-11-08 ``@ddalle``: Version 1.0
        """
        # Process "component" folder to read from
        comp = self.make_comp(comp)
        # Check if found
        if comp is None:
            return None, None
        # Assemble file names
        fcomp = os.path.join("outputs", "BodyTracking", comp)
        fcore = os.path.join(fcomp, "cfd.core.dat")
        fturb = os.path.join(fcomp, "cfd.turb.dat")
        # Output
        return fcore, fturb


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
        * 2021-11-08 ``@ddalle``: Version 1.0
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


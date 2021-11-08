# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.dataBook`: Kestrel data book module
=====================================================

This module provides Kestrel-specific interfaces to the various CFD
outputs tracked by the :mod:`cape` package.

"""

# Standard library


# Third-party imports


# Local imports
from ..cfdx import dataBook as cdbook


# Kestrel output column names
COLNAMES_KESTREL_STATE = {
    "ITER": "i",
    "TIME": "t",
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
}


# Iterative history
class CaseFM(cdbook.CaseFM):
    r"""Iterative force & moment history for one component, one case

    :Call:
        >>> fm = CaseFM(comp)
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
        # Initialize attributes
        self.init_data()
        # Check for empty input
        if not comp:
            return
        # File name to read
        fdat = self.genr8_fname_coeff_dat()
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
        nhdr, cols, coeffs, inds = fm.read_colnames(fdat)
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
                l = f.readline().strip()
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
                    vals = re.findall('"[\w ]+"', l)
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
            # Save coefficient
            inds.append(j)
            cols.append(ycol)
            coeffs.append(ycol)
        # Output
        return nhdr, cols, coeffs, inds

   # --- Files ---
    def genr8_fname_coeff_dat(self, comp=None):
        r"""Generate full file name for ``coeff.dat``

        :Call:
            >>> fdat = self.genr8_fname_coeff_dat(comp=None)
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


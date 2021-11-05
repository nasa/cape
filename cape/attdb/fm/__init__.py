#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`attdb.fm`: Aero Task Team force & moment database modules
==================================================================

This module provides several classes to interface with various *point* or *0D*
databases.  That is, each piece of data is a scalar in all of the columns.  A
common example of this is a force & moment database, for which the module is
named, where there are several scalar force or moment coefficients at each
flight condition.

An example of a database that does not fit into this paradigm would be a
surface pressure database, in which there is a vector of surface pressures for
each flight condition.  In the terminology of these database tools, that is an
example of a 1D database.

"""

# System module
import os
import sys
# Numerics
import numpy as np
# Plotting
import matplotlib.pyplot as plt
# More powerful interpolation
import scipy.interpolate
# Mat interface
import scipy.io as sio
import scipy.io.matlab.mio5_params as siom

# Spreadsheet interfaces
import xlrd

# CAPE modules
import cape.convert    as convert
# Toolkit modules
import cape.tnakit.db.db1        as db1
import cape.tnakit.plotutils.mpl as mpl
import cape.tnakit.statutils     as stats
# Local modules
from .. import trajectory


# Force and moment class
class DBCoeff(db1.DBCoeff):
    """Generic coefficient database and interpolation class
    
    :Call:
        >>> DBc = DBCoeff(mat=None, xls=None, csv=None, **kw)
    :Inputs:
        *mat*: :class:`str`
            Name of Matlab file to read
        *xls*: :class:`str` | :class:`xlrd.book.Book`
            Name of spreadsheet or interface to open workbook
        *csv*: :class:`str`
            Name of CSV file to read
        *sheet*: {``None``} | :class:`str`
            Name of worksheet to read if reading from spreadsheet
    :Outputs:
        *DBc*: :class:`attdb.fm.DBCoeff`
            Coefficient lookup database
        *DBc.coeffs*: :class:`list` (:class:`str` | :class:`unicode`)
            List of coefficients present in the database
        *DBc[coeff]*: :class:`np.ndarray`\ [:class:`float`]
            Data for coefficient named *coeff*
    :Versions:
        * 2018-06-08 ``@ddalle``: First version
    """
  # ==========
  # Config
  # ==========
  # <
  # >
  
  # ============
  # Data
  # ============
  # <
   # --- Statistics ---
    # Get coverage
    def GetCoverage(self, DBT, coeff, I, cov=0.95, **kw):
        """Calculate Student's t-distribution confidence region
        
        If the nominal application of the Student's t-distribution fails to
        cover a high enough fraction of the data, the bounds are extended until
        the data is covered.
        
        :Call:
            >>> vmu, width = DBc.GetCoverage(DBT, coeff, cov=0.95, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                First coefficient database
            *DBT*: :class:`attdb.fm.DBCoeff`
                Target coefficient database
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Array of case indices from *DBc*
            *cov*: {``0.95``} | 0 < :class:`float` < 1
                Coverage percentage
            *keys*: {``["mach","alpha_t","phi"]`` | :class:`list`
                List of keys to use for testing if cases are equal
            *tol*: {``1e-8``} | :class:`float`
                Default tolerance for matching conditions
        :Outputs:
            *vmu*: :class:`float`
                Mean delta of non-excluded deltas
            *width*: :class:`float`
                Half-width of confidence region
        :Versins:
            * 2018-09-28 ``@ddalle``: First version
        """
       # --- Setup ---
        # Find indices of matches
        I, J = self.get_MatchIndices(DBT, I, **kw)
        # Get values
        V1 = self[coeff][I]
        V2 = DBT[coeff][J]
        # Deltas (signed)
        dV = V2 - V1
        # Degrees of freedom
        df = J.size
        # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
        ksig = stats.student.ppf(0.5+0.5*cov, df)
        # Outlier flag
        osig = kw.get("OutlierSigma", 1.5*ksig)
       # --- Initial Stats ---
        # Mean
        vmu = np.mean(dV)
        # Standard deviation
        vstd = np.std(dV - vmu)
        # Find outliers
        I = np.abs(dV-vmu)/vstd > osig
        # Check outliers
        while np.any(I):
            # Filter
            dV = dV[np.logical_not(I)]
            # Recalculate statistics
            vmu  = np.mean(dV)
            vstd = np.std(dV - vmu)
            # Find outliers
            I = np.abs(dV-vmu)/vstd > osig
        # Nominal width
        width = ksig*vstd
       # --- Coverage Check ---
        # Limits
        vmin = vmu - width
        vmax = vmu + width
        # Filter cases that are outside bounds
        J = np.logical_or(dV<vmin, dV>vmax)
        # Count cases outside the bounds
        nout = np.count_nonzero(J)
        # Check coverage
        if float(nout)/df > 1-cov:
            # Sort the cases outside by distance from *vmu*
            dVo = np.sort(np.abs(dV[J]-vmu))
            # Number of cases allowed to be uncovered
            na = int(df - np.ceil(cov*df))
            # Number of additional cases that must be covered
            n1 = nout - na
            # The new width is the delta to the last newly included point
            width = dVo[n1-1]
       # --- Output ---
        return vmu, width
       # ---
        
    
    # Get range
    def GetRangeCoverage(self, DBT, coeff, I, cov=0.95, **kw):
        """Calculate Student's t-distribution confidence region centered at 0
        
        If the nominal application of the Student's t-distribution fails to
        cover a high enough fraction of the data, the bounds are extended until
        the data is covered.
        
        :Call:
            >>> width = DBc.GetRangeCoverage(DBT, coeff, I, cov=0.95, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                First coefficient database
            *DBT*: :class:`attdb.fm.DBCoeff`
                Target coefficient database
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Array of case indices from *DBc*
            *cov*: {``0.95``} | 0 < :class:`float` < 1
                Coverage percentage
            *keys*: {``["mach","alpha_t","phi"]`` | :class:`list`
                List of keys to use for testing if cases are equal
            *tol*: {``1e-8``} | :class:`float`
                Default tolerance for matching conditions
        :Outputs:
            *width*: :class:`float`
                Half-width of confidence region
        :Versions:
            * 2018-09-28 ``@ddalle``: First version
            * 2019-01-30 ``@ddalle``: Offloaded to :func:`get_range`
        """
        # Find indices of matches
        I, J = self.get_MatchIndices(DBT, I, **kw)
        # Get values
        V1 = self[coeff][I]
        V2 = DBT[coeff][J]
        # Deltas (signed)
        dV = V2 - V1
        # Ranges
        R = np.abs(dV)
        # Calculate range
        return get_range(R, cov, **kw)
   
    # Get indices of matches
    def get_MatchIndices(self, DBT, I, **kw):
        """Get indices of cases from two databases with common conditions
        
        :Call:
            >>> I, J = DBc.get_MatchIndices(DBT, I, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                First coefficient database
            *DBT*: :class:`attdb.fm.DBCoeff`
                Target coefficient database
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Array of case indices from *DBc*
            *keys*: {``["mach","alpha_t","phi"]`` | :class:`list`
                List of keys to use for testing if cases are equal
            *tol*: {``1e-8``} | :class:`float`
                Default tolerance for matching conditions
        :Outputs:
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Array of case indices from *DBc* with matches in *DBT*
            *J*: :class:`np.ndarray`\ [:class:`int`]
                Array of case indices from *DBT* with matches in *DBc*
        :Versins:
            * 2018-09-28 ``@ddalle``: First version
        """
        # Initialize matches
        J = -1 * np.ones(I.size, dtype="int")
        # Comparison key list
        kw.setdefault("keys", ["mach", "alpha_t", "phi"])
        # Loop through indices
        for k in range(I.size):
            # Get case index
            i = I[k]
            # Find the match
            j = DBT.x.FindMatch(self.x, i, **kw)
            # Check for a match
            if j:
                J[k] = j
        # Find non-NaN entries
        mask = np.where(J>=0)[0]
        # Filter the case lists
        I = I[mask]
        J = J[mask]
        # Output
        return I, J
  # >
  
  # ===============
  # Interpolation
  # ===============
  # <
   # --- Radial Basis Function ---
    # Create a radial basis function for one coefficient
    def CreateRBF(self, coeff, keys, I=None, func=None, **kw):
        """Create a radial basis function interpolant
        
        :Call:
            >>> f = DBc.CreateRBF(coeff, keys, func=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to interpolate
            *keys*: :class:`list` (:class:`str`) 
                List of lookup keys, must be at least two
            *func*: {*coeff*} | :class:`str`
                If used, the RBF is saved as *DBc.rbf[func]*
            *function*: {``"cubic"``} | ``"multiquadric"`` | ``"linear"``
                Basis function for :func:`scipy.interpolate.Rbf`
            *slices*: {``{}``} | :class:`dict`
                Dictionary of other lookup keys to constrain to specific values
            *tol*: {``1e-4``}  | :class:`float`
                Default tolerance to use in combination with *slices*
            *tols*: {``{}``} | :class:`dict`
                Dictionary of specific tolerances for single keys in *slices*
        :Outputs:
            *f*: :class:`scipy.interpolate.Rbf.rbf`
                Interpolation function, also saved in *DBc.rbf[func]*
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        # Check coefficient
        if coeff not in self.coeffs:
            raise KeyError("No coefficient called '%s'" % coeff)
        # Check keys
        nkey = len(keys)
        if nkey < 2:
            raise ValueError("Need at least two input keys " +
                "to create radial basis function interpolant.")
        # Get lengths
        n = len(self[coeff])
        # Default list
        if I is None:
            # Use all
            I = np.arange(n)
        # Check for constraints
        slices = kw.get("slices", {})
        # Tolerances for any slices
        dtol = kw.get("tol", 1e-4)
        tols = kw.get("tols", {})
        # Loop through slice keys
        for k in slices:
            # Get target value
            targ = slices[k]
            # Get tolerance
            tol = tols.get(k, dtol)
            # Apply the constraint
            J = np.where(np.abs(self[k] - targ) <= tol)[0]
            # Combine constraints
            I = np.intersect1d(I, J)
        # Tolerances for x-keys
        xtol = kw.get("xtol")
        # Initialize inputs
        vals = {}
        # Loop through inputs
        for k in keys:
            # Check presence
            if k not in self.coeffs:
                raise KeyError("No lookup key called '%s'" % k)
            # Get value
            V = self[k]
            # Count
            nk = len(V)
            # Check consistency
            if nk != n:
                raise ValueError(
                    ("Input key '%s' has %s entries; " % (k, nk)) +
                    ("output key '%s' has %s" % (coeff, n)))
            # Check for filtering
            # Save the value
            vals[k] = V[I]
        # Save output values
        vals[coeff] = self[coeff][I]
        # Convert to tuple
        args = tuple([vals[k] for k in (keys + [coeff])])
        # Create the interpolant
        f = scipy.interpolate.Rbf(*args,
            function=kw.get("function", "cubic"),
            smooth=kw.get("smooth", 0.0))
        # Create Radial Basis Function
        try:
            self.rbf
        except AttributeError:
            self.rbf = {}
        # Save key
        if func is None:
            # Use coefficient name
            self.rbf[coeff] = f
        else:
            # Use provided name
            self.rbf[func] = f
        # Output
        return f
    
    # Regularization
    def RegularizeByRBF(self, keys=None, coeffs=None, skey=None, **kw):
        """Regularize data to regular matrix of break points using RBFs
        
        The list of break points to which to regularize, usually generated by
        :func:`attdb.fm.GetBreakPoints`, must be present before running this
        function.
        
        :Call:
            >>> DBi = DBc.RegularizeByRBF(keys=None, coeffs=None, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient database interface
            *keys*: {``None``} | :class:`list` (:class:`str`)
                List of (ordered) input keys, default is from *DBc.bkpts*
            *coeffs*: {``None``} | :class:`list` (:class:`str`)
                List of coefficients to interpolate
            *skey*: {``None``} | :class:`str` | :class:`list` (:class:`str`)
                Optional name of slicing key(s) for RBF interpolation
            *cokeys*: {``None``} | :class:`list` (:class:`str`)
                List of dependent input keys, default is from *DBc.bkpts*
            *copykeys*: {``None``} | :class:`list` (:class:`str`)
                List of additional keys to copy (default is all)
            *function*: {``"cubic"``} | ``"multiquadric"`` | ``"linear"``
                Basis function for :func:`scipy.interpolate.Rbf`
            *tol*: {``1e-4``}  | :class:`float`
                Default tolerance to use in combination with *slices*
            *tols*: {``{}``} | :class:`dict`
                Dictionary of specific tolerances for single keys in *slices*
            *cls*: {``"DBCoeff"``} | ``"DBFM"``
                Name of class to use
        :Outputs:
            *DBi*: :class:`attdb.fm.DBCoeff`
                Coefficient database interface with regularized data
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        # Ensure presence of break points
        try:
            self.bkpts
        except AttributeError:
            raise AttributeError("Break points must be present." +
                "See GetBreakPoints()")
        # Input keys
        if keys is None:
            keys = self.bkpts.keys()
        # Number of input keys
        nkey = len(keys)
        # Get list of coefficients
        if coeffs is None:
            # Default: everything that's not a break point
            coeffs = np.setdiff1d(self.coeffs, keys)
        elif coeffs.__class__.__name__ not in ["list", "ndarray"]:
            # Convert single input to list
            coeffs = [coeff]
            
        # Get the type for the slice key
        ts = skey.__class__.__name__
        # Check for list
        if ts in ["list", "ndarray"]:
            # Get additional slice keys
            subkeys = skey[1:]
            # Single slice key
            mainkey = skey[0]
        elif skey is None:
            # No slices at all
            subkeys = []
            mainkey = None
            skey = None
        else:
            # No additional slice keys
            subkeys = []
            mainkey = skey
            # List of slice keys
            skey = [skey]
        # Interpolation keys do not include *mainkey*
        ikeys = list(keys)
        # Check for slice information to remove from interpolation list
        if skey:
            # Loop through slice keys
            for k in skey:
                # Check if present
                if k in ikeys:
                    ikeys.remove(k)
                    
        # Get full-factorial matrix at the current slice value
        X, slices = self.GetFullFactorialMatrix(skey, keys=keys)
        # Number of output points
        nX = X[keys[0]].size
        
        # Create blank database (of correct type)
        DBi = self.__class__()
        # Save the break points
        DBi.bkpts = {}
        for k in keys:
            DBi.bkpts[k] = self.bkpts[k]
        # Save the lookup values
        for k in keys:
            # Save values
            DBi[k] = X[k]
            # Append to coefficient list
            DBi.coeffs.append(k)
            
        # Perform interpolations
        for c in coeffs:
            # Status update
            if kw.get("v"):
                print("  Interpolating coefficient '%s'" % c)
            # Check for slices
            if skey is None:
                # One interpolant
                f = self.CreateRBF(c, keys, **kw)
                # Create tuple of input arguments
                args = tuple(X[k] for k in keys)
                # Evaluate coefficient
                DBi[c] = f(*args)
                
            else:
                
                # Number of slices
                nslice = slices[mainkey].size
                # Initialize data
                V = np.zeros_like(X[mainkey])
                # Loop through slices
                for i in range(nslice):
                    # Status update
                    if kw.get("v"):
                        # Get main key value
                        m = slices[mainkey][i]
                        # Get value in fixed number of characters
                        sv = ("%6g" % m)[:6]
                        # In-place status update
                        sys.stdout.write("    Slice %s=%s (%i/%i)\r"
                            % (mainkey, sv, i+1, nslice))
                        sys.stdout.flush()
                        
                    # Initialize mask
                    J = np.ones(nX, dtype="bool")
                    # Initialize slice
                    slice_i = {}
                    # Loop through keys
                    for k in skey:
                        # Get value
                        vk = slices[k][i]
                        # Constrain
                        J = np.logical_and(J, X[k]==vk)
                        # Save local slice
                        slice_i[k] = vk
                    # Get indices of slice
                    I = np.where(J)[0]
                    # Create interpolant for fixed value of *skey*
                    f = self.CreateRBF(c, ikeys, slices=slice_i, **kw)
                    # Create tuple of input arguments
                    args = tuple(X[k][I] for k in ikeys)
                    # Evaluate coefficient
                    V[I] = f(*args)
                # Save the values
                DBi[c] = V
                # Clean up prompt
                if kw.get("v"):
                    print("")
            # Add to coefficient list
            DBi.coeffs.append(c)
        # Trajectory co-keys
        cokeys = kw.get("cokeys", self.bkpts.keys())
        # Map other breakpoint keys
        for k in cokeys:
            # Skip if already present
            if k in DBi.bkpts: continue
            # Check for slices
            if mainkey is None: break
            # Check size
            if self[mainkey].size != self[k].size:
                continue
            # Regular matrix values of slice key
            M = X[mainkey]
            # Initialize data
            V = np.zeros_like(M)
            # Initialize break points
            T = []
            # Status update
            if kw.get("v"):
                print("  Mapping key '%s'" % k)
            # Loop through slice values
            for m in DBi.bkpts[mainkey]:
                # Find value of slice key matching that parameter
                i = np.where(self[mainkey] == m)[0][0]
                # Output value
                v = self[k][i]
                # Get the indices of break points with that value
                J = np.where(M == m)[0]
                # Evaluate coefficient
                V[J] = v
                # Save break point
                T.append(v)
            # Save the values
            DBi[k] = V
            # Save break points
            DBi.bkpts[k] = np.array(T)
            # Save key name if needed
            if k not in DBi.coeffs: DBi.coeffs.append(k)
            
        # Copy any other keys
        for k in kw.get("copykeys", self.coeffs):
            # Skip if already processed
            if k in DBi: continue
            # Otherwise, copy
            DBi[k] = self[k].copy()
            # Save the coefficient to the list, too
            DBi.coeffs.append(k)
            
        # Normalize trajectory
        DBi.GetTrajectory()
        # Output
        return DBi
        
    # Fill out a slice matrix
    def GetFullFactorialMatrix(self, skey=None, keys=None):
        """Get breakpoint dictionary for slice at value *v*
        
        This allows some of the break points to be scheduled.
        
        :Call:
            >>> X, slices = DBc.GetSliceMatrix(skey=None, keys=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient database interface
            *skey*: {``None``} | :class:`str` | :class:`list` (:class:`str`)
                Optional name of slicing key(s) for RBF interpolation
            *keys*: {``None``} | :class:`list` (:class:`str`)
                List of (ordered) input keys, default is from *DBc.bkpts*
        :Outputs:
            *X*: :class:`dict`
                Dictionary of full-factorial matrix
            *slices*: :class:`dict` (:class:`ndarray`)
                Array of slice values for each key in *skey*
        :Versions:
            * 2018-11-16 ``@ddalle``: First version
        """
        # Get the type for the slice key
        ts = skey.__class__.__name__
        # Check for list
        if ts in ["list", "ndarray"]:
            # Get additional slice keys
            subkeys = skey[1:]
            # Single slice key
            mainkey = skey[0]
        elif skey is None:
            # No slices at all
            subkeys = []
            mainkey = None
            skey = None
        else:
            # No additional slice keys
            subkeys = []
            mainkey = skey
            # List of slice keys
            skey = [skey]
        
        # Default key list
        if keys is None:
            # Default list
            keys = self.bkpts.keys()
        else:
            # Make a copy
            keys = list(keys)
        # Eliminate *skey* if in key list
        if mainkey in keys:
            keys.remove(mainkey)
        # Number of keys
        nkey = len(keys)
        
        # Initialize slice dictionary
        slices = {mainkey: np.zeros(0)}
        # Loop through slice keys
        for k in subkeys:
            # Initialize slice
            slices[k] = np.zeros(0)
        # Number of slice keys
        if skey is None:
            # No slices
            nskey = None
        else:
            # Get length
            nskey = len(skey)
        
        # Initialize dictionary of full-factorial matrix
        X = {}
        # Set values
        for k in keys:
            X[k] = np.zeros(0)
        
        # Slice check
        if mainkey is None:
            # No values to check
            M = np.zeros(1)
        else:
            # Get breakpoints for specified value
            M = self.bkpts[mainkey]
            # Also keep track of slice key values
            X[mainkey] = np.zeros(0)
        
        # Loop through slice values
        for (im, m) in enumerate(M):
            # Initialize matrix for this slice
            Xm = {}
            # Initialize slice values for this slice
            Xs = {}
            if mainkey:
                Xs[mainkey] = np.array([m])
        
            # Copy values
            for k in keys:
                # Get values
                Vm = self.bkpts[k]
                # Type of first entry
                t = self.bkpts[k][0].__class__.__name__
                # Check if it's a scheduled key; will be a list
                if t in ["list", "ndarray"]:
                    # Get break points for this slice key value
                    Vm = Vm[im]
                # Save the values
                Xm[k] = Vm
                # Save slice if appropriate
                if k in subkeys:
                    Xs[k] = Vm
                
            # Loop through break point keys to create full-factorial inputs
            for i in range(1, nkey):
                # Name of first key
                k1 = keys[i]
                # Loop through keys 0 to *i*-1
                for j in range(i):
                    # Name of second key
                    k2 = keys[j]
                    # Create N+1 dimensional interpolation
                    x1, x2 = np.meshgrid(Xm[k1], Xm[k2])
                    # Flatten
                    Xm[k2] = x2.flatten()
                    # Save first key if *j* ix 0
                    if j == i-1:
                        Xm[k1] = x1.flatten()
                
            # Loop through slice keys to create full-factorial inputs
            for i in range(1, nskey):
                # Name of first key
                k1 = skey[i]
                # Loop through keys 0 to *i*-1
                for j in range(i):
                    # Name of second key
                    k2 = skey[j]
                    # Create N+1 dimensional interpolation
                    x1, x2 = np.meshgrid(Xs[k1], Xs[k2])
                    # Flatten
                    Xs[k2] = x2.flatten()
                    # Save first key if *j* ix 0
                    if j == i-1:
                        Xs[k1] = x1.flatten()
            
            # Save values
            for k in keys:
                X[k] = np.hstack((X[k], Xm[k]))
            # Process slices
            if mainkey is not None:
                # Append to *skey* matrix
                X[mainkey] = np.hstack((X[mainkey], m*np.ones_like(Xm[k])))
                # Save slice full-factorial matrix
                for k in skey:
                    slices[k] = np.hstack((slices[k], Xs[k]))
                
        # Output
        return X, slices
        
    # Regularization
    def SaveRBFs(self, keys=None, coeffs=None, skey=None, **kw):
        """Regularize data to regular matrix of break points using RBFs
        
        The list of break points to which to regularize, usually generated by
        :func:`attdb.fm.GetBreakPoints`, must be present before running this
        function.
        
        :Call:
            >>> DBi = DBc.RegularizeByRBF(keys=None, coeffs=None, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient database interface
            *keys*: {``None``} | :class:`list` (:class:`str`)
                List of (ordered) input keys, default is from *DBc.bkpts*
            *coeffs*: {``None``} | :class:`list` (:class:`str`)
                List of coefficients to interpolate
            *skey*: {``None``} | :class:`str` | :class:`list` (:class:`str`)
                Optional name of slicing key(s) for RBF interpolation
            *cokeys*: {``None``} | :class:`list` (:class:`str`)
                List of dependent input keys, default is from *DBc.bkpts*
            *copykeys*: {``None``} | :class:`list` (:class:`str`)
                List of additional keys to copy (default is all)
            *function*: {``"cubic"``} | ``"multiquadric"`` | ``"linear"``
                Basis function for :func:`scipy.interpolate.Rbf`
            *tol*: {``1e-4``}  | :class:`float`
                Default tolerance to use in combination with *slices*
            *tols*: {``{}``} | :class:`dict`
                Dictionary of specific tolerances for single keys in *slices*
            *cls*: {``"DBCoeff"``} | ``"DBFM"``
                Name of class to use
        :Outputs:
            *DBi*: :class:`attdb.fm.DBCoeff`
                Coefficient database interface with regularized data
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        # Ensure presence of break points
        try:
            self.bkpts
        except AttributeError:
            raise AttributeError("Break points must be present." +
                "See GetBreakPoints()")
        # Input keys
        if keys is None:
            keys = self.bkpts.keys()
        # Number of input keys
        nkey = len(keys)
        # Get list of coefficients
        if coeffs is None:
            # Default: everything that's not a break point
            coeffs = np.setdiff1d(self.coeffs, keys)
        elif coeffs.__class__.__name__ not in ["list", "ndarray"]:
            # Convert single input to list
            coeffs = [coeff]
            
        # Get the type for the slice key
        ts = skey.__class__.__name__
        # Check for list
        if ts in ["list", "ndarray"]:
            # Get additional slice keys
            subkeys = skey[1:]
            # Single slice key
            mainkey = skey[0]
        elif skey is None:
            # No slices at all
            subkeys = []
            mainkey = None
            skey = None
        else:
            # No additional slice keys
            subkeys = []
            mainkey = skey
            # List of slice keys
            skey = [skey]
        # Interpolation keys do not include *mainkey*
        ikeys = list(keys)
        # Check for slice information to remove from interpolation list
        if skey:
            # Loop through slice keys
            for k in skey:
                # Check if present
                if k in ikeys:
                    ikeys.remove(k)
        
        # Create blank database (of correct type)
        DBi = self.__class__()
        # Save the break points
        DBi.bkpts = {}
        for k in keys:
            DBi.bkpts[k] = self.bkpts[k]
        
        
        # Save the lookup values
        for k in keys:
            # Append to coefficient list
            DBi.coeffs.append(k)
            
        # Perform interpolations
        for c in coeffs:
            # Status update
            if kw.get("v"):
                print("  Creating RBF for coefficient '%s'" % c)
            # Check for slices
            if skey is None:
                # One interpolant
                f = self.CreateRBF(c, keys, **kw)
                # Save the RBF instead of evaluating it
                DBi[c] = f
                
            else:
                
                # Number of slices
                nslice = self.bkpts[mainkey].size
                # Loop through slices
                for i in range(nslice):
                    # Initialize dictionary of RBFs
                    F = {}
                    # Get main key value
                    m = slices[mainkey][i]
                    # Create the slice
                    slice_i = {mainkey: m}
                    # Status update
                    if kw.get("v"):
                        # Get value in fixed number of characters
                        sv = ("%6g" % m)[:6]
                        # In-place status update
                        sys.stdout.write("    Slice %s=%s (%i/%i)\r"
                            % (mainkey, sv, i+1, nslice))
                        sys.stdout.flush()
                        
                    # Create interpolant for fixed value of *skey*
                    f = self.CreateRBF(c, ikeys, slices=slice_i, **kw)
                    # Save the RBF
                    F[m] = f
                # Save the RBF
                DBi[c] = F
                # Clean up prompt
                if kw.get("v"):
                    print("")
            # Add to coefficient list
            DBi.coeffs.append(c)
            
        # Output
        return DBi
   
   # --- Multilinear ---
    # Linear interpolation function
    def InterpMonolinear(self, m, k1, coeffs=None):
        """Perform simple linear interpolation
        
        :Call:
            >>> c = DBc.InterpMonolinear(m, k1, coeff)
            >>> C = DBc.InterpMonolinear(m, k1, coeffs=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient database interface
            *m*: :class:`float`
                Lookup value 1, for example Mach number
            *k1*: :class:`str`
                Name of lookup parameter 1
            *coeff*: :class:`str`
                Request for single coefficient lookup
            *coeffs*: {``None``} | :class:`list` (:class:`str`)
                List of coefficients to interpolate
            *DBc[coeff]*: :class:`np.ndarray`
                Properly sized array of output data
        :Outputs:
            *C* :class:`list`\ [:class:`float`]
                Interpolated coefficient at each coefficient in *coeffs*
            *c*: :class:`float`
                Interpolated coefficient for *coeff*
        :Versions:
            * 2018-06-11 ``@ddalle``: First object-orientation version
        """
        # Check for break points
        try:
            self.bkpts
        except AttributeError:
            raise AttributeError("No break point dictionary present")
        # Input types
        t1 = (m.__class__.__name__ in ["list", "ndarray"])
        # Linear interpolation
        if t1:
            # Beta vector
            n = len(m)
            return np.array([interp_monolinear(m[i],
                self.bkpts, k1, self, coeffs=coeffs)
                for i in range(n)])
        else:
            # Sclar
            return interp_monolinear(
                m, self.bkpts, k1, self, coeffs=coeffs)
        
    # Bilinear interpolation function
    def InterpBilinear(self, a, b, k1, k2, coeffs=None):
        """Perform bilinear interpolation
        
        :Call:
            >>> c = DBc.InterpBilinear(a, b, k1, k2, coeff)
            >>> C = DBc.InterpBilinear(a, b, k1, k2, coeffs=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient database interface
            *a*: :class:`float`
                Lookup value 1, for example angle of attack
            *b*: :class:`float`
                Lookup value 2, for example sideslip angle
            *k1*: :class:`str`
                Name of lookup parameter 1
            *k2*: :class:`str`
                Name of lookup parameter 2
            *coeff*: :class:`str`
                Request for single coefficient lookup
            *coeffs*: {``None``} | :class:`list` (:class:`str`)
                List of coefficients to interpolate
            *DBc[coeff]*: :class:`np.ndarray`
                Properly sized array of output data
        :Outputs:
            *C* :class:`list`\ [:class:`float`]
                Interpolated coefficient at each coefficient in *coeffs*
            *c*: :class:`float`
                Interpolated coefficient for *coeff*
        :Versions:
            * 2018-06-11 ``@ddalle``: First object-oriented version
        """
        # Check for break points
        try:
            self.bkpts
        except AttributeError:
            raise AttributeError("No break point dictionary present")
        # Input types
        t2 = (a.__class__.__name__ in ["list", "ndarray"])
        t3 = (b.__class__.__name__ in ["list", "ndarray"])
        # Bilinear interpolation
        if t2:
            # Number of points
            n = len(a)
            if t3:
                # Alpha and beta vectors
                return np.array([interp_bilinear(a[i], b[i],
                    self.bkpts, k1, k2, self, coeffs=coeffs)
                    for i in range(n)])
            else:
                # Alpha vector
                return np.array([interp_bilinear(a[i], b,
                    self.bkpts, k1, k2, self, coeffs=coeffs)
                    for i in range(n)])
        else:
            if t3:
                # Beta vector
                n = len(b)
                return np.array([interp_bilinear(a, b[i],
                    self.bkpts, k1, k2, self, coeffs=coeffs)
                    for i in range(n)])
            else:
                # Sclar
                return interp_bilinear(
                    a, b, self.bkpts, k1, k2, self, coeffs=coeffs)
        
    # Trilinear interpolation function
    def InterpTrilinear(self, m, a, b, k1, k2, k3, coeffs=None):
        """Perform trilinear interpolation
        
        :Call:
            >>> c = DBc.InterpTrilinear(m, a, b, k1, k2, k3, coeff)
            >>> C = DBc.InterpTrilinear(m, a, b, k1, k2, k3, coeffs=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient database interface
            *m*: :class:`float`
                Lookup value 1, for example Mach number
            *a*: :class:`float`
                Lookup value 2, for example angle of attack
            *b*: :class:`float`
                Lookup value 3, for example sideslip angle
            *bkpt*: :class:`dict`
                Dictionary of interpolation break points
            *k1*: :class:`str`
                Name of lookup parameter 1
            *k2*: :class:`str`
                Name of lookup parameter 2
            *k3*: :class:`str`
                Name of lookup parameter 3
            *coeff*: :class:`str`
                Request for single coefficient lookup
            *coeffs*: {``None``} | :class:`list` (:class:`str`)
                List of coefficients to interpolate
            *DBc[coeff]*: :class:`np.ndarray`
                Properly sized array of output data
        :Outputs:
            *C* :class:`list`\ [:class:`float`]
                Interpolated coefficient at each coefficient in *coeffs*
            *c*: :class:`float`
                Interpolated coefficient for *coeff*
        :Versions:
            * 2017-01-30 ``@ddalle``: First version
            * 2017-07-13 ``@ddalle``: Generic version
        """
        # Check for break points
        try:
            self.bkpts
        except AttributeError:
            raise AttributeError("No break point dictionary present")
        # Input types
        t1 = (m.__class__.__name__ in ["list", "ndarray"])
        t2 = (a.__class__.__name__ in ["list", "ndarray"])
        t3 = (b.__class__.__name__ in ["list", "ndarray"])
        # Trilinear interpolation
        if t1:
            # Number of points
            n = len(m)
            if t2:
                if t3:
                    # All three are vectors
                    return np.array([interp_trilinear(m[i], a[i], b[i],
                        self.bkpts, k1, k2, k3, self, coeffs=coeffs)
                        for i in range(n)])
                else:
                    # Mach and alpha vectors
                    return np.array([interp_trilinear(m[i], a[i], b,
                        self.bkpts, k1, k2, k3, self, coeffs=coeffs)
                        for i in range(n)])
            else:
                if t3:
                    # Mach and beta vectors
                    return np.array([interp_trilinear(m[i], a, b[i],
                        self.bkpts, k1, k2, k3, self, coeffs=coeffs)
                        for i in range(n)])
                else:
                    # Mach vector
                    return np.array([interp_trilinear(m[i], a, b,
                        self.bkpts, k1, k2, k3, self, coeffs=coeffs)
                        for i in range(n)])
        else:
            if t2:
                # Number of points
                n = len(a)
                if t3:
                    # Alpha and beta vectors
                    return np.array([interp_trilinear(m, a[i], b[i],
                        self.bkpts, k1, k2, k3, self, coeffs=coeffs)
                        for i in range(n)])
                else:
                    # Alpha vector
                    return np.array([interp_trilinear(m, a[i], b,
                        self.bkpts, k1, k2, k3, self, coeffs=coeffs)
                        for i in range(n)])
            else:
                if t3:
                    # Beta vector
                    n = len(b)
                    return np.array([interp_trilinear(m, a, b[i],
                        self.bkpts, k1, k2, k3, self, coeffs=coeffs)
                        for i in range(n)])
                else:
                    # Sclar
                    return interp_trilinear(
                        m, a, b, self.bkpts, k1, k2, k3, self, coeffs=coeffs)

  # >
# class DBCoeff


# Estimate *xCLM*
def estimate_xCLM(self, DCLM, DCN):
    """Estimate reference *x* for *UCLM* calculations
    
    :Call:
        >>> xCLM = estimate_xCLM(self, DCLM, DCN)
    :Inputs:
        *self*: :class:`DBFM`
            Force & moment database with *self.xMRP* and *self.Lref*
        *DCLM*: :class:`np.ndarray`\ [:class:`float`]
            Deltas between two databases' *CLM* values
        *DCN*: :class:`np.ndarray`\ [:class:`float`]
            Deltas between two databases' *CN* values
    :Outputs:
        *xCLM*: :class:`float`
            Reference *x* that minimizes *UCLM*
    :Versions:
        * 2019-02-20 ``@ddalle``: First version
    """
    # Original MRP in nondimensional coordinates
    xMRP = self.xMRP / self.Lref
    # Calculate mean deltas
    muDCN = np.mean(DCN)
    muDCLM = np.mean(DCLM)
    # Reduced deltas
    DCN1  = DCN  - muDCN
    DCLM1 = DCLM - muDCLM
    # Calculate reference *xCLM* for *UCLM*
    xCLM = xMRP - np.sum(DCLM1*DCN1)/np.sum(DCN1*DCN1)
    # Output
    return xCLM
    
# Estimate *xCLN*
def estimate_xCLN(self, DCLN, DCY):
    """Estimate reference *x* for *UCLN* calculations
    
    :Call:
        >>> xCLN = estimate_xCLM(self, DCLN, DCY)
    :Inputs:
        *self*: :class:`DBFM`
            Force & moment database with *self.xMRP* and *self.Lref*
        *DCLN*: :class:`np.ndarray`\ [:class:`float`]
            Deltas between two databases' *CLN* values
        *DCY*: :class:`np.ndarray`\ [:class:`float`]
            Deltas between two databases' *CY* values
    :Outputs:
        *xCLN*: :class:`float`
            Reference *x* that minimizes *UCLN*
    :Versions:
        * 2019-02-20 ``@ddalle``: First version
    """
    # Original MRP in nondimensional coordinates
    xMRP = self.xMRP / self.Lref
    # Calculate mean deltas
    muDCY = np.mean(DCY)
    muDCLN = np.mean(DCLN)
    # Reduced deltas
    DCY1  = DCY  - muDCY
    DCLN1 = DCLN - muDCLN
    # Calculate reference *xCLM* for *UCLM*
    xCLN = xMRP - np.sum(DCLN1*DCY1)/np.sum(DCY1*DCY1)
    # Output
    return xCLN

# Shift *CLM* deltas to some *x*
def shift_DCLM(self, DCLM, DCN, xCLM):
    """Shift *CLM* deltas to reference *x*
    
    :Call:
        >>> DCLM2 = shift_DCLM(self, DCLM, DCN, xCLM)
    :Inputs:
        *self*: :class:`DBFM`
            Force & moment database with *self.xMRP* and *self.Lref*
        *DCLM*: :class:`np.ndarray`\ [:class:`float`]
            Deltas between two databases' *CLM* values
        *DCN*: :class:`np.ndarray`\ [:class:`float`]
            Deltas between two databases' *CN* values
        *xCLM*: :class:`float`
            Reference *x* that minimizes *UCLM*
    :Outputs:
        *DCLM2*: :class:`np.ndarray`\ [:class:`float`]
            Deltas in *CLM* about *xCLM*
    :Versions:
        * 2019-02-20 ``@ddalle``: First version
    """
    # Original MRP in nondimensional coordinates
    xMRP = self.xMRP / self.Lref
    # Shift deltas to *xCLM*
    DCLM2 = DCLM + (xCLM-xMRP)*DCN
    # Output
    return DCLM2

# Shift *CLN* deltas to some *x*
def shift_DCLN(self, DCLN, DCY, xCLN):
    """Shift *CLN* deltas to reference *x*
    
    :Call:
        >>> DCLN2 = shift_DCLN(self, DCLN, DCY, xCLN)
    :Inputs:
        *self*: :class:`DBFM`
            Force & moment database with *self.xMRP* and *self.Lref*
        *DCLN*: :class:`np.ndarray`\ [:class:`float`]
            Deltas between two databases' *CLN* values
        *DCY*: :class:`np.ndarray`\ [:class:`float`]
            Deltas between two databases' *CY* values
        *xCLN*: :class:`float`
            Reference *x* that minimizes *UCLN*
    :Outputs:
        *DCLN2*: :class:`np.ndarray`\ [:class:`float`]
            Deltas in *CLN* about *xCLN*
    :Versions:
        * 2019-02-20 ``@ddalle``: First version
    """
    # Original MRP in nondimensional coordinates
    xMRP = self.xMRP / self.Lref
    # Shift deltas to *xCLM*
    DCLN2 = DCLN + (xCLN-xMRP)*DCY
    # Output
    return DCLN2
# def shift_DCLN


# Standard converters: alpha
def convert_alpha(*a, **kw):
    """Estimate angle of attack from other parameters
    
    :Call:
        >>> alph = convert_alpha(*a, **kw)
    :Inputs:
        *aoap*, *alpha_t*: :class:`float` | :class:`np.ndarray`
            Total angle of attack [deg]
        *phip*, *phi*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Vertical-to-wind roll angle [deg]
    :Outputs:
        *alph*: :class:`float` | :class:`np.ndarray`
            Angle of attack [deg]
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
    """
    # Get total angle of attack and roll angle
    aoap = kw.get("alpha_t", kw.get("aoap"))
    phip = kw.get("phi",     kw.get("phip", 0.0))
    # check if both are present
    if (aoap is not None) and (phip is not None):
        # Convert to alpha/beta
        alph, beta = convert.AlphaTPhi2AlphaBeta(aoap, phip)
        # Save value
        return alph

# Standard converters: beta
def convert_beta(*a, **kw):
    """Estimate sideslip angle from other parameters
    
    :Call:
        >>> beta = convert_beta(*a, **kw)
    :Inputs:
        *aoap*, *alpha_t*: :class:`float` | :class:`np.ndarray`
            Total angle of attack [deg]
        *phip*, *phi*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Vertical-to-wind roll angle [deg]
    :Outputs:
        *beta*: :class:`float` | :class:`np.ndarray`
            Sideslip angle [deg]
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
    """
    # Get total angle of attack and roll angle
    aoap = kw.get("alpha_t", kw.get("aoap"))
    phip = kw.get("phi",     kw.get("phip", 0.0))
    # check if both are present
    if (aoap is not None) and (phip is not None):
        # Convert to alpha/beta
        alph, beta = convert.AlphaTPhi2AlphaBeta(aoap, phip)
        # Save value
        return beta

# Standard converters: aoap
def convert_aoap(*a, **kw):
    """Estimate total angle of attack from other parameters
    
    :Call:
        >>> aoap = convert_aoap(*a, **kw)
    :Inputs:
        *alph*: :class:`float` | :class:`np.ndarray`
            Angle of attack [deg]
        *beta*: :class:`float` | :class:`np.ndarray`
            Sideslip angle [deg]
    :Outputs:
        *aoap*: :class:`float` | :class:`np.ndarray`
            Total angle of attack [deg]
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
    """
    # Get total angle of attack and roll angle
    alph = kw.get("alpha", kw.get("aoa", kw.get("alph")))
    beta = kw.get("beta",  kw.get("aos"))
    # check if both are present
    if (alph is not None) and (beta is not None):
        # Convert to alpha/beta
        aoap, phip = convert.AlphaBeta2AlphaTPhi(alph, beta)
        # Save value
        return aoap

# Standard converters: phip
def convert_phip(*a, **kw):
    """Estimate velocity roll angle from other parameters
    
    :Call:
        >>> aoap = convert_aoap(*a, **kw)
    :Inputs:
        *alph*: :class:`float` | :class:`np.ndarray`
            Angle of attack [deg]
        *beta*: :class:`float` | :class:`np.ndarray`
            Sideslip angle [deg]
    :Outputs:
        *phip*: :class:`float` | :class:`np.ndarray`
            Vertical-to-wind roll angle [deg]
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
    """
    # Get total angle of attack and roll angle
    alph = kw.get("alpha", kw.get("aoa", kw.get("alph")))
    beta = kw.get("beta",  kw.get("aos"))
    # check if both are present
    if (alph is not None) and (beta is not None):
        # Convert to alpha/beta
        aoap, phip = convert.AlphaBeta2AlphaTPhi(alph, beta)
        # Save value
        return phip
# def convert_phip

# Special evaluators: CLM vs x
def eval_CLMX(FM, *a, **kw):
    """Evaluate *CLM* about arbitrary *x* moment reference point
    
    :Call:
        >>> CLMX = eval_CLMX(FM, *a, **kw)
    :Inputs:
        *FM*: :class:`DBCoeffFM`
            Force & moment database instance
        *a*: :class:`tuple`
            Arguments to call ``FM("CLM", *a)``, with *xMRP* additional input
        *kw*: :class:`dict`
            Keywords used as alternate definition of *a*
    :Outputs:
        *CLMX*: :class:`float` | :class:`np.ndarray`
            Pitching moment about arbitrary *xMRP*
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
    """
    # *xMRP* of original data
    xmrp = FM.xMRP / FM.Lref
    # Number of original arguments
    nf = len(FM.eval_args["CLM"])
    # Get value for *xMRP*
    xMRP = FM.get_arg_value(nf, "xMRP", *a, **kw)
    # Check for an *xhat*
    xhat = kw.get("xhat", xMRP/FM.Lref)
    # Evaluate main functions
    CLM = FM("CLM", *a, **kw)
    CN  = FM("CN",  *a, **kw)
    # Transfer
    return CLM + (xhat-xmrp)*CN

# Special evaluators: CLN vs x
def eval_CLNX(FM, *a, **kw):
    """Evaluate *CLN* about arbitrary *x* moment reference point
    
    :Call:
        >>> CLNX = eval_CLNX(FM, *a, **kw)
    :Inputs:
        *FM*: :class:`DBCoeffFM`
            Force & moment database instance
        *a*: :class:`tuple`
            Arguments to call ``FM("CLM", *a)``, with *xMRP* additional input
        *kw*: :class:`dict`
            Keywords used as alternate definition of *a*
    :Outputs:
        *CLNX*: :class:`float` | :class:`np.ndarray`
            Pitching moment about arbitrary *xMRP*
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
    """
    # *xMRP* of original data
    xmrp = FM.xMRP / FM.Lref
    # Number of original arguments
    nf = len(FM.eval_args["CLN"])
    # Get value for *xMRP*
    xMRP = FM.get_arg_value(nf, "xMRP", *a, **kw)
    # Check for an *xhat*
    xhat = kw.get("xhat", xMRP/FM.Lref)
    # Evaluate main functions
    CLN = FM("CLN", *a, **kw)
    CY  = FM("CY",  *a, **kw)
    # Transfer
    return CLN + (xhat-xmrp)*CY
# def eval_CLNX

# Evaluate *UCLM* about different x
def eval_UCLMX(FM, *a, **kw):
    """Evaluate *UCLM* about arbitrary *x* moment reference point
    
    :Call:
        >>> UCLMX = eval_UCLMX(FM, *a, **kw)
    :Inputs:
        *FM*: :class:`DBCoeffFM`
            Force & moment database instance
        *a*: :class:`tuple`
            Arguments to call ``FM("CLM", *a)``, with *xMRP* additional input
        *kw*: :class:`dict`
            Keywords used as alternate definition of *a*
    :Outputs:
        *UCLMX*: :class:`float` | :class:`np.ndarray`
            Pitching moment uncertainty about arbitrary *xMRP*
    :Versions:
        * 2019-03-13 ``@ddalle``: First version
    """
    # *xMRP* of original data
    xmrp = FM.xMRP / FM.Lref
    # Number of original arguments
    nf = len(FM.eval_args["CLM"])
    # Get value for *xMRP*
    xMRP = FM.get_arg_value(nf, "xMRP", *a, **kw)
    # Check for an *xhat*
    xhat = kw.get("xhat", xMRP/FM.Lref)
    # Evaluate main functions
    UCLM = FM("UCLM", *a, **kw)
    xCLM = FM("xCLM", *a, **kw)
    UCN  = FM("UCN",  *a, **kw)
    # Transfer
    UCLMX = np.sqrt(UCLM*UCLM + ((xCLM-xhat)*UCN)**2)
    # Output
    return UCLMX

# Evaluate *UCLN* about different x
def eval_UCLNX(FM, *a, **kw):
    """Evaluate *UCLN* about arbitrary *x* moment reference point
    
    :Call:
        >>> UCLNX = eval_UCLNX(FM, *a, **kw)
    :Inputs:
        *FM*: :class:`DBCoeffFM`
            Force & moment database instance
        *a*: :class:`tuple`
            Arguments to call ``FM("CLM", *a)``, with *xMRP* additional input
        *kw*: :class:`dict`
            Keywords used as alternate definition of *a*
    :Outputs:
        *UCLNX*: :class:`float` | :class:`np.ndarray`
            Pitching moment uncertainty about arbitrary *xMRP*
    :Versions:
        * 2019-03-13 ``@ddalle``: First version
    """
    # *xMRP* of original data
    xmrp = FM.xMRP / FM.Lref
    # Number of original arguments
    nf = len(FM.eval_args["CLN"])
    # Get value for *xMRP*
    xMRP = FM.get_arg_value(nf, "xMRP", *a, **kw)
    # Check for an *xhat*
    xhat = kw.get("xhat", xMRP/FM.Lref)
    # Evaluate main functions
    UCLN = FM("UCLN", *a, **kw)
    xCLN = FM("xCLN", *a, **kw)
    UCY  = FM("UCY",  *a, **kw)
    # Transfer
    UCLNX = np.sqrt(UCLN*UCLN + ((xCLN-xhat)*UCY)**2)
    # Output
    return UCLNX
# def eval_UCLNX
    


# Generic force & moment class
class DBCoeffFM(DBCoeff):
    """Generic coefficient database and interpolation class
    
    :Call:
        >>> DBc = DBCoeffFM(mat=None, xls=None, csv=None, **kw)
    :Inputs:
        *mat*: :class:`str`
            Name of Matlab file to read
        *xls*: :class:`str` | :class:`xlrd.book.Book`
            Name of spreadsheet or interface to open workbook
        *csv*: :class:`str`
            Name of CSV file to read
        *sheet*: {``None``} | :class:`str`
            Name of worksheet to read if reading from spreadsheet
        *trajectory*: {``{}``} | ``{"mach": "MACH", ...}`` | :class:`dict`
            Dictionary of alternative names for standard trajectory keys
    :Outputs:
        *DBc*: :class:`attdb.fm.DBCoeffFM`
            Coefficient lookup database
        *DBc.coeffs*: :class:`list` (:class:`str` | :class:`unicode`)
            List of coefficients present in the database
        *DBc[coeff]*: :class:`np.ndarray`\ [:class:`float`]
            Data for coefficient named *coeff*
    :Versions:
        * 2018-06-08 ``@ddalle``: First version
    """
  # ==========
  # Config
  # ==========
  # <
   # --- Builtins ---
    # Initialization method
    def __init__(self, mat=None, xls=None, csv=None, **kw):
        """Initialization method
        
        :Versions:
            * 2018-06-11 ``@ddalle``: First version
            * 2018-07-20 ``@ddalle``: :class:`DBFM` --> :class:`DBCoeffMAB`
            * 2019-02-27 ``@ddalle``: New :class:`DBCoeffFM`
            * 2019-02-28 ``@ddalle``: Moved to separate method
        """
        # Refer to parent class's initialization method
        self.init_DBCoeff(mat=mat, xls=xls, csv=csv, **kw)
        
   # --- Init ---
    # Initialization method
    def init_DBCoeff(self, mat=None, xls=None, csv=None, **kw):
        """Initialization method for :class:`DBCoeffFM`
        
        :Call:
            >>> DBc.init_DBCoeff(mat=None, xls=None, csv=None, **kw)
        :Inputs:
            *mat*: :class:`str`
                Name of Matlab file to read
            *xls*: :class:`str` | :class:`xlrd.book.Book`
                Name of spreadsheet or interface to open workbook
            *csv*: :class:`str`
                Name of CSV file to read
            *sheet*: {``None``} | :class:`str`
                Name of worksheet to read if reading from spreadsheet
            *trajectory*: {``{}``} | ``{"mach": "MACH", ...}`` | :class:`dict`
                Dictionary of alternative names for standard trajectory keys
        :Versions:
            * 2018-06-11 ``@ddalle``: First version
            * 2018-07-20 ``@ddalle``: :class:`DBFM` --> :class:`DBCoeffMAB`
            * 2019-02-27 ``@ddalle``: New :class:`DBCoeffFM`
            * 2019-02-28 ``@ddalle``: Moved to separate method
        """
        # Refer to parent class's initialization method
        self.read_db1_DBCoeff(mat=mat, xls=xls, csv=csv, **kw)
        # Argument conversions
        self.set_arg_converter("alpha",   convert_alpha)
        self.set_arg_converter("aoa",     convert_alpha)
        self.set_arg_converter("beta",    convert_beta)
        self.set_arg_converter("aos",     convert_beta)
        self.set_arg_converter("alpha_t", convert_aoap)
        self.set_arg_converter("aoap",    convert_aoap)
        self.set_arg_converter("phi",     convert_phip)
        self.set_arg_converter("phip",    convert_phip)
        # Default values (in case they could be read from file)
        Lref = getattr(self, "Lref", 333.0)
        Aref = getattr(self, "Aref", 87092.01694098)
        xMRP = getattr(self, "xMRP", 4759.68)
        yMRP = getattr(self, "yMRP", 0.0)
        zMRP = getattr(self, "zMRP", 0.0)
        # Save reference parameters
        self.Lref = kw.get("Lref", kw.get("RefLength", Lref))
        self.Aref = kw.get("Aref", kw.get("RefArea", Aref))
        # Save moment reference point
        self.xMRP = kw.get("xMRP", xMRP)
        self.yMRP = kw.get("yMRP", yMRP)
        self.zMRP = kw.get("zMRP", zMRP)
        # Set default argument values
        self.set_arg_default("xMRP", xMRP)
        self.set_arg_default("yMRP", yMRP)
        self.set_arg_default("zMRP", zMRP)
        # UQ keys
        self.uq_coeffs = {
            "CA": "UCA",
            "CY": "UCY",
            "CN": "UCN",
            "CLL": "UCLL",
            "CLM": "UCLM",
            "CLN": "UCLN",
        }
        # Extra keys
        self.uq_keys_extra = {
            "UCLM": ["xCLM"],
            "UCLN": ["xCLN"],
        }
        # Shift keys
        self.uq_keys_shift = {
            "CLM": ["CN"],
            "CLN": ["CY"],
        }
        # Functions to calculate extra keys
        self.uq_funcs_extra = {
            "xCLM": estimate_xCLM,
            "xCLN": estimate_xCLN,
        }
        # Functions to perform shifting
        self.uq_funcs_shift = {
            "UCLM": shift_DCLM,
            "UCLN": shift_DCLN,
        }
        # Remove previously unused kwargs
        kw.pop("v", None)
        kw.pop("txt", None)
        # Form a trajectory
        self.GetTrajectory(**kw)
   
   # --- Copy ---
    # Copy
    def copy(self):
        """Copy a coefficient lookup database
        
        :Call:
            >>> DBi = DBc.copy()
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
        :Outputs:
            *DBi*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        # Form a new database
        DBi = self.__class__()
        # Copy relevant parts
        self.copy_DBCoeffFM(DBi)
        # Output
        return DBi
    
    # Case-specific copy
    def copy_DBCoeffFM(self, DBi):
        """Copy methods for :class:`DBFM` and subclasses
        
        :Call:
            >>> DBc.copy_DBCoeffFM(DBi)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeffFM`
                Coefficient lookup database
            *DBi*: :class:`attdb.fm.DBCoeffFM`
                Coefficient lookup database
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        # Form a new database
        DBi = self.__class__()
        # Copy relevant parts
        self.copy_db1_DBCoeff(DBi)
        # Copy MRP and ref values
        DBi.Lref = self.Lref
        DBi.Aref = self.Aref
        DBi.xMRP = self.xMRP
        DBi.yMRP = self.yMRP
        DBi.zMRP = self.zMRP
        # Copy trajectory
        DBi.x = self.x.copy()
        # Output
        return DBi
  # >
  
  # ==============
  # Eval/CALL
  # ==============
  # <
   # --- Moments ---
    # Declare functions for shifting moments in *x*
    def set_xMRP_shift_funcs(self):
        # Get inputs to *CLM*
        args_CLM = self.eval_args["CLM"]
        args_UCLM = self.eval_args["UCLM"]
        # Append *xMRP* to it
        args_CLMX = args_CLM + ["xMRP"]
        args_UCLMX = args_UCLM + ["xMRP"]
        # Declare function
        self.SetEvalMethod(["CLMX"], "function", args_CLMX, eval_CLMX)
        self.SetEvalMethod(["CLNX"], "function", args_CLMX, eval_CLNX)
        # Declare UQ functions
        self.SetEvalMethod(["UCLMX"], "function", args_UCLMX, eval_UCLMX)
        self.SetEvalMethod(["UCLNX"], "function", args_UCLMX, eval_UCLNX)
  # >
  
  
  # ===========
  # I/O
  # ===========
  # <
   # --- Matlab Files ---
    # Read a Matlab file
    def ReadMat(self, fmat, **kw):
        """Read a Matlab ``.mat`` file to import data
        
        :Call:
            >>> DBc.ReadMat(fmat, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient database interface
            *fmat*: :class:`str`
                Name of Matlab data file
        :Versions:
            * 2018-07-06 ``@ddalle``: First version
        """
        # Call the parent method
        DBCoeff.ReadMat(self, fmat, **kw)
        # Loop through any additional keys
        for k in self.mat:
            # Skip if regular key
            if k in ["DB", "bkpts", "x"]: continue
            # Save values (hopefully a scalar)
            setattr(self,k, self.mat[k])
        
    # Prepare a Matlab file output
    def PrepareMat(self, **kw):
        """Write a generic Matlab ``.mat`` file as a struct
        
        :Call:
            >>> DBc.WriteMat(fmat, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *fmat*: :class:`str`
                Name of Matlab data file
            *FM*: :class:`dict` (:class:`mat_struct`)
                Partially constructed Matlab output object
        :Versions:
            * 2018-07-06 ``@ddalle``: First version
            * 2019-02-27 ``@ddalle``: 
        """
        # Create a struct of variables to save
        FM = self.prepmat_db1_DBCoeff(**kw)
        
        # Create trajectory
        try:
            # Call the instance to see if it exists
            self.x
            # Create struct
            x = siom.mat_struct()
            # Loop through trajectory keys
            for k in self.x.keys():
                # Set the value
                setattr(x,k, self.x[k])
            # Save the key names: Python keys --> Matlab fields
            x._fieldnames = self.x.keys()
            # Save to output dictionary
            FM["x"] = x
        except AttributeError:
            # No trajectory?
            pass
        
        # Save references
        FM["Lref"] = self.Lref
        FM["Aref"] = self.Aref
        FM["xMRP"] = self.xMRP
        FM["yMRP"] = self.yMRP
        FM["zMRP"] = self.zMRP
        
        # Output
        return FM
  # >
  
  # =============
  # Filtering
  # =============
  # <
   # --- Repeats ---
    # Replace repeat points with their average
    def FilterRepeats(self, TolCons={}, EqCons=[], **kw):
        """Remove duplicate points (e.g. before creating RBF)
        
        :Call:
            >>> DB = DBc.FilterRepeats(TolCons={}, EqCons=[], **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeffMAB`
                Coefficient database interface
            *TolCons*: {``{}``} | :class:`dict`\ [:class:`float`]
                Dictionary of tolerance constraints for points considered equal
            *EqCons*: {``[]``} | :class:`list` (:class:`str`)
                List of keys only considered equal if exact
            *I*: {``None``} | :class:`np.ndarray`\ [:class:`int`]
                Only consider specified indices
            *f*: {``None``} | :class:`func`
                Averaging function, default is to use :func:`np.mean`
        :Outputs:
            *DB*: :class:`attdb.fm.DBCoeffMAB`
                Possibly smaller version of database
        :Versions:
            * 2018-07-20 ``@ddalle``: First version
        """
        # Get sweeps
        J = self.x.GetSweeps(
            EqCons=EqCons,
            TolCons=TolCons,
            I=kw.get("I"))
        # Averaging function
        f = kw.get("f", np.mean)
        # Create output
        DB = self.copy()
        
        # Number of sweeps
        nJ = len(J)
        # Length of each sweep
        N = np.array([Ji.size for Ji in J])
        # Create coefficient list
        coeffs = kw.get("coeffs", self.coeffs)
        # Basic length
        n0 = self[coeffs[0]].size
        
        # Loop through coefficients
        for k in coeffs:
            # Check length
            if self[k].size != n0: continue
            # Initialize output data
            DB[k] = np.zeros(nJ)
            # Cumulative count
            nk = 0
            # Loop through subsweeps
            for j in range(nJ):
                # Get list of conditions
                Ji = J[j]
                # Copy data if only one point
                if N[j] == 1:
                    DB[k][j] = self[k][Ji[0]]
                    continue
                # Get the values
                V = self[k][Ji]
                # Compute the mean
                DB[k][j] = f(V)
        
        #  Update trajectory
        DB.GetTrajectory()
        # Output
        return DB
        
  # >
  
  # ===========
  # Plot
  # ===========
  # <
   # --- Plot Lookup ---
    # Get values by name and other options
    def get_CoeffValues(self, coeff, I, **kw):
        """Get values of a coefficient with optional additional operations
        
        :Call:
            >>> yv = DBc.get_CoeffValues(coeff, I, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                An individual item data book
            *coeff*: :class:`str`
                Name of coefficient
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
        :Keyword Arguments:
            *Lref*: {``DBc.Lref``} | :class:`float`
                Reference length
            *dxk*: {``None``} | :class:`str`
                Name of key to apply as a shift in MRP x-coordinate
            *xref*: {``None``} | :class:`float`
                Fixed xMRP to use
            *dxMRP*: {``None``} | :class:`float`
                Fixed shift in xMRP to use
            *XMRPFunction*: {``None``} | :class:`function`
                Function to use to use MRP dependent on Mach number
        :Versions:
            * 2018-09-28 ``@ddalle``: First version
        """
       # --------------------
       # Process y-axis data
       # --------------------
        # Extract the mean values.
        if coeff in self:
            # Read the coefficient directly
            yv = self[coeff][I]
        else:
            raise ValueError("Unrecognized coefficient '%s'" % coeff)
       # ---------------------
       # Reference Parameters
       # ---------------------
        # Get reference quantities
        Lref = kw.get("Lref", getattr(self, "Lref", 1.0))
        Aref = kw.get("Aref", getattr(self, "Aref", 1.0))
        xMRP = kw.get("xMRP", getattr(self, "xMRP", 0.0))
        yMRP = kw.get("yMRP", getattr(self, "yMRP", 0.0))
        zMRP = kw.get("zMRP", getattr(self, "zMRP", 0.0))
       # -----------------------
       # Process x-shift in MRP
       # -----------------------
        # Coefficients for shifting moments
        dxk = kw.get("dxk")
        # Process sign
        if dxk and dxk.startswith("-"):
            # Negative shift
            sgnx = -1.0
            dxk  = dxk[1:]
        else:
            # Positive shift
            sgnx = 1.0
        # Check for special cases
        if dxk and (dxk in self.coeffs):
            # Check for MRP shift
            xmrp  = kw.get("xref")
            dxmrp = kw.get("dxMRP")
            fxmrp = kw.get("XMRPFunction")
            # Shift if necessary
            if (xmrp is not None):
                # Check type
                if xmrp.__class__.__name__ == "list":
                    xmrp = np.array(xmrp)
                # Shift moment to specific point
                yv = yv + sgnx*(xmrp-xMRP)/Lref*self[dxk][I]
            if (dxmrp is not None):
                # Check type
                if dxmrp.__class__.__name__ == "list":
                    dxmrp = np.array(dxmrp)
                # Shift the moment reference point
                yv = yv + sgnx*dxmrp/Lref*self[dxk][I]
            if (fxmrp is not None):
                # Use a function to evaluate new MRP (may vary by index)
                try:
                    # Lookup by index
                    xmrp = fxmrp(self, I)
                except Exception:
                    # Lookup by Mach number
                    mk = self.get_mach_key()
                    # Evaluate based on Mach number
                    xmrp = fxmrp(self[mk][I])
                # Shift the moment to specific point
                yv = yv + sgnx*(xmrp-xMRP)/Lref*self[dxk][I]
       # -----------------------
       # Process y-shift in MRP
       # -----------------------
        # Coefficients for shifting moments
        dyk = kw.get("dyk")
        # Process sign
        if dyk and dyk.startswith("-"):
            # Negative shift
            sgny = -1.0
            dyk  = dyk[1:]
        else:
            # Positive shift
            sgny = 1.0
        # Check for special cases
        if dyk and (dyk in self.coeffs):
            # Check for MRP shift
            ymrp  = kw.get("yref")
            dymrp = kw.get("dyMRP")
            fymrp = kw.get("YMRPFunction")
            # Shift if necessary
            if (ymrp is not None):
                # Check type
                if ymrp.__class__.__name__ == "list":
                    ymrp = np.array(ymrp)
                # Shift moment to specific point
                yv = yv + sgny*(yrmp-yMRP)/Lref*self[dyk][I]
            if (dymrp is not None):
                # Check type
                if dymrp.__class__.__name__ == "list":
                    dymrp = np.array(dymrp)
                # Shift the moment reference point
                yv = yv + sgny*dymrp/Lref*self[dyk][I]
            if (fymrp is not None):
                # Use a function to evaluate new MRP (may vary by index)
                try:
                    # Lookup by index
                    ymrp = fymrp(self, I)
                except Exception:
                    # Lookup by Mach number
                    mk = self.get_mach_key()
                    # Evaluate based on Mach number
                    ymrp = fymrp(self[mk][I])
                # Shift the moment to specific point
                yv = yv + sgny*(ymrp-yMRP)/Lref*self[dyk][I]
       # -----------------------
       # Process z-shift in MRP
       # -----------------------
        # Coefficients for shifting moments
        dzk = kw.get("dzk")
        # Process sign
        if dzk and dzk.startswith("-"):
            # Negative shift
            sgnz = -1.0
            dzk  = dzk[1:]
        else:
            # Positive shift
            sgnz = 1.0
        # Check for special cases
        if dzk and (dzk in self.coeffs):
            # Check for MRP shift
            zmrp  = kw.get("zref")
            dzmrp = kw.get("dzMRP")
            fzmrp = kw.get("ZMRPFunction")
            # Shift if necessary
            if (zmrp is not None):
                # Check type
                if zmrp.__class__.__name__ == "list":
                    zmrp = np.array(zmrp)
                # Shift moment to specific point
                yv = yv + sgnz*(zrmp-zMRP)/Lref*self[dzk][I]
            if (dzmrp is not None):
                # Check type
                if dzmrp.__class__.__name__ == "list":
                    dzmrp = np.array(dzmrp)
                # Shift the moment reference point
                yv = yv + sgnz*dzmrp/Lref*self[dzk][I]
            if (fzmrp is not None):
                # Use a function to evaluate new MRP (may vary by index)
                try:
                    # Lookup by index
                    zmrp = fzmrp(self, I)
                except Exception:
                    # Lookup by Mach number
                    mk = self.get_mach_key()
                    # Evaluate based on Mach number
                    zmrp = fzmrp(self[mk][I])
                # Shift the moment to specific point
                yv = yv + sgnz*(zmrp-zMRP)/Lref*self[dzk][I]
       # ---------
       # Output
       # ---------
        return yv
   
   # --- Plotting Raw Data ---
    # Delta histogram
    def PlotDelta(self, DBT, coeff, I, **kw):
        """Plot raw deltas between two databases
        
        :Call:
            >>> h = DBc.PlotDelta(DBT, coeff, I, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                An individual item data book
            *DBT*: :class:`attdb.fm.DBCoeff`
                Target values for data book
            *coeff*: :class:`str`
                Name of coefficient
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
        :Keyword Arguments:
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Target*: {``None``} | :class:`DBBase` | :class:`list`
                Target database or list thereof
            *TargetValue*: :class:`float` | :class:`list`\ [:class:`float`]
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *OutlierSigma*: {``7.0``} | :class:`float`
                Standard deviation multiplier for determining outliers
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowError*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *ShowTarget*: :class:`bool`
                Option to show target value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *TargetFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the target value
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``Iteration Number``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2016-04-04 ``@ddalle``: Moved from point sensor to data book
        """
       # --- Initial Options ---
        # Get horizontal key.
        xk = kw.get('xk')
        # Get MRP shift keys
        dxk = kw.get('dxk')
        dyk = kw.get('dyk')
        dzk = kw.get('dzk')
       # --- Reference Parameters ---
        # Get reference quantities
        Lref = kw.get("Lref", getattr(self, "Lref", 1.0))
        Aref = kw.get("Aref", getattr(self, "Aref", 1.0))
        xMRP = kw.get("xMRP", getattr(self, "xMRP", 0.0))
        yMRP = kw.get("yMRP", getattr(self, "yMRP", 0.0))
        zMRP = kw.get("zMRP", getattr(self, "zMRP", 0.0))
       # --- Process Targets ---
        # Find matching cases in *DBT* and filter
        I, J = self.get_MatchIndices(DBT, I, **kw)
       # --- Get Database Values ---
        # Get database values
        V1 = self.get_CoeffValues(coeff, I, **kw)
        # Get target values
        V2 = DBT.get_CoeffValues(coeff, J, **kw)
        # Calculate delta
        yv = V2 - V1
       # --- Process x-axis data ---
        # Extract the values for the x-axis.
        if xk is None or xk == 'Index':
            # Use the indices as the x-axis
            xv = I
            # Label
            xk = 'Index'
        elif xk in self:
            # Extract the values.
            xv = self[xk][I]
        elif xk == "alpha":
            # Get angles of attack
            xv = self.x.GetAlpha(I)
        elif xk == "beta":
            # Get sideslip angles
            xv = self.x.GetBeta(I)
        elif xk in ["alpha_t", "aoav"]:
            # Get maneuver angle of attack
            xv = self.x.GetAlphaTotal(I)
        elif xk in ["phi", "phiv"]:
            # Get maneuver roll angles
            xv = self.x.GetPhi(I)
        elif xk in ["alpha_m", "aoam"]:
            # Get maneuver angle of attack
            xv = self.x.GetAlphaManeuver(I)
        elif xk in ["phi_m", "phim"]:
            # Get maneuver roll angles
            xv = self.x.GetPhiManeuver(I)
        # Sorting order for *xv*
        ixv = np.argsort(xv)
        xv = xv[ixv]
       # --- Plot calls ---
        # Make sure "coeff" option is not double defined
        coeff = kw.pop("coeff", coeff)
        # Call the main method
        h = self.PlotCoeffBase(xv, yv, coeff, **kw)
        # Output
        return h
       # ---
       
    # Delta histogram
    def PlotRange(self, DBT, coeff, I, **kw):
        """Plot raw ranges (absolute value of difference) between two databases
        
        :Call:
            >>> h = DBc.PlotRange(DBT, coeff, I, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                An individual item data book
            *DBT*: :class:`attdb.fm.DBCoeff`
                Target values for data book
            *coeff*: :class:`str`
                Name of coefficient
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
        :Keyword Arguments:
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Target*: {``None``} | :class:`DBBase` | :class:`list`
                Target database or list thereof
            *TargetValue*: :class:`float` | :class:`list`\ [:class:`float`]
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *OutlierSigma*: {``7.0``} | :class:`float`
                Standard deviation multiplier for determining outliers
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowError*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *ShowTarget*: :class:`bool`
                Option to show target value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *TargetFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the target value
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``Iteration Number``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2016-04-04 ``@ddalle``: Moved from point sensor to data book
        """
       # --- Initial Options ---
        # Get horizontal key.
        xk = kw.get('xk')
        # Get MRP shift keys
        dxk = kw.get('dxk')
        dyk = kw.get('dyk')
        dzk = kw.get('dzk')
       # --- Reference Parameters ---
        # Get reference quantities
        Lref = kw.get("Lref", getattr(self, "Lref", 1.0))
        Aref = kw.get("Aref", getattr(self, "Aref", 1.0))
        xMRP = kw.get("xMRP", getattr(self, "xMRP", 0.0))
        yMRP = kw.get("yMRP", getattr(self, "yMRP", 0.0))
        zMRP = kw.get("zMRP", getattr(self, "zMRP", 0.0))
       # --- Process Targets ---
        # Find matching cases in *DBT* and filter
        I, J = self.get_MatchIndices(DBT, I, **kw)
       # --- Get Database Values ---
        # Get database values
        V1 = self.get_CoeffValues(coeff, I, **kw)
        # Get target values
        V2 = DBT.get_CoeffValues(coeff, J, **kw)
        # Calculate delta
        yv = np.abs(V2 - V1)
       # --- Process x-axis data ---
        # Extract the values for the x-axis.
        if xk is None or xk == 'Index':
            # Use the indices as the x-axis
            xv = I
            # Label
            xk = 'Index'
        elif xk in self:
            # Extract the values.
            xv = self[xk][I]
        elif xk == "alpha":
            # Get angles of attack
            xv = self.x.GetAlpha(I)
        elif xk == "beta":
            # Get sideslip angles
            xv = self.x.GetBeta(I)
        elif xk in ["alpha_t", "aoav"]:
            # Get maneuver angle of attack
            xv = self.x.GetAlphaTotal(I)
        elif xk in ["phi", "phiv"]:
            # Get maneuver roll angles
            xv = self.x.GetPhi(I)
        elif xk in ["alpha_m", "aoam"]:
            # Get maneuver angle of attack
            xv = self.x.GetAlphaManeuver(I)
        elif xk in ["phi_m", "phim"]:
            # Get maneuver roll angles
            xv = self.x.GetPhiManeuver(I)
        # Sorting order for *xv*
        ixv = np.argsort(xv)
        xv = xv[ixv]
       # --- Plot calls ---
        # Make sure "coeff" option is not double defined
        coeff = kw.pop("coeff", coeff)
        # Call the main method
        h = self.PlotCoeffBase(xv, yv, coeff, **kw)
        # Output
        return h
       # ---
        
    # Plot a sweep of one or more coefficients
    def PlotContour(self, coeff, I, **kw):
        """Create a contour plot of selected data points
        
        :Call:
            >>> h = DBc.PlotContour(coeff, I, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup databook
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *xk*: :class:`str`
                Trajectory key for *x* axis
            *yk*: :class:`str`
                Trajectory key for *y* axis
            *ContourType*: {"tricontourf"} | "tricontour" | "tripcolor"
                Contour plotting function to use
            *LineType*: {"plot"} | "triplot" | "none"
                Line plotting function to highlight data points
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *ColorMap*: {``"jet"``} | :class:`str`
                Name of color map to use
            *ColorBar*: [ {``True``} | ``False`` ]
                Whether or not to use a color bar
            *ContourOptions*: :class:`dict`
                Plot options to pass to contour plotting function
            *LineOptions*: :class:`dict`
                Plot options for the line plot
            *FigWidth*: :class:`float`
                Width of figure in inches
            *FigHeight*: :class:`float`
                Height of figure in inches
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2017-04-17 ``@ddalle``: First version
        """
       # ------
       # Inputs
       # ------
        # Get horizontal key.
        xk = kw.get('xk')
        yk = kw.get('yk')
        # Check for axis variables
        if xk is None:
            raise ValueError("No x-axis key given")
        if yk is None:
            raise ValueError("No y-axis key given")
        # Extract the values for the x-axis
        if xk in self:
            # Get values directly
            xv = self[xk][I]
        elif xk.lower() == "alpha":
            # Angle of attack
            xv = self.x.GetAlpha(I)
        elif xk.lower() == "beta":
            # Angle of sideslip
            xv = self.x.GetBeta(I)
        elif xk.lower() in ["alpha_m", "aoam"]:
            # Maneuver angle of attack
            xv = self.x.GetAlphaManeuver(I)
        # Extract the values for the y-axis
        if yk in self:
            # Get values directly
            yv = self[yk][I]
        elif yk.lower() == "alpha":
            # Angle of attack
            yv = self.x.GetAlpha(I)
        elif yk.lower() == "beta":
            # Angle of sideslip
            yv = self.x.GetBeta(I)
        elif yk.lower() in ["alpha_m", "aoam"]:
            # Maneuver angle of attack
            yv = self.x.GetAlphaManeuver(I)
        # Extract the values to plot
        zv = self.get_CoeffValues(coeff, I, **kw)
        # Contour type, line type
        ctyp = kw.get("ContourType", "tricontourf")
        ltyp = kw.get("LineType", "plot")
        # Convert to lower case
        if type(ctyp).__name__ in ['str', 'unicode']:
            ctyp = ctyp.lower()
        if type(ltyp).__name__ in ['str', 'unicode']:
            ltyp = ltyp.lower()
        # Figure dimensions
        fw = kw.get('FigWidth', 6)
        fh = kw.get('FigHeight', 4.5)
        # Initialize output
        h = {}
        # Default label starter
        try:
            # Name of component
            dlbl = self.comp
        except AttributeError:
            # Backup default
            try:
                # Name of object
                dlbl = self.Name
            except AttributeError:
                # No default
                dlbl = ''
        # Initialize label.
        lbl = kw.get('Label', dlbl)
       # ------------
       # Contour Plot
       # ------------
        # Get colormap
        ocmap = kw.get("ColorMap", "jet")
        # Initialize plot options for contour plot
        kw_c = dict(cmap=ocmap)
        # Controu options
        for k in denone(kw.get("ContourOptions")):
            # Option
            o_k = kw["ContourOptions"][k]
            # Override
            if o_k is not None: kw_c[k] = o_k
        # Label
        kw_c.setdefault('label', lbl)
        # Fix aspect ratio...
        if kw.get("AxisEqual", True):
            plt.axis('equal')
        # Check plot type
        if ctyp == "tricontourf":
            # Filled contour
            h['contour'] = plt.tricontourf(xv, yv, zv, **kw_c)
        elif ctyp == "tricontour":
            # Contour lines
            h['contour'] = plt.tricontour(xv, yv, zv, **kw_c)
        elif ctyp == "tripcolor":
            # Triangulation
            h['contour'] = plt.tripcolor(xv, yv, zv, **kw_c)
        else:
            # Unrecognized
            raise ValueError("Unrecognized ContourType '%s'" % ctyp)
       # ----------------
       # Line or Dot Plot
       # ----------------
        # Check for a line plot
        if ltyp and ltyp != "none":
            # Initialize plot options for primary plot
            kw_p = dict(color='k', marker='^', zorder=9)
            # Set default line style
            if ltyp == "plot":
                kw_p["ls"] = ''
            # Plot options
            for k in denone(kw.get("LineOptions")):
                # Option
                o_k = kw["LineOptions"][k]
                # Override the default option.
                if o_k is not None: kw_p[k] = o_k
            # Label
            kw_p.setdefault('label', lbl)
            # Plot it
            if ltyp in ["plot", "line", "dot"]:
                # Regular plot
                h['line'] = plt.plot(xv, yv, **kw_p)
            elif ltyp == "triplot":
                # Plot triangles
                h['line'] = plt.triplot(xv, yv, **kw_p)
            else:
                # Unrecognized
                raise ValueError("Unrecognized LineType '%s'" % ltyp)
       # ----------
       # Formatting
       # ----------
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Labels.
        h['x'] = plt.xlabel(xk)
        h['y'] = plt.ylabel(yk)
        # Get limits that include all data (and not extra)
        xmin, xmax = get_xlim(h['ax'], pad=0.05)
        ymin, ymax = get_ylim(h['ax'], pad=0.05)
        # Make sure data is included
        h['ax'].set_xlim(xmin, xmax)
        h['ax'].set_ylim(ymin, ymax)
        # Legend.
        if kw.get('ColorBar', True):
            # Font size checks.
            fsize = 9
            # Activate the color bar
            h['cbar'] = plt.colorbar()
            # Set font size
            h['cbar'].ax.tick_params(labelsize=fsize)
        # Figure dimensions.
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes.
        try: plt.tight_layout()
        except Exception: pass
        # Output
        return h
   
   # --- Histogram ---
    # Plot a histogram of a coefficient
    def PlotCoeffHist(self, coeff, I, **kw):
        """Plot a histogram of one coefficient over several cases
        
        :Call:
            >>> h = DBc.PlotCoeffHist(coeff, I, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                An individual item data book
            *coeff*: :class:`str`
                Name of coefficient
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
        :Keyword Arguments:
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Target*: {``None``} | :class:`DBBase` | :class:`list`
                Target database or list thereof
            *TargetValue*: :class:`float` | :class:`list`\ [:class:`float`]
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *OutlierSigma*: {``7.0``} | :class:`float`
                Standard deviation multiplier for determining outliers
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowError*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *ShowTarget*: :class:`bool`
                Option to show target value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *TargetFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the target value
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``Iteration Number``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2016-04-04 ``@ddalle``: Moved from point sensor to data book
        """
       # --- Get Database Values ---
        # Get values
        V = self.get_CoeffValues(coeff, I, **kw)
       # --- Plot calls ---
        # Make sure "coeff" option is not double defined
        coeff = kw.pop("coeff", coeff)
        # Call the main method
        h = self.PlotHistBase(V, coeff=coeff, **kw)
        # Output
        return h
       # ---
        
    # Delta histogram
    def PlotDeltaHist(self, DBT, coeff, I, **kw):
        """Plot a histogram of deltas between two databases
        
        :Call:
            >>> h = DBc.PlotDeltaHist(DBT, coeff, I, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                An individual item data book
            *DBT*: :class:`attdb.fm.DBCoeff`
                Target values for data book
            *coeff*: :class:`str`
                Name of coefficient
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
        :Keyword Arguments:
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Target*: {``None``} | :class:`DBBase` | :class:`list`
                Target database or list thereof
            *TargetValue*: :class:`float` | :class:`list`\ [:class:`float`]
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *OutlierSigma*: {``7.0``} | :class:`float`
                Standard deviation multiplier for determining outliers
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowError*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *ShowTarget*: :class:`bool`
                Option to show target value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *TargetFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the target value
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``Iteration Number``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2016-04-04 ``@ddalle``: Moved from point sensor to data book
        """
       # --- Process Targets ---
        # Find matching cases in *DBT* and filter
        I, J = self.get_MatchIndices(DBT, I, **kw)
       # --- Get Database Values ---
        # Get database values
        V1 = self.get_CoeffValues(coeff, I, **kw)
        # Get target values
        V2 = DBT.get_CoeffValues(coeff, J, **kw)
        # Calculate delta
        V = V2 - V1
       # --- Plot calls ---
        # Make sure "coeff" option is not double defined
        coeff = kw.pop("coeff", coeff)
        # Set target flag
        kw["target"] = True
        # Call the main method
        h = self.PlotHistBase(V, coeff=coeff, **kw)
        # Output
        return h
       # ---
        
    # Range histogram
    def PlotRangeHist(self, DBT, coeff, I, **kw):
        """Plot a histogram of ranges (absolute delta) between two databases
        
        :Call:
            >>> h = DBc.PlotRangeHist(DBT, coeff, I, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                An individual item data book
            *coeff*: :class:`str`
                Name of coefficient
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
        :Keyword Arguments:
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Target*: {``None``} | :class:`DBBase` | :class:`list`
                Target database or list thereof
            *TargetValue*: :class:`float` | :class:`list`\ [:class:`float`]
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *OutlierSigma*: {``6/sqrt(pi)``} | :class:`float`
                Standard deviation multiplier for determining outliers
            *FilterSigma*: {``9/sqrt(pi)``} | :class:`float`
                Multiple of standard deviation to leave out of histogram
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowError*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *ShowTarget*: :class:`bool`
                Option to show target value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *TargetFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the target value
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``Iteration Number``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2016-04-04 ``@ddalle``: Moved from point sensor to data book
        """
       # --- Process Targets ---
        # Find matching cases in *DBT* and filter
        I, J = self.get_MatchIndices(DBT, I, **kw)
       # --- Get Database Values ---
        # Get database values
        V1 = self.get_CoeffValues(coeff, I, **kw)
        # Get target values
        V2 = DBT.get_CoeffValues(coeff, J, **kw)
        # Calculate delta
        V = V2 - V1
       # --- Plot calls ---
        # Make sure "coeff" option is not double defined
        coeff = kw.pop("coeff", coeff)
        # Set target flag
        kw["target"] = True
        # Call the main method
        h = self.PlotRangeHistBase(V, coeff=coeff, **kw)
        # Output
        return h
       # ---
  # >
# class DBCoeffFM


# Generic coefficient class with trilinear Mach, alpha, beta tools
class DBCoeffMAB(DBCoeffFM):
    """Generic coefficient database and interpolation class
    
    :Call:
        >>> DBc = DBCoeffMAB(mat=None, xls=None, csv=None, **kw)
    :Inputs:
        *mat*: :class:`str`
            Name of Matlab file to read
        *xls*: :class:`str` | :class:`xlrd.book.Book`
            Name of spreadsheet or interface to open workbook
        *csv*: :class:`str`
            Name of CSV file to read
        *sheet*: {``None``} | :class:`str`
            Name of worksheet to read if reading from spreadsheet
        *trajectory*: {``{}``} | ``{"mach": "MACH", ...}`` | :class:`dict`
            Dictionary of alternative names for standard trajectory keys
    :Outputs:
        *DBc*: :class:`attdb.fm.DBCoeffMAB`
            Coefficient lookup database
        *DBc.coeffs*: :class:`list` (:class:`str` | :class:`unicode`)
            List of coefficients present in the database
        *DBc[coeff]*: :class:`np.ndarray`\ [:class:`float`]
            Data for coefficient named *coeff*
    :Versions:
        * 2018-06-08 ``@ddalle``: First version
    """
  # ==========
  # Config
  # ==========
  # <
   # --- Builtins ---
    # Initialization method
    def __init__(self, mat=None, xls=None, csv=None, **kw):
        """Initialization method
        
        :Versions:
            * 2019-02-28 ``@ddalle``: First version
        """
        # Refer to parent class's initialization method
        self.init_DBCoeffMAB(mat=mat, xls=xls, csv=csv, **kw)
        
    
   # --- Init ---
    # Initialization method
    def init_DBCoeffMAB(self, **kw):
        # Initialize
        self.init_DBCoeff(**kw)
        # Coefficient list
        coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
        # Names for Mach number, angle of attack, sideslip
        xkeys = [
            self.get_mach_key(error=False),
            self.get_alpha_key(error=False),
            self.get_beta_key(error=False)
        ]
        # Names of UQ keys
        uxkeys = [xkeys[0]]
        # Make sure *eval_args* exists
        try:
            self.eval_args
        except AttributeError:
            self.eval_args = {}
        # Set args
        for coeff in coeffs:
            # Set evaluation arguments
            self.eval_args.setdefault(coeff, xkeys)
            # Set UQ arguments
            self.eval_args.setdefault("U"+coeff, uxkeys)
        # Apply shifting xMRP
        self.set_xMRP_shift_funcs()
        
   # --- Copy ---
    # Copy
    def copy(self):
        """Copy a coefficient lookup database
        
        :Call:
            >>> DBi = DBc.copy()
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeffMAB`
                Coefficient lookup database
        :Outputs:
            *DBi*: :class:`attdb.fm.DBCoeffMAB`
                Coefficient lookup database
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        # Form a new database
        DBi = self.__class__()
        # Copy relevant parts
        self.copy_db1_DBCoeff(DBi)
        # Copy MRP and ref values
        DBi.Lref = self.Lref
        DBi.Aref = self.Aref
        DBi.xMRP = self.xMRP
        DBi.yMRP = self.yMRP
        DBi.zMRP = self.zMRP
        # Copy trajectory
        DBi.x = self.x.copy()
        # Output
        return DBi
  # >
  
  # =============
  # Trajectory
  # =============
  # <
   # --- Wrappers ---
    # Form a trajectory from keywords
    def GetTrajectory(self, **kw):
        """Generic coefficient database and interpolation class
        
        :Call:
            >>> DBc.GetTrajectory(**kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient lookup database
        :Versions:
            * 2018-06-29 ``@ddalle``: Split from main function
        """
        # Dictionary of translators for key names
        kwt = kw.get("trajectory", kw.get("Trajectory", {}))
        # Initialize information for trajectory
        kwx = dict(kw)
        # Get expected trajectory key names
        kmach = kwt.get("mach",    "mach")
        kaoav = kwt.get("alpha_t", "alpha_t")
        kphiv = kwt.get("phi",     "phi")
        ka    = kwt.get("alpha",   "alpha")
        kb    = kwt.get("beta",    "beta")
        kq    = kwt.get("q",       "q")
        kp    = kwt.get("p",       "p")
        kT    = kwt.get("T",       "T")
        kRey  = kwt.get("Rey",     "Rey")
        # Check for Mach number
        if kmach in self.coeffs:
            # Save the Mach number
            kwx["mach"] = self[kmach]
        # Handle flow angles
        if (kaoav in self.coeffs) and (kphiv in self.coeffs):
            # Save total angle of attack and roll
            kwx["alpha_t"] = self[kaoav]
            kwx["phi"]     = self[kphiv]
        elif (ka in self.coeffs) and (kb in self.coeffs):
            # Save angle of attack and angle of sideslip
            kwx["alpha"] = self[ka]
            kwx["beta"]  = self[kb]
        # Dynamic pressure
        if kq in self.coeffs:
            kwx["q"] = self[kq]
        # Static pressure
        if kp in self.coeffs:
            kwx["p"] = self[kp]
        # Reynolds number
        if kRey in self.coeffs:
            kwx["Rey"] = self[kRey]
        # Freestream static temperature
        if kT in self.coeffs:
            kwx["T"] = self[kT]
        # Create trajectory
        self.x = trajectory.Trajectory(**kwx)
    
   # --- Velocity Axis ---
    # Add *alpha_t* and *phi*
    def GetAlphaTPhi(self, k2=None, k3=None, **kw):
        """Add *alpha_t* and *phi*
        
        :Call:
            >>> ca1, ca2 = DBc.GetAlphaTPhi(k2=None, k3=None, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient lookup database
            *k2*: {``None``} | :class:`str`
                Non-default name of angle of attack key
            *k3*: {``None``} | :class:`str`
                Non-default name of sideslip key
            *c1*: {``None``} | ``"alpha_t"`` | ``"aoav"`` | :class:`str`
                Name of total angle of attack key
            *c2*: {``None``} | ``"phi"`` | ``"phiv"`` | :class:`str`
                Name of roll angle between nose and velocity
        :Outputs:
            *ca1*: {``"alpha_t"``} | ``"aoav"`` | :class:`str`
                Name of total angle of attack key
            *ca2*: {``"phi"``} | ``"phiv"`` | :class:`str`
                Name of roll angle between nose and velocity
        :Versions:
            * 2018-06-12 ``@ddalle``: First version
        """
        # Get independent variable key names
        k2, k3 = self.get_bilinear_keys(k2=k2, k3=k3)
        # Output coefficient name overrides
        c1 = kw.get("c1")
        c2 = kw.get("c2")
        # Get total angle of attack key
        if c1 is None:
            if "alpha_t" in self.coeffs:
                c1 = "alpha_t"
            elif "aoav" in self.coeffs:
                c1 = "aoav"
            elif "AOAV" in self.coeffs:
                c1 = "AOAV"
            elif "AOAP" in self.coeffs:
                c1 = "AOAP"
            else:
                c1 = "alpha_t"
        # Get total roll angle key
        if c2 is None:
            if "phi" in self.coeffs:
                c2 = "phi"
            elif "phiv" in self.coeffs:
                c2 = "phiv"
            elif "PHIV" in self.coeffs:
                c2 = "PHIV"
            elif "AOPHIP" in self.coeffs:
                c2 = "PHIP"
            else:
                c2 = "phi"
        # Get values
        a = self[k2]
        b = self[k3]
        # Perform conversions
        aoav, phiv = convert.AlphaBeta2AlphaTPhi(a, b)
        # Save parameters
        self[c1] = aoav
        self[c2] = phiv
        # Save coefficients
        if c1 not in self.coeffs: self.coeffs.append(c1)
        if c2 not in self.coeffs: self.coeffs.append(c2)
        # Return names
        return c1, c2
   
   # --- Body Axes ---
    # Add *alpha_t* and *phi*
    def GetAlphaBeta(self, k2=None, k3=None, **kw):
        """Add *alpha_t* and *phi*
        
        :Call:
            >>> ca1, ca2 = DBc.GetAlphaTPhi(k2=None, k3=None, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBCoeff`
                Coefficient lookup database
            *k2*: {``None``} | :class:`str`
                Non-default name of angle of attack key
            *k3*: {``None``} | :class:`str`
                Non-default name of sideslip key
            *c1*: {``None``} | ``"alpha_t"`` | ``"aoav"`` | :class:`str`
                Name of total angle of attack key
            *c2*: {``None``} | ``"phi"`` | ``"phiv"`` | :class:`str`
                Name of roll angle between nose and velocity
        :Outputs:
            *ca1*: {``"alpha_t"``} | ``"aoav"`` | :class:`str`
                Name of total angle of attack key
            *ca2*: {``"phi"``} | ``"phiv"`` | :class:`str`
                Name of roll angle between nose and velocity
        :Versions:
            * 2018-06-12 ``@ddalle``: First version
        """
        # Get independent variable key names
        kaoav = self.get_alpha_t_key(error=True)
        kphiv = self.get_phi_key(error=True)
        # Output coefficient name overrides
        c1 = kw.get("c1", "alpha")
        c2 = kw.get("c2", "beta")
        # Get values
        aoav = self[kaoav]
        phiv = self[kphiv]
        # Perform conversions
        a, b = convert.AlphaTPhi2AlphaBeta(aoav, phiv)
        # Save parameters
        self[c1] = a
        self[c2] = b
        # Save coefficients
        if c1 not in self.coeffs: self.coeffs.append(c1)
        if c2 not in self.coeffs: self.coeffs.append(c2)
        # Return names
        return c1, c2
  # >
  
  # ===============
  # Interpolation
  # ===============
  # <
   # --- Key Lookup ---
    # Key name processing
    def get_mach_key(self, k1=None, error=True):
        """Get default name of key for Mach number
        
        :Call:
            >>> ka1 = DBc.get_mach_key(k1=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k1*: {``None``} | :class:`str`
                Name of Mach number key; if ``None``, automatic value
            *error*: {``True``} | ``False``
                Whether or not to raise an exception if key is not found
        :Outputs:
            *ka1*: *k1* | ``"mach"`` | ``"Mach"`` | ``"MACH"``
                Name of Mach number key in *DBc.coeffs*
        :Versions:
            * 2018-06-12 ``@ddalle``: First version
        """
        return self.get_key(k1,
            defs=["mach", "Mach", "MACH"],
            title="Mach number",
            error=error)
    
    # Key name processing
    def get_alpha_key(self, k2=None, error=True):
        """Get default name of key for angle of attack
        
        :Call:
            >>> ka2 = DBc.get_alpha_key(k2=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k2*: {``None``} | :class:`str`
                Name of angle of attack key; if ``None``, automatic value
            *error*: {``True``} | ``False``
                Whether or not to raise an exception if key is not found
        :Outputs:
            *ka2*: *k2* | ``"alph"`` | ``"alpha"`` | ``"ALPHA"`` | ``"aoa"``
                Name of angle of attack key in *DBc.coeffs*
        :Versions:
            * 2018-06-22 ``@ddalle``: First version
        """
        return self.get_key(k2,
            defs=["alpha", "alph", "Alpha", "ALPH", "ALPHA", "aoa", "AOA"],
            title="angle of attack",
            error=error)
    
    # Key name processing
    def get_beta_key(self, k3=None, error=True):
        """Get default name of key for angle of sideslip
        
        :Call:
            >>> ka3 = DBc.get_beta_key(k3=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k3*: {``None``} | :class:`str`
                Name of angle of sideslip key; if ``None``, automatic value
            *error*: {``True``} | ``False``
                Whether or not to raise an exception if key is not found
        :Outputs:
            *ka3*: *k3* | ``"beta"`` | ``"BETA"`` | ``"aos"``
                Name of sideslip angle key in *DBc.coeffs*
        :Versions:
            * 2018-06-22 ``@ddalle``: First version
        """
        return self.get_key(k3,
            defs=["beta", "BETA", "Beta", "aos", "AOS"],
            title="angle of sideslip",
            error=error)
        
    # Key name processing
    def get_alpha_t_key(self, k2=None, error=False):
        """Get default name of key for total angle of attack
        
        :Call:
            >>> ka2 = DBc.get_alpha_t_key(k2=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k2*: {``None``} | :class:`str`
                Name of total angle of attack key; if ``None``, automatic value
            *error*: ``True`` | {``False``}
                Whether or not to raise an exception if key is not found
        :Outputs:
            *ka2*: *k2* | ``"alpha_t"`` | ``"aoav"`` | ``"AOAV"``
                Name of total angle of attack key in *DBc.coeffs*
        :Versions:
            * 2018-06-29 ``@ddalle``: First version
        """
        return self.get_key(k2,
            defs=["alpha_t", "Alpha_t", "Alpha_T", "aoav", "AOAV"],
            title="total angle of attack",
            error=error)
    
    # Key name processing
    def get_phi_key(self, k3=None, error=False):
        """Get default name of key for velocity roll angle
        
        :Call:
            >>> ka3 = DBc.get_beta_key(k3=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k3*: {``None``} | :class:`str`
                Name of roll angle key key; if ``None``, automatic value
            *error*: ``True`` | {``False``}
                Whether or not to raise an exception if key is not found
        :Outputs:
            *ka3*: *k3* | ``"phi"`` | ``"PHI"`` | ``"phiv"`` | ``"PHIV"``
                Name of roll angle key in *DBc.coeffs*
        :Versions:
            * 2018-06-22 ``@ddalle``: First version
        """
        return self.get_key(k3,
            defs=["phi", "PHI", "Phi", "phiv", "PHIV"],
            title="velocity roll angle",
            error=error)
        
    # Key name processing
    def get_trilinear_keys(self, k1=None, k2=None, k3=None):
        """Get default names of keys for Mach, alpha, and beta
        
        :Call:
            >>> ka1, ka2, ka3 = DBc.get_trilinear_keys(k1=None, k2=None, k3=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k1*: {``None``} | :class:`str`
                Name of Mach number key; if ``None``, automatic value
            *k2*: {``None``} | :class:`str`
                Name of angle of attack key; if ``None``, automatic value
            *k3*: {``None``} | :class:`str`
                Name of sideslip angle key; if ``None``, automatic value
        :Outputs:
            *ka1*: *k1* | ``"mach"`` | ``"Mach"`` | ``"MACH"``
                Name of Mach number key in *DBc.coeffs*
            *ka2*: *k2* | ``"alph"`` | ``"alpha"`` | ``"ALPHA"`` | ``"aoa"``
                Name of angle of attack key in *DBc.coeffs*
            *ka3*: *k3* | ``"beta"`` | ``"BETA"`` | ``"aos"``
                Name of sideslip angle key in *DBc.coeffs*
        :Versions:
            * 2018-06-11 ``@ddalle``: First version
        """
        # Get Mach key
        k1 = self.get_mach_key(k1=k1)
        # Get angle of attack key
        k2 = self.get_alpha_key(k2=k2)
        # Get sideslip key
        k3 = self.get_beta_key(k3=k3)
        # Output
        return k1, k2, k3
        
    # Key name processing
    def get_mach_uq_key(self, k1=None):
        """Get default name of key for Mach number
        
        :Call:
            >>> ka1 = DBc.get_mach_uq_key(k1=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k1*: {``None``} | :class:`str`
                Name of Mach number key; if ``None``, automatic value
        :Outputs:
            *ka1*: *k1* | ``"mach"`` | ``"Mach"`` | ``"MACH"``
                Name of Mach number key in *DBc.coeffs*
        :Versions:
            * 2018-06-12 ``@ddalle``: First version
        """
        return self.get_key(k1,
            defs=["Umach", "UMACH", "mach", "Mach", "MACH"],
            title="Mach number")
    
    # Key name processing
    def get_alpha_uq_key(self, k2=None):
        """Get default name of key for angle of attack
        
        :Call:
            >>> ka2 = DBc.get_alpha_uq_key(k2=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k2*: {``None``} | :class:`str`
                Name of angle of attack key; if ``None``, automatic value
        :Outputs:
            *ka2*: *k2* | ``"alph"`` | ``"alpha"`` | ``"ALPHA"`` | ``"aoa"``
                Name of angle of attack key in *DBc.coeffs*
        :Versions:
            * 2018-06-22 ``@ddalle``: First version
        """
        return self.get_key(k2,
            defs=["Ualph", "Ualpha", "UALPH", "UALPHA", "Uaoa", "UAOA",
                "alph", "alpha", "ALPH", "ALPHA", "aoa", "AOA"],
            title="angle of attack",
            error=False)
    
    # Key name processing
    def get_beta_uq_key(self, k3=None):
        """Get default name of key for angle of attack
        
        :Call:
            >>> ka3 = DBc.get_beta_uq_key(k3=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k3*: {``None``} | :class:`str`
                Name of angle of sideslip key; if ``None``, automatic value
        :Outputs:
            *ka3*: *k3* | ``"beta"`` | ``"BETA"`` | ``"aos"``
                Name of sideslip angle key in *DBc.coeffs*
        :Versions:
            * 2018-06-22 ``@ddalle``: First version
        """
        return self.get_key(k3,
            defs=["Ubeta","UBETA","Uaos","UAOS","beta","BETA","aos","AOS"],
            title="angle of sideslip",
            error=False)
        
    # Key name processing
    def get_trilinear_uq_keys(self, k1=None, k2=None, k3=None, **kw):
        """Get default names of keys for Mach, alpha, and beta
        
        :Call:
            >>> ka1, ka2, ka3 = DBc.get_trilinear_uq_keys(**kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k1*: {``None``} | :class:`str`
                Name of Mach number key; if ``None``, automatic value
            *k2*: {``None``} | :class:`str`
                Name of angle of attack key; if ``None``, automatic value
            *k3*: {``None``} | :class:`str`
                Name of sideslip angle key; if ``None``, automatic value
        :Outputs:
            *ka1*: *k1* | ``"mach"`` | ``"Mach"`` | ``"MACH"``
                Name of Mach number key in *DBc.coeffs*
            *ka2*: *k2* | ``"alph"`` | ``"alpha"`` | ``"ALPHA"`` | ``"aoa"``
                Name of angle of attack key in *DBc.coeffs*
            *ka3*: *k3* | ``"beta"`` | ``"BETA"`` | ``"aos"``
                Name of sideslip angle key in *DBc.coeffs*
        :Versions:
            * 2018-06-11 ``@ddalle``: First version
        """
        # Get Mach key
        k1 = self.get_mach_uq_key(k1=k1)
        # Get angle of attack key
        k2 = self.get_alpha_uq_key(k2=k2)
        # Get sideslip key
        k3 = self.get_beta_uq_key(k3=k3)
        # Output
        return k1, k2, k3
    
   
   # --- Trilinear Interpolation ---
    # General trilinear interpolation (no shift)
    def GetCoeffMAB(self, coeff, m, a, b, **kw):
        """Perform trilinear interpolation of a generic coefficient
        
        :Call:
            >>> v = DBc.GetCoeffMAB(coeff, m, a, b, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *coeff*: :class:`str`
                Name of coefficient to look up
            *m*: :class:`float`
                Lookup Mach number
            *a*: :class:`float`
                Lookup angle of attack
            *b*: :class:`float`
                Lookup angle of sideslip
            *k1*: {``None``} | :class:`str`
                Non-default name of Mach number key
            *k2*: {``None``} | :class:`str`
                Non-default name of angle of attack key
            *k3*: {``None``} | :class:`str`
                Non-default name of sideslip angle key
            *c*: {``None``} | :class:`str`
                Non-default name of *coeff* in *DBc*
            *title*: {*coeff*} | :class:`str`
                Name of coefficient to use in any error message
        :Outputs:
            *CA*: :class:`float`
                Trilinear interpolation of *CA* database
        :Versions:
            * 2018-06-11 ``@ddalle``: First version
        """
        # Lookups
        k1 = kw.get('k1')
        k2 = kw.get('k2')
        k3 = kw.get('k3')
        # Error title
        ttl = kw.get("title", coeff)
        # Get default list
        defs = kw.get("defs", [])
        # Add primary name to default list
        defs.insert(0, coeff)
        # Get independent variable key names
        k1, k2, k3 = self.get_trilinear_keys(k1=k1, k2=k2, k3=k3)
        # Get lookup key
        c = self.get_key(defs=defs, title=ttl, error=kw.get("error",True))
        # Input types
        t1 = (m.__class__.__name__ in ["list", "ndarray"])
        t2 = (a.__class__.__name__ in ["list", "ndarray"])
        t3 = (b.__class__.__name__ in ["list", "ndarray"])
        # Trilinear interpolation
        if t1:
            # Number of points
            n = len(m)
            if t2:
                if t3:
                    # All three are vectors
                    return np.array([self.InterpTrilinear(m[i], a[i], b[i],
                        k1, k2, k3, c) for i in range(n)])
                else:
                    # Mach and alpha vectors
                    return np.array([self.InterpTrilinear(m[i], a[i], b,
                        k1, k2, k3, c) for i in range(n)])
            else:
                if t3:
                    # Mach and beta vectors
                    return np.array([self.InterpTrilinear(m[i], a, b[i],
                        k1, k2, k3, c) for i in range(n)])
                else:
                    # Mach vector
                    return np.array([self.InterpTrilinear(m[i], a, b,
                        k1, k2, k3, c) for i in range(n)])
        else:
            if t2:
                # Number of points
                n = len(a)
                if t3:
                    # Alpha and beta vectors
                    return np.array([self.InterpTrilinear(m, a[i], b[i],
                        k1, k2, k3, c) for i in range(n)])
                else:
                    # Alpha vector
                    return np.array([self.InterpTrilinear(m, a[i], b,
                        k1, k2, k3, c) for i in range(n)])
            else:
                if t3:
                    # Beta vector
                    n = len(b)
                    return np.array([self.InterpTrilinear(m, a, b[i],
                        k1, k2, k3, c) for i in range(n)])
                else:
                    # Sclar
                    return self.InterpTrilinear(m, a, b, k1, k2, k3, c)
   
   # --- Bilinear Interpolation ---
    # Key name processing
    def get_bilinear_keys(self, k2=None, k3=None, **kw):
        """Get default names of keys for alpha and beta
        
        :Call:
            >>> ka2, ka3 = DBc.get_bilinear_keys(k2=None, k3=None)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *k2*: {``None``} | :class:`str`
                Name of angle of attack key; if ``None``, automatic value
            *k3*: {``None``} | :class:`str`
                Name of sideslip angle key; if ``None``, automatic value
        :Outputs:
            *ka2*: *k2* | ``"alph"`` | ``"alpha"`` | ``"ALPHA"`` | ``"aoa"``
                Name of angle of attack key in *DBc.coeffs*
            *ka3*: *k3* | ``"beta"`` | ``"BETA"`` | ``"aos"``
                Name of sideslip angle key in *DBc.coeffs*
        :Versions:
            * 2018-06-12 ``@ddalle``: First version
        """
        # Get angle of attack key
        k2 = self.get_alpha_key(k2)
        # Get sideslip key
        k3 = self.get_beta_key(k3)
        # Output
        return k2, k3
   
   # --- Special UQ ---
    # Generic UQ interpolation
    def interp_uq13(self, m, a=None, b=None, c=None, defs=[], **kw):
        """Interpolate either as linear or trilinear as appropriate
        
        :Call:
            >>> UC = DBc.interp_uq13(m, a=None, b=None, c=None, defs=[], **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup database
            *m*: :class:`float`
                Lookup Mach number
            *c*: {``None``} | :class:`str`
                Non-default name of *UCA* or similar key
            *defs*: {``[]``} | :class:`list`
                List of default names applicable to lookup key
            *k1*: {``None``} | :class:`str`
                Non-default name of Mach number (for UQ) key
            *k2*: {``None``} | :class:`str`
                Non-default name of angle of attack (for UQ) key
            *k3*: {``None``} | :class:`str`
                Non-default name of angle of sideslip (for UQ) key
        :Outputs:
            *UCA*: :class:`float`
                Linear interpolation of *UCA* database
        :Versions:
            * 2018-06-12 ``@ddalle``: First version
        """
        # Get independent variable key name
        k1, k2, k3 = self.get_trilinear_uq_keys(**kw)
        # Get lookup key
        c = self.get_key(c, defs, title=kw.get("title"))
        # Check if keys are present
        if (a is None) or (b is None) or (k2 is None) or (k3 is None):
            # Linear interpolation
            return self.InterpMonolinear(m, k1, c)
        else:
            # Get dimensions
            nm = len(self[k1])
            na = len(self[k2])
            nb = len(self[k3])
            # Check if this is really trilinear
            if (nm == na) and (nm == nb):
                # Trilinear interpolation
                return self.InterpTrilinear(m, a, b, k1, k2, k3, c)
            else:
                # Linear interpolation
                return self.InterpMonolinear(m, k1, c)
  # >
  
  # ===========
  # Plot
  # ===========
  # <
   # --- Interpolated Sweep ---
    # Plot a fixed set of Mach, alpha, beta points
    def PlotCoeffMAB(self, coeff, M, A, B, **kw):
        """Make a line plot of a coefficient at specified Mach, alpha, beta pts
        
        :Call:
            >>> h = DBc.PlotCoeffMAB(coeff, M, A, B, **kw)
        :Inputs:
            *DBc*: :class:`attdb.fm.DBFM`
                Coefficient lookup databook
            *coeff*: :class:`str`
                Coefficient being plotted
            *M*: :class:`np.ndarray`\ [:class:`float`]
                Array of Mach numbers
            *A*: :class:`np.ndarray`\ [:class:`float`]
                Array of angle of attack values
            *B*: :class:`np.ndarray`\ [:class:`float`]
                Array of sideslip angles
        :Keyword Arguments:
            *x*: {``None``} | :class:`str`
                Trajectory key for *x* axis (or plot against index if ``None``)
            *Label*: {*comp*} | :class:`str`
                Manually specified label
            *Legend*: {``True``} | ``False``
                Whether or not to use a legend
            *StDev*: {``None``} | :class:`float`
                Multiple of iterative history standard deviation to plot
            *MinMax*: {``False``} | ``True``
                Whether to plot minimum and maximum over iterative history
            *Uncertainty*: {``False``} | ``True``
                Whether to plot direct uncertainty
            *LineOptions*: :class:`dict`
                Plot options for the primary line(s)
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *MinMaxOptions*: :class:`dict`
                Dictionary of plot options for the min/max plot
            *UncertaintyOptions*: :class:`dict`
                Dictionary of plot options for the uncertainty plot
            *FigWidth*: :class:`float`
                Width of figure in inches
            *FigHeight*: :class:`float`
                Height of figure in inches
            *PlotTypeStDev*: {``'FillBetween'``} | ``'ErrorBar'``
                Plot function to use for standard deviation plot
            *PlotTypeMinMax*: {``'FillBetween'``} | ``'ErrorBar'``
                Plot function to use for min/max plot
            *PlotTypeUncertainty*: ``'FillBetween'`` | {``'ErrorBar'``}
                Plot function to use for uncertainty plot
            *LegendFontSize*: {``9``} | :class:`int` > 0 | :class:`float`
                Font size for use in legends
            *Grid*: {``None``} | ``True`` | ``False``
                Turn on/off major grid lines, or leave as is if ``None``
            *GridStyle*: {``{}``} | :class:`dict`
                Dictionary of major grid line line style options
            *MinorGrid*: {``None``} | ``True`` | ``False``
                Turn on/off minor grid lines, or leave as is if ``None``
            *MinorGridStyle*: {``{}``} | :class:`dict`
                Dictionary of minor grid line line style options
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2018-06-25 ``@ddalle``: First version
        """
       # --- Initial Options ---
        # Get horizontal key
        xk = kw.get('xk', "mach")
       # --- Reference Parameters ---
        # Get reference quantities
        Lref = kw.get("Lref", getattr(self, "Lref", 1.0))
        Aref = kw.get("Aref", getattr(self, "Aref", 1.0))
        xMRP = kw.get("xMRP", getattr(self, "xMRP", 0.0))
        yMRP = kw.get("yMRP", getattr(self, "yMRP", 0.0))
        zMRP = kw.get("zMRP", getattr(self, "zMRP", 0.0))
       # --- Process x-axis data ---
        # 
        # Extract the values for the x-axis.
        if xk in ["mach", "Mach", "MACH"]:
            # Use Mach number
            xv = M
        if xk == "alpha":
            # Get angles of attack
            xv = A
        elif xk == "beta":
            # Get sideslip angles
            xv = B
        elif xk in ["alpha_t", "aoav"]:
            # Get maneuver angle of attack
            xv, phiv = convert.AlphaBeta2AlphaTPhi(A, B)
        elif xk in ["phi", "phiv"]:
            # Get maneuver roll angles
            aoav, xv = convert.AlphaBeta2AlphaTPhi(A, B)
        elif xk in ["alpha_m", "aoam"]:
            # Get maneuver angle of attack
            xv, phip = convert.AlphaBeta2AlphaMPhi(A, B)
        elif xk in ["phi_m", "phim"]:
            # Get maneuver roll angles
            aoap, xv = convert.AlphaBeta2AlphaMPhi(A, B)
       # --- Process y-axis data ---
        # Interpolate values
        yv = self.GetCoeffMAB(coeff, M, A, B)
        # Process scale if needed
        if xv.__class__.__name__.startswith("float"):
            # Create an array of one value repeated
            xv = yv*np.ones_like(yv)
       # --- Process x-shift in MRP ---
        # Coefficients for shifting moments
        dxk = kw.get("dxk")
        # Process sign
        if dxk and dxk.startswith("-"):
            # Negative shift
            sgnx = -1.0
            dxk  = dxk[1:]
        else:
            # Positive shift
            sgnx = 1.0
        # Check for special cases
        if dxk and (dxk in self.coeffs):
            # Check for MRP shift
            xmrp  = kw.get("xref")
            dxmrp = kw.get("dxMRP")
            fxmrp = kw.get("XMRPFunction")
            # Shift if necessary
            if (xmrp is not None):
                # Check type
                if xmrp.__class__.__name__ == "list":
                    xmrp = np.array(xmrp)
                # Evaluate *dx* key
                dyv = self.GetCoeffMAB(dxk, M, A, B)
                # Shift moment to specific point
                yv = yv + sgnx*(xmrp-xMRP)/Lref*dyv
            if (dxmrp is not None):
                # Check type
                if dxmrp.__class__.__name__ == "list":
                    dxmrp = np.array(dxmrp)
                # Evaluate *dx* key
                dyv = self.GetCoeffMAB(dxk, M, A, B)
                # Shift the moment reference point
                yv = yv + sgnx*dxmrp/Lref*dyv
            if (fxmrp is not None):
                # Use a function to evaluate new MRP (may vary by index)
                try:
                    xmrp = fxmrp(M, A, B)
                except Exception:
                    xmrp = fxmrp(M)
                # Evaluate *dx* key
                dyv = self.GetCoeffMAB(dxk, M, A, B)
                # Shift the moment to specific point
                yv = yv + sgnx*(xmrp-xMRP)/Lref*dyv
       # --- Process y-shift in MRP ---
        # Coefficients for shifting moments
        dyk = kw.get("dyk")
        # Process sign
        if dyk and dyk.startswith("-"):
            # Negative shift
            sgny = -1.0
            dyk  = dyk[1:]
        else:
            # Positive shift
            sgny = 1.0
        # Check for special cases
        if dyk and (dyk in self.coeffs):
            # Check for MRP shift
            ymrp  = kw.get("yref")
            dymrp = kw.get("dyMRP")
            fymrp = kw.get("YMRPFunction")
            # Shift if necessary
            if (ymrp is not None):
                # Check type
                if ymrp.__class__.__name__ == "list":
                    ymrp = np.array(ymrp)
                # Evaluate *dx* key
                dyv = self.GetCoeffMAB(dyk, M, A, B)
                # Shift moment to specific point
                yv = yv + sgny*(yrmp-yMRP)/Lref*dyv
            if (dymrp is not None):
                # Check type
                if dymrp.__class__.__name__ == "list":
                    dymrp = np.array(dymrp)
                # Evaluate *dx* key
                dyv = self.GetCoeffMAB(dyk, M, A, B)
                # Shift the moment reference point
                yv = yv + sgny*dymrp/Lref*dyv
            if (fymrp is not None):
                # Use a function to evaluate new MRP (may vary by index)
                try:
                    ymrp = fymrp(M, A, B)
                except Exception:
                    ymrp = fymrp(M)
                # Evaluate *dx* key
                dyv = self.GetCoeffMAB(dyk, M, A, B)
                # Shift the moment to specific point
                yv = yv + sgny*(ymrp-yMRP)/Lref*dyv
       # --- Process z-shift in MRP ---
        # Coefficients for shifting moments
        dzk = kw.get("dzk")
        # Process sign
        if dzk and dzk.startswith("-"):
            # Negative shift
            sgnz = -1.0
            dzk  = dzk[1:]
        else:
            # Positive shift
            sgnz = 1.0
        # Check for special cases
        if dzk and (dzk in self.coeffs):
            # Check for MRP shift
            zmrp  = kw.get("zref")
            dzmrp = kw.get("dzMRP")
            fzmrp = kw.get("ZMRPFunction")
            # Shift if necessary
            if (zmrp is not None):
                # Check type
                if zmrp.__class__.__name__ == "list":
                    zmrp = np.array(zmrp)
                # Evaluate *dx* key
                dyv = self.GetCoeffMAB(dzk, M, A, B)
                # Shift moment to specific point
                yv = yv + sgnz*(zrmp-zMRP)/Lref*dyv
            if (dzmrp is not None):
                # Check type
                if dzmrp.__class__.__name__ == "list":
                    dzmrp = np.array(dzmrp)
                # Evaluate *dx* key
                dyv = self.GetCoeffMAB(dzk, M, A, B)
                # Shift the moment reference point
                yv = yv + sgnz*dzmrp/Lref*dyv
            if (fzmrp is not None):
                # Use a function to evaluate new MRP (may vary by index)
                try:
                    zmrp = fzmrp(M, A, B)
                except Exception:
                    zmrp = fzmrp(M)
                # Evaluate *dx* key
                dyv = self.GetCoeffMAB(dzk, M, A, B)
                # Shift the moment to specific point
                yv = yv + sgnz*(zmrp-zMRP)/Lref*dyv
       # --- Uncertainty Plot ---
        # Name of uncertainty coefficient
        uk = kw.get("uk")
        # Keys to shift UQ
        if dxk:
            dxuk = kw.get("dxuk", "U"+dxk)
        else:
            dxuk = None
        if dyk:
            dyuk = kw.get("dyuk", "U"+dyk)
        else:
            dyuk = None
        if dzk:
            dzuk = kw.get("dzuk", "U"+dzk)
        else:
            dzuk = None
        # Get values if applicable
        if (uk in self.coeffs):
            # Get dimensions
            ny = len(self[coeff])
            nu = len(self[uk])
            # Check dimensions
            if ny == nu:
                # Get values directly
                uv = self.interp_uq13(M, A, B, c=uk)
            else:
                # Lookup values for UQ
                uv = self.interp_uq13(M, c=uk)
            # Shift MRP: x
            if dxuk in self.coeffs:
                # Evaluate shift values
                if ny == nu:
                    duvx = self.interp_uq13(M, A, B, c=dxuk)
                else:
                    duvx = self.interp_uq13(M, c=dxuk)
                # Apply shift
                uv = uv + sgnx*(xmrp-xMRP)/Lref*duvx
            # Shift MRP: y
            if dyuk in self.coeffs:
                # Evaluate shift values
                if ny == nu:
                    duvy = self.interp_uq13(M, A, B, c=dyuk)
                else:
                    duvy = self.interp_uq13(M, c=dyuk)
                # Apply shift
                uv = uv + sgny*(ymrp-yMRP)/Lref*duvy
            # Shift MRP: z
            if dzuk in self.coeffs:
                # Evaluate shift values
                if ny == nu:
                    duvz = self.interp_uq13(M, A, B, c=dzuk)
                else:
                    duvz = self.interp_uq13(M, c=dzuk)
                # Apply shift
                uv = uv + sgnz*(zmrp-zMRP)/Lref*duvz
            # Set the uncertainty values
            kw["uv"] = uv
        # Name of plus/minums UQ coeffs
        ukP = kw.get("ukP")
        ukM = kw.get("ukM")
        # Get values if applicable
        if (ukP in self.coeffs) and (ukM in self.coeffs):
            # Get dimensions
            ny = len(self[coeff])
            nu = len(self[ukP])
            # Check dimensions
            if ny == nu:
                # Get values directly
                uvP = self.interp_uq13(M, A, B, c=uk)
                uvM = self.interp_uq13(M, A, B, c=uk)
            else:
                # Lookup values for UQ
                uvP = self.interp_uq13(M, c=ukP)
                uvM = self.interp_uq13(M, c=ukM)
            # Shift MRP: x
            if dxuk in self.coeffs:
                # Evaluate shift values
                if ny == nu:
                    duvx = self.interp_uq13(M, A, B, c=dxuk)
                else:
                    duvx = self.interp_uq13(M, c=dxuk)
                # Apply shift
                uvP = uvP + sgnx*(xmrp-xMRP)/Lref*duvx
                uvM = uvM - sgnx*(xmrp-xMRP)/Lref*duvx
            # Shift MRP: y
            if dyuk in self.coeffs:
                # Evaluate shift values
                if ny == nu:
                    duvy = self.interp_uq13(M, A, B, c=dyuk)
                else:
                    duvy = self.interp_uq13(M, c=dyuk)
                # Apply shift
                uvP = uvP + sgny*(ymrp-yMRP)/Lref*duvy
                uvM = uvM - sgny*(ymrp-yMRP)/Lref*duvy
            # Shift MRP: z
            if dzuk in self.coeffs:
                # Evaluate shift values
                if ny == nu:
                    duvz = self.interp_uq13(M, A, B, c=dzuk)
                else:
                    duvz = self.interp_uq13(M, c=dzuk)
                # Apply shift
                uvP = uvP + sgnz*(zmrp-zMRP)/Lref*duvz
                uvM = uvM - sgnz*(zmrp-zMRP)/Lref*duvz
            # Set the uncertainty values
            kw["uvP"] = uvP
            kw["uvM"] = uvM
       # --- Standard Deviation ---
        # Name of sigma key
        sk = kw.get("sk")
        # Get values if applicable
        if (sk in self.coeffs):
            # Get values
            sv = self.GetCoeffMAB(sk, M, A, B)
            # Save it
            kw["sv"] = sv
       # --- Min/Max Plot ---
        # Key names
        kmin = kw.get("kmin")
        kmax = kw.get("kmax")
        # Get values if applicable
        if (kmin in self.coeffs) and (kmax in self.coeffs):
            # Get values
            ymin = self.GetCoeffMAB(kmin, M, A, B)
            ymax = self.GetCoeffMAB(kmax, M, A, B)
            # Save it
            kw["ymin"] = ymin
            kw["ymax"] = ymax
       # --- Plot calls ---
        # Call the main method
        h = self.PlotCoeffBase(xv, yv, coeff, **kw)
        # Output
        return h
       # ---
  # >
# class DBCoeffMAB



# Generic coefficient class with trilinear Mach, alpha, beta tools
class DBFM(DBCoeffMAB):
    """Generic coefficient database and interpolation class
    
    :Call:
        >>> DBc = DBFM(mat=None, xls=None, csv=None, **kw)
    :Inputs:
        *mat*: :class:`str`
            Name of Matlab file to read
        *xls*: :class:`str` | :class:`xlrd.book.Book`
            Name of spreadsheet or interface to open workbook
        *csv*: :class:`str`
            Name of CSV file to read
        *sheet*: {``None``} | :class:`str`
            Name of worksheet to read if reading from spreadsheet
        *trajectory*: {``{}``} | ``{"mach": "MACH", ...}`` | :class:`dict`
            Dictionary of alternative names for standard trajectory keys
    :Outputs:
        *DBc*: :class:`attdb.fm.DBFM`
            Coefficient lookup database
        *DBc.coeffs*: :class:`list` (:class:`str` | :class:`unicode`)
            List of coefficients present in the database
        *DBc[coeff]*: :class:`np.ndarray`\ [:class:`float`]
            Data for coefficient named *coeff*
    :Versions:
        * 2018-06-08 ``@ddalle``: First version
    """
  # ==========
  # Config
  # ==========
  # <
    pass
  # >
# class DBFM



# Function to fix "NoneType is not iterable" nonsense
def denone(x):
    """Replace ``None`` with ``[]`` to avoid iterative problems
    
    :Call:
        >>> y = denone(x)
    :Inputs:
        *x*: any
            Any variable
    :Outputs:
        *y*: any
            Same as *x* unless *x* is ``None``, then ``[]``
    :Versions:
        * 2015-03-09 ``@ddalle``: First version
        * 2018-06-22 ``@ddalle``: Copied from :mod:`cape.util`
    """
    if x is None:
        return []
    else:
        return x
# def denone

# Function to get interpolation weights for uq
def get_weights_monolinear(m, bkpt, key):
    """Get interpolation weights for 1D linear interpolation
    
    :Call:
        >>> I, W = get_weights_monolinear(m, bkpt, key)
    :Inputs:
        *m*: :class:`float`
            Value of lookup parameter, for example Mach number
        *bkpt*: :class:`dict`
            Dictionary of interpolation break points
        *key*: :class:`str`
            Name of lookup parameter
        *bkpt[key]*: :class:`np.ndarray`
            Array of lookup values (strictly ascending)
    :Outputs:
        *I*: :class:`np.ndarray` (:class:`int`, shape=(8,))
            List of indices to use in lookup
        *W*: :class:`np.ndarray` (:class:`float`, shape=(8,))
            List of trilinear interpolation weights
    :Versions:
        * 2017-01-30 ``@ddalle``: First version
        * 2017-07-13 ``@ddalle``: Generalized to 1D interpolant
    """
    # Extract values
    try:
        # Naive extractions
        M = bkpt[key]
    except KeyError as e:
        # Missing key
        raise KeyError(
            "Lookup key '%s' is not present in break point dict" % e[0])
    except Exception:
        # Other error: wrong type
        raise TypeError("Break point dictionary given has type '%s'"
            % bkpt.__class__.__name__)
    # Get min/max
    mmin = np.min(M)
    mmax = np.max(M)
    # Check input values
    if m < mmin or m > mmax:
        raise ValueError(
            ("Lookup value '%s' is " % m) +
            ("outside available range [%s,%s]" % (mmin, mmax)))
    # Get indices above and below lookup values
    im = np.where(M <= m)[0][-1]
    # Check for last index
    if m == mmax: im -= 1
    # Interpolation values
    ma = M[im]
    mb = M[im+1]
    # Monolinear interpolation weights
    wmb = (m - ma) / (mb - ma); wma = 1 - wmb
    # Create indices of valid lookup points
    I = np.array([im, im+1], dtype=int)
    # Create weights
    W = np.array([wma, wmb])
    # Output
    return I, W
    

# Function to get interpolation weights and indices
def get_weights_bilinear(a, b, bkpt, k1, k2):
    """Get interpolation weights for bilinear interpolation of regular data
    
    :Call:
        >>> I, W = get_weights_bilinear(a, b, bkpt, k1, k2)
    :Inputs:
        *a*: :class:`float`
            Lookup value 1, for example angle of attack
        *b*: :class:`float`
            Lookup value 2, for example sideslip angle
        *bkpt*: :class:`dict`
            Dictionary of interpolation break points
        *k1*: :class:`str`
            Name of lookup parameter 1
        *k2*: :class:`str`
            Name of lookup parameter 2
        *bkpt[k1]*: :class:`np.ndarray`
            Array of lookup values for *k1* (strictly ascending)
    :Outputs:
        *I*: :class:`np.ndarray` (:class:`int`, shape=(8,))
            List of indices to use in lookup
        *W*: :class:`np.ndarray` (:class:`float`, shape=(8,))
            List of trilinear interpolation weights
    :Versions:
        * 2017-07-13 ``@ddalle``: First version
    """
    # Extract values
    try:
        # Naive extractions
        A = bkpt[k1]
        B = bkpt[k2]
    except KeyError as e:
        # Missing key
        raise KeyError(
            "Lookup key '%s' is not present in break point dict" % e[0])
    except Exception:
        # Other error: wrong type
        raise TypeError("Break point dictionary given has type '%s'"
            % bkpt.__class__.__name__)
    # Get extrema
    amin = np.min(A); amax = np.max(A)
    bmin = np.min(B); bmax = np.max(B)
    # Get dimensions
    na = len(A)
    # Check input values
    if a < amin or a > amax:
        raise ValueError(
            ("Lookup value 1 '%s' is " % a) +
            ("outside available range [%s,%s]" % (amin, amax)))
    elif b < bmin or b > bmax:
        raise ValueError(
            ("Lookup value 2 '%s' is " % b) +
            ("outside available range [%s,%s]" % (bmin, bmax)))
    # Get indices above and below lookup values
    ia = np.where(A <= a)[0][-1]
    ib = np.where(B <= b)[0][-1]
    # Check for last index
    if a == amax: ia -= 1
    if b == bmax: ib -= 1
    # Interpolation values
    aa = A[ia]; ab = A[ia+1]
    ba = B[ib]; bb = B[ib+1]
    # Monolinear interpolation weights
    wab = (a - aa) / (ab - aa); waa = 1 - wab
    wbb = (b - ba) / (bb - ba); wba = 1 - wbb
    # Create indices of valid lookup points
    I = np.array([
        ia*nb + ib,       (ia+1)*nb + ib,   
        ia*nb + ib+1,     (ia+1)*nb + ib+1 
    ], dtype=int)
    # Create weights
    W = np.array([
        waa*wba, wab*wba, 
        waa*wbb, wab*wbb 
    ])
    # Output
    return I, W
    

# Function to get interpolation weights and indices
def get_weights_trilinear(m, a, b, bkpt, k1, k2, k3):
    """Get interpolation weights for trilinear interpolation of regular data
    
    :Call:
        >>> I, W = get_weights_trilinear(m, a, b, bkpt, k1, k2, k3)
    :Inputs:
        *m*: :class:`float`
            Lookup value 1, for example Mach number
        *a*: :class:`float`
            Lookup value 2, for example angle of attack
        *b*: :class:`float`
            Lookup value 3, for example sideslip angle
        *bkpt*: :class:`dict`
            Dictionary of interpolation break points
        *k1*: :class:`str`
            Name of lookup parameter 1
        *k2*: :class:`str`
            Name of lookup parameter 2
        *k3*: :class:`str`
            Name of lookup parameter 3
        *bkpt[k1]*: :class:`np.ndarray`
            Array of lookup values for *k1* (strictly ascending)
    :Outputs:
        *I*: :class:`np.ndarray` (:class:`int`, shape=(8,))
            List of indices to use in lookup
        *W*: :class:`np.ndarray` (:class:`float`, shape=(8,))
            List of trilinear interpolation weights
    :Versions:
        * 2017-01-30 ``@ddalle``: First version
        * 2017-07-13 ``@ddalle``: Generalized to 1D interpolant
    """
    # Extract values
    try:
        # Naive extractions
        M = bkpt[k1]
        A = bkpt[k2]
        B = bkpt[k3]
    except KeyError as e:
        # Missing key
        raise KeyError(
            "Lookup key '%s' is not present in break point dict" % e[0])
    except Exception:
        # Other error: wrong type
        raise TypeError("Break point dictionary given has type '%s'"
            % bkpt.__class__.__name__)
    # Get extrema
    mmin = np.min(M); mmax = np.max(M)
    amin = np.min(A); amax = np.max(A)
    bmin = np.min(B); bmax = np.max(B)
    # Get dimensions
    na = len(A)
    nb = len(B)
    # Check input values
    if m < mmin or m > mmax:
        raise ValueError(
            ("Lookup value 1 '%s' is " % m) +
            ("outside available range [%s,%s]" % (mmin, mmax)))
    elif a < amin or a > amax:
        raise ValueError(
            ("Lookup value 2 '%s' is " % a) +
            ("outside available range [%s,%s]" % (amin, amax)))
    elif b < bmin or b > bmax:
        raise ValueError(
            ("Lookup value 3 '%s' is " % b) +
            ("outside available range [%s,%s]" % (bmin, bmax)))
    # Get indices above and below lookup values
    im = np.where(M <= m)[0][-1]
    ia = np.where(A <= a)[0][-1]
    ib = np.where(B <= b)[0][-1]
    # Check for last index
    if m == mmax: im -= 1
    if a == amax: ia -= 1
    if b == bmax: ib -= 1
    # Interpolation values
    ma = M[im]; mb = M[im+1]
    aa = A[ia]; ab = A[ia+1]
    ba = B[ib]; bb = B[ib+1]
    # Monolinear interpolation weights
    wmb = (m - ma) / (mb - ma); wma = 1 - wmb
    wab = (a - aa) / (ab - aa); waa = 1 - wab
    wbb = (b - ba) / (bb - ba); wba = 1 - wbb
    # Create indices of valid lookup points
    I = np.array([
        im*na*nb + ia*nb + ib,       (im+1)*na*nb + ia*nb + ib,
        im*na*nb + (ia+1)*nb + ib,   (im+1)*na*nb + (ia+1)*nb + ib,
        im*na*nb + ia*nb + ib+1,     (im+1)*na*nb + ia*nb + ib+1,
        im*na*nb + (ia+1)*nb + ib+1, (im+1)*na*nb + (ia+1)*nb + ib+1
    ], dtype=int)
    # Create weights
    W = np.array([
        wma*waa*wba, wmb*waa*wba,
        wma*wab*wba, wmb*wab*wba,
        wma*waa*wbb, wmb*waa*wbb,
        wma*wab*wbb, wmb*wab*wbb
    ])
    # Output
    return I, W
    
# Get coefficient from data
def get_DB_coeff(FM, coeff, nV):
    """Extract coefficient from data object with error checks
    
    :Call:
        >>> V = get_DB_coeff(FM, coeff, nV)
    :Inputs:
        *FM*: :class:`dict` | :class:`DBCoeff`
            Dictionary of coefficient values
        *coeff*: :class:`str`
            Name of coefficient to look up
        *nV*: :class:`int`
            Expected size of coefficient array
    :Outputs:
        *V*: :class:`np.ndarray` shape=(nV,)
            Array of lookup data from *FM[coeff]*
    :Versions:
        * 2017-07-13 ``@ddalle``: First version
        * 2018-06-11 ``@ddalle``: Changed title
    """
    # Get data
    try:
        # Naive retrieval
        V = FM[coeff]
    except KeyError as e:
        # Missing key
        raise KeyError("No output data for key '%s'" % e[0])
    except Exception:
        # Other error: wrong type
        raise TypeError("Output data FM must be 'dict', not '%s'"
            % FM.__class__.__name__)
    # Check dimensions
    try:
        # Get actual dimensions
        ni = V.size
        mi = V.ndim
    except Exception:
        # Not an array
        raise TypeError("Output data FM['%s'] must be array, not '%s'"
            % (coeff, V.__class__.__name__))
    # Test dimensions
    if (ni!=nV) or (mi!=1):
        raise IndexError(
            "Output data FM['%s'] expected shape [%s] but received %s"
            % (coeff, nV, list(V.shape)))
    # Output
    return V
    
# Linear interpolation function
def interp_monolinear(m, bkpt, k1, FM, coeffs=None):
    """Perform simple linear interpolation
    
    :Call:
        >>> c = interp_monolinear(m, bkpt, k1, FM, coeff)
        >>> C = interp_monolinear(m, bkpt, k1, FM, coeffs=None)
    :Inputs:
        *m*: :class:`float`
            Lookup value 1, for example Mach number
        *bkpt*: :class:`dict`
            Dictionary of interpolation break points
        *k1*: :class:`str`
            Name of lookup parameter 1
        *FM*: :class:`dict` | :class:`DBCoeff`
            Dictionary of values
        *coeff*: ``CA`` | ``CY`` | ``CN`` | ``CLL`` | ``CLM`` | ``CLN``
            Request for single coefficient lookup
        *coeffs*: {``None``} | :class:`list` (:class:`str`)
            List of coefficients to interpolate
        *FM[coeff]*: :class:`np.ndarray`
            Properly sized array of output data
    :Outputs:
        *C* :class:`list`\ [:class:`float`]
            Interpolated coefficient at each coefficient in *coeffs*
        *c*: :class:`float`
            Interpolated coefficient for *coeff*
    :Versions:
        * 2017-01-30 ``@ddalle``: First version
        * 2017-07-13 ``@ddalle``: Generic version
    """
    # Get interpolation weights
    I, W = get_weights_monolinear(m, bkpt, k1)
    # Assuming that worked, get expected dimension
    nV = len(bkpt[k1])
    # Default coefficient list
    if coeffs is None:
        # Default coefficient list
        coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
    # Check for single coefficient (output scalar)
    if type(coeffs).__name__ in ['str', 'unicode']:
        # Scalar flag
        q_scalar = True
        # Enlist
        coeffs = [coeffs]
    else:
        # List flag
        q_scalar = False
    # List of coefficients
    C = []
    # Loop through coefficients
    for coeff in coeffs:
        # Get data
        V = get_DB_coeff(FM, coeff, nV)
        # Interpolate
        C.append(np.dot(W, V[I]))
    # Output
    if q_scalar:
        return C[0]
    else:
        return C
    
# Bilinear interpolation function
def interp_bilinear(a, b, bkpt, k1, k2, FM, coeffs=None):
    """Perform bilinear interpolation
    
    :Call:
        >>> c = interp_bilinear(a, b, bkpt, k1, k2, FM, coeff)
        >>> C = interp_bilinear(a, b, bkpt, k1, k2, FM, coeffs=None)
    :Inputs:
        *a*: :class:`float`
            Lookup value 1, for example angle of attack
        *b*: :class:`float`
            Lookup value 2, for example sideslip angle
        *bkpt*: :class:`dict`
            Dictionary of interpolation break points
        *k1*: :class:`str`
            Name of lookup parameter 1
        *k2*: :class:`str`
            Name of lookup parameter 2
        *FM*: :class:`dict` | :class:`DBCoeff`
            Dictionary of values
        *coeff*: ``CA`` | ``CY`` | ``CN`` | ``CLL`` | ``CLM`` | ``CLN``
            Request for single coefficient lookup
        *coeffs*: {``None``} | :class:`list` (:class:`str`)
            List of coefficients to interpolate
        *FM[coeff]*: :class:`np.ndarray`
            Properly sized array of output data
    :Outputs:
        *C* :class:`list`\ [:class:`float`]
            Interpolated coefficient at each coefficient in *coeffs*
        *c*: :class:`float`
            Interpolated coefficient for *coeff*
    :Versions:
        * 2017-07-13 ``@ddalle``: First version
    """
    # Get interpolation weights
    I, W = get_weights_bilinear(a, b, bkpt, k1, k2)
    # Assuming that worked, get expected dimension
    nV = len(bkpt[k1]) * len(bkpt[k2])
    # Default coefficient list
    if coeffs is None:
        # Default coefficient list
        coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
    # Check for single coefficient (output scalar)
    if type(coeffs).__name__ in ['str', 'unicode']:
        # Scalar flag
        q_scalar = True
        # Enlist
        coeffs = [coeffs]
    else:
        # List flag
        q_scalar = False
    # List of coefficients
    C = []
    # Loop through coefficients
    for coeff in coeffs:
        # Get data
        V = get_DB_coeff(FM, coeff, nV)
        # Interpolate
        C.append(np.dot(W, V[I]))
    # Output
    if q_scalar:
        return C[0]
    else:
        return C
    
# Trilinear interpolation function
def interp_trilinear(m, a, b, bkpt, k1, k2, k3, FM, coeffs=None):
    """Perform trilinear interpolation
    
    :Call:
        >>> c = interp_trilinear(m, a, b, bkpt, k1, k2, k3, FM, coeff)
        >>> C = interp_trilinear(m, a, b, bkpt, k1, k2, k3, FM, coeffs=None)
    :Inputs:
        *m*: :class:`float`
            Lookup value 1, for example Mach number
        *a*: :class:`float`
            Lookup value 2, for example angle of attack
        *b*: :class:`float`
            Lookup value 3, for example sideslip angle
        *bkpt*: :class:`dict`
            Dictionary of interpolation break points
        *k1*: :class:`str`
            Name of lookup parameter 1
        *k2*: :class:`str`
            Name of lookup parameter 2
        *k3*: :class:`str`
            Name of lookup parameter 3
        *FM*: :class:`dict` | :class:`DBCoeff`
            Dictionary of values
        *coeff*: ``CA`` | ``CY`` | ``CN`` | ``CLL`` | ``CLM`` | ``CLN``
            Request for single coefficient lookup
        *coeffs*: {``None``} | :class:`list` (:class:`str`)
            List of coefficients to interpolate
        *FM[coeff]*: :class:`np.ndarray`
            Properly sized array of output data
    :Outputs:
        *C* :class:`list`\ [:class:`float`]
            Interpolated coefficient at each coefficient in *coeffs*
        *c*: :class:`float`
            Interpolated coefficient for *coeff*
    :Versions:
        * 2017-01-30 ``@ddalle``: First version
        * 2017-07-13 ``@ddalle``: Generic version
    """
    # Get interpolation weights
    I, W = get_weights_trilinear(m, a, b, bkpt, k1, k2, k3)
    # Assuming that worked, get expected dimension
    nV = len(bkpt[k1]) * len(bkpt[k2]) * len(bkpt[k3])
    # Default coefficient list
    if coeffs is None:
        # Default coefficient list
        coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
    # Check for single coefficient (output scalar)
    if type(coeffs).__name__ in ['str', 'unicode']:
        # Scalar flag
        q_scalar = True
        # Enlist
        coeffs = [coeffs]
    else:
        # List flag
        q_scalar = False
    # List of coefficients
    C = []
    # Loop through coefficients
    for coeff in coeffs:
        # Get data
        V = get_DB_coeff(FM, coeff, nV)
        # Interpolate
        C.append(np.dot(W, V[I]))
    # Output
    if q_scalar:
        return C[0]
    else:
        return C

            
# Function to automatically get inclusive data limits.
def get_ylim(ha, pad=0.05):
    """Calculate appropriate *y*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.
    
    :Call:
        >>> ymin, ymax = get_ylim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Get the y data for this line
            ydata = h.get_ydata()
            # Check the min and max data
            if len(ydata) > 0:
                ymin = min(ymin, min(h.get_ydata()))
                ymax = max(ymax, max(h.get_ydata()))
        elif t in ['PathCollection', 'PolyCollection']:
            # Loop through paths
            for P in h.get_paths():
                # Get the coordinates
                ymin = min(ymin, min(P.vertices[:,1]))
                ymax = max(ymax, max(P.vertices[:,1]))
    # Check for identical values
    if ymax - ymin <= 0.1*pad:
        # Expand by manual amount,.
        ymax += pad*abs(ymax)
        ymin -= pad*abs(ymin)
    # Add padding.
    yminv = (1+pad)*ymin - pad*ymax
    ymaxv = (1+pad)*ymax - pad*ymin
    # Output
    return yminv, ymaxv
    
# Function to automatically get inclusive data limits.
def get_xlim(ha, pad=0.05):
    """Calculate appropriate *x*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.
    
    :Call:
        >>> xmin, xmax = get_xlim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Get data
            xdata = h.get_xdata()
            # Check the min and max data
            if len(xdata) > 0:
                xmin = min(xmin, min(h.get_xdata()))
                xmax = max(xmax, max(h.get_xdata()))
        elif t in ['PathCollection', 'PolyCollection']:
            # Loop through paths
            for P in h.get_paths():
                # Get the coordinates
                xmin = min(xmin, min(P.vertices[:,1]))
                xmax = max(xmax, max(P.vertices[:,1]))
    # Check for identical values
    if xmax - xmin <= 0.1*pad:
        # Expand by manual amount,.
        xmax += pad*abs(xmax)
        xmin -= pad*abs(xmin)
    # Add padding.
    xminv = (1+pad)*xmin - pad*xmax
    xmaxv = (1+pad)*xmax - pad*xmin
    # Output
    return xminv, xmaxv


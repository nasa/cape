#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------
:mod:`attdb.stats`: Statistics for ATT Database Tools
-------------------------------------------------------

This module includes team-specific statistics tools for aero databases in the
context of the SLS Aero Task Team.  This includes analysis of ranges with
outlier checks and Gaussian-based coverage analysis.

"""

# Common third-party modules
import numpy as np

# Statistics modules from SciPy
try:
    from scipy.stats import norm
    from scipy.stats import t as student
except ImportError:
    pass
# stats


# Calculate range
def get_range(R, cov, **kw):
    """Calculate Student's t-distribution confidence range
        
    If the nominal application of the Student's t-distribution fails to
    cover a high enough fraction of the data, the bounds are extended until
    the data is covered.
    
    :Call:
        >>> width = get_range(R, cov, **kw)
    :Inputs:
        *R*: :class:`np.ndarray` (:class:`float`)
            Array of ranges (absolute values of deltas)
        *cov*: ``0.95`` | 0 < :class:`float` < 1
            Coverage percentage
        *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
            CDF if no extra coverage needed
        *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
            Multiple of standard deviation to identify outliers; default is
            150% of the nominal coverage calculated using t-distribution.
    :Outputs:
        *width*: :class:`float`
            Half-width of confidence region
    :Versions:
        * 2018-09-28 ``@ddalle``: First version
        * 2019-01-30 ``@ddalle``: Offloaded to :func:`get_range`
        * 2019-02-13 ``@ddalle``: Moved to :mod:`stats`
    """
   # --- Setup ---
    # Enforce array
    R = np.asarray(np.abs(R))
    # Degrees of freedom
    df = R.size
    # Probability
    cdf = kw.get("CoverageCDF", kw.get("cdf", cov))
    # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
    ksig = np.sqrt(2)*student.ppf(0.5+0.5*cdf, df)
    kcov = np.sqrt(2)*student.ppf(0.5+0.5*cov, df)
    # Outlier cutoff
    osig = kw.get('OutlierSigma', kw.get("osig", 1.5*ksig))
   # --- Initial Stats ---
    # Calculate mean of ranges
    vmu = np.mean(R)
    # Standard deviation of underlying distribution
    vstd = 0.5*np.sqrt(np.pi)*vmu
   # --- Outliers ---
    # Find outliers
    I = R/vstd > osig
    # Check outliers
    while np.any(I):
        # Filter
        R = R[np.logical_not(I)]
        # Update degrees of freedom
        df = R.size
        # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
        ksig = np.sqrt(2)*student.ppf(0.5+0.5*cdf, df)
        kcov = np.sqrt(2)*student.ppf(0.5+0.5*cov, df)
        # Recalculate statistics
        vmu = np.mean(R)
        vstd = 0.5*np.sqrt(np.pi)*vmu
        # Find outliers
        I = R/vstd > osig
    # Nominal width
    width = ksig*vstd
   # --- Coverage Check ---
    # Filter cases that are outside bounds
    J = R > width
    # Count cases outside the bounds
    nout = np.count_nonzero(J)
    # Check coverage
    if float(nout)/df > 1-cov:
        # Sort the cases outside by distance from ``0``
        Ro = np.sort(R[J])
        # Number of cases allowed to be uncovered
        na = int(df - np.ceil(cov*df))
        # Number of additional cases that must be covered
        n1 = nout - na
        # The new width is the delta to the last newly included point
        width = Ro[n1-1]
        # Apply any difference between *cov* and *cdf*
        width = max(1,ksig/kcov) * width
   # --- Output ---
    return width


# Calculate interval
def get_cov_interval(dx, cov, **kw):
    """Calculate Student's *t*\ -distribution confidence range
        
    If the nominal application of the Student's t-distribution fails to
    cover a high enough fraction of the data, the bounds are extended until
    the data is covered.
    
    :Call:
        >>> a, b = get_cov_interval(dx, cov, **kw)
    :Inputs:
        *dx*: :class:`np.ndarray` (:class:`float`)
            Array of signed deltas
        *cov*: 0 < :class:`float` < 1
            Coverage percentage
        *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
            CDF if no extra coverage needed
        *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
            Multiple of standard deviation to identify outliers; default is
            150% of the nominal coverage calculated using t-distribution.
    :Outputs:
        *a*: :class:`float`
            Lower bound of coverage interval
        *b*: :class:`float`
            Upper bound of coverage interval
    :Versions:
        * 2019-02-04 ``@ddalle``: First version
        * 2019-02-13 ``@ddalle``: Moved to :mod:`stats`
    """
   # --- Setup ---
    # Enforce array
    dx = np.asarray(dx)
    # Degrees of freedom
    df = dx.size
    # Probability
    cdf = kw.get("CoverageCDF", kw.get("cdf", cov))
    # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
    ksig = student.ppf(0.5+0.5*cdf, df)
    kcov = student.ppf(0.5+0.5*cov, df)
    # Outlier cutoff
    osig = kw.get('OutlierSigma', kw.get("osig", 1.5*ksig))
   # --- Initial Stats ---
    # Calculate mean and sigma
    vmu = np.mean(dx)
    vstd = np.std(dx)
   # --- Outliers ---
    # Find outliers
    I = np.abs(dx-vmu)/vstd > osig
    # Check outliers
    while np.any(I):
        # Filter
        dx = dx[np.logical_not(I)]
        # Update degrees of freedom
        df = dx.size
        # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
        ksig = student.ppf(0.5+0.5*cdf, df)
        kcov = student.ppf(0.5+0.5*cov, df)
        # Recalculate statistics
        vmu = np.mean(dx)
        vstd = np.std(dx)
        # Find outliers
        I = np.abs(dx-vmu)/vstd > osig
    # Nominal width
    width = ksig*vstd
    # Interval
    a = vmu - width
    b = vmu + width
   # --- Coverage Check ---
    # Margin due to difference between *cov* and *cdf*
    ai = vmu - kcov/ksig*width
    bi = vmu + kcov/ksig*width
    # Filter cases that are outside bounds
    J = np.logical_and(ai<=dx, dx<=bi)
    # Count cases outside the bounds
    ncov = np.count_nonzero(J)
    # Check coverage
    if float(ncov)/df < cov:
        # Sort the cases outside by distance from ``0``
        Ro = np.sort(np.abs(dx[np.logical_not(J)]-vmu))
        # Number of additional cases that must be covered
        n1 = int(np.ceil(cov*df)) - ncov
        # The new width is the delta to the last newly included point
        width = Ro[n1-1]*ksig/kcov
        # Recalculate interval
        a = vmu - width
        b = vmu + width
   # --- Output ---
    return a, b


# Filter outliers
def check_outliers(dx, cov, **kw):
    """Find outliers in a data set
    
    :Call:
        >>> I = check_outliers(dx, cov, **kw)
    :Inputs:
        *dx*: :class:`np.ndarray` (:class:`float`)
            Array of signed deltas
        *cov*: ``0.95`` | 0 < :class:`float` < 1
            Coverage percentage
        *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
            CDF if no extra coverage needed
        *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
            Multiple of standard deviation to identify outliers; default is
            150% of the nominal coverage calculated using t-distribution.
    :Outputs:
        *I*: :class:`np.ndarray` (:class:`bool`)
            Flags for non-outlier cases,  ``False`` if case is an outlier
    :Versions:
        * 2019-02-04 ``@ddalle``: First version
        * 2019-02-13 ``@ddalle``: Moved to :mod:`stats`
    """
   # --- Setup ---
    # Enforce array
    dx = np.asarray(dx)
    # Degrees of freedom
    N = dx.size
    df = N
    # Probability
    cdf = kw.get("CoverageCDF", kw.get("cdf", cov))
    # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
    ksig = student.ppf(0.5+0.5*cdf, df)
    kcov = student.ppf(0.5+0.5*cov, df)
    # Outlier cutoff
    osig = kw.get('OutlierSigma', kw.get("osig", 1.5*ksig))
   # --- Initial Stats ---
    # Calculate mean and sigma
    vmu = np.mean(dx)
    vstd = np.std(dx)
   # --- Outliers ---
    # Find outliers
    I = np.abs(dx-vmu)/vstd <= osig
    J = np.logical_not(I)
    # Number of outliers
    n0 = 0
    n1 = np.count_nonzero(J)
    # Check outliers
    while n1 > n0:
        # Save old outlier count
        n0 = n1
        # Update degrees of freedom
        df = N - n1
        # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
        ksig = student.ppf(0.5+0.5*cdf, df)
        kcov = student.ppf(0.5+0.5*cov, df)
        # Recalculate statistics
        vmu = np.mean(dx[I])
        vstd = np.std(dx[I])
        # Find outliers
        I = np.abs(dx-vmu)/vstd <= osig
        J = np.logical_not(I)
        # Count outliers
        n1 = np.count_nonzero(J)
    # Output
    return I


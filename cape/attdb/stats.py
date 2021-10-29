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

# Numerics
import numpy as np


# Statistics modules from SciPy
try:
    from scipy.stats import norm
    from scipy.stats import t as student
except ImportError:
    pass

# Get ordered stats
def get_ordered_stats(V, cov=None, onesided=False, **kw):
    r"""Calculate coverage using ordered statistics

    :Call:
        >>> vmin, vmax = get_ordered_stats(V, cov)
        >>> vmin, vmax = get_ordered_stats(V, **kw)
        >>> vlim = get_ordered_stats(V, cov, onesided=True)
        >>> vlim = get_ordered_stats(V, onsided=True, **kw)
    :Inputs:
        *V*: :class:`np.ndarray`\ [:class:`float`]
            Array of scalar values
        *cov*: :class:`float`
            Coverage fraction, 0 < *cov* <= 1
        *onsided*: ``True`` | {``False``}
            Option to find coverage of one-sided distribution
        *ksig*: {``None``} | :class:`float`
            Option to calculate *cov* based on Gaussian distribution
        *tsig*: {``None``} | :class:`float`
            Option to calculate *cov* based on Student's t-distribution
    :Outputs:
        *vmin*: :class:`float`
            Lower limit of two-sided coverage interval
        *vmax*: :class:`float`
            Upper limit of two-sided coverage interval
        *vlim*: :class:`float`
            Upper limit of one-sided coverage interval
    :Versions:
        * 2021-09-30 ``@ddalle``: Version 1.0
    """
    # Get standard deviation counts
    ksig = kw.get("ksig")
    tsig = kw.get("tsig")
    # Input size
    n = len(V)
    # Check for trivial case
    if n == 1:
        if onesided:
            return V[0]
        else:
            return V[0], V[0]
    # Check for *ksig* option
    if ksig:
        # Check for conflict
        if cov is not None:
            raise ValueError("Specified both *cov* and *ksig*")
        # Calculate coverage fraction using 3-sigma or similar
        cov = norm.cdf(ksig)
    # Check for *tsig* option
    if tsig:
        # Check for conflict
        if ksig is not None:
            raise ValueError("Specified both *ksig* and *tsig*")
        if cov is not None:
            raise ValueError("Specified both *cov* and *ksig*")
        # Calculate coverage fraction using 3-sigma or similar
        cov = student.cdf(tsig, n)
    # Check for one-sided option
    if onesided:
        # Simple coverage
        return get_ordered_upper(V, 2*cov - 1)
    else:
        # Two-sided coverages
        vmin = get_ordered_lower(V, cov)
        vmax = get_ordered_upper(V, cov)
        # Output
        return vmin, vmax


# Calculate coverage using ordered stats
def get_ordered_lower(V, cov):
    r"""Calculate value less than fraction *cov* of *V*'s values

    :Call:
        >>> v = get_ordered_lower(V, cov)
    :Inputs:
        *V*: :class:`np.ndarray`\ [:class:`float`]
            Array of scalar values
        *cov*: :class:`float`
            Coverage fraction, 0 < *cov* <= 1
    :Outputs:
        *v*: :class:`float`
            Value such that ``cov*V.size`` entries in *V* are greater
            than or equal to *v*; may be interpolated between sorted
            values of *V*
    :Versions:
        * 2021-09-30 ``@ddalle``: Version 1.0
    """
    # Get size
    n = len(V)
    # Check for trivial input
    if n == 0:
        return np.nan
    elif n == 1:
        return V[0]
    # Get sorted values
    U = np.sort(V)
    # Calculate indices (first exact)
    i = cov * U.size
    # Get neighboring integer indices for interpolation
    ia = int(i)
    ib = int(np.ceil(i))
    # Check for trivial case (exact coverage)
    if ia == ib:
        return U[n-ia]
    # Otherwise interpolate
    va = U[n-ia]
    vb = U[n-ib]
    # Get fraction covered by *ia*
    cova = ia / float(n)
    # Interpolate ... n = 1 / (covb-cova)
    return va + n*(cov-cova)*(vb-va)


# Calculate coverage using ordered stats
def get_ordered_upper(V, cov):
    r"""Calculate value greater than fraction *cov* of *V*'s values

    :Call:
        >>> v = get_ordered_upper(V, cov)
    :Inputs:
        *V*: :class:`np.ndarray`\ [:class:`float`]
            Array of scalar values
        *cov*: :class:`float`
            Coverage fraction, 0 < *cov* <= 1
    :Outputs:
        *v*: :class:`float`
            Value such that ``cov*V.size`` entries in *V* are less than
            or equal to *v*; may be interpolated between sorted values
            of *V*
    :Versions:
        * 2021-09-30 ``@ddalle``: Version 1.0
    """
    # Get size
    n = len(V)
    # Check for trivial input
    if n == 0:
        return np.nan
    elif n == 1:
        return V[0]
    # Get sorted values
    U = np.sort(V)
    # Calculate indices (first exact)
    i = cov * U.size
    # Get neighboring integer indices for interpolation
    ia = int(i)
    ib = int(np.ceil(i))
    # Check for trivial case (exact coverage)
    if ia == ib:
        return U[ia-1]
    # Otherwise interpolate
    va = U[ia-1]
    vb = U[ib-1]
    # Get fraction covered by *ia*
    cova = ia / float(n)
    # Interpolate ... n = 1 / (covb-cova)
    return va + n*(cov-cova)*(vb-va)
    

# Calculate range
def get_range(R, cov=None, **kw):
    r"""Calculate Student's t-distribution confidence range
        
    If the nominal application of the Student's t-distribution fails to
    cover a high enough fraction of the data, the bounds are extended
    until the data is covered.
    
    :Call:
        >>> width = get_range(R, cov, **kw)
    :Inputs:
        *R*: :class:`np.ndarray`\ [:class:`float`]
            Array of ranges (absolute values of deltas)
        *cov*, *Coverage*: {``None``} | 0 < :class:`float` < 1
            Strict coverage fraction
        *ksig*, *CoverageSigma*: {``None``} | :class:`float`
            Number of standard deviations to cover (default based on
            *cov*; user must supply either *cov* or *ksig* or both)
        *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
            Fraction to use to define *ksig*
        *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
            Multiple of standard deviation to identify outliers
    :Outputs:
        *width*: :class:`float`
            Half-width of confidence region
    :Versions:
        * 2018-09-28 ``@ddalle``: Version 1.0
        * 2021-09-20 ``@ddalle``: Version 1.1
            - use :func:`_parse_options`
            - allow 100% coverage
            - remove confusing *kcov* vs *ksig* scaling
    """
   # --- Setup ---
    # Enforce array
    R = np.asarray(R)
    # Check for outliers
    mask = check_outliers_range(R, cov, **kw)
    # Filter outliers
    R = R[mask]
    # Degrees of freedom
    df = R.size
    # Options
    opts = _parse_options(df, cov=cov, **kw)
    # Coverage fraction
    cov = opts["cov"]
    # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
    ksig = opts["ksig"]
   # --- Initial Stats ---
    # Check for invalid *ksig*
    if ksig is None:
        # Use null *ksig* and just raw coverage
        ksig = 0.0
    else:
        # Multiply by sqrt(2) to get range PDF sigma from underlying
        ksig = np.sqrt(2)*ksig
    # Calculate mean of ranges
    vmu = np.mean(R)
    # Standard deviation of underlying distribution
    vstd = 0.5*np.sqrt(np.pi)*vmu
    # Nominal width
    width = ksig*vstd
   # --- Coverage Check ---
    # Filter cases already covered
    J = R <= width
    # Number of covered cases
    ncov = np.count_nonzero(J)
    # Check coverage
    if float(ncov)/df < cov:
        # Number of additional cases that must be covered
        n1 = int(np.ceil(df*cov)) - ncov
        # Sort the cases outside by distance from ``0``
        Ro = np.sort(R[np.logical_not(J)])
        # The new width is the delta to the last newly included point
        width = Ro[n1-1]
   # --- Output ---
    return width


# Calculate interval
def get_coverage(dx, cov=None, **kw):
    r"""Calculate Student's *t*\ -distribution confidence range
        
    If the nominal application of the Student's t-distribution fails to
    cover a high enough fraction of the data, the bounds are extended
    until *cov* (user-defined fraction) of the data is covered.
    
    :Call:
        >>> width = get_coverage(dx, cov, **kw)
    :Inputs:
        *dx*: :class:`np.ndarray`\ [:class:`float`]
            Array of signed deltas
        *cov*, *Coverage*: {``None``} | 0 < :class:`float` < 1
            Strict coverage fraction
        *ksig*, *CoverageSigma*: {``None``} | :class:`float`
            Number of standard deviations to cover (default based on
            *cov*; user must supply either *cov* or *ksig* or both)
        *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
            Fraction to use to define *ksig*
        *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
            Multiple of standard deviation to identify outliers
    :Outputs:
        *width*: :class:`float`
            Half-width of confidence region
    :Versions:
        * 2019-02-04 ``@ddalle``: Version 1.0
        * 2021-09-20 ``@ddalle``: Version 1.1
            - use :func:`_parse_options`
            - allow 100% coverage
            - remove confusing *kcov* vs *ksig* scaling
    """
    # Get full interval
    a, b = get_cov_interval(dx, cov=cov, **kw)
    # Get larger of -a or +b
    return max(-a, b)


# Calculate interval
def get_cov_interval(dx, cov=None, **kw):
    r"""Calculate Student's *t*\ -distribution confidence range
        
    If the nominal application of the Student's t-distribution fails to
    cover a high enough fraction of the data, the bounds are extended
    until *cov* (user-defined fraction) of the data is covered.
    
    :Call:
        >>> a, b = get_cov_interval(dx, cov, **kw)
    :Inputs:
        *dx*: :class:`np.ndarray`\ [:class:`float`]
            Array of signed deltas
        *cov*, *Coverage*: {``None``} | 0 < :class:`float` < 1
            Strict coverage fraction
        *ksig*, *CoverageSigma*: {``None``} | :class:`float`
            Number of standard deviations to cover (default based on
            *cov*; user must supply either *cov* or *ksig* or both)
        *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
            Fraction to use to define *ksig*
        *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
            Multiple of standard deviation to identify outliers
    :Outputs:
        *a*: :class:`float`
            Lower bound of coverage interval
        *b*: :class:`float`
            Upper bound of coverage interval
    :Versions:
        * 2019-02-04 ``@ddalle``: Version 1.0
        * 2021-09-20 ``@ddalle``: Version 1.1
            - use :func:`_parse_options`
            - allow 100% coverage
            - remove confusing *kcov* vs *ksig* scaling
    """
   # --- Setup ---
    # Enforce array
    dx = np.asarray(dx)
    # Check for outliers
    mask = check_outliers(dx, cov, **kw)
    # Filter outliers
    dx = dx[mask]
    # Degrees of freedom
    df = dx.size
    # Options
    opts = _parse_options(df, cov=cov, **kw)
    # Coverage fraction
    cov = opts["cov"]
    # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
    ksig = opts["ksig"]
   # --- Initial Stats ---
    # Check for 100% coverage
    if ksig is None:
        # Just cover all points
        return np.min(dx), np.max(dx)
    # Calculate mean and sigma
    vmu = np.mean(dx)
    vstd = np.std(dx)
    # Nominal width
    width = ksig*vstd
    # Interval
    a = vmu - width
    b = vmu + width
   # --- Coverage Check ---
    # Filter cases that are outside bounds
    J = np.logical_and(a<=dx, dx<=b)
    # Count cases outside the bounds
    ncov = np.count_nonzero(J)
    # Check coverage
    if float(ncov)/df < cov:
        # Number of additional cases that must be covered
        n1 = int(np.ceil(cov*df)) - ncov
        # Convert *dx* to deltas from mean
        ddx = dx[np.logical_not(J)] - vmu
        adx = np.abs(ddx)
        # Sort the cases outside by distance from ``0``
        Jdx = np.argsort(adx)
        # Additional cases to cover
        dx_extra = ddx[Jdx[:n1]] + vmu
        # Recalculate interval to ensure those *n1* are covered
        a = min(a, np.min(dx_extra))
        b = max(b, np.max(dx_extra))
   # --- Output ---
    return a, b


# Filter outliers
def check_outliers_range(R, cov=None, **kw):
    r"""Find outliers in an array of ranges
    
    :Call:
        >>> I = check_outliers_range(R, cov, **kw)
    :Inputs:
        *R*: :class:`np.ndarray`\ [:class:`float`]
            Array of ranges (unsigned deltas)
        *cov*, *Coverage*: {``None``} | 0 < :class:`float` < 1
            Strict coverage fraction
        *ksig*, *CoverageSigma*: {``None``} | :class:`float`
            Number of standard deviations to cover (default based on
            *cov*; user must supply either *cov* or *ksig* or both)
        *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
            Fraction to use to define *ksig*
        *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
            Multiple of standard deviation to identify outliers
    :Outputs:
        *I*: :class:`np.ndarray`\ [:class:`bool`]
            Flags for non-outlier cases,  ``False`` if case is an outlier
    :Versions:
        * 2021-02-20 ``@ddalle``: Version 1.0
    """
   # --- Setup ---
    # Enforce array (copy to preserve original data)
    R = np.asarray(R)
    # Degrees of freedom
    N = R.size
    df = N
    # Options
    opts = _parse_options(df, cov=cov, **kw)
    # Outlier cutoff (in standard deviations)
    osig = opts["osig"]
    # Check for infinite cutoff
    if osig is None:
        return np.full(df, True)
    # Expand *osig*, which is cutoff for underlying distribution
    # We want a cutoff for the range distribution
    osig = np.sqrt(2) * osig
   # --- Initial Stats ---
    # Calculate mean of ranges
    vmu = np.mean(R)
    # Standard deviation of underlying distribution
    vstd = 0.5*np.sqrt(np.pi)*vmu
   # --- Outliers ---
    # Find outliers
    I = R / vstd <= osig
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
        # Recalculate statistics
        vmu = np.mean(dx[I])
        vstd = np.std(dx[I])
        # Find outliers
        I = R / vstd <= osig
        J = np.logical_not(I)
        # Count outliers
        n1 = np.count_nonzero(J)
    # Output
    return I


# Filter outliers
def check_outliers(dx, cov=None, **kw):
    r"""Find outliers in a data set
    
    :Call:
        >>> I = check_outliers(dx, cov, **kw)
    :Inputs:
        *dx*: :class:`np.ndarray`\ [:class:`float`]
            Array of signed deltas
        *cov*, *Coverage*: {``None``} | 0 < :class:`float` < 1
            Strict coverage fraction
        *ksig*, *CoverageSigma*: {``None``} | :class:`float`
            Number of standard deviations to cover (default based on
            *cov*; user must supply either *cov* or *ksig* or both)
        *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
            Fraction to use to define *ksig*
        *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
            Multiple of standard deviation to identify outliers
    :Outputs:
        *I*: :class:`np.ndarray`\ [:class:`bool`]
            Flags for non-outlier cases,  ``False`` if case is an outlier
    :Versions:
        * 2019-02-04 ``@ddalle``: Version 1.0
        * 2021-09-20 ``@ddalle``: Version 1.1
            - use :func:`_parse_options`
            - allow 100% coverage
    """
   # --- Setup ---
    # Enforce array (copy to preserve original data)
    dx = np.asarray(dx)
    # Degrees of freedom
    N = dx.size
    df = N
    # Options
    opts = _parse_options(df, cov=cov, **kw)
    # Outlier cutoff (in standard deviations)
    osig = opts["osig"]
    # Check for infinite cutoff
    if osig is None:
        return np.full(df, True)
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


# Parse options
def _parse_options(df, **kw):
    # Actual coverage fraction(s)
    cov = kw.get("Coverage", kw.get("cov"))
    cdf = kw.get("CoverageCDF", kw.get("cdf"))
    # User-specified CDF
    ksig = kw.get("CoverageSigma", kw.get("ksig"))
    # Check if we have *ksig* in place of *cov*
    if cov is None:
        # Check for *something* to start with
        if ksig is None:
            if cdf is not None:
                cov = cdf
            else:
                raise ValueError("Must specify either *cov* or *ksig*")
        else:
            # Calculate *cov* from *ksig*
            cov = student.cdf(ksig, df)
    # Cumulative distribution function coverage
    cdf = kw.get("CoverageCDF", kw.get("cdf", cov))
    # Checks
    if not isinstance(cov, float):
        raise TypeError(
            "Coverage fraction (cov) must be float, got '%s'" %
            cov.__class__.__name__)
    elif cov <= 0:
        raise ValueError(
            "Coverage fraction (cov=%.2f) must be positive)" % cov)
    elif cov > 1:
        raise ValueError(
            "Coverage fraction (cov=%.2f) cannot be greater than 1)" % cov)
    # Check *cdf*
    if not isinstance(cdf, float):
        raise TypeError(
            "CDF coverage (cdf) must be float, got '%s'" %
            cdf.__class__.__name__)
    elif cdf <= 0:
        raise ValueError(
            "CDF coverage (cdf=%.2f) must be positive)" % cdf)
    elif cdf > 1:
        raise ValueError(
            "CDF coverage (cdf=%.2f) cannot be greater than 1)" % cdf)
    # Nominal coverage in standard deviations
    if 1 - cdf < 1e-6:
        # Infinite
        ksig = None
    else:
        # Nominal
        ksig = student.ppf(0.5+0.5*cdf, df)
    # Parse standard deviation multipliers
    ksig = kw.get("CoverageSigma", kw.get("ksig", ksig))
    # Default outlier bound
    if ksig is None:
        osig = None
    else:
        osig = 1.5 * ksig
    # Outlier bound
    osig = kw.get("OutlierSigma", kw.get("osig", osig))
    # Return dict of valid options
    return {
        "cov": cov,
        "cdf": cdf,
        "ksig": ksig,
        "osig": osig,
    }


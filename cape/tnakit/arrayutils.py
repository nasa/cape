# -*- coding: utf-8 -*-
"""
This includes tools for modifying or analyzing vectors, including
tools to display the vectors/arrays.
"""


# Third-party modules
import numpy as np

# Local modules
from . import typeutils

# Determine print flag from a vector or list
def get_printf_fmt(V, prec=6, emax=4, emin=-2, echar="e"):
    r"""Determine an appropriate print format for a column vector
    
    :Call:
        >>> fmt = get_printf_fmt(V, prec=6, emax=4, emin=-2, echar="e")
    :Inputs:
        *V*: :class:`list` | :class:`np.ndarray`
            1D vector of values for printing as column in CSV file
        *prec*: {``6``} | :class:`int` > 0
            Number of digits to use after ``.`` in float format
        *emax*: {``4``} | :class:`int` > 0
            Maximum number of digits to left of ``.`` in float
        *emin*: {``0``} | :class:`int` < 0
            Maximum number of zeros to right of ``.`` in float
        *echar*: {``"e"``} | ``"E"`` | ``"d"`` | ``"D"``
            Character to use for exponential format
    :Outputs:
        *fmt*: :class:`str`
            String flag such that ``fmt % v`` has same number of
            digits for each *v* in *V*
    :Versions:
        * 2019-07-12 ``@ddalle``: First version
    """
    # Check for list
    if isinstance(V, list):
        # Convert to array
        U = np.asarray(V)
    elif isinstance(V, np.ndarray):
        # No reason to copy
        U = V
    else:
        # Invalid type
        raise TypeError(
            "Expected list or np.ndarray, got '%s'" % V.__class__.__name__)
    # Get data type
    dt = U.dtype.name
    # Test for a string
    if dt.startswith("str"):
        # Determine maximum length
        maxlen = max([len(u) for u in U])
        # Create string format
        return "%%-%is" % maxlen
    # Don't consider "NaN" entries
    mask = np.logical_not(np.isnan(U))
    # Get min and max values (with padding)
    umin = 1.0001 * np.min(U[mask])
    umax = 1.0001 * np.max(U[mask])
    # Maximum magnitude
    uabs = max(-umin, umax)
    # Logarithm
    if uabs == 0:
        # Use exponent of 0
        uexp = 1
    else:
        # Round down to nearest power of 10
        uexp = int(np.floor(np.log10(uabs))) + 1
    # Get first (non-NaN) value
    u = U[mask][0]
    # Check format
    if dt.startswith("int"):
        # Check sign
        if umin < 0:
            # Give appropriate number of integer digits
            return "%%%ii" % (uexp + 1)
        else:
            # Don't leave room for - sign
            return "%%%ii" % uexp
    elif isinstance(u, float):
        # Check magnitude and sign
        if uexp > emax and umin < 0:
            # Use engineering notation but make room for sign
            return "%%%i.%i%s" % (prec + 7, prec, echar)
        elif uexp > emax:
            # Use engineering notation but all positive
            return "%%%i.%i%s" % (prec + 6, prec, echar)
        elif uexp - 1 < emin and umin < 0:
            # Engineering with negative exponent and minus sign
            return "%%%i.%i%s" % (prec + 7, prec, echar)
        elif uexp - 1 < emin:
            # Engineering with negative exponent
            return "%%%i.%i%s" % (prec + 6, prec, echar)
        elif umin < 0:
            # Float notation with minus sign and plenty of digits
            return "%%%i.%if" % (prec + max(1, uexp) + 2, prec)
        else:
            # Float, positive, greater than 1.0
            return "%%%i.%if" % (prec + max(1, uexp) + 1, prec)

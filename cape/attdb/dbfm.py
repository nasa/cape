#!/us/bin/env python
# -*- coding: utf-8 -*-
r"""
-----------------------------------------------------------------
:mod:`cape.attdb.dbfm`: Aero Task Team Force & Moment Databases
-----------------------------------------------------------------

This module provides customizations of :mod:`cape.attdb.rdb` that are
especially useful for launch vehicle force & moment databases.  The
force & moment coefficient names follow common missile naming
conventions:

    =======  ===============  ============================
    Name     Symbol           Description
    =======  ===============  ============================
    *CA*     :math:`C_A`      Axial force
    *CY*     :math:`C_Y`      Side force
    *CN*     :math:`C_N`      Normal force
    *CLL*    :math:`C_\ell`   Rolling moment
    *CLM*    :math:`C_m`      Pitching moment
    *CLN*    :math:`C_n`      Yawing moment
    =======  ===============  ============================


"""

# Standard library modules
import os
import sys

# Third-party modules
import numpy as np

# CAPE modules
import cape.attdb.convert as convert
import cape.tnakit.kwutils as kwutils

# Local modules
from . import rdb
from . import rdbaero


# Sets of common variable names
_alph_cols = rdbaero.AeroDataKit._tagcols["alpha"]
_aoap_cols = rdbaero.AeroDataKit._tagcols["aoap"]
_aoav_cols = rdbaero.AeroDataKit._tagcols["aoav"]
_beta_cols = rdbaero.AeroDataKit._tagcols["beta"]
_phip_cols = rdbaero.AeroDataKit._tagcols["phip"]
_phiv_cols = rdbaero.AeroDataKit._tagcols["phiv"]

    
# Standard converters: alpha
def convert_alpha(*a, **kw):
    r"""Determine angle of attack from *kwargs*

    All keyword args are listed by ``"Tag"``. For example, instead of
    *alpha*, the user may also use *ALPHA*, *aoa*, or any col in
    *AeroDataKit._tagcols["alpha"]*.
    
    :Call:
        >>> alpha = convert_alpha(*a, **kw)
    :Inputs:
        *a*: :class:`tuple`
            Positional args; discarded here
        *alpha*: :class:`float` | :class:`np.ndarray`
            Direct alias from *_alph_cols*
        *aoap*: :class:`float` | :class:`np.ndarray`
            Total angle of attack [deg]
        *phip*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Vertical-to-wind roll angle [deg]
        *aoav*: :class:`float` | :class:`np.ndarray`
            Missile-axis angle of attack [deg]
        *phiv*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Missile-axis roll angle [deg]
    :Outputs:
        *alpha*: ``None`` | :class:`float` | :class:`np.ndarray`
            Angle of attack [deg]
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
        * 2020-03-19 ``@ddalle``: Using :class:`AeroDataKit`
    """
   # --- Direct ---
    # Check for alias
    for col in _alph_cols:
        # Check if present
        if col in kw:
            # Return that
            return kw[col]
   # --- aoap, phip ---
    # Get total angle of attack
    for col in _aoap_cols:
        # Get total angle of attack
        aoap = kw.get(col)
        # Check
        if aoap is not None:
            break
    # Get roll angle
    for col in _phip_cols:
        # Get total angle of attack
        phip = kw.get(col)
        # Check
        if phip is not None:
            break
    # Check if both are present
    if (aoap is not None):
        # Default roll angle
        if phip is None:
            phip = 0.0
        # Convert to alpha/beta
        alpha, _ = convert.AlphaTPhi2AlphaBeta(aoap, phip)
        # Return *alpha*; discard *beta*
        return alpha
   # --- aoav, phiv ---
    # Get missile-axis angle of attack
    for col in _aoav_cols:
        # Get total angle of attack
        aoav = kw.get(col)
        # Check
        if aoav is not None:
            break
    # Get missile-axis roll angle
    for col in _phiv_cols:
        # Get total angle of attack
        phiv = kw.get(col)
        # Check
        if phiv is not None:
            break
    # Check if both are present
    if (aoav is not None):
        # Default roll angle
        if phiv is None:
            phiv = 0.0
        # Convert to alpha/beta
        alpha, _ = convert.AlphaTPhi2AlphaBeta(aoav, phiv)
        # Return *alpha*; discard *beta*
        return alpha


# Standard converters: beta
def convert_beta(*a, **kw):
    r"""Determine sideslip angle from *kwargs*

    All keyword args are listed by ``"Tag"``. For example, instead of
    *alpha*, the user may also use *ALPHA*, *aoa*, or any col in
    *AeroDataKit._tagcols["alpha"]*.
    
    :Call:
        >>> beta = convert_beta(*a, **kw)
    :Inputs:
        *a*: :class:`tuple`
            Positional args; discarded here
        *beta*: :class:`float` | :class:`np.ndarray`
            Direct alias from *_alph_cols*
        *aoap*: :class:`float` | :class:`np.ndarray`
            Total angle of attack [deg]
        *phip*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Vertical-to-wind roll angle [deg]
        *aoav*: :class:`float` | :class:`np.ndarray`
            Missile-axis angle of attack [deg]
        *phiv*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Missile-axis roll angle [deg]
    :Outputs:
        *beta*: :class:`float` | :class:`np.ndarray`
            Sideslip angle [deg]
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
        * 2020-03-19 ``@ddalle``: Using :class:`AeroDataKit`
    """
   # --- Direct ---
    # Check for alias
    for col in _beta_cols:
        # Check if present
        if col in kw:
            # Return that
            return kw[col]
   # --- aoap, phip ---
    # Get total angle of attack
    for col in _aoap_cols:
        # Get total angle of attack
        aoap = kw.get(col)
        # Check
        if aoap is not None:
            break
    # Get roll angle
    for col in _phip_cols:
        # Get total angle of attack
        phip = kw.get(col)
        # Check
        if phip is not None:
            break
    # Check if both are present
    if (aoap is not None):
        # Default roll angle
        if phip is None:
            phip = 0.0
        # Convert to alpha/beta
        _, beta = convert.AlphaTPhi2AlphaBeta(aoap, phip)
        # Discard *alpha*; return *beta*
        return beta
   # --- aoav, phiv ---
    # Get missile-axis angle of attack
    for col in _aoav_cols:
        # Get total angle of attack
        aoav = kw.get(col)
        # Check
        if aoav is not None:
            break
    # Get missile-axis roll angle
    for col in _phiv_cols:
        # Get total angle of attack
        phiv = kw.get(col)
        # Check
        if phiv is not None:
            break
    # Check if both are present
    if (aoav is not None):
        # Default roll angle
        if phiv is None:
            phiv = 0.0
        # Convert to alpha/beta
        _, beta = convert.AlphaTPhi2AlphaBeta(aoav, phiv)
        # Discard *alpha*; return *beta*
        return beta


# Standard converters: aoap
def convert_aoap(*a, **kw):
    r"""Determine total angle of attack from *kwargs*

    All keyword args are listed by ``"Tag"``. For example, instead of
    *alpha*, the user may also use *ALPHA*, *aoa*, or any col in
    *AeroDataKit._tagcols["alpha"]*.
    
    :Call:
        >>> aoap = convert_aoap(*a, **kw)
    :Inputs:
        *a*: :class:`tuple`
            Positional args; discarded here
        *aoap*: :class:`float` | :class:`np.ndarray`
            Total angle of attack [deg]
        *alpha*: :class:`float` | :class:`np.ndarray`
            Angle of attack [deg]
        *beta*: :class:`float` | :class:`np.ndarray`
            Direct alias from *_alph_cols*
        *aoav*: :class:`float` | :class:`np.ndarray`
            Missile-axis angle of attack [deg]
        *phiv*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Missile-axis roll angle [deg]
    :Outputs:
        *aoap*: :class:`float` | :class:`np.ndarray`
            Total angle of attack [deg]
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
        * 2020-03-19 ``@ddalle``: Using :class:`AeroDataKit`
    """
   # --- Direct ---
    # Check for alias
    for col in _aoap_cols:
        # Check if present
        if col in kw:
            # Return that
            return kw[col]
   # --- alpha, beta ---
    # Get angle of attack
    for col in _alph_cols:
        # Get total angle of attack
        a = kw.get(col)
        # Check
        if a is not None:
            break
    # Get sideslip angle
    for col in _beta_cols:
        # Get total angle of attack
        b = kw.get(col)
        # Check
        if b is not None:
            break
    # Check if both are present
    if (a is not None):
        # Default roll angle
        if b is None:
            b = 0.0
        # Convert to aoap/phip
        aoap, _ = convert.AlphaBeta2AlphaTPhi2(a, b)
        # Return *aoap*; discard *phip*
        return aoap
   # --- aoav, phiv ---
    # Get missile-axis angle of attack
    for col in _aoav_cols:
        # Get total angle of attack
        aoav = kw.get(col)
        # Check
        if aoav is not None:
            break
    # Get missile-axis roll angle
    for col in _phiv_cols:
        # Get total angle of attack
        phiv = kw.get(col)
        # Check
        if phiv is not None:
            break
    # Check if both are present
    if (aoav is not None):
        # Default roll angle
        if phiv is None:
            phiv = 0.0
        # Convert to alpha/beta
        aoap, _ = convert.AlphaMPhi2AlphaTPhip(aoav, phiv)
        # Return *aoap*; discard *phip*
        return aoap


# Standard converters: phip
def convert_phip(*a, **kw):
    r"""Determine body-to-wind roll angle from *kwargs*

    All keyword args are listed by ``"Tag"``. For example, instead of
    *alpha*, the user may also use *ALPHA*, *aoa*, or any col in
    *AeroDataKit._tagcols["alpha"]*.
    
    :Call:
        >>> phip = convert_phip(*a, **kw)
    :Inputs:
        *a*: :class:`tuple`
            Positional args; discarded here
        *phip*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Vertical-to-wind roll angle [deg]
        *alpha*: :class:`float` | :class:`np.ndarray`
            Angle of attack [deg]
        *beta*: :class:`float` | :class:`np.ndarray`
            Direct alias from *_alph_cols*
        *aoav*: :class:`float` | :class:`np.ndarray`
            Missile-axis angle of attack [deg]
        *phiv*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Missile-axis roll angle [deg]
    :Outputs:
        *phip*: {``0.0``} | :class:`float` | :class:`np.ndarray`
            Vertical-to-wind roll angle [deg]
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
        * 2020-03-19 ``@ddalle``: Using :class:`AeroDataKit`
    """
   # --- Direct ---
    # Check for alias
    for col in _phip_cols:
        # Check if present
        if col in kw:
            # Return that
            return kw[col]
   # --- alpha, beta ---
    # Get angle of attack
    for col in _alph_cols:
        # Get total angle of attack
        a = kw.get(col)
        # Check
        if a is not None:
            break
    # Get sideslip angle
    for col in _beta_cols:
        # Get total angle of attack
        b = kw.get(col)
        # Check
        if b is not None:
            break
    # Check if both are present
    if (a is not None):
        # Default roll angle
        if b is None:
            b = 0.0
        # Convert to aoap/phip
        _, phip = convert.AlphaBeta2AlphaTPhi2(a, b)
        # Discard *aoap*; return *phip*
        return phip
   # --- aoav, phiv ---
    # Get missile-axis angle of attack
    for col in _aoav_cols:
        # Get total angle of attack
        aoav = kw.get(col)
        # Check
        if aoav is not None:
            break
    # Get missile-axis roll angle
    for col in _phiv_cols:
        # Get total angle of attack
        phiv = kw.get(col)
        # Check
        if phiv is not None:
            break
    # Check if both are present
    if (aoav is not None):
        # Default roll angle
        if phiv is None:
            phiv = 0.0
        # Convert to alpha/beta
        _, phip = convert.AlphaMPhi2AlphaTPhip(aoav, phiv)
        # Discard *aoap*; return *phip*
        return phip


# Special evaluators: CLM vs x
def eval_CLMX(db, col1, col2, *a, **kw):
    r"""Evaluate *CLM* about arbitrary *x* moment reference point
    
    :Call:
        >>> CLMX = eval_CLMX(db, col1, col2, *a, **kw)
    :Inputs:
        *db*: :class:`DBFM`
            Force & moment data kit
        *col1*: ``"CLM"`` | :class:`str`
            Name of pitching moment column
        *col2*: ``"CN"`` | :class:`str`
            Name of normal force column
        *a*: :class:`tuple`
            Arguments to call ``db("CLM", *a)`` [plus *xMRP*]
        *kw*: :class:`dict`
            Keywords used as alternate definition of *a*
    :Outputs:
        *CLMX*: :class:`float` | :class:`np.ndarray`
            Pitching moment about arbitrary *xMRP*
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
        * 2020-03-20 ``@ddalle``: :class:`DataKit` version
        * 2020-03-25 ``@ddalle``: Added *col1*, *col2* args
    """
    # *xMRP* of original data
    xmrp = db.xMRP / db.Lref
    # Number of original arguments
    nf = len(db.get_response_args(col1))
    # Get value for *xMRP*
    xMRP = db.get_arg_value(nf, "xMRP", *a, **kw)
    # Check for an *xhat*
    xhat = kw.get("xhat", xMRP/db.Lref)
    # Evaluate main functions
    CLM = db(col1, *a, **kw)
    CN  = db(col2, *a, **kw)
    # Transfer
    return CLM + (xhat-xmrp)*CN


# Special evaluators: CLN vs x
def eval_CLNX(db, col1, col2, *a, **kw):
    r"""Evaluate *CLN* about arbitrary *x* moment reference point
    
    :Call:
        >>> CLNX = eval_CLNX(db, col1, col2, *a, **kw)
    :Inputs:
        *db*: :class:`DBFM`
            Force & moment data kit
        *col1*: ``"CLN"`` | :class:`str`
            Name of yawing moment column
        *col2*: ``"CY"`` | :class:`str`
            Name of side force column
        *a*: :class:`tuple`
            Arguments to call ``FM("CLM", *a)`` [plus *xMRP*]
        *kw*: :class:`dict`
            Keywords used as alternate definition of *a*
    :Outputs:
        *CLNX*: :class:`float` | :class:`np.ndarray`
            Pitching moment about arbitrary *xMRP*
    :Versions:
        * 2019-02-28 ``@ddalle``: First version
        * 2020-03-20 ``@ddalle``: :class:`DataKit` version
    """
    # *xMRP* of original data
    xmrp = db.xMRP / db.Lref
    # Number of original arguments
    nf = len(db.get_response_args(col1))
    # Get value for *xMRP*
    xMRP = db.get_arg_value(nf, "xMRP", *a, **kw)
    # Check for an *xhat*
    xhat = kw.get("xhat", xMRP/db.Lref)
    # Evaluate main functions
    CLN = db(col1, *a, **kw)
    CY  = db(col2, *a, **kw)
    # Transfer
    return CLN + (xhat-xmrp)*CY


# Create evaluator
def genr8_fCLMX(col1="CLM", col2="CLN"):
    r"""Generate an evaluator for *CLMX* with specified *cols*

    :Call:
        >>> func = genr8_fCLMX(col1="CLM", col2="CN")
    :Inputs:
        *col1*: {``"CLM"``} | :class:`str`
            Name of *CLM* column to evaluate
        *col2*: {``"CN"``} | :class:`str`
            Name of *CN* column to evaluate
    :Outputs:
        *func*: :class:`function`
            Function to evaluate *col1* about arbitrary *xMRP*
    :Output Call:
        >>> CLMX = func(db, *a, **kw)
    :Output Args:
        *db*: :class:`DBFM`
            Force and moment data kit
        *a*: :class:`tuple`\ [:class:`float` | :class:`np.ndarray`]
            Args to *col1* and *col2*, plus optional *xMRP*
        *kw*: :class:`dict`
            Keyword args to *col1* and *col2*, plus optional *xMRP*
    :Versions:
        * 2020-03-26 ``@ddalle``: First version
    """
    # Define the function
    def func(db, *a, **kw):
        return eval_CLMX(db, col1, col2, *a, **kw)
    # Return the function
    return func


# Create evaluator
def genr8_fCLNX(col1="CLN", col2="CY"):
    r"""Generate an evaluator for *CLNX* with specified *cols*

    :Call:
        >>> func = genr8_fCLNX(col1="CLN", col2="CN")
    :Inputs:
        *col1*: {``"CLN"``} | :class:`str`
            Name of yawing moment column
        *col2*: {``"CY"``} | :class:`str`
            Name of side force column
    :Outputs:
        *func*: :class:`function`
            Function to evaluate *col1* about arbitrary *xMRP*
    :Output Call:
        >>> CLNX = func(db, *a, **kw)
    :Output Args:
        *db*: :class:`DBFM`
            Force and moment data kit
        *a*: :class:`tuple`\ [:class:`float` | :class:`np.ndarray`]
            Args to *col1* and *col2*, plus optional *xMRP*
        *kw*: :class:`dict`
            Keyword args to *col1* and *col2*, plus optional *xMRP*
    :Versions:
        * 2020-03-26 ``@ddalle``: First version
    """
    # Define the function
    def func(db, *a, **kw):
        return eval_CLNX(db, col1, col2, *a, **kw)
    # Return the function
    return func


# Evaluate *UCLM* about different x
def eval_UCLMX(db, col1, col2, col3, *a, **kw):
    r"""Evaluate *UCLM* about arbitrary *x* moment reference point

    :Call:
        >>> UCLMX = eval_UCLMX(db, col1, col2, col3, *a, **kw)
    :Inputs:
        *db*: :class:`DBFM`
            Force & moment data kit
        *col1*: ``"UCLM"`` | :class:`str`
            Name of pitching moment uncertainty column
        *col2*: ``"UCN"`` | :class:`str`
            Name of normal force uncertainty column
        *col3*: ``"xCLM"`` | :class:`str`
            Name of *UCLM* reference MRP column
        *a*: :class:`tuple`
            Arguments to call ``FM(col1, *a)`` [plus *xMRP*]
        *kw*: :class:`dict`
            Keywords used as alternate definition of *a*
    :Outputs:
        *UCLMX*: :class:`float` | :class:`np.ndarray`
            Pitching moment uncertainty about arbitrary *xMRP*
    :Versions:
        * 2019-03-13 ``@ddalle``: First version
        * 2020-03-20 ``@ddalle``: :class:`DataKit` version
        * 2020-03-26 ``@ddalle``: Added *col* args
    """
    # *xMRP* of original data
    xmrp = db.xMRP / db.Lref
    # Number of original arguments
    nf = len(db.get_response_args(col1))
    # Get value for *xMRP*
    xMRP = db.get_arg_value(nf, "xMRP", *a, **kw)
    # Check for an *xhat*
    xhat = kw.get("xhat", xMRP/db.Lref)
    # Evaluate main functions
    UCLM = db(col1, *a, **kw)
    xCLM = db(col3, *a, **kw)
    UCN  = db(col2, *a, **kw)
    # Transfer
    UCLMX = np.sqrt(UCLM*UCLM + ((xCLM-xhat)*UCN)**2)
    # Output
    return UCLMX


# Evaluate *UCLN* about different x
def eval_UCLNX(db, col1, col2, col3, *a, **kw):
    r"""Evaluate *UCLN* about arbitrary *x* moment reference point
    
    :Call:
        >>> UCLNX = eval_UCLNX(db, col1, col2, col3, *a, **kw)
    :Inputs:
        *db*: :class:`DBFM`
            Force & moment data kit
        *col1*: ``"UCLN"`` | :class:`str`
            Name of yawing moment uncertainty column
        *col2*: ``"UCY"`` | :class:`str`
            Name of side force uncertainty column
        *col3*: ``"xCLN"`` | :class:`str`
            Name of *UCLN* reference MRP column
        *a*: :class:`tuple`
            Arguments to call ``FM(col1, *a)`` [plus *xMRP*]
        *kw*: :class:`dict`
            Keywords used as alternate definition of *a*
    :Outputs:
        *UCLNX*: :class:`float` | :class:`np.ndarray`
            Pitching moment uncertainty about arbitrary *xMRP*
    :Versions:
        * 2019-03-13 ``@ddalle``: First version
        * 2020-03-20 ``@ddalle``: :class:`DataKit` version
    """
    # *xMRP* of original data
    xmrp = db.xMRP / db.Lref
    # Number of original arguments
    nf = len(db.get_response_args(col1))
    # Get value for *xMRP*
    xMRP = db.get_arg_value(nf, "xMRP", *a, **kw)
    # Check for an *xhat*
    xhat = kw.get("xhat", xMRP/db.Lref)
    # Evaluate main functions
    UCLN = db(col1, *a, **kw)
    xCLN = db(col3, *a, **kw)
    UCY  = db(col2, *a, **kw)
    # Transfer
    UCLNX = np.sqrt(UCLN*UCLN + ((xCLN-xhat)*UCY)**2)
    # Output
    return UCLNX


# Create evaluator for *UCLMX*
def genr8_fUCLMX(col1="UCLM", col2="UCN", col3="xCLM"):
    r"""Generate an evaluator for *UCLMX* with specified *cols*

    :Call:
        >>> func = genr8_fUCLMX(col1="UCLM", col2="UCN", col3="xCLM")
    :Inputs:
        *col1*: ``"UCLM"`` | :class:`str`
            Name of pitching moment uncertainty column
        *col2*: ``"UCN"`` | :class:`str`
            Name of normal force uncertainty column
        *col3*: ``"xCLM"`` | :class:`str`
            Name of *UCLM* reference MRP column
    :Outputs:
        *func*: :class:`function`
            Function to evaluate *col1* about arbitrary *xMRP*
    :Output Call:
        >>> UCLMX = func(db, *a, **kw)
    :Output Args:
        *db*: :class:`DBFM`
            Force and moment data kit
        *a*: :class:`tuple`\ [:class:`float` | :class:`np.ndarray`]
            Args to *col1*, plus optional *xMRP*
        *kw*: :class:`dict`
            Keyword args to *col1*, plus optional *xMRP*
    :Versions:
        * 2020-03-26 ``@ddalle``: First version
    """
    # Define the function
    def func(db, *a, **kw):
        return eval_UCLMX(db, col1, col2, col3, *a, **kw)
    # Return the function
    return func


# Create evaluator for *UCLNX*
def genr8_fUCLNX(col1="UCLN", col2="UCY", col3="xCLN"):
    r"""Generate an evaluator for *UCLNX* with specified *cols*

    :Call:
        >>> func = genr8_fUCLNX(col1="UCLN", col2="UCY", col3="xCLN")
    :Inputs:
        *col1*: ``"UCLN"`` | :class:`str`
            Name of yawing moment uncertainty column
        *col2*: ``"UCY"`` | :class:`str`
            Name of side force uncertainty column
        *col3*: ``"xCLN"`` | :class:`str`
            Name of *UCLN* reference MRP column
    :Outputs:
        *func*: :class:`function`
            Function to evaluate *col1* about arbitrary *xMRP*
    :Output Call:
        >>> UCLNX = func(db, *a, **kw)
    :Output Args:
        *db*: :class:`DBFM`
            Force and moment data kit
        *a*: :class:`tuple`\ [:class:`float` | :class:`np.ndarray`]
            Args to *col1*, plus optional *xMRP*
        *kw*: :class:`dict`
            Keyword args to *col1*, plus optional *xMRP*
    :Versions:
        * 2020-03-26 ``@ddalle``: First version
    """
    # Define the function
    def func(db, *a, **kw):
        return eval_UCLNX(db, col1, col2, col3, *a, **kw)
    # Return the function
    return func


# Estimate *xCLM*
def estimate_xCLM(self, DCLM, DCN):
    r"""Estimate reference *x* for *UCLM* calculations
    
    :Call:
        >>> xCLM = estimate_xCLM(self, DCLM, DCN)
    :Inputs:
        *self*: :class:`DBFM`
            Force & moment database with *self.xMRP* and *self.Lref*
        *DCLM*: :class:`np.ndarray` (:class:`float`)
            Deltas between two databases' *CLM* values
        *DCN*: :class:`np.ndarray` (:class:`float`)
            Deltas between two databases' *CN* values
    :Outputs:
        *xCLM*: :class:`float`
            Reference *x* that minimizes *UCLM*
    :Versions:
        * 2019-02-20 ``@ddalle``: First version
        * 2020-05-04 ``@ddalle``: Copied from :mod:`attdb.fm`
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
    r"""Estimate reference *x* for *UCLN* calculations
    
    :Call:
        >>> xCLN = estimate_xCLN(self, DCLN, DCY)
    :Inputs:
        *self*: :class:`DBFM`
            Force & moment database with *self.xMRP* and *self.Lref*
        *DCLN*: :class:`np.ndarray` (:class:`float`)
            Deltas between two databases' *CLN* values
        *DCY*: :class:`np.ndarray` (:class:`float`)
            Deltas between two databases' *CY* values
    :Outputs:
        *xCLN*: :class:`float`
            Reference *x* that minimizes *UCLN*
    :Versions:
        * 2019-02-20 ``@ddalle``: First version
        * 2020-05-04 ``@ddalle``: Copied from :mod:`attdb.fm`
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


# DBFM options
class _DBFMOpts(rdb.DataKitOpts):
   # --- Global Options ---
    # List of options
    _optlist = {
        "RefLength",
        "RefArea",
        "xMRP",
        "yMRP",
        "zMRP",
    }

    # Alternate names
    _optmap = {
        "Aref": "RefArea",
        "Lref": "RefLength",
        "XMRP": "xMRP",
        "YMRP": "yMRP",
        "ZMRP": "zMRP",
    }


# DBFM definition
class _DBFMDefn(rdb.DataKitDefn):
    pass


# Combine options with parent class
_DBFMOpts.combine_optdefs()


# Create class
class DBFM(rdbaero.AeroDataKit):
    r"""Database class for launch vehicle force & moment

    :Call:
        >>> db = dbfm.DBFM(fname=None, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            File name; extension is used to guess data format
        *csv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVFile` read
        *textdata*: {``None``} | :class:`str`
            Explicit file name for :class:`TextDataFile`
        *simplecsv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVSimple`
        *xls*: {``None``} | :class:`str`
            File name for :class:`XLSFile`
        *mat*: {``None``} | :class:`str`
            File name for :class:`MATFile`
    :Outputs:
        *db*: :class:`cape.attdb.dbfm.DBFM`
            LV force & moment database
    :Versions:
        * 2020-03-20 ``@ddalle``: First version
    """
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Options ---
    # Class for options
    _optscls = _DBFMOpts
    # Class for definitions
    _defncls = _DBFMDefn

   # --- Tags ---
    # Additional tags
    _tagmap = {
        "CA":   "CA",
        "CLL":  "CLL",
        "CLM":  "CLM",
        "CLN":  "CLN",
        "CN":   "CN",
        "CY":   "CY",
        "Cl":   "CLL",
        "Cm":   "CLM",
        "Cn":   "CLN",
    }
  # >

  # ==================
  # Config
  # ==================
  # <
   # --- Init ---
    # Initialization method
    def __init__(self, fname=None, **kw):
        r"""Initialization method

        :Versions:
            * 2020-03-20 ``@ddalle``: First version
            * 2020-05-08 ``@ddalle``: Split :func:`_make_dbfm`
        """
        # Call parent's init method
        rdbaero.AeroDataKit.__init__(self, fname, **kw)
        # Initialize special aspects for DBFM
        self._make_dbfm()

    # Do special init methods for FM databases
    def _make_dbfm(self):
        r"""Perform special aspects of :class:`DBFM` init

        :Call:
            >>> db._make_dbfm()
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
        :Versions:
            * 2020-05-08 ``@ddalle``: Forked from :func:`__init__`
        """
        # Set default arg converters
        self._make_arg_converters_aero()
        # Set UQ cols
        self._make_uq_cols_FM()
        self._make_uq_ecols_FM()
        self._make_uq_acols_FM()
        # Save reference properties
        self.Lref = self.opts.get_option("RefLength", 1.0)
        self.Aref = self.opts.get_option("RefArea", 0.25*np.pi)
        self.xMRP = self.opts.get_option("xMRP", 0.0)
        self.yMRP = self.opts.get_option("yMRP", 0.0)
        self.zMRP = self.opts.get_option("zMRP", 0.0)
        # Set default arguments
        self.set_arg_default("xMRP", self.xMRP)
        self.set_arg_default("yMRP", self.yMRP)
        self.set_arg_default("zMRP", self.zMRP)
  # >

  # ==================
  # Attributes
  # ==================
  # <
   # --- Arg Converters ---
    # Automate arg converters for preset tags
    def _make_arg_converters_aero(self):
        r"""Set default arg converters for cols with known tags

        :Call:
            >>> db._make_arg_converters_aero()
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
        :Versions:
            * 2020-03-19 ``@ddalle``: First version
        """
        # Loop through any keys with "alpha" tag
        for col in self.get_cols_by_tag("alpha"):
            # Set converter
            self.set_arg_converter(col, convert_alpha)
        # Loop through any keys with "beta" tag
        for col in self.get_cols_by_tag("beta"):
            # Set converter
            self.set_arg_converter(col, convert_beta)
        # Loop through any keys with "aoap" tag
        for col in self.get_cols_by_tag("aoap"):
            # Set converter
            self.set_arg_converter(col, convert_aoap)
        # Loop through any keys with "phip" tag
        for col in self.get_cols_by_tag("phip"):
            # Set converter
            self.set_arg_converter(col, convert_phip)

   # --- UQ Columns ---
    # Set UQ col for all FM cols
    def _make_uq_cols_FM(self):
        r"""Set default UQ col names for forces & moments

        :Call:
            >>> db._make_uq_cols_FM()
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
        :Versions:
            * 2020-03-20 ``@ddalle``: First version
        """
        # List FM tags
        tags = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
        # Loop through them
        for tag in tags:
            # Get cols with matching tag
            for col in self.get_cols_by_tag(tag):
                # Name of [default] UQ col
                ucol = self.prepend_colname(col, "U")
                # Set it
                self.set_uq_col(col, ucol)

    # Set UQ extra cols for all FM cols
    def _make_uq_ecols_FM(self):
        r"""Set default UQ extra cols for reference UQ MRP

        :Call:
            >>> db._make_uq_ecols_FM()
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
        :Versions:
            * 2020-03-23 ``@ddalle``: First version
        """
        # List moment tags
        tags = ["CLM", "CLN"]
        # Loop through them
        for tag in tags:
            # Get cols with matching tag
            for col in self.get_cols_by_tag(tag):
                # Get actual UQ col
                ucol = self.get_uq_col(col)
                # Default UQ col if needed
                if ucol is None:
                    ucol = self.prepend_colname(col, "U")
                # Form ecol name
                ecol = self.substitute_prefix(col, "U", "x")
                # Set it
                self.set_uq_ecol(ucol, ecol)
                # Set function
                if tag == "CLM":
                    # Set efunc
                    self.set_uq_efunc(ecol, estimate_xCLM)
                elif tag == "CLN":
                    # Set efunc
                    self.set_uq_efunc(ecol, estimate_xCLN)

    # Set UQ aux cols for moment cols
    def _make_uq_acols_FM(self):
        r"""Set default UQ aux cols for reference UQ MRP

        :Call:
            >>> db._make_uq_acols_FM()
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
        :Versions:
            * 2020-03-23 ``@ddalle``: First version
        """
        # List moment tags
        tags = ["CLM", "CLN"]
        # Loop through them
        for tag in tags:
            # Get cols with matching tag
            for col in self.get_cols_by_tag(tag):
                # Check col names' ending
                if col.endswith("LM"):
                    # CLM -> CN
                    acol = col[:-2] + "N"
                elif col.endswith("LN"):
                    # CLN -> CY
                    acol = col[:-2] + "Y"
                elif col.endswith("m"):
                    # Cm -> CN
                    acol = col[:-1] + "N"
                elif col.endswith("n"):
                    # Cn -> CY
                    acol = col[:-1] + "Y"
                # Get actual UQ col
                ucol = self.get_uq_col(col)
                # Default UQ col if needed
                if ucol is None:
                    ucol = self.prepend_colname(col, "U")
                # Set it
                self.set_uq_acol(ucol, acol)

   # --- Derived Data Columns ---
    # Make *CLMX* evaluators
    def make_CLMX(self):
        r"""Build and save evaluators for *CLMX* cols

        :Call:
            >>> db.make_CLMX()
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
        :Versions:
            * 2020-03-26 ``@ddalle``: First version
        """
        # Loop through *CLM* cols
        for col in self.get_cols_by_tag("CLM"):
            # Args
            args = self.get_response_args(col)
            # Check if set
            if args is None:
                return
            # Append MRP location
            args += ["xMRP"]
            # Get aux column name
            acol = self._getcol_CN_from_CLM(col)
            # Name of shifted col
            scol = self.append_colname(col, "X")
            # Test if *scol* is already present
            if self.get_response_method(scol):
                continue
            # Set aux cols for "CLM" to "CN"
            self.set_response_acol(col, [acol])
            # Set aux cols for "CLMX" to ["CLM", "CN"]
            self.set_response_acol(scol, [col, acol])
            # Generate *CLMX* function
            func = genr8_fCLMX(col, acol)
            # Save it
            self.make_response(scol, "function", args, func=func)

    # Make *CLNX* evaluators
    def make_CLNX(self):
        r"""Build and save evaluators for *CLNX* cols

        :Call:
            >>> db.make_CLNX()
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
        :Versions:
            * 2020-03-26 ``@ddalle``: First version
        """
        # Loop through *CLM* cols
        for col in self.get_cols_by_tag("CLN"):
            # Args
            args = self.get_response_args(col)
            # Check if set
            if args is None:
                return
            # Append MRP location
            args += ["xMRP"]
            # Get aux column name
            acol = self._getcol_CY_from_CLN(col)
            # Name of shifted col
            scol = self.append_colname(col, "X")
            # Test if *scol* is already present
            if self.get_response_method(scol):
                continue
            # Set aux cols for "CLN" to "CY"
            self.set_response_acol(col, [acol])
            # Set aux cols for "CLNX" to ["CLN", "CY"]
            self.set_response_acol(scol, [col, acol])
            # Generate *CLMX* function
            func = genr8_fCLNX(col, acol)
            # Save it
            self.make_response(scol, "function", args, func=func)

    # Make *UCLMX* evaluators
    def make_UCLMX(self):
        r"""Build and save evaluators for *UCLMX* cols

        :Call:
            >>> db.make_UCLMX()
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
        :Versions:
            * 2020-05-04 ``@ddalle``: First version
        """
        # Loop through *CLM* cols
        for col in self.get_cols_by_tag("CLM"):
            # Uncertainty column
            ucol0 = self.get_uq_col(col)
            # Check if set
            if ucol0 is None:
                continue
            # Args
            args = self.get_response_args(ucol0)
            # Check if set
            if args is None:
                continue
            # Append MRP location
            args += ["xMRP"]
            # Get aux column name
            acol = self._getcol_CN_from_CLM(col)
            # Check it
            if acol is None:
                continue
            # UQ col for *CN*
            ucol1 = self.get_uq_col(acol)
            # Check it
            if ucol1 is None:
                continue
            # Name of col for UQ at shifted location
            ucol = self.append_colname(ucol0, "X")
            # Test if *scol* is already present
            if self.get_response_method(ucol):
                continue
            # Name for UQ reference MRP
            ecols = self.get_uq_ecol(ucol0)
            # Check it
            if len(ecols) != 1:
                continue
            else:
                # Unpack
                ucol2, = ecols
            # Generate *CLMX* function
            func = genr8_fUCLMX(ucol0, ucol1, ucol2)
            # Save it
            self.make_response(ucol, "function", args, func=func)

    # Make *UCLNX* evaluators
    def make_UCLNX(self):
        r"""Build and save evaluators for *UCLNX* cols

        :Call:
            >>> db.make_UCLNX()
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
        :Versions:
            * 2020-05-04 ``@ddalle``: First version
        """
        # Loop through *CLM* cols
        for col in self.get_cols_by_tag("CLN"):
            # Uncertainty column
            ucol0 = self.get_uq_col(col)
            # Check if set
            if ucol0 is None:
                continue
            # Args
            args = self.get_response_args(ucol0)
            # Check if set
            if args is None:
                continue
            # Append MRP location
            args += ["xMRP"]
            # Get aux column name
            acol = self._getcol_CY_from_CLN(col)
            # Check it
            if acol is None:
                continue
            # UQ col for *CN*
            ucol1 = self.get_uq_col(acol)
            # Check it
            if ucol1 is None:
                continue
            # Name of col for UQ at shifted location
            ucol = self.append_colname(ucol0, "X")
            # Test if *scol* is already present
            if self.get_response_method(ucol):
                continue
            # Name for UQ reference MRP
            ecols = self.get_uq_ecol(ucol0)
            # Check it
            if len(ecols) != 1:
                continue
            else:
                # Unpack
                ucol2, = ecols
            # Generate *CLNX* function
            func = genr8_fUCLNX(ucol0, ucol1, ucol2)
            # Save it
            self.make_response(ucol, "function", args, func=func)

    # Get *CN* col name from *CLM*
    def _getcol_CN_from_CLM(self, col):
        r"""Form *CN* col name from *CLM* col name

        :Call:
            >>> acol = db._getcol_CN_from_CLM(col)
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
            *col*: ``"CLM"`` | :class:`str`
                Name of *CLM* column
        :Outputs:
            *acol*: ``"CN"`` | :class:`str`
                Name of *CN* column
        :Versions:
            * 2020-03-26 ``@ddalle``: First version
        """
        # Split component and coeff
        parts = col.split(".")
        # Get coeff
        coeff = parts[-1]
        # Get aux column name
        if coeff.endswith("LM"):
            # CLM -> CN
            acoeff = coeff[:-2] + "N"
        elif coeff.endswith("m"):
            # Cm -> CN
            acoeff = coeff[:-1] + "N"
        elif coeff.startswith("CLM"):
            # CLMF -> CNF
            acoeff = "CN" + coeff[3:]
        elif coeff.startswith("Cm"):
            # Cmf -> CNf
            acoeff = "CN" + coeff[2:]
        elif coeff.startswith("MLM"):
            # MLM -> FN
            acoeff = "FN" + acoeff[3:]
        elif coeff.startswith("My"):
            # My -> Fz
            acoeff = "Fz" + coeff[2:]
        else:
            # Just add N
            acoeff = coeff + "N"
        # Save updated coeff
        parts[-1] = acoeff
        # Output
        return ".".join(parts)

    # Get *CY* col name from *CLN*
    def _getcol_CY_from_CLN(self, col):
        r"""Form *CY* col name from *CLN* col name

        :Call:
            >>> acol = db._getcol_CY_from_CLN(col)
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
            *col*: ``"CLM"`` | :class:`str`
                Name of *CLM* column
        :Outputs:
            *acol*: ``"CN"`` | :class:`str`
                Name of *CN* column
        :Versions:
            * 2020-03-26 ``@ddalle``: First version
        """
        # Split component and coeff
        parts = col.split(".")
        # Get coeff
        coeff = parts[-1]
        # Get aux column name
        if coeff.endswith("LN"):
            # CLN -> CY
            acoeff = coeff[:-2] + "Y"
        elif coeff.endswith("n"):
            # Cn -> CY
            acoeff = coeff[:-1] + "Y"
        elif coeff.startswith("CLN"):
            # CLNF -> CYF
            acoeff = "CY" + coeff[3:]
        elif coeff.startswith("Cn"):
            # Cnf -> CYf
            acoeff = "CY" + coeff[2:]
        elif coeff.startswith("MLN"):
            # MLN -> FY
            acoeff = "FY" + acoeff[3:]
        elif coeff.startswith("Mz"):
            # Mz -> Fy
            acoeff = "Fy" + coeff[2:]
        else:
            # Just add N
            acoeff = coeff + "Y"
        # Save updated coeff
        parts[-1] = acoeff
        # Output
        return ".".join(parts)

    # Get *CLL* col name from *CN*
    def _getcol_CLL_from_CN(self, col):
        r"""Form *CLL* col name from *CN* col name

        :Call:
            >>> acol = db._getcol_CLL_from_CN(col)
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
            *col*: ``"CN"`` | :class:`str`
                Name of *CN* column
        :Outputs:
            *acol*: ``"CLL"`` | :class:`str`
                Name of *CLM* column
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Split component and coeff
        parts = col.split(".")
        # Get coeff
        coeff = parts[-1]
        # Get aux column name
        if coeff.endswith("N"):
            # CN -> CLL
            acoeff = coeff[:-1] + "LL"
        elif coeff.startswith("CN"):
            # CNF -> CLLF
            acoeff = "CLL" + coeff[2:]
        elif coeff.startswith("FN"):
            # FN -> MLL
            acoeff = "MLL" + acoeff[2:]
        elif coeff.startswith("Fz"):
            # Fz -> Mx
            acoeff = "Mx" + coeff[2:]
        else:
            # Just add LL
            acoeff = coeff + "LL"
        # Save updated coeff
        parts[-1] = acoeff
        # Output
        return ".".join(parts)

    # Get *CLL* col name from *CY*
    def _getcol_CLL_from_CY(self, col):
        r"""Form *CLL* col name from *CY* col name

        :Call:
            >>> acol = db._getcol_CLL_from_CY(col)
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
            *col*: ``"CY"`` | :class:`str`
                Name of *CY* column
        :Outputs:
            *acol*: ``"CLL"`` | :class:`str`
                Name of *CLN* column
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Split component and coeff
        parts = col.split(".")
        # Get coeff
        coeff = parts[-1]
        # Get aux column name
        if coeff.endswith("Y"):
            # CY -> CLN
            acoeff = coeff[:-1] + "LL"
        elif coeff.startswith("CY"):
            # CYF -> CLLF
            acoeff = "CLL" + coeff[2:]
        elif coeff.startswith("FY"):
            # FY -> MLL
            acoeff = "MLL" + acoeff[2:]
        elif coeff.startswith("Fy"):
            # Fy -> Mx
            acoeff = "Mx" + coeff[2:]
        else:
            # Just add LN
            acoeff = coeff + "LL"
        # Save updated coeff
        parts[-1] = acoeff
        # Output
        return ".".join(parts)

    # Get *CLM* col name from *CN*
    def _getcol_CLM_from_CN(self, col):
        r"""Form *CLM* col name from *CN* col name

        :Call:
            >>> acol = db._getcol_CLM_from_CN(col)
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
            *col*: ``"CN"`` | :class:`str`
                Name of *CN* column
        :Outputs:
            *acol*: ``"CLM"`` | :class:`str`
                Name of *CLM* column
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Split component and coeff
        parts = col.split(".")
        # Get coeff
        coeff = parts[-1]
        # Get aux column name
        if coeff.endswith("N"):
            # CN -> CLM
            acoeff = coeff[:-1] + "LM"
        elif coeff.startswith("CN"):
            # CNF -> CLMF
            acoeff = "CLM" + coeff[2:]
        elif coeff.startswith("FN"):
            # FN -> MLM
            acoeff = "MLM" + acoeff[2:]
        elif coeff.startswith("Fz"):
            # My -> Fz
            acoeff = "My" + coeff[2:]
        else:
            # Just add LM
            acoeff = coeff + "LM"
        # Save updated coeff
        parts[-1] = acoeff
        # Output
        return ".".join(parts)

    # Get *CY* col name from *CLN*
    def _getcol_CLN_from_CY(self, col):
        r"""Form *CLN* col name from *CY* col name

        :Call:
            >>> acol = db._getcol_CLN_from_CY(col)
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
            *col*: ``"CY"`` | :class:`str`
                Name of *CY* column
        :Outputs:
            *acol*: ``"CLN"`` | :class:`str`
                Name of *CLN* column
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Split component and coeff
        parts = col.split(".")
        # Get coeff
        coeff = parts[-1]
        # Get aux column name
        if coeff.endswith("Y"):
            # CY -> CLN
            acoeff = coeff[:-1] + "LM"
        elif coeff.startswith("CY"):
            # CYF -> CLNF
            acoeff = "CLN" + coeff[2:]
        elif coeff.startswith("FY"):
            # FY -> MLN
            acoeff = "MLN" + acoeff[2:]
        elif coeff.startswith("Fy"):
            # Mz -> Fy
            acoeff = "Mz" + coeff[2:]
        else:
            # Just add LN
            acoeff = coeff + "LN"
        # Save updated coeff
        parts[-1] = acoeff
        # Output
        return ".".join(parts)

    # Get *CN* col name from *dCN*
    def _getcol_CX_from_dCX(self, col):
        r"""Form *CN* col name from *dCN* col name

        :Call:
            >>> acol = db._getcol_CX_from_dCX(col)
        :Inputs:
            *db*: :class:`cape.attdb.dbfm.DBFM`
                LV force & moment database
            *col*: ``"dCN"`` | :class:`str`
                Name of *CY* column
        :Outputs:
            *acol*: ``"CN"`` | :class:`str`
                Name of *CLN* column
        :Versions:
            * 2020-06-11 ``@ddalle``: First version
        """
        # Split component and coeff
        parts = col.split(".")
        # Get coeff
        coeff = parts[-1]
        # Get aux column name
        if coeff.startswith("d"):
            # dCN -> CN
            acoeff = coeff[1:]
        elif coeff.startswith("D"):
            # DCN -> CN
            acoeff = coeff[1:]
        else:
            # Just add *F* for integral
            acoeff = coeff + "F"
        # Save updated coeff
        parts[-1] = acoeff
        # Output
        return ".".join(parts)
  # >


# Combine options
kwutils._combine_val(DBFM._tagmap, rdbaero.AeroDataKit._tagmap)

# Invert the _tagmap
DBFM.create_tagcols()

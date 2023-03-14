#!/usr/bin/env python
"""
---------------------------------------------------------
:mod:`tnakit.convert`: Conversion Utilities
---------------------------------------------------------

This module provides several conversion functions for the SLS Aero Task Team
database module set.  This includes conversions from angle of attack and
sideslip angle to total angle of attack and roll, and vice versa.

    .. code-block:: pycon
    
        >>> tnakit.convert.AlphaTPhi2AlphaBeta(2.0, 90.0)
        (0.0, 2.0)
        >>> tnakit.convert.AlphaBeta2AlphaTPhi(4.0, 4.0)
        (5.6545547205742643, 45.069869884780957)

It also includes several atmospheric calculations and unit conversions. An
important example is calculating dynamic viscosity (in FPS units) using
Sutherland's Law. The output is in units of slug/ft*s.

    .. code-block:: pycon
    
        >>> tnakit.convert.SutherlandFPS(459.67)
        3.39730346788531e-07
"""

# Numerics
import numpy as np

# Degrees to radians
deg = np.pi / 180.0

# Step function
def fstep(x):
    """Return -1 for negative numbers and +1 for nonnegative inputs
    
    :Call:
        >>> y = fstep(x)
    :Inputs:
        *x*: :class:`float` | :class:`np.ndarray`
            Input or array of inputs
    :Outputs:
        *y*: ``-1.0`` | ``1.0`` | :class:`np.ndarray`
            Output step function or array of outputs
    :Versions:
        * 2017-06-27 ``@ddalle``: First version
    """
    return -1 + 2*np.ceil(0.5+0.5*np.sign(x))


# Floor Step function
def fstep1(x):
    """Return -1 for nonpositive numbers and +1 for negative inputs
    
    :Call:
        >>> y = fstep(x)
    :Inputs:
        *x*: :class:`float` | :class:`np.ndarray`
            Input or array of inputs
    :Outputs:
        *y*: ``-1.0`` | ``1.0`` | :class:`np.ndarray`
            Output step function or array of outputs
    :Versions:
        * 2017-06-27 ``@ddalle``: First version
    """
    return -1 + 2*np.floor(0.5+0.5*np.sign(x))


# Convert (total angle of attack, total roll angle) to (aoa, aos)
def AlphaTPhi2AlphaBeta(alpha_t, phi):
    """
    Convert total angle of attack and total roll angle to angle of attack and
    sideslip angle.
    
    :Call:
        >>> alpha, beta = cape.AlphaTPhi2AlphaBeta(alpha_t, beta)
    :Inputs:
        *alpha_t*: :class:`float` | :class:`numpy.array`
            Total angle of attack
        *phi*: :class:`float` | :class:`numpy.array`
            Total roll angle
    :Outputs:
        *alpha*: :class:`float` | :class:`numpy.array`
            Angle of attack
        *beta*: :class:`float` | :class:`numpy.array`
            Sideslip angle
    :Versions:
        * 2014-06-02 ``@ddalle``: First version
    """
    # Trig functions.
    ca = np.cos(alpha_t*deg); cp = np.cos(phi*deg)
    sa = np.sin(alpha_t*deg); sp = np.sin(phi*deg)
    # Get the components of the normalized velocity vector.
    u = 1.0e-8*np.fix(1e8*ca)
    v = 1.0e-8*np.fix(1e8*sa * sp)
    w = 1.0e-8*np.fix(1e8*sa * cp)
    # Convert to alpha, beta
    alpha = np.arctan2(w, u) / deg
    beta = np.arcsin(v) / deg
    # Output
    return alpha, beta
    
    
# Convert (aoa, aos) to (total angle of attack, total roll angle)
def AlphaBeta2AlphaTPhi(alpha, beta):
    """Convert angle of attack and sideslip to total angle of attack and roll
    
    :Call:
        >>> alpha_t, phi = cape.AlphaBeta2AlphaTPhi(alpha, beta)
    :Inputs:
        *alpha*: :class:`float` | :class:`numpy.array`
            Angle of attack
        *beta*: :class:`float` | :class:`numpy.array`
            Sideslip angle
    :Outputs:
        *alpha_t*: :class:`float` | :class:`numpy.array`
            Total angle of attack
        *phi*: :class:`float` | :class:`numpy.array`
            Total roll angle
    :Versions:
        * 2014-06-02 ``@ddalle``: First version
        * 2014-11-05 ``@ddalle``: Transposed alpha and beta in *w* formula
    """
    # Trig functions.
    ca = np.cos(alpha*deg); cb = np.cos(beta*deg)
    sa = np.sin(alpha*deg); sb = np.sin(beta*deg)
    # Get the components of the normalized velocity vector.
    u = cb * ca
    v = sb
    w = cb * sa
    # Convert to alpha_t, phi
    phi = 180 - np.arctan2(v, -w) / deg
    alpha_t = np.arccos(u) / deg
    # Output
    return alpha_t, phi
    
# Convert (aoa, aos) to (maneuver angle of attack, maneuver roll angle)
def AlphaBeta2AlphaMPhi(alpha, beta):
    """Convert angle of attack and sideslip to maneuver axis
    
    :Call:
        >>> alpha_m, phi_m = cape.AlphaBeta2AlphaMPhi(alpha, beta)
    :Inputs:
        *alpha*: :class:`float` | :class:`numpy.array`
            Angle of attack
        *beta*: :class:`float` | :class:`numpy.array`
            Sideslip angle
    :Outputs:
        *alpha_m*: :class:`float` | :class:`numpy.array`
            Signed total angle of attack [deg]
        *phi_m*: :class:`float` | :class:`numpy.array`
            Total roll angle
    :Versions:
        * 2017-06-26 ``@ddalle``: First version
    """
    # Trig functions.
    ca = np.cos(alpha*deg); cb = np.cos(beta*deg)
    sa = np.sin(alpha*deg); sb = np.sin(beta*deg)
    # Get the components of the normalized velocity vector.
    u = cb * ca
    v = sb
    w = cb * sa
    # Convert to alpha_t, phi
    phi1 = 180 - np.arctan2(w, -v) / deg
    aoa1 = np.arccos(u) / deg
    # Apply different signs
    alpha_m = aoa1 * fstep1(180-phi1)
    phi_m = 180 - phi1 - 90*fstep(alpha_m)
    # Ensure phi==0 for alpha==0
    phi_m *= np.sign(np.abs(alpha_m))
    # Output
    return alpha_m, phi_m
    
# Convert (total angle of attack, total roll angle) to (aoam, phim)
def AlphaTPhi2AlphaMPhi(alpha_t, phi):
    """
    Convert total angle of attack and total roll angle to angle of attack and
    sideslip angle.
    
    :Call:
        >>> alpha_m, phi_m = cape.AlphaTPhi2AlphaMPhi(alpha_t, phi)
    :Inputs:
        *alpha_t*: :class:`float` | :class:`numpy.array`
            Total angle of attack
        *phi*: :class:`float` | :class:`numpy.array`
            Total roll angle
    :Outputs:
        *alpha_m*: :class:`float` | :class:`numpy.array`
            Signed total angle of attack [deg]
        *phi_m*: :class:`float` | :class:`numpy.array`
            Total roll angle
    :Versions:
        * 2017-06-27 ``@ddalle``: First version
    """
    # Trig functions.
    ca = np.cos(alpha_t*deg); cp = np.cos(phi*deg)
    sa = np.sin(alpha_t*deg); sp = np.sin(phi*deg)
    # Get the components of the normalized velocity vector.
    u = ca
    v = sa * sp
    w = sa * cp
    # Convert to alpha_t, phi
    phi1 = 180 - np.arctan2(w, -v) / deg
    aoa1 = np.arccos(u) / deg
    # Apply different signs
    alpha_m = aoa1 * fstep1(180-phi1)
    phi_m = 180 - phi1 - 90*fstep(alpha_m)
    # Ensure phi==0 for alpha==0
    phi_m *= np.sign(np.abs(alpha_m))
    # Output
    return alpha_m, phi_m
    
# Convert (aoam, phim) to (aoav, phiv)
def AlphaMPhi2AlphaTPhi(alpha_m, phi_m):
    """
    Convert total angle of attack and total roll angle to angle of attack and
    sideslip angle.
    
    :Call:
        >>> alpha_t, phi = cape.AlphaTPhi2AlphaMPhi(alpha_m, phi_m)
    :Inputs:
        *alpha_m*: :class:`float` | :class:`numpy.array`
            Signed total angle of attack [deg]
        *phi_m*: :class:`float` | :class:`numpy.array`
            Total roll angle
    :Outputs:
        *alpha_t*: :class:`float` | :class:`numpy.array`
            Total angle of attack
        *phi*: :class:`float` | :class:`numpy.array`
            Total roll angle
    :Versions:
        * 2017-06-27 ``@ddalle``: First version
    """
    # Trig functions.
    ca = np.cos(alpha_m*deg); cp = np.cos(phi_m*deg)
    sa = np.sin(alpha_m*deg); sp = np.sin(phi_m*deg)
    # Get the components of the normalized velocity vector.
    u = ca
    v = sa * sp
    w = sa * cp
    # Convert to alpha_t, phi
    phi = 180 - np.arctan2(v, -w) / deg
    alpha_t = np.arccos(u) / deg
    # Output
    return alpha_t, phi
# def AlphaMPhie2AlphaTPhi


# Sutherland's law (FPS)
def SutherlandFPS(T, mu0=None, T0=None, C=None):
    """Calculate viscosity using Sutherland's law using imperial units
    
    This returns
    
        .. math::
        
            \\mu = \\mu_0 \\frac{T_0+C}{T+C}\\left(\\frac{T}{T_0}\\right)^{3/2}
    
    :Call:
        >>> mu = SutherlandFPS(T)
        >>> mu = SutherlandFPS(T, mu0=None, T0=None, C=None)
    :Inputs:
        *T*: :class:`float`
            Static temperature in degrees Rankine
        *mu0*: {``3.58394e-7``} | :class:`float`
            Reference viscosity [slug/ft*s]
        *T0*: {``491.67``} | :class:`float`
            Reference temperature [R]
        *C*: {``198.6``} | :class:`float`
            Reference temperature [R]
    :Outputs:
        *mu*: :class:`float`
            Dynamic viscosity [slug/ft*s]
    :Versions:
        * 2016-03-23 ``@ddalle``: First version
    """
    # Reference viscosity
    if mu0 is None: mu0 = 3.58394e-7
    # Reference temperatures
    if T0 is None: T0 = 491.67
    if C  is None: C = 198.6
    # Sutherland's law
    return mu0 * (T0+C)/(T+C) * (T/T0)**1.5
# def SutherlandFPS

# Get Reynolds number
def ReynoldsPerFoot(p, T, M, R=None, gam=None, mu0=None, T0=None, C=None):
    """Calculate Reynolds number per foot using Sutherland's Law
    
    :Call:
        >>> Re = ReynoldsPerFoot(p, T, M)
        >>> Re = ReynoldsPerFoot(p, T, M, gam=None, R=None, T0=None, C=None)
    :Inputs:
        *p*: :class:`float`
            Static pressure [psf]
        *T*: :class:`float`
            Static temperature [R]
        *M*: :class:`float`
            Mach number
        *R*: :class:`float`
            Gas constant [ft^2/s^2*R]
        *gam*: :class:`float`
            Ratio of specific heats
        *mu0*: :class:`float`
            Reference viscosity [slug/ft*s]
        *T0*: :class:`float`
            Reference temperature [R]
        *C*: :class:`float`
            Reference temperature [R]
    :Outputs:
        *Re*: :class:`float`
            Reynolds number per foot
    :See also:
        * :func:`SutherlandFPS`
        * :func:`PressureFPSFromRe`
    :Versions:
        * 2016-03-23 ``@ddalle``: First version
    """
    # Gas constant
    if R is None: R = 1716.0
    # Ratio of specific heats
    if gam is None: gam = 1.4
    # Calculate density
    rho = p / (R*T)
    # Sound speed
    a = np.sqrt(gam*R*T)
    # Velocity
    U = M*a
    # Calculate viscosity
    mu = SutherlandFPS(T, mu0=mu0, T0=T0, C=C)
    # Reynolds number per foot
    return rho*U/mu
# def ReynoldsPerFoot

# Calculate pressure from Reynolds number
def PressureFPSFromRe(Re, T, M, R=None, gam=None, mu0=None, T0=None, C=None):
    """Calculate pressure from Reynolds number
    
    :Call:
        >>> p = PressureFPSFromRe(Re, T, M)
        >>> p = PressureFPSFromRe(Re, T, M, R=None, gam=None, **kw)
    :Inputs:
        *Re*: :class:`float`
            Reynolds number per foot
        *T*: :class:`float`
            Static temperature [R]
        *M*: :class:`float`
            Mach number
        *R*: :class:`float`
            Gas constant [ft^2/s^2*R]
        *gam*: :class:`float`
            Ratio of specific heats
        *mu0*: :class:`float`
            Reference viscosity [slug/ft*s]
        *T0*: :class:`float`
            Reference temperature [K]
        *C*: :class:`float`
            Reference temperature [K]
    :Outputs:
        *p*: :class:`float`
            Static pressure [psf]
    :See also:
        * :func:`SutherlandFPS`
        * :func:`ReynoldsPerFoot`
    :Versions:
        * 2016-03-24 ``@ddalle``: First version
    """
    # Gas constant
    if R is None: R = 1716.0
    # Ratio of specific heats
    if gam is None: gam = 1.4
    # Sound speed
    a = np.sqrt(gam*R*T)
    # Velocity
    U = M*a
    # Viscosity
    mu = SutherlandFPS(T, mu0=mu0, T0=T0, C=C)
    # Pressure
    return Re*mu*R*T/U
# def PressureFPSFromRe


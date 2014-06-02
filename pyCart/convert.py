"""
Conversion Tools for pyCart
===========================

Perform conversions such as (alpha total, phi) to (alpha, beta).
"""

# Need NumPy for trig.
import numpy as _np

# Convert (total angle of attack, total roll angle) to (aoa, aos)
def AlphaTPhi2AlphaBeta(alpha_t, phi):
    """
    Convert total angle of attack and total roll angle to angle of attack and
    sideslip angle.
    
    :Call:
        >>> alpha, beta = pyCart.AlphaTPhi2AlphaBeta(alpha_t, beta)
        
    :Inputs:
        *alpha_t*: :class:`float` or :class:`numpy.array`
            Total angle of attack
        *phi*: :class:`float` or :class:`numpy.array`
            Total roll angle
            
    :Outputs:
        *alpha*: :class:`float` or :class:`numpy.array`
            Angle of attack
        *beta*: :class:`float` or :class:`numpy.array`
            Sideslip angle
    """
    # Versions:
    #  2014.06.02 @ddalle : First version
    
    # Trig functions.
    ca = _np.cos(alpha_t*_np.pi/180); cp = _np.cos(phi*_np.pi/180)
    sa = _np.sin(alpha_t*_np.pi/180); sp = _np.sin(phi*_np.pi/180)
    # Get the components of the normalized velocity vector.
    u = ca
    v = sa * sp
    w = sa * cp
    # Convert to alpha, beta
    alpha = _np.arctan2(w, u) * 180/_np.pi
    beta = _np.arcsin(v) * 180/_np.pi
    # Output
    return alpha, beta
    
    
# Convert (total angle of attack, total roll angle) to (aoa, aos)
def AlphaBeta2AlphaTPhi(alpha, beta):
    """
    Convert total angle of attack and total roll angle to angle of attack and
    sideslip angle.
    
    :Call:
        >>> alpha_t, phi = pyCart.AlphaBeta2AlphaTPhi(alpha, beta)
        
    :Inputs:
        *alpha*: :class:`float` or :class:`numpy.array`
            Angle of attack
        *beta*: :class:`float` or :class:`numpy.array`
            Sideslip angle
            
    :Outputs:
        *alpha_t*: :class:`float` or :class:`numpy.array`
            Total angle of attack
        *phi*: :class:`float` or :class:`numpy.array`
            Total roll angle
    """
    # Versions:
    #  2014.06.02 @ddalle : First version
    
    # Trig functions.
    ca = _np.cos(alpha*_np.pi/180); cb = _np.cos(beta*_np.pi/180)
    sa = _np.sin(alpha*_np.pi/180); sb = _np.sin(beta*_np.pi/180)
    # Get the components of the normalized velocity vector.
    u = ca * cb
    v = sb
    w = ca * sb
    # Convert to alpha, beta
    phi = _np.arctan2(v, w) * 180/_np.pi
    alpha_t = _np.arcsin(v) * 180/_np.pi
    # Output
    return alpha_t, phi
    

    
    


#!/usr/bin/env python
"""
Volume Computation Tools
========================

Chimera Grid Tools ``volcomp.F`` converted to Python.

"""

# Numerics
import numpy as np

    
# Volume of a pyramid
def VOLPYM(XP,YP,ZP, XA,YA,ZA, XB,YB,ZB, XC,YC,ZC, XD,YD,ZD):
    """Compute the volume of a pentahedral pyramid
    
    The base of the points are A,B,C,D counterclockwise viewed from apex P.
    All inputs can be either scalars or vectors, but each input that *is* a
    vector must have the same shape.  The output is a scalar if and only if all
    15 inputs are scalar.
    
    :Call:
        >>> V = VOLPYM(XP,YP,ZP, XA,YA,ZA, XB,YB,ZB, XC,YC,ZC, XD,YD,ZD)
    :Inputs:
        *XP*: :class:`float` | :class:`np.ndarray`
            X-coordinates of vertex point(s)
        *YP*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of vertex point(s)
        *ZP*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of vertex point(s)
        *XA*: :class:`float` | :class:`np.ndarray`
            X-coordinates of base point(s) A
        *YA*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of base point(s) A
        *ZA*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of base point(s) A
    :Outputs:
        *V*: :class:`float` | :class:`np.ndarray`
            Volume of each pyramid(s)
    :Versions:
        * 2017-02-12 ``@ddalle``: Translated from CGT ``lib/volcomp.F``
    """
    # Vectors (for use in cross products)
    XAC = XA-XC
    YAC = YA-YC
    ZAC = ZA-ZC
    XBD = XB-XD
    YBD = YB-YD
    ZBD = ZB-ZD
    # Calculate Volume
    VOL = (1.0/6.0) * (
        (XP - 0.25*(XA+XB+XC+XD))*(YAC*ZBD - ZAC*YBD) +
        (YP - 0.25*(YA+YB+YC+YD))*(ZAC*XBD - XAC*ZBD) +
        (ZP - 0.25*(ZA+ZB+ZC+ZD))*(XAC*YBD - YAC*XBD))
    # Output
    return VOL
    
# Volume of tetrahedron
def VOLTET(XA,YA,ZA, XB,YB,ZB, XC,YC,ZC, XD,YD,ZD):
    """Compute the volume of a tetrahedron
    
    All inputs can be either scalars or vectors, but each input that *is* a
    vector must have the same shape.  The output is a scalar if and only if all
    15 inputs are scalar.
    
    :Call:
        >>> V = VOLTET(XA,YA,ZA, XB,YB,ZB, XC,YC,ZC, XD,YD,ZD)
    :Inputs:
        *XA*: :class:`float` | :class:`np.ndarray`
            X-coordinates of base point(s) A
        *YA*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of base point(s) A
        *ZA*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of base point(s) A
    :Outputs:
        *V*: :class:`float` | :class:`np.ndarray`
            Volume of each pyramid(s)
    :Versions:
        * 2017-02-12 ``@ddalle``: Translated from CGT ``lib/volcomp.F``
    """
    # Vectors (for use in cross products)
    XBD = XB-XD
    YBD = YB-YD
    ZBD = ZB-ZD
    XCD = XC-XD
    YCD = YC-YD
    ZCD = ZC-ZD
    # Calculate volume
    VOL = (1.0/6.0) * (
        (XA-XD)*(YBD*ZCD - ZBD*YCD) +
        (YA-YD)*(ZBD*XCD - XBD*ZCD) +
        (ZA-ZD)*(XBD*YCD - YBD*XCD))
    # Output
    return VOL
    
# Volume of triangular prism
def VOLPRIS(X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3, X4,Y4,Z4, X5,Y5,Z5, X6,Y6,Z6):
    """Compute the volume of a triangular prism
    
    The prism is constructed so that 1,2,3 are counterclockwise at the base and
    4,5,6 at the top counterclockwise.  The volumes of the three pyramids
    (*V1*,*V2*,*V3*), two tetrahedra (*V4*,*V5*), and prism (*V*) are returned.
    
    All inputs can be either scalars or vectors, but each input that *is* a
    vector must have the same shape.  The outputs are a scalar if and only if
    all 18 inputs are scalar.
    
    :Call:
        >>> V1,V2,V3,V4,V5,V = VOLPRIS(X1,Y1,Z1,X2,Y2,Z2, ..., X6,Y6,Z6)
    :Inputs:
        *X1*: :class:`float` | :class:`np.ndarray`
            X-coordinate(s) of point(s) 1 at base of prism(s)
        *Y1*: :class:`float` | :class:`np.ndarray`
            Y-coordinate(s) of point(s) 1 at base of prism(s)
        *Z1*: :class:`float` | :class:`np.ndarray`
            Z-coordinate(s) of point(s) 1 at base of prism(s)
        *X2*: :class:`float` | :class:`np.ndarray`
            X-coordinate(s) of point(s) 2 at base of prism(s)
        *X3*: :class:`float` | :class:`np.ndarray`
            X-coordinate(s) of point(s) 3 at base of prism(s)
        *X4*: :class:`float` | :class:`np.ndarray`
            X-coordinate(s) of point(s) 1 at top of prism(s)
    :Outputs:
        *V1*: :class:`float` | :class:`np.ndarray`
            Volume(s) of first pyramid(s), nodes C,1,4,5,2
        *V2*: :class:`float` | :class:`np.ndarray`
            Volume(s) of second pyramid(s), nodes C,1,3,6,4
        *V3*: :class:`float` | :class:`np.ndarray`
            Volume(s) of third pyramid(s), nodes C,2,5,6,3
        *V4*: :class:`float` | :class:`np.ndarray`
            Volume(s) of first tetrahedron(s), nodes C,1,2,3
        *V5*: :class:`float` | :class:`np.ndarray`
            Volume(s) of first tetrahedron(s), nodes C,6,5,4
        *V*: :class:`float` | :class:`np.ndarray`
            Volume(s) of prism(s)
    :Versions:
        * 2017-02-12 ``@ddalle``: Translated from ``lib/volcomp.F``
    """
    # Middle of the prism
    ONESIX = 1.0/6.0
    X7 = ONESIX*(X1+X2+X3+X4+X5+X6)
    Y7 = ONESIX*(Y1+Y2+Y3+Y4+Y5+Y6)
    Z7 = ONESIX*(Z1+Z2+Z3+Z4+Z5+Z6)
    # Compute volumes of the three pyramids
    V1 = VOLPYM(X7,Y7,Z7, X1,Y1,Z1, X4,Y4,Z4, X5,Y5,Z5, X2,Y2,Z2)
    V2 = VOLPYM(X7,Y7,Z7, X1,Y1,Z1, X3,Y3,Z3, X6,Y6,Y6, X4,Y4,Z4)
    V3 = VOLPYM(X7,Y7,Z7, X2,Y2,Z2, X5,Y5,Z5, X6,Y6,Z6, X3,Y3,Z3)
    V4 = VOLTET(X7,Y7,Z7, X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3)
    V5 = VOLTET(X7,Y7,Z7, X6,Y6,Z6, X5,Y5,Z5, X4,Y4,Z4)
    # Total volume
    V = V1 + V2 + V3 + V4 + V5
    # Output
    return V1, V2, V3, V4, V5, V
    
# Volume of triangular prism
def VolTriPrism(X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3, X4,Y4,Z4, X5,Y5,Z5, X6,Y6,Z6):
    """Compute the volume of a triangular prism
    
    The prism is constructed so that 1,2,3 are counterclockwise at the base and
    4,5,6 at the top counterclockwise.
    
    All inputs can be either scalars or vectors, but each input that *is* a
    vector must have the same shape.  The outputs are a scalar if and only if
    all 18 inputs are scalar.
    
    :Call:
        >>> V = VolTriPrism(X1,Y1,Z1,X2,Y2,Z2, ..., X6,Y6,Z6)
    :Inputs:
        *X1*: :class:`float` | :class:`np.ndarray`
            X-coordinate(s) of point(s) 1 at base of prism(s)
        *Y1*: :class:`float` | :class:`np.ndarray`
            Y-coordinate(s) of point(s) 1 at base of prism(s)
        *Z1*: :class:`float` | :class:`np.ndarray`
            Z-coordinate(s) of point(s) 1 at base of prism(s)
        *X2*: :class:`float` | :class:`np.ndarray`
            X-coordinate(s) of point(s) 2 at base of prism(s)
        *X3*: :class:`float` | :class:`np.ndarray`
            X-coordinate(s) of point(s) 3 at base of prism(s)
        *X4*: :class:`float` | :class:`np.ndarray`
            X-coordinate(s) of point(s) 1 at top of prism(s)
    :Outputs:
        *V*: :class:`float` | :class:`np.ndarray`
            Volume(s) of prism(s)
    :Versions:
        * 2017-02-12 ``@ddalle``: Modified from ``lib/volcomp.F``
    """
    # Middle of the prism
    ONESIX = 1.0/6.0
    X7 = ONESIX*(X1+X2+X3+X4+X5+X6)
    Y7 = ONESIX*(Y1+Y2+Y3+Y4+Y5+Y6)
    Z7 = ONESIX*(Z1+Z2+Z3+Z4+Z5+Z6)
    # Compute volumes of the three pyramids
    V1 = VOLPYM(X7,Y7,Z7, X1,Y1,Z1, X4,Y4,Z4, X5,Y5,Z5, X2,Y2,Z2)
    V2 = VOLPYM(X7,Y7,Z7, X1,Y1,Z1, X3,Y3,Z4, X6,Y6,Y6, X4,Y4,Z4)
    V3 = VOLPYM(X7,Y7,Z7, X2,Y2,Z2, X5,Y5,Z5, X6,Y6,Z6, X3,Y3,Z3)
    V4 = VOLTET(X7,Y7,Z7, X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3)
    V5 = VOLTET(X7,Y7,Z7, X6,Y6,Z6, X5,Y5,Z5, X4,Y4,Z4)
    # Output total volume
    return V1 + V2 + V3 + V4 + V5
            
    
    
    
    

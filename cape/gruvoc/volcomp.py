r"""
:mod:`gruvoc.volcomp`: Functions to calculate element volumes
==============================================================

This module contains functions to compute the volumes of the four basic
unstructured volume types.
"""


# Volume of a hex
def hexvol(
        x1, y1, z1,
        x2, y2, z2,
        x3, y3, z3,
        x4, y4, z4,
        x5, y5, z5,
        x6, y6, z6,
        x7, y7, z7,
        x8, y8, z8):
    r"""Compute the volume of a hex cell

    The hex is constructed so that 1,2,3,4 are counterclockwise at the
    base and 5,6,7,8 at the top counterclockwise.

    This function works by dividing the 8-node hexahedron by splitting
    it into two triangular prisms.

    :Call:
        >>> v = hexvol(x1, y1, z1, x2, ..., x8, y8, z8)
    :Inputs:
        *x1*: :class:`float` | :class:`np.ndarray`
            X-coordinates of point 1
        *y1*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of point 1
        *z1*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of point 1
        *x8*: :class:`float` | :class:`np.ndarray`
            X-coordinates of point 8
        *y8*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of point 8
        *z8*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of point 8
    :Outputs:
        *v*: :class:`float` | :class:`np.ndarray`
            Volume of each prism
    """
    # Split into to prisms
    v1 = privol(
        x1, y1, z1, x2, y2, z2, x3, y3, z3,
        x5, y5, z5, x6, y6, z6, x7, y7, z7)
    v2 = privol(
        x1, y1, z1, x3, y3, z3, x4, y4, z4,
        x5, y5, z5, x7, y7, z7, x8, y8, z8)
    # Return the sum
    return v1 + v2


# Volume of a (triangular) prism
def privol(
        x1, y1, z1,
        x2, y2, z2,
        x3, y3, z3,
        x4, y4, z4,
        x5, y5, z5,
        x6, y6, z6):
    r"""Compute the volume of a triangular prism

    The prism is constructed so that 1,2,3 are counterclockwise at the
    base and 4,5,6 at the top counterclockwise.

    This function works by dividing the 6-node pentahedron by splitting
    it into a pyramid and tetrahedron.

    :Call:
        >>> v = privol(x1, y1, z1, x2, ..., x6, y6, z6)
    :Inputs:
        *x1*: :class:`float` | :class:`np.ndarray`
            X-coordinates of point 1
        *y1*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of point 1
        *z1*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of point 1
        *x6*: :class:`float` | :class:`np.ndarray`
            X-coordinates of point 6
        *y6*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of point 6
        *z6*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of point 6
    :Outputs:
        *v*: :class:`float` | :class:`np.ndarray`
            Volume of each prism
    """
    # Divide into pyramid and tet and prism
    v1 = pyrvol(
        x1, y1, z1,
        x3, y3, z3,
        x6, y6, z6,
        x4, y4, z4,
        x2, y2, z2)
    # Tetrahedron
    v2 = tetvol(
        x4, y4, z4,
        x5, y5, z5,
        x6, y6, z6,
        x2, y2, z2)
    # Add them up
    return v1 + v2


# Volume of a pyramid
def pyrvol(
        xa, ya, za,
        xb, yb, zb,
        xc, yc, zc,
        xd, yd, zd,
        xp, yp, zp):
    r"""Compute the volume of a pentahedral pyramid

    The base of the points are A,B,C,D counterclockwise viewed from apex
    P. All inputs can be either scalars or vectors, but each input that
    *is* a vector must have the same shape. The output is a scalar if
    and only if all 15 inputs are scalar.

    :Call:
        >>> v = pyrvol(xa, ya, za, xb, ..., zd, xp, yp, zp)
    :Inputs:
        *xa*: :class:`float` | :class:`np.ndarray`
            X-coordinates of base point(s) A
        *ya*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of base point(s) A
        *za*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of base point(s) A
        *xp*: :class:`float` | :class:`np.ndarray`
            X-coordinates of vertex point(s)
        *yp*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of vertex point(s)
        *zp*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of vertex point(s)
    :Outputs:
        *v*: :class:`float` | :class:`np.ndarray`
            Volume of each pyramid
    """
    # Deltas for use in cross products
    xac = xa - xc
    yac = ya - yc
    zac = za - zc
    xbd = xb - xd
    ybd = yb - yd
    zbd = zb - zd
    # Calculate volume
    v = (1.0/6.0) * (
        (xp - 0.25*(xa+xb+xc+xd)) * (yac*zbd - zac*ybd) +
        (yp - 0.25*(ya+yb+yc+yd)) * (zac*xbd - xac*zbd) +
        (zp - 0.25*(za+zb+zc+zd)) * (xac*ybd - yac*xbd))
    # Output
    return v


# Volume of tetrahedron
def tetvol(
        xa, ya, za,
        xb, yb, zb,
        xc, yc, zc,
        xd, yd, zd):
    r"""Compute the volume of a tetrahedron
    All inputs can be either scalars or vectors, but each input that
    *is* a vector must have the same shape. The output is a scalar if
    and only if all 15 inputs are scalar.

    :Call:
        >>> v = tetvol(xa, ya, za, xb, ..., zc, xd, yd, zd)
    :Inputs:
        *xa*: :class:`float` | :class:`np.ndarray`
            X-coordinates of base point(s) A
        *ya*: :class:`float` | :class:`np.ndarray`
            Y-coordinates of base point(s) A
        *za*: :class:`float` | :class:`np.ndarray`
            Z-coordinates of base point(s) A
    :Outputs:
        *v*: :class:`float` | :class:`np.ndarray`
            Volume of each tetrahedron
    """
    # Vectors (for use in cross products)
    xbd = xb - xd
    ybd = yb - yd
    zbd = zb - zd
    xcd = xc - xd
    ycd = yc - yd
    zcd = zc - zd
    # Calculate volume
    v = (1.0/6.0) * (
        (xa-xd) * (ybd*zcd - zbd*ycd) +
        (ya-yd) * (zbd*xcd - xbd*zcd) +
        (za-zd) * (xbd*ycd - ybd*xcd))
    # Output
    return v

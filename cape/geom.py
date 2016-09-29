"""
Generic Cape Geometry Module: :mod:`cape.geom`
==============================================

This module provides several methods for modifying points or performing other
geometric manipulations in a way accessible to each of the subclasses.

:Versions:
    * 2015-09-30 ``@ddalle``: First version
"""

# Numerics
import numpy as np

# Function to rotate a triangulation about an arbitrary vector
def RotatePoints(X, v1, v2, theta):
    """Rotate a list of points
    
    :Call:
        >>> Y = RotatePoints(X, v1, v2, theta)
    :Inputs:
        *X*: :class:`numpy.ndarray`(:class:`float`), *shape* = (N,3)
            List of node coordinates
        *v1*: :class:`numpy.ndarray`, *shape* = (3,)
            Start point of rotation vector
        *v2*: :class:`numpy.ndarray`, *shape* = (3,)
            End point of rotation vector
        *theta*: :class:`float`
            Rotation angle in degrees
    :Outputs:
        *Y*: :class:`numpy.ndarray`(:class:`float`), *shape* = (N,3)
            List of rotated node coordinates
    :Versions:
        * 2014-10-07 ``@ddalle``: Copied from previous TriBase.Rotate()
    """
    # Convert points to NumPy.
    v1 = np.array(v1)
    v2 = np.array(v2)
    # Ensure array.
    if type(X).__name__ != 'ndarray':
        X = np.array(X)
    # Check for points.
    if X.size == 0:
        return X
    # Ensure list of points.
    if len(X.shape) == 1:
        X = np.array([X])
    # Extract the coordinates and shift origin.
    x = X[:,0] - v1[0]
    y = X[:,1] - v1[1]
    z = X[:,2] - v1[2]
    # Make the rotation vector
    v = (v2-v1) / np.linalg.linalg.norm(v2-v1)
    # Dot product of points with rotation vector
    k1 = v[0]*x + v[1]*y + v[2]*z
    # Trig functions
    c_th = np.cos(theta*np.pi/180.)
    s_th = np.sin(theta*np.pi/180.)
    # Initialize output.
    Y = X.copy()
    # Apply Rodrigues' rotation formula to get the rotated coordinates.
    Y[:,0] = x*c_th+(v[1]*z-v[2]*y)*s_th+v[0]*k1*(1-c_th)+v1[0]
    Y[:,1] = y*c_th+(v[2]*x-v[0]*z)*s_th+v[1]*k1*(1-c_th)+v1[1]
    Y[:,2] = z*c_th+(v[0]*y-v[1]*x)*s_th+v[2]*k1*(1-c_th)+v1[2]
    # Output
    return Y
    
# Function to rotate a triangulation about an arbitrary vector
def TranslatePoints(X, dR):
    """Translate the nodes of a triangulation object.
        
    The offset coordinates may be specified as individual inputs or a
    single vector of three coordinates.
    
    :Call:
        >>> TranslatePoints(X, dR)
    :Inputs:
        *X*: :class:`numpy.ndarray`(:class:`float`), *shape* = (N,3)
            List of node coordinates
        *dR*: :class:`numpy.ndarray` or :class:`list`
            List of three coordinates to use for translation
    :Outputs:
        *Y*: :class:`numpy.ndarray`(:class:`float`), *shape* = (N,3)
            List of translated node coordinates
    :Versions:
        * 2014-10-08 ``@ddalle``: Copied from previous TriBase.Translate()
    """
    # Convert points to NumPy.
    dR = np.array(dR)
    # Ensure array.
    if type(X).__name__ != 'ndarray':
        X = np.array(X)
    # Ensure list of points.
    if len(X.shape) == 1:
        X = np.array([X])
    # Initialize output.
    Y = X.copy()
    # Offset each coordinate.
    Y[:,0] += dR[0]
    Y[:,1] += dR[1]
    Y[:,2] += dR[2]
    # Output
    return Y
    
# Distance from a point to a line segment
def DistancePointToLine(x, x1, x2):
    """Get distance from a point to a line segment
    
    :Call:
        >>> d = DistancePointToLine(x, x1, x2)
    :Inputs:
        *x*: :class:`np.ndarray` shape=(3,)
            Test point
        *x1*: :class:`np.ndarray` shape=(3,)
            Segment start point
        *x2*: :class:`np.ndarray` shape=(3,)
            Segment end point
    :Outputs:
        *d*: :class:`float`
            Distance from segment to point
    :Versions:
        * 2016-09-29 ``@ddalle``: First version
    """
    # Vector of the segment
    dx = x2 - x1
    # Vector to the point from both segment ends
    d1 = x - x1
    d2 = x - x2
    # Dot products
    c1 = dx[0]*d1[0] + dx[1]*d1[1] + dx[2]*d1[2]
    c2 = dx[0]*d2[0] + dx[1]*d2[1] + dx[2]*d2[2]
    # Test the location of the point relative to the segment
    if c1 <= 0:
        # Point is upstream of the segment; return distance to *x1*
        return np.sqrt(d1[0]**2 + d1[1]**2 + d1[2]**2)
    elif c2 >= 0:
        # Point is downstream of the segment, return distance to *x2*
        return np.sqrt(d2[0]**2 + d2[1]**2 + d2[2]**2)
    else:
        # Point is within segment
        # Length of segment
        ds = np.sqrt(dx[0]**2 + dx[1]**2 + dx[2]**2)
        # Compute cross product
        A0 = dx[1]*d1[2] - dx[2]*d1[1]
        A1 = dx[2]*d1[0] - dx[0]*d1[2]
        A2 = dx[0]*d1[1] - dx[1]*d1[0]
        # Distance = (Area of parallelogram) / (Length of base)
        return np.sqrt(A0*A0+A1*A1+A2*A2) / ds
    
# Distance from a point to a group of line segments
def DistancePointToCurve(x, X):
    """Get distance from a point to each segment of a piecewise linear curve
    
    :Call:
        >>> D = DistancePointToCurve(x, X)
    :Inputs:
        *x*: :class:`np.ndarray` shape=(3,)
            Test point
        *X*: :class:`np.ndarray` shape=(n,3)
            Array of curve break points
    :Outputs:
        *D*: :class:`np.ndarray` shape=(n-1,)
            Distance from *x* to each segment
    :Versions:
        * 2016-09-29 ``@ddalle``: First version
    """
    # Vector segments
    dX = X[1:,:] - X[:-1,:]
    # Vector to the point from both segment ends
    dx = x[0] - X[:,0]
    dy = x[1] - X[:,1]
    dz = x[2] - X[:,2]
    # Dot products of end-to-end and end-to-*x* vectors
    c = dX[:,0]*dx[:-1] + dX[:,1]*dy[:-1] + dX[:,2]*dz[:-1]
    # Distance from *x* to each vertex
    di = np.sqrt(dx*dx + dy*dy + dz*dz)
    # Initialize with distance to first point
    D = np.fmin(di[:-1], di[1:])
    # Test for interior points
    I = np.logical_and(c[:-1]>0, c[1:]<0)
    # Compute cross products for those segments
    A0 = dX[I,1]*dz[I] - dX[I,2]*dy[I]
    A1 = dX[I,2]*dx[I] - dX[I,0]*dz[I]
    A2 = dX[I,0]*dy[I] - dX[I,1]*dx[I]
    # Arc lengths
    ds = np.sqrt(np.sum(dX[I,:]**2, axis=1))
    # Apply interior distances
    D[I] = np.sqrt(A0*A0 + A1*A1 + A2*A2) / ds
    # Output
    return D
    
        
        


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
    

# Check for intersection between lines
def lines_int_line(X1, Y1, X2, Y2, x1, y1, x2, y2, **kw):
    """Check if a set of line segments intersects another line segment
    
    :Call:
        >>> Q = lines_int_line(X1, Y1, X2, Y2, x1, y1, x2, y2, **kw)
    :Inputs:
        *X1*: :class:`np.ndarray` (:class:`float`, any shape)
            Matrix of *x*-coordinates of start points of several line segments
        *Y1*: :class:`np.ndarray` (:class:`float`, shape=*X1.shape*)
            Matrix of *y*-coordinates of start poitns of several line segments
        *X2*: :class:`np.ndarray` (:class:`float`, shape=*X1.shape*)
            Matrix of *x*-coordinates of end points of several line segments
        *Y2*: :class:`np.ndarray` (:class:`float`, shape=*X1.shape*)
            Matrix of *y*-coordinates of end points of several line segments
        *x1*: :class:`float`
            Start point *x*-coordinate of test segment
        *y1*: :class:`float`
            Start point *y*-coordinate of test segment
        *x2*: :class:`float`
            End point *x*-coordinate of test segment
        *y2*: :class:`float`
            End point *y*-coordinate of test segment
    :Outputs:
        *Q*: :class:`np.ndarray` (:class:`bool`, shape=*X1.shape*)
            Matrix of whether or not each segment intersects test segment
    :Versions:
        * 2017-02-06 ``@ddalle``: First version
    """
    # Length of test segment
    L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    # No intersections with null segments
    if L < 1e-8: return np.zeros_like(X1)
    # Tangent vector
    tx = (x2 - x1) / L
    ty = (y2 - y1) / L
    # Convert the list of segments into coordinates 
    XI1 = (X1-x1)*tx + (Y1-y1)*ty
    XI2 = (X2-x1)*tx + (Y2-y1)*ty
    YI1 = (Y1-y1)*tx - (X1-x1)*ty
    YI2 = (Y2-y1)*tx - (X2-x1)*ty
    # Initial test: crosses y=0 line?
    Q = (YI1*YI2 <= 0)
    # Special filter for segments parallel to test segment
    I0 = np.logical_and(YI1==0, YI2==0)
    Q[I0] = False
    # For segments that cross y==0, find coordinate
    X0 = XI1[Q] - YI1[Q]*(XI2[Q]-XI1[Q]) / (YI2[Q]-YI1[Q])
    # Test those segments
    Q[Q] = np.logical_and(X0>=0, X0<=L)
    # Come back for the segments that are parallel to the test segment
    Q[I0] = np.logical_or(
        # Check if *x1* inside [0,L]
        (XI1[I0]-L)*XI1[I0] <= 0,
        # Check if *x2* inside [0,L]
        (XI2[I0]-L)*XI2[I0] <= 0,
        # Check if [x1,x2] or [x2,x1] contains [0,L]
        XI2[I0]*XI1[I0] <=0)
    # Output
    return Q

# Check for intersection between lines
def edges_int_line(X1, Y1, X2, Y2, x1, y1, x2, y2, **kw):
    """Check if a set of edges intersects another line segment
    
    Intersections between the test segment and the start point of any edge are
    not counted as an intersection.
    
    :Call:
        >>> Q = edges_int_line(X1, Y1, X2, Y2, x1, y1, x2, y2, **kw)
    :Inputs:
        *X1*: :class:`np.ndarray` (:class:`float`, any shape)
            Matrix of *x*-coordinates of start points of several line segments
        *Y1*: :class:`np.ndarray` (:class:`float`, shape=*X1.shape*)
            Matrix of *y*-coordinates of start poitns of several line segments
        *X2*: :class:`np.ndarray` (:class:`float`, shape=*X1.shape*)
            Matrix of *x*-coordinates of end points of several line segments
        *Y2*: :class:`np.ndarray` (:class:`float`, shape=*X1.shape*)
            Matrix of *y*-coordinates of end points of several line segments
        *x1*: :class:`float`
            Start point *x*-coordinate of test segment
        *y1*: :class:`float`
            Start point *y*-coordinate of test segment
        *x2*: :class:`float`
            End point *x*-coordinate of test segment
        *y2*: :class:`float`
            End point *y*-coordinate of test segment
    :Outputs:
        *Q*: :class:`np.ndarray` (:class:`bool`, shape=*X1.shape*)
            Matrix of whether or not each segment intersects test segment
    :Versions:
        * 2017-02-06 ``@ddalle``: First version
    """
    # Length of test segment
    L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    # No intersections with null segments
    if L < 1e-8: return np.zeros_like(X1)
    # Tangent vector
    tx = (x2 - x1) / L
    ty = (y2 - y1) / L
    # Convert the list of segments into coordinates 
    XI1 = (X1-x1)*tx + (Y1-y1)*ty
    XI2 = (X2-x1)*tx + (Y2-y1)*ty
    YI1 = (Y1-y1)*tx - (X1-x1)*ty
    YI2 = (Y2-y1)*tx - (X2-x1)*ty
    # Initial test: crosses y=0 line?
    Q = (YI1*YI2 <= 0)
    # Special filter for segments parallel to test segment
    I0 = np.logical_and(YI1==0, YI2==0)
    Q[I0] = False
    # For segments that cross y==0, find coordinate
    X0 = XI1[Q] - YI1[Q]*(XI2[Q]-XI1[Q]) / (YI2[Q]-YI1[Q])
    # Test those segments
    Q[Q] = np.logical_and(X0>=0, X0<=L)
    # Filter *OUT* intersections with (XI1, YI1)
    Q[YI1==0] = False
    # Come back for the segments that are parallel to the test segment
    Q[I0] = np.logical_or(
        # Check if *x1* strictly inside (0,L)
        (XI1[I0]-L)*XI1[I0] < 0,
        # Check if *x2* inside [0,L]
        (XI2[I0]-L)*XI2[I0] <= 0,
        # Check if [x1,x2] or [x2,x1] contains [0,L]
        XI2[I0]*XI1[I0] <=0)
    # Output
    return Q
    
# Check if a triangle contains a point
def tris_have_pt(X, Y, x, y, **kw):
    """Check if each triangle in a list contains a specified point
    
    :Call:
        >>> Q = tris_have_pt(X, Y, x, y, **kw)
    :Inputs:
        *X*: :class:`np.ndarray` (:class:`float`, shape=(n,3))
            *x*-coordinates of vertices of *n* triangles
        *Y*: :class:`np.ndarray` (:class:`float`, shape=(n,3))
            *y*-coordinates of vertices of *n* triangles
        *x*: :class:`float`
            *x*-coordinate of test point
        *y*: :class:`float`
            *y*-coordinate of test point
    :Outputs:
        *Q*: :class:`np.ndarray` (:class:`bool`, shape=(n,))
            Whether or not each triangle contains the test point
    :Versions:
        * 2017-02-06 ``@ddalle``: First version
    """
    # Get types
    tX = type(X).__name__
    tY = type(Y).__name__
    # Check input type
    if tX == "list":
        # Convert to array
        X = np.array(X)
    elif tX != "ndarray":
        # Convert to array
        raise TypeError("Triangles must be arrays")
    if tY == "list":
        # Convert to array
        Y = np.array(Y)
    elif tY != "ndarray":
        # Convert to array
        raise TypeError("Triangles must be arrays")
    # Test for single triangle
    if X.ndim == 1:
        X = np.array([X])
    if Y.ndim == 1:
        Y = np.array([Y])
    # Check dimensions
    if X.shape[1] != 3:
        raise IndexError("Triangle arrays must have three columns")
    # Construct test point to the left of all triangles
    x0 = np.min(X) - 1.0
    y0 = y
    # Construct test point below all the triangles
    x1 = x
    y1 = np.min(Y) - 1.0
    # Construct test point above all the triangles
    x2 = x
    y2 = np.max(Y) + 1.0
    # Construct list of segments that consists of each triangle edge
    # Use the inputs as the start points; rotate vertices to get end points
    X2 = np.transpose([X[:,1], X[:,2], X[:,0]])
    Y2 = np.transpose([Y[:,1], Y[:,2], Y[:,0]])
    # Draw three lines starting with (x,y)
    #   Line 0: (x,y) to a point directly left of and outside tris
    #   Line 1: (x,y) to a point directly below and outside tris
    #   Line 2: (x,y) to a point directly above and outside tris
    Q0 = edges_int_line(X, Y, X2, Y2, x, y, x0, y0)
    Q1 = edges_int_line(X, Y, X2, Y2, x, y, x1, y1)
    Q2 = edges_int_line(X, Y, X2, Y2, x, y, x2, y2)
    # Count up intersections; each line must intersect odd # of edges
    n0 = np.sum(Q0, axis=1)
    n1 = np.sum(Q1, axis=1)
    n2 = np.sum(Q2, axis=1)
    # Check for odd number of intersections with all three test lines
    Q = np.logical_and(n0%2==1, np.logical_and(n1%2==1, n2%2==1))
    # Output
    return Q
    
    
    


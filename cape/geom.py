"""
Generic CAPE Geometry Module: :mod:`cape.geom`
==============================================

:Versions:
    * 2015-09-30 ``@ddalle``: First version
"""

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

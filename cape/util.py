"""
Cape utilities: :mod:`cape.util`
================================

This module provides several utilities used throughout the Cape system,
including :func:`SigmaMean` to compute statistical sampling error for iterative
histories and :func:`readline` to process special space-or-comma-separated lines
for run matrix files.
"""

# Numerics
import numpy as np
# File system
import subprocess as sp
# Import path utilities
import os.path, sys
# Text
import re



# cape base folder
capeFolder = os.path.split(os.path.abspath(__file__))[0]
rootFolder = os.path.split(capeFolder)[0]
# Folder containing TecPlot templates
TecFolder = os.path.join(rootFolder, "templates", "tecplot")
# Folder containing Paraview templates
ParaviewFolder = os.path.join(rootFolder, "templates", "paraview")

# Stack vectors
def stackcol(cols):
    """Create a matrix out of vectors that are assumed to be columns
    
    :Call:
        >>> A = stackcols(cols)
    :Inputs:
        *cols*: :class:`list` | :class:`tuple`
            List of vectors
        *cols[0]*: :class:`list` | :class:`np.ndarray`
            First column vector
    :Outputs:
        *A*: :class:`np.ndarray`
            Matrix with ``A[:,0]==cols[0]``, ``A[:,1]==cols[1]``, etc.
    :Versions:
        * 2017-02-17 ``@ddalle``: First version
    """
    # We have to do this stupid VSTACK thing for old versions of NUMPY
    # First, create tuple of 1xN row vector matrices
    V = ([c] for c in cols)
    # Stack as row vectors and then transpose
    return np.transpose(np.vstack(V))
            

# Split text by either comma or space
def SplitLineGeneral(line):
    """Split a string in which uses a mix of commas and spaces as delimiters
    
    :Call:
        >>> V = SplitLineGeneral(line)
    :Inputs:
        *line*: :class:`str`
            Text with commas, spaces, or a combination as delimiters
    :Outputs:
        *V*: :class:`list` (:class:`str`)
            List of values split by delimiters
    :Versions:
        * 2016-12-29 ``@ddalle``: First version
    """
    # Split using regular expressions (after stripping white space)
    V = re.split("[\s\,]+", line.strip())
    # Check for empty
    if (len(V) == 1) and (V[0] == ""):
        # Return an empty state instead
        return []
    else:
        # Return the list
        return V

# Convert a list of numbers to a compact string
def RangeString(rng):
    """Convert a list of ascending integers to a string like "1-10,12,14-15"
    
    :Call:
        >>> txt = RangeString(rng)
    :Inputs:
        *rng*: :class:`list` (:class:`int`)
            Range of integers
    :Outputs:
        *txt*: :class:`str`
            Nicely formatted string combining contiguous ranges with ``"-"``
    :Versions:
        * 2016-10-20 ``@ddalle``: First version
    """
    # Number of components
    n = len(rng)
    # Check for single component or no components
    if n == 0:
        return ""
    if n == 1:
        return ("%s" % rng[0])
    # Initialize the string and indices
    txt = []
    ibeg = rng[0]
    iend = rng[0]
    # Loop through the grid numbers, which are ascending and unique.
    for i in range(1,n):
        # Get the compID
        icur = rng[i]
        # Check if this is one greater than the previous one
        if icur == iend + 1:
            # Add to the current list
            iend += 1
        # Write if appropriate
        if i == n-1 or icur > iend+1:
            # Check if single element or list
            if ibeg == iend:
                # Write single
                txt.append("%s" % ibeg)
            else:
                # Write list
                txt.append("%s-%s" % (ibeg, iend))
            # Check if last entry is single
            if i == n-1 and icur > iend+1:
                txt.append("%s" % icur)
            # Reset.
            ibeg = icur
            iend = icur
    # Output
    return ",".join(txt)
    
# Eliminate unused nodes
def TrimUnused(T):
    """Remove any node numbers that are not used 
    
    For example:
    
        .. code-block:: none
        
            [[1, 4, 5], [4, 8, 90]] --> [[1, 2, 3], [2, 6, 7]]
    
    :Call:
        >>> U = cape.util.TrimUnused(T)
    :Inputs:
        *T*: :class:`np.ndarray` (:class:`int`)
            Nodal index matrix or similar
    :Outputs:
        *U*: :class:`np.ndarray` (:class:`int`)
            Nodal matrix with nodes 1 to *n* with same dimensions as *T*
    :Versions:
        * 2017-02-10 ``@ddalle``: First version
        * 2017-03-30 ``@ddalle``: From :func:`cape.tri.Tri.TrimUnusedNodes`
    """
    # Get nodes that are used
    N = np.unique(T)
    # New number of nodes
    nNode = len(N)
    nMax = np.max(N)
    # Renumbered nodes
    I = np.arange(1, nNode+1)
    M = np.zeros(nMax+1, dtype="int")
    M[N] = I
    # Create flat arrays
    V = T.flatten()
    # Initialize output
    U = np.zeros_like(V)
    # Loop through the nodes that are used
    for j in np.arange(V.size):
        # Make replacement
        U[j] = M[V[j]]
    # Output
    return np.reshape(U, T.shape)

# Convert matrix of truth values to BC lists
def GetBCBlock2(I):
    """Get largest rectangle of boundary conditions
    
    :Call:
        >>> js, je, ks, ke = GetBCBlock(I)
    :Inputs:
        *I*: :class:`np.ndarray` (:class:`bool`, shape=(NJ,NK))
            Matrix of whether or not each grid point is in the family
    :Outputs:
        *js*: {``None``} | :class:`int`
            Start *j*-index of block
        *je*: {``None``} | :class:`int`
            End *j*-index of block
        *ks*: {``None``} | :class:`int`
            Start *k*-index of block
        *ke*: {``None``} | :class:`int`
            End *k*-index of block
    :Versions:
        * 2017-02-08 ``@ddalle``: First version
    """
    # Initialize indices
    js = None
    je = None
    ks = None
    ke = None
    # Check for NO MATCHES
    if not np.any(I):
        return js, je, ks, ke
    # Get dimensions
    nj, nk = I.shape
    # Get first column with finds
    ks = np.where(np.any(I, axis=0))[0][0]
    # Loop through columns
    for k in range(ks,nk):
        # Get indices of matches
        J = np.where(I[:,k])[0]
        # Check for matches in this column
        if len(J) == 0:
            ke = k - 1
            break
        # Process start of matches
        if k == ks:
            # First column; found start index for the block
            js = J[0]
        elif js != J[0]:
            # Mismatch for row start index: END OF BLOCK
            ke = k - 1
            break
        # Find gaps
        IE = np.where(np.diff(J) > 1)[0]
        # Check for gaps
        if len(IE) == 0:
            # No gaps
            jek = J[-1]
        else:
            # Get the end of the first continuous block
            jek = J[IE[0]]
        # Process end index
        if k == ks:
            # First column; found index for the block
            je = jek
        if jek != je:
            # Mismatch for end row index: END OF BLOCK
            ke = k - 1
            break
        # If reaching here; update value of *ke*
        ke = k
    # Return
    return js, je+1, ks, ke+1

# Function to get uncertainty in the mean
def SigmaMean(x):
    """Calculate standard deviation of mean of an array of values
    
    Specifically, this returns the standard deviation of an array generated in
    the following way.  If you created 100 sets with the same statistical
    properties as *x* and created an array *X* which contained the means of each
    of those 100 sets, the purpose of this function is to estimate what the
    standard deviation of *X* would be.
    
    :Call:
        >>> sig = cape.util.SigmaMean(x)
    :Inputs:
        *x*: :class:`numpy.ndarray` or :class:`list`
            Array of points
    :Outputs:
        *sig*: :class:`float`
            Estimated standard deviation of the mean
    :Versions:
        * 2015-02-21 ``@ddalle``: First version
    """
    # Length of list
    n = len(x)
    # Best length to break list into
    ni = int(np.sqrt(n))
    # Number of sublists
    mi = n / ni
    # Split into chunks
    X = np.array([np.mean(x[i*ni:(i+1)*ni]) for i in range(mi)])
    # Standard deviation of the sub-means
    si = np.std(X)
    # Output
    return si * np.sqrt(float(ni)/float(n))
    
# Function to get a non comment line
def readline(f, comment='#'):
    """Read line that is nonempty and not a comment
    
    :Call:
        >>> line = readline(f, comment='#')
    :Inputs:
        *f*: :class:`file`
            File instance
        *comment*: :class:`str`
            Character(s) that begins a comment
    :Outputs:
        *line*: :class:`str`
            Nontrivial line or `''` if at end of file
    :Versions:
        * 2015-11-19 ``@ddalle``: First version
    """
    # Read a line.
    line = f.readline()
    # Check for empty line (EOF)
    if line == '': return line
    # Process stripped line
    lstrp = line.strip()
    # Check if otherwise empty or a comment
    while (lstrp=='') or lstrp.startswith(comment):
        # Read the next line.
        line = f.readline()
        # Check for empty line (EOF)
        if line == '': return line
        # Process stripped line
        lstrp = line.strip()
    # Return the line.
    return line
    
# Function to get Tecplot command
def GetTecplotCommand():
    """Return the Tecplot 360 command on the current system
    
    The preference is 'tec360EX', 'tec360', 'tecplot'.  An exception is raised
    if none of these commands can be found.
    
    :Call:
        >>> cmd = cape.util.GetTecplotCommand()
    :Outputs:
        *cmd*: :class:`str`
            Name of the command to the current 'tec360' command
    :Versions:
        * 2015-03-02 ``@ddalle``: First version
    """
    # Shut up about output.
    f = open('/dev/null', 'w')
    # Loop through list of possible commands
    for cmd in ['tec360EX', 'tec360', 'tecplot']:
        # Use `which` to see where the command might be.
        ierr = sp.call(['which', cmd], stdout=f, stderr=f)
        # Check.
        if ierr: continue
        # If this point is reached, we found the command.
        return cmd
    # If this point is reached, no command was found.
    raise SystemError('No Tecplot360 command found')

# Function to fix "NoneType is not iterable" nonsense
def denone(x):
    """Replace ``None`` with ``[]`` to avoid iterative problems
    
    :Call:
        >>> y = cape.util.denone(x)
    :Inputs:
        *x*: any
            Any variable
    :Outputs:
        *y*: any
            Same as *x* unless *x* is ``None``, then ``[]``
    :Versions:
        * 2015-03-09 ``@ddalle``: First version
    """
    if x is None:
        return []
    else:
        return x
        
# Check if an object is a list.
def islist(x):
    """Check if an object is a list or not
    
    :Call:
        >>> q = cape.util.islist(x)
    :Inputs:
        *x*: any
            Any variable
    :Outputs:
        *q*: :class:`bool`
            Whether or not *x* is in [:class:`list` or :class:`numpy.ndarray`]
    :Versions:
        * 2015-06-01 ``@ddalle``: First version
    """
    return type(x).__name__ in ['list', 'ndarray']

# Function to automatically get inclusive data limits.
def get_ylim(ha, ypad=0.05, **kw):
    """Calculate appropriate *y*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.
    
    :Call:
        >>> ymin, ymax = get_ylim(ha, ypad=0.05, ym=None, yp=None)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *ypad*: {``0.05``} | :class:`float`
            Extra padding to min and max values to plot
        *ym*: :class:`float`
            Padding on minimum side
        *yp*: :class:`float`
            Padding on maximum side
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
        * 2016-06-10 ``@ddalle``: Moved to :mod:`cape.util`
    """
    # Initialize limits.
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check for empty
            if len(h.get_xdata()) == 0: continue
            # Check the min and max data
            ymin = min(ymin, min(h.get_ydata()))
            ymax = max(ymax, max(h.get_ydata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Get the coordinates.
            ymin = min(ymin, min(P.vertices[:,1]))
            ymax = max(ymax, max(P.vertices[:,1]))
    # Process margins
    ym = kw.get('ym', ypad)
    yp = kw.get('yp', ypad)
    # Check for identical values
    if ymax - ymin <= 0.05*(ym+yp):
        # Expand by manual amount.
        ymax += yp*ymax
        ymin -= ym*ymin
    # Add padding.
    yminv = (1+ym)*ymin - ym*ymax
    ymaxv = (1+yp)*ymax - yp*ymin
    # Output
    return yminv, ymaxv
    
# Function to automatically get inclusive data limits.
def get_xlim(ha, xpad=0.05, **kw):
    """Calculate appropriate *x*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.
    
    :Call:
        >>> xmin, xmax = get_xlim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *xpad*: :class:`float`
            Extra padding to min and max values to plot.
        *xm*: :class:`float`
            Padding on minimum side
        *xp*: :class:`float`
            Padding on maximum side
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check for empty
            if len(h.get_xdata()) == 0: continue
            # Check the min and max data
            xmin = min(xmin, min(h.get_xdata()))
            xmax = max(xmax, max(h.get_xdata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Get the coordinates.
            xmin = min(xmin, min(P.vertices[:,0]))
            xmax = max(xmax, max(P.vertices[:,0]))
    # Process margins
    xm = kw.get('xm', xpad)
    xp = kw.get('xp', xpad)
    # Check for identical values
    if xmax - xmin <= 0.05*(xm+xp):
        # Expand by manual amount.
        xmax += xp*xmax
        xmin -= xm*xmin
    # Add padding.
    xminv = (1+xm)*xmin - xm*xmax
    xmaxv = (1+xp)*xmax - xp*xmin
    # Output
    return xminv, xmaxv
    
# Function to automatically get inclusive data limits.
def get_ylim_ax(ha, ypad=0.05, **kw):
    """Calculate appropriate *y*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.
    
    This version is specialized for equal-aspect ratio axes.
    
    :Call:
        >>> ymin, ymax = get_ylim_ax(ha, ypad=0.05, ym=None, yp=None)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *ypad*: {``0.05``} | :class:`float`
            Extra padding to min and max values to plot
        *ym*: :class:`float`
            Padding on minimum side
        *yp*: :class:`float`
            Padding on maximum side
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
        * 2016-06-10 ``@ddalle``: Moved to :mod:`cape.util`
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check for empty
            if len(h.get_xdata()) == 0: continue
            # Check the min and max data
            xmin = min(xmin, min(h.get_xdata()))
            xmax = max(xmax, max(h.get_xdata()))
            ymin = min(ymin, min(h.get_ydata()))
            ymax = max(ymax, max(h.get_ydata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Get the coordinates.
            xmin = min(xmin, min(P.vertices[:,0]))
            xmax = max(xmax, max(P.vertices[:,0]))
            ymin = min(ymin, min(P.vertices[:,1]))
            ymax = max(ymax, max(P.vertices[:,1]))
    # Process margins
    ym = kw.get('ym', ypad)
    yp = kw.get('yp', ypad)
    # Check for identical values
    if ymax - ymin <= 0.05*(ym+yp):
        # Expand by manual amount.
        ymax += yp*ymax
        ymin -= ym*ymin
    # Add padding.
    yminv = ymin - ym*max(xmax-xmin, ymax-ymin)
    ymaxv = ymax + yp*max(xmax-xmin, ymax-ymin)
    # Output
    return yminv, ymaxv
    
# Function to automatically get inclusive data limits.
def get_xlim_ax(ha, xpad=0.05, **kw):
    """Calculate appropriate *x*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.
    
    This version is specialized for equal-aspect ratio axes.
    
    :Call:
        >>> xmin, xmax = get_xlim_ax(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *xpad*: :class:`float`
            Extra padding to min and max values to plot.
        *xm*: :class:`float`
            Padding on minimum side
        *xp*: :class:`float`
            Padding on maximum side
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check for empty
            if len(h.get_xdata()) == 0: continue
            # Check the min and max data
            xmin = min(xmin, min(h.get_xdata()))
            xmax = max(xmax, max(h.get_xdata()))
            ymin = min(ymin, min(h.get_ydata()))
            ymax = max(ymax, max(h.get_ydata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Get the coordinates.
            xmin = min(xmin, min(P.vertices[:,0]))
            xmax = max(xmax, max(P.vertices[:,0]))
            ymin = min(ymin, min(P.vertices[:,1]))
            ymax = max(ymax, max(P.vertices[:,1]))
    # Process margins
    xm = kw.get('xm', xpad)
    xp = kw.get('xp', xpad)
    # Check for identical values
    if xmax - xmin <= 0.05*(xm+xp):
        # Expand by manual amount.
        xmax += xp*xmax
        xmin -= xm*xmin
    # Add padding.
    xminv = xmin - xm*max(xmax-xmin, ymax-ymin)
    xmaxv = xmax + xp*max(xmax-xmin, ymax-ymin)
    # Output
    return xminv, xmaxv


            
    

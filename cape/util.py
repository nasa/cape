"""
:mod:`cape.util`: Cape utilities
=================================

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

# Would like to use scipy, but let's not have a strict dependency
try:
    import scipy.signal
except ImportError:
    pass


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
    V = tuple([c] for c in cols)
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
        * 2017-03-30 ``@ddalle``: From :func:`cape.tri.Tri.RemoveUnusedNodes`
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
        *x*: :class:`numpy.ndarray` | :class:`list`
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
    
# Use Welch's method (or a crude estimate) to get a primary frequency
def GetBestFrequency(y, fs=1.0, **kw):
    """Get best frequency using :func:`scipy.signal.welch` if available
    
    If SciPy is not available, use a crude count of how many times the signal
    crosses the mean value (with a window to avoid overcounting small
    oscillations right around the mean value).  The dimensions of this output
    are such that the signal matches sinusoids such as :math:`\sin(\omega x)`.
    To meet this format, the output is 2 times the peak frequency from
    :func:`scipy.signal.welch`.
    
    :Call:
        >>> w = GetBestFrequency(y, fs=1.0)
    :Inputs:
        *y*: :class:`np.ndarray` shape=(*n*,)
            Input signal to process
        *fs*: {``1.0``} | :class:`float`
            Sampling frequency of *y*; usually 1 as in 1 per iteration
    :Outputs:
        *w*: :class:`float`
            Dominant frequency
    :Versions:
        * 2017-09-29 ``@ddalle``: First version
    """
    # Length of signal
    n = len(y)
    # Number of points per segment
    npwerseg = kw.get("nperseg", min(n,256))
    # Attempt to use Welch's method from SciPy method
    try:
        # Estimate power spectral density
        f, a = scipy.signal.welch(y, fs=fs, nperseg=nperseg)
        # Return the peak frequency (disallow w==0)
        return 2*f[1+np.argmax(a[1:])]
    except Exception:
        pass
    # Calculate mean
    m = np.mean(y)
    # Calculate amplitude
    sig = np.std(y)
    # Initialize array for zero crossings
    I = np.zeros(n)
    # Set indices for above-mean and below-mean samples
    I[y > m+0.15*sig] = 1
    I[y < m-0.15*sig] = -1
    # Eliminate zeros from the array (samples close to the mean)
    J = I[I!=0]
    # Count crossings (j_i * j_{i+1} == -1)
    k = np.count_nonzero(J[1:]*J[:-1] == -1)
    # Convert to a frequency
    return float(k)*np.pi/n
    
# Function to fit a line plus a sinusoid
def FitLinearSinusoid(x, y, w):
    """Find the best fit of a line plus a sinusoid with a specified frequency
    
    The function returns the best fit for
    
        .. math::
        
                y = a_0 + a_1x + a_2\\cos(\\omega x) + a_3\\sin(\\omega x)
                
    :Call:
        >>> a = FitLinearSinusoid(x, y, w)
        >>> a0, a1, a2, a3 = FitLinearSinusoid(x, y, w)
    :Inputs:
        *x*: :class:`np.ndarray`
            Array of independent variable samples (e.g. iteration number)
        *y*: :class:`np.ndarray`
            Signal to be fit
        *w*: :class:`float`
            Specified frequency of the sinusoid
    :Outputs:
        *a*: :class:`np.ndarray` (:class:`float`)
            Array of *a0*, *a1*, *a2*, *a3*
        *a0*: :class:`float`
            Constant offset
        *a1*: :class:`float`
            Linear slope
        *a2*: :class:`float`
            Magnitude of cosine signal
        *a3*: :class:`float`
            Magnitude of sine signal
    :Versions:
        * 2017-09-29 ``@ddalle``: First version
    """
    # Length of signal
    n = len(x)
    # Check consistency
    if len(y) != n:
        raise ValueError(
            ("Input signal has length %i, but " % len(y)) +
            ("time sample array has length %i" % n))
    # Calculate trig functions
    cx = np.cos(w*x)
    sx = np.sin(w*x)
    # Calculate relevant sums
    x1  = np.sum(x)
    x2  = np.sum(x*x)
    c1  = np.sum(cx)
    c2  = np.sum(cx*cx)
    xc1 = np.sum(x*cx)
    s1  = np.sum(sx)
    s2  = np.sum(sx*sx)
    xs1 = np.sum(x*sx)
    cs1 = np.sum(cx*sx)
    # Right-hand side sums
    y1  = np.sum(y)
    yx1 = np.sum(y*x)
    yc1 = np.sum(y*cx)
    ys1 = np.sum(y*sx)
    # Create matrices
    A = np.array([
        [n,  x1,  c1,  s1],
        [x1, x2,  xc1, xs1],
        [c1, xc1, c2,  cs1],
        [s1, xs1, cs1, s2]
    ])
    Y = np.array([y1, yx1, yc1, ys1])
    # Solve linear system to get best fit
    try:
        a = np.linalg.solve(A, Y)
    except Exception:
        # This can easily happen for flat signals
        A = np.array([[n,x1],[x1,x2]])
        Y = np.array([y1, yx1])
        # Simpler linear system (ignore sinusoid)
        try:
            a = np.hstack((np.linalg.solve(A,Y), [0.,0.0]))
        except Exception:
            # That shouldn't happen, but just use the mean if needed
            a = np.array([y1/n, 0.0, 0.0, 0.0])
    # Output
    return a
    
# Function to select the best best line+sine fit
def SearchSinusoidFitRange(x, y, nAvg, nMax=None, dn=None, nMin=0, **kw):
    """Find the best window size to minimize the slope of a linear/sine fit
    
    :Call:
        >>> F = SearchSinusoidFitRange(x, y, nAvg, nMax, dn=None, **kw)
    :Inputs:
        *x*: :class:`np.ndarray`
            Array of independent variable samples (e.g. iteration number)
        *y*: :class:`np.ndarray`
            Signal to be fit
        *nAvg*: :class:`int`
            Minimum candidate window size
        *nMax*: {*nAvg*} | :class:`int`
            Maximum candidate window size
        *dn*: {*nAvg*} | :class:`int`
            Candidate interval size
        *nMin*: {``0``} | :class:`int`
            First iteration allowed in the window
    :Outputs:
        *F*: :class:`dict`
            Dictionary of fit coefficients and statistics
        *F['n']*, *n*: :class:`int`
            Number of iterations in selected window
        *F['mu']*: :class:`float`
            Mean value over the window of size *n*
        *F['w']*, *w*: :class:`float`
            Estimated dominant frequency over the window
        *F['a']*, *a*, ``[a0, a1, a2, a3]``: :class:`np.ndarray`
            List of line+sinusoid fit coefficients
        *a0*: :class:`float`
            Constant offset of best line+sinusoid fit
        *a1*: :class:`float`
            Linear slope of best line+sinusoid fit
        *a2*: :class:`float`
            Amplitude of cosine contribution to line+sinusoid fit
        *a3*: :class:`float`
            Amplitude of sine contribution to line+sinusoid fit
        *F['sig']*: :class:`float`
            Raw standard deviation over window of size *n*
        *F['eps']*, *eps*: :class:`float`
            Sampling error; see :func:`SigmaMean`
        *F['dy']*, *dy*: :class:`float`
            Drift over the window, equal to ``a1*n``
        *F['u']*: :class:`float`
            Uncertainty estimate based on *dy* and ``3*eps``
        *F['np']*: :class:`float`
            Number of dominant-frequency periods in window
    :Versions:
        * 2017-09-29 ``@ddalle``: First version
    """
    # Process defaults
    if nMax is None: nMax = nAvg
    if dn   is None: dn = nAvg
    # Number of available iterations after *nMin*
    nAvail = np.count_nonzero(x>nMin)
    # Total number of iterations
    nx = len(x)
    # Number of available iterations
    nMax = min(nMax, nAvail)
    # Check for insufficient iterations for a single window
    if nAvg > nx:
        # Use all the iterations b/c there are less than *nAvg* after *nMin*
        nAvg = nx
        nMax = nx
    elif nAvg > nMax:
        # Use *nAvg* iterations, which reach before *nMin*
        nMax = nAvg
    # Create array of minimum window sizes
    N = nAvg + np.arange(max(1,np.ceil(float(nMax-nAvg)/dn)))*dn
    # Append *nMax* if not in *N*
    if np.max(N) < nMax: N = np.append(N, nMax)
    # Create one window if no range
    if len(N) == 1: N = np.append(N, nAvg)
    # Ensure integer
    N = np.array(N, dtype="int")
    # Number of candidate windows
    nw = len(N) - 1
    # Initialize candidates
    F = {}
    n = np.zeros(nw)
    u = np.zeros(nw)
    # Loop through windows
    for i in range(nw):
        # Get statistics
        F[i] = SearchSinusoidFit(x, y, N[i], N[i+1], **kw)
        # Save error
        u[i] = F[i]["u"]
    # Find best error
    i = np.argmin(u)
    # Output
    return F[i]
    
    
    
    
# Function to calculate best linear/sinusoidal fit within a range of windows
def SearchSinusoidFit(x, y, N1, N2, **kw):
    """Find the best window size to minimize the slope of a linear/sine fit
    
    :Call:
        >>> F = SearchSinusoidFit(x, y, N1, N2, **kw)
    :Inputs:
        *x*: :class:`np.ndarray`
            Array of independent variable samples (e.g. iteration number)
        *y*: :class:`np.ndarray`
            Signal to be fit
        *N1*: :class:`int`
            Minimum candidate window size
        *N2*: :class:`int`
            Maximum candidate window size
    :Outputs:
        *F*: :class:`dict`
            Dictionary of fit coefficients and statistics
        *F['n']*, *n*: :class:`int`
            Number of iterations in selected window
        *F['mu']*: :class:`float`
            Mean value over the window of size *n*
        *F['w']*, *w*: :class:`float`
            Estimated dominant frequency over the window
        *F['a']*, *a*, ``[a0, a1, a2, a3]``: :class:`np.ndarray`
            List of line+sinusoid fit coefficients
        *a0*: :class:`float`
            Constant offset of best line+sinusoid fit
        *a1*: :class:`float`
            Linear slope of best line+sinusoid fit
        *a2*: :class:`float`
            Amplitude of cosine contribution to line+sinusoid fit
        *a3*: :class:`float`
            Amplitude of sine contribution to line+sinusoid fit
        *F['sig']*: :class:`float`
            Raw standard deviation over window of size *n*
        *F['eps']*, *eps*: :class:`float`
            Sampling error; see :func:`SigmaMean`
        *F['dy']*, *dy*: :class:`float`
            Drift over the window, equal to ``a1*n``
        *F['u']*: :class:`float`
            Uncertainty estimate based on *dy* and ``3*eps``
        *F['np']*: :class:`float`
            Number of dominant-frequency periods in window
    :Versions:
        * 2017-09-29 ``@ddalle``: First version
    """
    # Switch inputs if necessary
    if N2 < N1:
        N1, N2 = N2, N1
    # Check for degenerate ranges
    if N2 < 5:
        # Just say it's a constant
        v = np.mean(y[-N2:])
        a = np.array([v, 0, 0, 0])
        # No statistics
        eps = 0.0
        sig = 0.0
        w = 0.0
        # Use the whole range as the drift
        ymin = np.min(y[-N2:])
        ymax = np.max(y[-N2:])
        dy = ymax - ymin
        u  = dy
        # Trivial output
        return {
            "n": N2,
            "a": a,
            "w": w,
            "u": u,
            "dy": dy,
            "mu": v,
            "np": 0.0,
            "eps": eps,
            "sig": sig,
            "min": ymin,
            "max": ymax,
        }
    # Use the maximum window size to get the best frequency
    w = GetBestFrequency(y[-N2:], **kw)
    # Calculate the half period based on this frequency
    if w == 0.0:
        # Use the whole interval (happens with perfectly flat signals)
        p = N2
    else:
        # Use an actual period
        p = np.pi/w
    # Get the largest window that's a whole or half multiple of period
    n = max(N1, int(N2/p) * int(p))
    # Get sample sizes
    xi = x[-n:]
    yi = y[-n:]
    # Calculate the line+sine fit
    a = FitLinearSinusoid(xi, yi, w)
    # Calculate mean value
    v = np.mean(yi)
    # Standard deviation
    sig = np.std(yi)
    # Sampling error
    eps = SigmaMean(yi)
    # Drift
    dy = n*a[1]
    # Overall uncertainty
    u = np.sqrt(9*eps*eps + dy*dy)
    # Output
    return {
        "n": n,
        "a": a,
        "w": w,
        "u": u,
        "np":  0.5*n/int(p),
        "mu":  v,
        "eps": eps,
        "sig": sig,
        "dy":  dy,
        "min": np.min(yi),
        "max": np.max(yi),
    }
    
    
    
    
# Function to calculate window with lowest linear fit
def BisectLinearFit(I, x, N1, N2, **kw):
    """Calculate window size that results in 
    
    :Call:
        >>> N, dx = BisectLinearFit(I, x, N1, N2, **kw)
    :Inputs:
        *I*: :class:`np.ndarray` (:class:`int` | :class:`float`)
            Iteration indices (in case of non-uniform spacing)
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of test values
        *N1*: :class:`int`
            Minimum candidate window size
        *N2*: :class:`int`
            Maximum candidate window size
    :Outputs:
        *N*: :class:`int`
            Window size with flattest linear fit
        *dx*: :class:`float`
            Absolute value of change in *x* over window *N*
    :Versions:
        * 20178-09-28 ``@ddalle``: First version
    """
    # Check inputs
    N1 = int(N1)
    N2 = int(N2)
    # Check if we need to switch
    if N1 > N2:
        N1, N2 = N2, N1
    # Add the bisection point to the mix
    N = int((N1+N2)/2)
    # Calculate linear fits
    a1, a0 = np.polyfit(I[-N1:], x[-N1:], 1)
    a2, a0 = np.polyfit(I[-N2:], x[-N2:], 1)
    a,  a0 = np.polyfit(I[-N:],  x[-N:],  1)
    print("k=%2i, a1=%6.4f, a2=%6.4f, a=%6.4f" % (0,a1,a2,a))
    print("       N1=%-5i,  N2=%-5i,  N=%-5i" % (N1, N2, N))
    # Check signs
    if a*a2 <= 0:
        # Use upper half
        a1, N1 = a, N
    elif a*a1 <= 0:
        # Use lower half
        a2, N2 = a, N
    elif abs(a1*N1) < abs(a2*N2):
        # Check for medium being better
        if abs(a1*N1) < abs(a*N):
            # Use the minimum window size
            return N1, a1*(I[-1]-I[-N1])
        else:
            # Use the middle size
            return N, a*(I[-1]-I[-N])
    else:
        # Check for medimum being better
        if abs(a*N) < abs(a2*N2):
            # Use the middle size
            return N, a*(I[-1]-I[-N])
        else:
            # Use the maximum window size
            return N2, a2*(I[-1]-I[-N2])
    # Initialize iteration count
    k = 0
    kmax = kw.get("kmax", 6)
    # Perform bisection/secant method
    while (k < kmax) and (N2-N1 > 5):
        # Iteration count
        k += 1
        # Calculate the intermediate value
        N = int(N1 - a1/(a2-a1)*(N2-N1))
        # Calculate new fit
        a, a0 = np.polyfit(I[-N:], x[-N:], 1)
        print("k=%2i, a1=%6.4f, a2=%6.4f, a=%6.4f" % (k,a1,a2,a))
        print("       N1=%-5i,  N2=%-5i,  N=%-5i" % (N1, N2, N))
        # Check side
        if a*a1 <= 0:
            # Update upper bound (use lower half)
            a2, N2 = a, N
        else:
            # Update lower bound (use upper half)
            a1, N1 = a, N
    # Output
    return N, a*(I[-1] - I[-N])
        
        
    
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


            
    

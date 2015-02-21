"""
Utilities for pyCart: :mod:`pyCart.util`

"""

# Numerics
import numpy as np

# Function to get uncertainty in the mean
def SigmaMean(x):
    """Calculate standard deviation of mean of an array of values
    
    Specifically, this returns the standard deviation of an array generated in
    the following way.  If you created 100 sets with the same statistical
    properties as *x* and created an array *X* which contained the means of each
    of those 100 sets, the purpose of this function is to estimate what the
    standard deviation of *X* would be.
    
    :Call:
        >>> sig = pyCart.util.SigmaMean(x)
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
    



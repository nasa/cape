"""
Utilities for pyCart: :mod:`pyCart.util`

"""

# Numerics
import numpy as np
# File system
import subprocess as sp
# Import path utilities
import os.path



# pyCart base folder
pyCartFolder = os.path.split(os.path.abspath(__file__))[0]
# Folder containing TecPlot templates
TecFolder = os.path.split(pyCartFolder)[0]
TecFolder = os.path.join(TecFolder, "templates", "tecplot")


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
    
# Function to get Tecplot command
def GetTecplotCommand():
    """Return the Tecplot 360 command on the current system
    
    The preference is 'tec360EX', 'tec360', 'tecplot'.  An exception is raised
    if none of these commands can be found.
    
    :Call:
        >>> cmd = pyCart.util.GetTecplotCommand()
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
        >>> y = pycart.util.denone(x)
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

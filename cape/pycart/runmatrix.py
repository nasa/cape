"""
:mod:`pyCart.runmatrix`: Cart3D Run Matrix Module 
==================================================

This module handles classes and methods that are specific to the list of run
cases (i.e., the "trajectory").  It is defined in the ``"RunMatrix"`` section
of the master JSON file (e.g. :file:`pyCart.json`), and is usually read from a
modified comma-separated or space-separated text file.

The primary Cart3D state variables are Mach number, angle of attack, and
sidelsip.  To this it is common practice to add a ``"config"`` variable (which
sets the name of the group folder) and a ``"Label"`` which can be used to give
an extension to the name of a case.  A run matrix using only these variables
could be defined as follows.

    .. code-block:: javascript
    
        "RunMatrix": {
            "Keys": ["mach", "alpha", "beta"],
            "File": "inputs/matrix.csv"
        }
        
Then the matrix file :file:`inputs/matrix.csv` could have contents such as the
following.

    .. code-block:: none
    
        # mach, alpha, beta
        0.8, 0.0, 0.0, ascent,
        1.2, 2.0, 0.0, ascent, maxq

Then the following example API commands could be used to show the case names.
In this case *cart3d* is the interface to a Cart3D run configuration read from
a JSON file, and *cart3d.x* is the run matrix.

    .. code-block:: pycon
    
        >>> import pyCart
        >>> cart3d = pyCart.Cart3d("pyCart.json")
        >>> cart3d.x.GetFullFolderNames()
        ['ascent/m0.8a0.0b0.0', 'ascent/m1.2a2.0b0.0_maxq']
        
For this module, there are no methods that are particular to Cart3D.  All
functionality is inherited from :class:`cape.runmatrix.RunMatrix`.

:See Also:
    * :mod:`cape.runmatrix`
    * :mod:`pyCart.cart3d`
"""

# Import the cape module.
import cape.runmatrix

# RunMatrix class
class RunMatrix(cape.runmatrix.RunMatrix):
    """Read a run matrix
    
    :Call:
        >>> x = pyCart.RunMatrix(**traj)
        >>> x = pyCart.RunMatrix(File=fname, Keys=keys)
    :Inputs:
        *traj*: :class:`dict`
            Dictionary of options from ``cart3d.Options["RunMatrix"]``
    :Keyword arguments:
        *File*: :class:`str`
            Name of file to read, defaults to ``'RunMatrix.dat'``
        *Keys*: :class:`list` of :class:`str` items
            List of variable names, defaults to ``['Mach','alpha','beta']``
        *Prefix*: :class:`str`
            Prefix to be used for each case folder name
        *GroupPrefix*: :class:`str`
            Prefix to be used for each grid folder name
        *GroupMesh*: :class:`bool`
            Whether or not cases in same group can share volume grids
        *Definitions*: :class:`dict`
            Dictionary of definitions for each key
    :Outputs:
        *x*: :class:`pyCart.runmatrix.RunMatrix`
            Instance of the trajectory class
    :Data members:
        *x.nCase*: :class:`int`
            Number of cases in the trajectory
        *x.prefix*: :class:`str`
            Prefix to be used in folder names for each case in trajectory
        *x.GroupPrefix*: :class:`str`
            Prefix to be used for each grid folder name
        *x.cols*: :class:`list` (:class:`str`)
            List of variable names used
        *x.text*: :class:`dict` (:class:`float`)
            Lists of variable values taken from trajectory file
        *x[key]*: :class:`numpy.ndarray`, *dtype=float*
            Vector of values of each variable specified in *x.cols*
    :Versions:
        2014-05-28 ``@ddalle``: First version
        2014-06-05 ``@ddalle``: Generalized for user-defined keys
    """
        
    # Function to display things
    def __repr__(self):
        """
        Return the string representation of a trajectory.
        
        This looks like ``<pyCart.RunMatrix(nCase=N, keys=['Mach','alpha'])>``
        """
        # Return a string.
        return '<pyCart.RunMatrix(nCase=%i, keys=%s)>' % (self.nCase,
            self.keys)
    
# class RunMatrix
    

"""
:mod:`pyOver.trajectory`: OVERFLOW run matrix module 
=====================================================

This module handles classes and methods that are specific to the list of run
cases (i.e., the "trajectory").  It is defined in the ``"Trajectory"`` section
of the master JSON file (e.g. :file:`pyOver.json`), and is usually read from a
modified comma-separated or space-separated text file.

The primary OVERFLOW state variables are Mach number, angle of attack, and
sidelsip.  To this it is common practice to add a ``"config"`` variable (which
sets the name of the group folder) and a ``"Label"`` which can be used to give
an extension to the name of a case.  A run matrix using only these variables
could be defined as follows.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["mach", "alpha", "beta", "q", "T", "config", "Label"],
            "File": "inputs/matrix.csv"
        }
        
Then the matrix file :file:`inputs/matrix.csv` could have contents such as the
following.

    .. code-block:: none
    
        # mach, alpha, beta, q, T, config, Label 
        0.8, 0.0, 0.0, 480.00, 425.7, ascent,
        1.2, 2.0, 0.0, 550.00, 425.7, ascent, a

Then the following example API commands could be used to show the case names.
In this case *ofl* is the interface to a OVERFLOW run configuration read from
a JSON file, and *ofl.x* is the run matrix.

    .. code-block:: pycon
    
        >>> import pyOver
        >>> ofl = pyOver.Overflow("pyOver.json")
        >>> ofl.x.GetFullFolderNames()
        ['ascent/m0.8a0.0b0.0', 'ascent/m1.2a2.0b0.0_a']
        
For this module, there are no methods that are particular to OVERFLOW.  All
functionality is inherited from :class:`cape.trajectory.Trajectory`.

:See Also:
    * :mod:`cape.trajectory`
    * :mod:`pyOver.cart3d`
"""

# Import the cape module.
import cape.trajectory


# Trajectory class
class Trajectory(cape.trajectory.Trajectory):
    """
    Read a simple list of configuration variables
    
    :Call:
        >>> x = pyOver.Trajectory(**traj)
        >>> x = pyOver.Trajectory(File=fname, Keys=keys)
    :Inputs:
        *traj*: :class:`dict`
            Dictionary of options from ``ofl.Options["Trajectory"]``
    :Keyword arguments:
        *File*: :class:`str`
            Name of file to read, defaults to ``'Trajectory.dat'``
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
        *x*: :class:`pyOver.trajectory.Trajectory`
            Instance of the trajectory class
    :Data members:
        *x.nCase*: :class:`int`
            Number of cases in the trajectory
        *x.prefix*: :class:`str`
            Prefix to be used in folder names for each case in trajectory
        *x.GroupPrefix*: :class:`str`
            Prefix to be used for each grid folder name
        *x.keys*: :class:`list`, *dtype=str*
            List of variable names used
        *x.text*: :class:`dict`, *dtype=list*
            Lists of variable values taken from trajectory file
        *x.Mach*: :class:`numpy.ndarray`, *dtype=float*
            Vector of Mach numbers in trajectory
        ``getattr(x, key)``: :class:`numpy.ndarray`, *dtype=float*
            Vector of values of each variable specified in *keys*
    :Versions:
        2015-12-29 ``@ddalle``: First version based on :mod:`cape.Trajectory`
    """
        
    # Function to display things
    def __repr__(self):
        """
        Return the string representation of a trajectory.
        
        This looks like ``<pyOver.Trajectory(nCase=N, keys=['Mach','alpha'])>``
        """
        # Return a string.
        return '<pyOver.Trajectory(nCase=%i, keys=%s)>' % (self.nCase,
            self.keys)
        
    

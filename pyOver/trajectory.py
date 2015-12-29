"""
OVERFLOW run case list module: :mod:`pyOver.trajectory`
=======================================================

This module handles classes and methods that are specific to the list of run
cases (i.e., the "trajectory").
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
            Dictionary of options from ``fun3d.Options["Trajectory"]``
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
        *x*: :class:`pyFun.trajectory.Trajectory`
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
        
    

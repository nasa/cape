"""
Cape run matrix module: :mod:`cape.trajectory`
==============================================

This module provides a class :class:`cape.trajectory.Trajectory` for interacting
with a list of cases.  Usually this is the list of cases defined as the run
matrix for a set of CFD solutions, and it is defined in the ``"Trajectory"``
:ref:`section of the JSON file <cape-json-trajectory>`.

However, the contents of the :class:`cape.trajectory.Trajectory` may have a list
of cases that differs from the run matrix, for example containing instead the
cases contained in a data book.

The key defining parameter of a run matrix is the list of independent variables,
which are referred to as "trajectory keys" within Cape.  For example, a common
list of trajectory keys for an inviscid setup is ``["mach", "alpha", "beta"]``. 
If the run matrix is loaded as *x*, then the value of the Mach number for case
number *i* would be ``x.mach[i]``.  If the name of the key to query is held
within a variable *k*, use the following syntax to get the value.

    .. code-block:: python
    
        # Name of trajectory key
        k = "alpha"
        # Case number
        i = 10
        # Value of that key for case *i*
        getattr(x,k)[i]

Each case has a particular folder name.  To get the name of the folder for case
*i*, use the syntax

    .. code-block:: python
        
        // Group folder name
        x.GetGroupFolderNames(i)
        // Case folder name
        x.GetFolderNames(i)
        // Combined group and case folder name
        x.GetFullFolderNames(i)

The trajectory class also contains several methods for filtering cases.  For
example, the user may get the list of indices of cases with a Mach number
greater than 1.0, or the user may restrict to cases containing the text "a1.0". 
These use the methods :func:`cape.trajectory.Trajectory.Filter` and
:func:`cape.trajectory.Trajectory.FilterString`, respectively.  Filtering
examples are shown below.

    .. code-block:: python
    
        // Constraints
        I = x.Filter(cons=['mach>=0.5', 'mach<1.0'])
        // Contains exact text
        I = x.FilterString('m0.9')
        // Contains text with wildcards (like file globs)
        I = x.FilterWildcard('m?.?a-*')
        // Contains a regular expression
        I = x.FilterRegex('m0\.[5-8]+a')
        
These methods are all combined in the
:func:`cape.trajectory.Trajectory.GetIndices` method.  The
:func:`cape.trajectory.Trajectory.GetSweeps` method provides a capability to
split a run matrix into groups in which the cases in each group satisfy
user-specified constraints, for example having the same angle of attack.

Also provided are methods such as :func:`cape.trajectory.Trajectory.GetAlpha`,
which allows the user to easily access the angle of attack for case *i* even if
the run matrix is defined using total angle of attack and roll angle.
Similarly, :func:`cape.trajectory.Trajectory.GetReynoldsNumber` returns the
Reynolds number per grid unit even if the run matrix uses static or dynamic
pressure, which relieves the user from having to do the conversions before
creating the run matrix conditions.
"""

# Basic numerics
import numpy as np
# More powerful text processing
import re, fnmatch
# File system and operating system management
import os
# Conversions and constants
from . import convert


# Trajectory class
class Trajectory:
    """
    Read a list of configuration variables
    
    :Call:
        >>> x = cape.Trajectory(**traj)
        >>> x = cape.Trajectory(File=fname, Keys=keys)
    :Inputs:
        *traj*: :class:`dict`
            Dictionary of options from ``opts["Trajectory"]``
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
        *x*: :class:`cape.trajectory.Trajectory`
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
        2014-05-28 ``@ddalle``: First version
        2014-06-05 ``@ddalle``: Generalized for user-defined keys
    """
    
    # Initialization method
    def __init__(self, **kwargs):
        """Initialization method"""
        # Check for an empty trajectory
        if kwargs.get('Empty', False): return
        # Process the inputs.
        fname = kwargs.get('File', None)
        keys = kwargs.get('Keys', ['Mach', 'alpha', 'beta'])
        prefix = kwargs.get('Prefix', "F")
        groupPrefix = kwargs.get('GroupPrefix', "Grid")
        # Process the definitions.
        defns = kwargs.get('Definitions', {})
        # Save properties.
        self.keys = keys
        self.prefix = prefix
        self.GroupPrefix = groupPrefix
        # List of PASS and ERROR markers
        self.PASS = []
        self.ERROR = []
        # Process the key definitions.
        self.ProcessKeyDefinitions(defns)
        # Read the file.
        if fname and os.path.isfile(fname):
            self.ReadTrajectoryFile(fname)
        # Loop through the keys to see if any were specified in the inputs.
        for key in keys:
            # Check inputs for that key.
            if key not in kwargs: continue
            # Check the specification type.
            if type(kwargs[key]).__name__ not in ['list']:
                # Use the same value for all cases
                self.text[key] = [str(kwargs[key])] * len(self.text[keys[0]])
            else:
                # Set it with the new value.
                self.text[key] = [str(v) for v in kwargs[key]]
        # Check if PASS markers are specified.
        if 'PASS' in kwargs:
            self.PASS = kwargs['PASS']
        # Check if ERROR markers are specified.
        if 'ERROR' in kwargs:
            self.ERROR = kwargs['ERROR']
        # Convert PASS and ERROR list to numpy.
        self.PASS  = np.array(self.PASS)
        self.ERROR = np.array(self.ERROR)
        # Save the number of cases.
        nCase = len(self.text[keys[0]])
        self.nCase = nCase
        # Number of entries
        nPass = len(self.PASS)
        nErr  = len(self.ERROR)
        # Make sure PASS and ERROR fields have correct length
        if nPass < nCase:
            self.PASS = np.hstack((self.PASS, False*np.ones(nCase-nPass)))
        if nErr < nCase:
            self.ERROR = np.hstack((self.ERROR, False*np.ones(nCase-nErr)))
        # Create the numeric versions.
        for key in keys:
            # Check the key type.
            if 'Value' not in self.defns[key]:
                raise KeyError(
                    "Definition for trajectory key '%s' is incomplete." % key)
            if self.defns[key]['Value'] == 'float':
                # Normal numeric value
                setattr(self, key,
                    np.array([float(v) for v in self.text[key]]))
            elif self.defns[key]['Value'] == 'int':
                # Normal numeric value
                setattr(self, key,
                    np.array([int(v) for v in self.text[key]]))
            elif self.defns[key]['Value'] == 'hex':
                # Hex numeric value
                setattr(self, key,
                    np.array([eval('0x'+v) for v in self.text[key]]))
            elif self.defns[key]['Value'] in ['oct', 'octal']:
                # Octal value
                setattr(self, key,
                    np.array([eval('0o'+v) for v in self.text[key]]))
            elif self.defns[key]['Value'] in ['bin', 'binary']:
                # Binary value
                setattr(self, key,
                    np.array([eval('0b'+v) for v in self.text[key]]))
            else:
                # Assume string
                setattr(self, key, np.array(self.text[key]))
        # Process the groups (conditions in a group can use same grid).
        self.ProcessGroups()
        
    # Function to display things
    def __repr__(self):
        """
        Return the string representation of a trajectory.
        
        This looks like ``<cape.Trajectory(nCase=N, keys=['Mach','alpha'])>``
        """
        # Return a string.
        return '<cape.Trajectory(nCase=%i, keys=%s)>' % (self.nCase,
            self.keys)
        
    # Copy the trajectory
    def Copy(self):
        """Return a copy of the trajectory
        
        :Call:
            >>> y = x.Copy()
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the trajectory class
        :Outputs:
            *y*: :class:`cape.trajectory.Trajectory`
                Separate trajectory with same data
        :Versions:
            * 2015-05-22 ``@ddalle``
        """
        # Initialize an empty trajectory.
        y = Trajectory(Empty=True)
        # Copy the fields.
        y.defns  = self.defns
        y.abbrv  = self.abbrv
        y.keys   = self.keys
        y.text   = self.text
        y.prefix = self.prefix
        y.PASS   = self.PASS
        y.ERROR  = self.ERROR
        y.nCase  = self.nCase
        # Group-related info
        y.GroupPrefix  = self.GroupPrefix
        y.GroupKeys    = self.GroupKeys
        y.NonGroupKeys = self.NonGroupKeys
        # Loop through keys to copy values.
        for k in self.keys:
            # Copy the array
            setattr(y,k, getattr(self,k).copy())
        # Process groups to make it a full trajectory.
        self.ProcessGroups()
        # Output
        return y
        
    # Function to read a file
    def ReadTrajectoryFile(self, fname):
        """Read trajectory variable values from file
        
        :Call:
            >>> x.ReadTrajectoryFile(fname)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the trajectory class
            *fname*: :class:`str`
                Name of trajectory file
        :Versions:
            * 2014-10-13 ``@ddalle``: Cut code from __init__ method
        """
        # Open the file.
        f = open(fname)
        # Extract the keys.
        keys = self.keys
        # Number of variables
        nVar = len(keys)
        # Loop through the lines.
        for line in f.readlines():
            # Strip the line.
            line = line.strip()
            # Check for empty line or comment
            if line.startswith('#') or len(line)==0:
                continue
            # Separate by commas and/or white space
            v = re.split("[\s\,]+", line)
            # Check v[0]
            if v[-1].lower() in ['$p', 'pass']:
                # Case is marked as passed.
                self.PASS.append(True)
                self.ERROR.append(False)
                # Shift the entries.
                v.pop()
            elif v[0].lower() in ['p', '$p', 'pass']:
                # Case is marked as passed.
                self.PASS.append(True)
                self.ERROR.append(False)
                # Shift the entries.
                v.pop(0)
            elif v[-1].lower() in ['$e', 'error']:
                # Case is marked as error.
                self.PASS.append(False)
                self.ERROR.append(True)
                # Shift the entries.
                v.pop()
            elif v[0].lower() in ['e', '$e', 'error']:
                # Case is marked as error.
                self.PASS.append(False)
                self.ERROR.append(True)
                # Shift the entries.
                v.pop(0)
            else:
                # Case is unmarked.
                self.PASS.append(False)
                self.ERROR.append(False)
            # Save the strings.
            for k in range(nVar):
                # Check for text.
                if k < len(v):
                    # Save the text.
                    self.text[keys[k]].append(v[k])
                elif self.defns[keys[k]]['Value'] == 'str':
                    # No text (especially useful for optional labels)
                    # Default value.
                    v0 = self.defns[keys[k]].get('Default', '')
                    self.text[keys[k]].append(v0)
                else:
                    # No text (especially useful for optional labels)
                    v0 = self.defns[keys[k]].get('Default', '0')
                    self.text[keys[k]].append(str(v0))
        # Close the file.
        f.close()
        
    # Function to process the role that each key name plays.
    def ProcessKeyDefinitions(self, defns):
        """
        Process definitions for the function of each trajectory variable
        
        Many variables have default definitions, such as ``'Mach'``,
        ``'alpha'``, etc.  For user-defined trajectory keywords, defaults will
        be used for aspects of the definition that are missing from the inputs.
        
        :Call:
            >>> x.ProcessKeyDefinitions(defns)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *defns*: :class:`dict`
                Dictionary of keyword definitions or partial definitions
        :Effects:
            *x.text*: :class:`dict`
                Text for each variable and each break point is initialized
            *x.defns*: :class:`dict`
                Definition dictionary is created after processing defaults
            *x.abbrv*: :class:`dict`
                Dictionary of abbreviations for each trajectory key
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
            * 2014-06-17 ``@ddalle``: Overhauled to read from ``defns`` dict
        """
        # Overall default key
        odefkey = defns.get('Default', {})
        # Process the mandatory fields.
        odefkey.setdefault('Group', True)
        odefkey.setdefault('Type', "Group")
        odefkey.setdefault('Format', '%s')
        # Initialize the dictionaries.
        self.text = {}
        self.defns = {}
        self.abbrv = {}
        # Initialize the fields.
        for key in self.keys:
            # Initialize the text for this key.
            self.text[key] = []
            # Process defaults.
            if key.lower() in ['m', 'mach']:
                # Mach number; non group
                defkey = {
                    "Group": False,
                    "Type": "Mach",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "m"
                }
            elif key in ['Alpha', 'alpha', 'aoa']:
                # Angle of attack; non group
                defkey = {
                    "Group": False,
                    "Type": "alpha",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "a"
                }
            elif key in ['Beta', 'beta', 'aos']:
                # Sideslip angle; non group
                defkey = {
                    "Group": False,
                    "Type": "beta",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "b"
                }
            elif key.lower() in ['alpha_t', 'alpha_total']:
                # Total angle of attack; non group
                defkey = {
                    "Group": False,
                    "Type": "alpha_t",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "a"
                }
            elif key.lower() in ['phi', 'phiv']:
                # Total roll angle; non group
                defkey = {
                    "Group": False,
                    "Type": "phi",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "r"
                }
            elif key.lower() in ['re', 'rey', 'reynolds', 'reynolds_number']:
                # Reynolds number per unit
                defkey = {
                    "Group": False,
                    "Type": "Re",
                    "Value": "float",
                    "Format": "%.2e",
                    "Label": False,
                    "Abbreviation": "Re"
                }
            elif key == "T" or key.lower() in ['tinf', 'temp', 'temperature']:
                # Static temperature
                defkey = {
                    "Group": False,
                    "Type": "T",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "T"
                }
            elif key in ['p', 'P', 'pinf']:
                # Static freestream pressure
                defkey = {
                    "Group": False,
                    "Type": "p",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "p"
                }
            elif key in ['q', 'qinf', 'qbar']:
                # Dynamic pressure
                defkey = {
                    "Group": False,
                    "Type": "q",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "q"
                }
            elif key.lower() in ['gamma']:
                # Freestream ratio of specific heats
                defkey = {
                    "Group": False,
                    "Type": "gamma",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "g"
                }
            elif key.lower() in ['p0', 'p_total'] or key.startswith('p0'):
                # Surface stagnation pressure ratio
                defkey = {
                    "Group": False,
                    "Type": "SurfBC",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "p0",
                    "RefPressure": 1.0,
                    "RefTemperature": 1.0,
                    "TotalPressure": None,
                    "TotalTemperature": "T0",
                    "CompID": []
                }
            elif key.startswith('CT'):
                # Thrust coefficient
                defkey = {
                    "Group": False,
                    "Type": "SurfCT",
                    "Value": "float",
                    "Format": "%s",
                    "Label": True,
                    "Abbreviation": "CT",
                    "RefDynamicPressure": None,
                    "RefArea": None,
                    "AreaRatio": 4.0,
                    "MachNumber": 1.0,
                    "TotalTemperature": "T0",
                    "CompID": []
                }
            elif key.lower() in ['t0', 't_total']:
                # Surface stagnation temperature ratio
                defkey = {
                    "Group": False,
                    "Type": "value",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "T0"
                }
            elif key.lower() in ['label', 'suffix']:
                # Extra label for case (non-group)
                defkey = {
                    "Group": False,
                    "Type": "Label",
                    "Value": "str",
                    "Format": "%s",
                    "Abbreviation": ""
                }
            elif key.lower() in ['config', 'GroupPrefix']:
                # Group name or prefix, e.g. 'poweroff', 'poweron', etc.
                defkey = {
                    "Group": True,
                    "Type": "Config",
                    "Value": "str",
                    "Format": "%s",
                    "Abbreviation": ""
                }
            elif key in ['GroupLabel', 'GroupSuffix']:
                # Extra label for groups
                defkey = {
                    "Group": True,
                    "Type": "GroupLabel",
                    "Value": "str",
                    "Format": "%s",
                    "Abbreviation": ""
                }
            else:
                # Start with default key
                defkey = odefkey
                # Set the abbreviation to the full name.
                defkey["Abbreviation"] = key
            # Check if the input has that key defined.
            optkey = defns.get(key, {})
            # Loop through properties.
            for k in defkey.keys():
                optkey.setdefault(k, defkey[k])
            # Save the definitions.
            self.defns[key] = optkey
            # Save the abbreviations.
            self.abbrv[key] = optkey.get("Abbreviation", key)
        
    # Process the groups that need separate grids.
    def ProcessGroups(self):
        """
        Split trajectory variables into groups.  A "group" is a set of
        trajectory conditions that can use the same mesh.
        
        :Call:
            >>> x.ProcessGroups()
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
        :Effects:
            Creates fields that save the properties of the groups.  These fields
            are called *x.GroupKeys*, *x.GroupX*, *x.GroupID*.
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
        """
        # Initialize matrix of group-generating key values.
        x = []
        # Initialize list of group variables.
        gk = []
        ngk = []
        # Loop through the keys to check for groups.
        for key in self.keys:
            # Check the definition for the grouping status.
            if self.defns[key]['Group']:
                # Append to matrix of group-only variables for ALL conditions.
                x.append(getattr(self,key))
                # Append to the list of group variables.
                gk.append(key)
            else:
                # Append to the list of groupable variables.
                ngk.append(key)
        # Save.
        self.GroupKeys = gk
        self.NonGroupKeys = ngk
        # Turn *x* into a proper matrix.
        x = np.transpose(np.array(x))
        # Unfortunately NumPy arrays do not support testing very well.
        y = [list(xi) for xi in x]
        # Initialize list of unique conditions.
        Y = []
        # Initialize the groupID numbers.
        gID = []
        # Test for case of now groups.
        if y == []:
            # List of group==0 nodes.
            gID = np.zeros(self.nCase)
        # Loop through the full set of conditions.
        for yi in y:
            # Test if it's in the existing set of unique conditions.
            if not (yi in Y):
                # If not, append it.
                Y.append(yi)
            # Save the index.
            gID.append(Y.index(yi))
        # Convert the result and save it.
        self.GroupX = np.array(Y)
        # Save the group index for *all* conditions.
        self.GroupID = np.array(gID)
        return None
        
    # Function to list directory names
    def GetFolderNames(self, i=None, prefix=None):
        """
        List folder names for each of the cases in a trajectory.
        
        The folder names will be of the form
    
            ``F_m2.0a0.0b-0.5/``
            
        if the prefix is ``'F'``, or
        
            ``m2.0a0.0b-0.5/``
            
        if the prefix is empty.
            
        Trajectory keys that require separate meshes for each value of the key
        will not be part of the folder name.  The number of digits used will
        match the number of digits in the trajectory file.
        
        :Call:
            >>> dname = x.GetFolderNames()
            >>> dname = x.GetFolderNames(i=None, prefix="F")
        :Inputs:
            *T*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
            *prefix*: :class:`str`
                Header for name of each folder
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        :Versions:
            * 2014-05-28 ``@ddalle``: First version
            * 2014-06-05 ``@ddalle``: Refined to variables that use common grid
        """
        # Process the prefix.
        if prefix is None: prefix = self.prefix
        # Process the index list.
        if i is None: i = range(self.nCase)
        # Get the variable names.
        keys = self.NonGroupKeys
        # Check for a list.
        if np.isscalar(i):
            # Get the name.
            dlist = self._AssembleName(keys, prefix, i)
        else:
            # Initialize the list.
            dlist = []
            # Loop through the conditions.
            for j in i:
                # Get the folder name.
                dname = self._AssembleName(keys, prefix, j)
                # Append to the list.
                dlist.append(dname)
        # Return the list.
        return dlist
        
    # Function to get grid folder names
    def GetGroupFolderNames(self, i=None):
        """
        Get names of folders that require separate meshes
        
        :Call:
            >>> x.GetGroupFolderNames()
            >>> x.GetGroupFolderNames(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
        """
        # Check for prefix variables.
        #if 
        # Set the prefix.
        prefix = self.GroupPrefix
        # Process the index list.
        if i is None: i = range(self.nCase)
        # Get the names of variables requiring separate grids.
        keys = self.GroupKeys
        # Check for a list.
        if np.isscalar(i):
            # Get the name.
            dlist = self._AssembleName(keys, prefix, i)
        else:
            # Initialize the list.
            dlist = []
            # Loop through the conditions.
            for j in i:
                # Get the folder name.
                dname = self._AssembleName(keys, prefix, j)
                # Append to the list.
                dlist.append(dname)
        # Return the list.
        return dlist
        
    # Function to get grid folder names
    def GetUniqueGroupFolderNames(self, i=None):
        """
        Get unique names of folders that require separate meshes
        
        :Call:
            >>> x.GetUniqueGroupFolderNames()
            >>> x.GetUniqueGroupFolderNames(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of group(s) to process
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        :Versions:
            * 2014-09-03 ``@ddalle``: First version
        """
        # Get all group folder names
        dlist = self.GetGroupFolderNames()
        # Transform to unique list.
        dlist = np.unique(dlist)
        # Check for an index filter.
        if i:
            # Return either a single value or sublist
            return dlist[i]
        else:
            # Return the whole list
            return dlist
    
    # Function to filter cases
    def Filter(self, cons, I=None):
        """Filter cases according to a set of constraints
        
        The constraints are specified as a list of strings that contain
        inequalities of variables that are in *x.keys*.
        
        For example, if *m* is the name of a key (presumably meaning Mach
        number), and *a* is a variable presumably representing angle of attack,
        the following example finds the indices of all cases with Mach number
        greater than 1.5 and angle of attack equal to ``2.0``.
        
            >>> i = x.Filter(['m>1.5', 'a==2.0'])
            
        A warning will be produces if one of the constraints does not correspond
        to a trajectory variable or cannot be evaluated for any other reason.
        
        :Call:
            >>> i = x.Filter(cons)
            >>> i = x.Fitler(cons, I)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *cons*: :class:`list` (:class:`str`)
                List of constraints
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                List of indices that match constraints
        :Versions:
            * 2014-12-09 ``@ddalle``: First version
        """
        # Initialize the conditions.
        if I is None:
            # Consider all indices
            i = np.arange(self.nCase) > -1
        else:
            # Start with all indices failed.
            i = np.arange(self.nCase) < -1
            # Set the specified indices to True
            i[I] = True
        # Check for None
        if cons is None: cons = []
        # Loop through constraints
        for con in cons:
            # Check for empty constraints.
            if len(con.strip()) == 0: continue
            # Constraint may fail with bad input.
            try:
                # Apply the constraint.
                i = np.logical_and(i, eval('self.' + con))
            except Exception:
                # Print a warning and move on.
                print("Constraint '%s' failed to evaluate." % con)
        # Output
        return np.where(i)[0]
        
    # Function to filter by checking if string is in the name
    def FilterString(self, txt, I=None):
        """Filter cases by whether or not they contain a substring
        
        :Call:
            >>> i = x.FilterString(txt)
            >>> i = x.FilterString(txt, I)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *txt*: :class:`str`
                Substring to use as filter
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                List of indices that match constraints
        :Versions:
            * 2015-11-02 ``@ddalle``: First version
        """
        # Initialize the conditions.
        if I is None:
            # Consider all indices
            i = np.arange(self.nCase) > -1
        else:
            # Start with all indices failed.
            i = np.arange(self.nCase) < -1
            # Set the specified indices to True
            i[I] = True
        # Loop through conditions
        for j in np.where(i)[0]:
            # Get the case name.
            frun = self.GetFullFolderNames(j)
            # Check if the string is in there.
            if txt not in frun:
                i[j] = False
        # Output
        return np.where(i)[0]
        
    # Function to filter by checking if string is in the name
    def FilterWildcard(self, txt, I=None):
        """Filter cases by whether or not they contain a substring
        
        This function uses file wild cards, so for example
        ``x.FilterWildcard('*m?.0*')`` matches any case whose name contains
        ``m1.0`` or ``m2.0``, etc.  To make sure the ``?`` is a number, use
        ``*m[0-9].0``.  To obtain a filter that matches both ``m10.0`` and
        ``m1.0``, see :func:`FilterRegex`.
        
        :Call:
            >>> i = x.FilterWildcard(txt)
            >>> i = x.FilterWildcard(txt, I)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *txt*: :class:`str`
                Wildcard to use as filter
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                List of indices that match constraints
        :Versions:
            * 2015-11-02 ``@ddalle``: First version
        """
        # Initialize the conditions.
        if I is None:
            # Consider all indices
            i = np.arange(self.nCase) > -1
        else:
            # Start with all indices failed.
            i = np.arange(self.nCase) < -1
            # Set the specified indices to True
            i[I] = True
        # Loop through conditions
        for j in np.where(i)[0]:
            # Get the case name.
            frun = self.GetFullFolderNames(j)
            # Check if the string is in there.
            if not fnmatch.fnmatch(frun, txt):
                i[j] = False
        # Output
        return np.where(i)[0]
        
    # Function to filter by checking if string is in the name
    def FilterRegex(self, txt, I=None):
        """Filter cases by whether or not they match a regular expression
        
        :Call:
            >>> i = x.FilterRegex(txt)
            >>> i = x.FilterRegex(txt, I)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *txt*: :class:`str`
                Wildcard to use as filter
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                List of indices that match constraints
        :Versions:
            * 2015-11-02 ``@ddalle``: First version
        """
        # Initialize the conditions.
        if I is None:
            # Consider all indices
            i = np.arange(self.nCase) > -1
        else:
            # Start with all indices failed.
            i = np.arange(self.nCase) < -1
            # Set the specified indices to True
            i[I] = True
        # Loop through conditions
        for j in np.where(i)[0]:
            # Get the case name.
            frun = self.GetFullFolderNames(j)
            # Check if the name matches the regular expression.
            if not re.search(txt, frun):
                i[j] = False
        # Output
        return np.where(i)[0]
        
        
    # Function to expand indices
    def ExpandIndices(self, itxt):
        """Expand string of subscripts into a list of indices
        
        :Call:
            >>> I = x.ExpandIndices(itxt)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *itxt*: :class:`str` or :class:`unicode`
                Text of subscripts, separated by ';'
        :Outputs:
            *I*: :class:`list` (:class:`int`)
                Array of indices matching any of the input indices
        :Examples:
            >>> x.ExpandIndices(':5')
            array([0, 1, 2, 3, 4])
            >>> x.ExpandIndices(':4;7,8')
            array([0, 1, 2, 3, 7, 8])
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Check the input.
        if type(itxt).__name__ in ['list', 'ndarray']:
            # Already split
            ITXT = itxt
        elif type(itxt).__name__ in ['str', 'unicode']:
            # Split.
            ITXT = itxt.split(';')
        else:
            # Invalid format
            return []
        # Get the full list of indices.
        I0 = range(self.nCase)
        # Initialize output
        I = []
        # Split the input by semicolons.
        for i in ITXT:
            # Ignore []
            i = i.lstrip('[').rstrip(']')
            try:
                # Check for a ':'
                if ':' in i:
                    # Add a range.
                    I += eval('I0[%s]' % i)
                elif ',' in i:
                    # List
                    I += list(eval(i))
                else:
                    # Individual case
                    I += [eval(i)]
            except Exception:
                # Status update.
                print("Index specification '%s' failed to evaluate." % i)
        # Return the matches.
        return I
        
    # Get indices
    def GetIndices(self, **kw):
        """Get indices from either list or constraints or both
        
        :Call:
            >>> I = x.GetIndices(I=None, cons=[], **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *I*: :class:`list` | :class:`str`
                Array of indices or text of indices
            *cons*: :class:`list` (:class:`str`) | :class:`str`
                List of constraints or text list using commas
            *re*: :class:`str`
                Regular expression to test against folder names
            *filter*: :class:`str`
                Exact test to test against folder names
            *glob*: :class:`str`
                Wild card to test against folder names
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                Array of indices
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
            * 2016-02-17 ``@ddalle``: Upgraded to handle text
        """
        # Get special kwargs
        I = kw.get("I")
        cons = kw.get("cons", [])
        # Check index for string
        if type(I).__name__ in ['str', 'unicode']:
            # Process indices
            I = self.ExpandIndices(I)
        # Check constraints for string
        if type(cons).__name__ in ['str', 'unicode']:
            # Separate into list of constraints
            cons = cons.split(',')
        # Initialize indices using "I" and "cons"
        I = self.Filter(cons, I)
        # Check for regular expression filter
        if kw.get('re') not in [None, '']:
            # Filter by regular expression
            I = self.FilterRegex(kw.get('re'), I)
        # Check for wildcard filter
        if kw.get('glob') not in [None, '']:
            # Filter by wildcard glob
            I = self.FilterWildcard(kw.get('glob'), I)
        # Check for simple substring
        if kw.get('filter') not in [None, '']:
            # Filter by substring
            I = self.FilterString(kw.get('filter'), I)
        # Output
        return I
            
    # Function to get sweep based on constraints
    def GetSweep(self, M, **kw):
        """
        Return a list of indices meeting sweep constraints
        
        The sweep uses the index of the first entry of ``True`` in *M*, i.e.
        ``i0=np.where(M)[0][0]``.  Then the sweep contains all other points that
        meet all criteria with respect to trajectory point *i0*.
        
        For example, using ``EqCons=['mach']`` will cause the method to return
        points with *x.mach* matching *x.mach[i0]*.
        
        :Call:
            >>> I = x.GetSweep(M, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *M*: :class:`numpy.ndarray` (:class:`bool`)
                Mask of which trajectory points should be considered
            *SortVar*: :class:`str`
                Variable by which to sort each sweep
            *EqCons*: :class:`list` (:class:`str`)
                List of trajectory keys which must match (exactly) the first
                point in the sweep
            *TolCons*: :class:`dict` (:class:`float`)
                Dictionary whose keys are trajectory keys which must match the
                first point in the sweep to a specified tolerance and whose
                values are the specified tolerances
            *IndexTol*: :class:`int`
                If specified, only trajectory points in the range
                ``[i0,i0+IndexTol]`` are considered for the sweep
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of trajectory point indices in the sweep
        :Versions:
            * 2015-05-24 ``@ddalle``: First version
        """
        # Check for an *i0* point.
        if not np.any(M): return np.array([])
        # Copy the mask.
        m = M.copy()
        # Sort key.
        xk = kw.get('SortVar')
        # Get constraints
        EqCons  = kw.get('EqCons',  [])
        TolCons = kw.get('TolCons', {})
        # Ensure no NoneType
        if EqCons  is None: EqCons = []
        if TolCons is None: TolCons = {}
        # Get the first index.
        i0 = np.where(M)[0][0]
        # Check for an IndexTol.
        itol = kw.get('IndexTol', self.nCase)
        # Max index to consider.
        if type(itol).__name__.startswith('int'):
            # Possible maximum index
            imax = min(self.nCase, i0+itol)
        else:
            # Do not reject points based on index.
            imax = self.nCase
        # Filter if necessary.
        if imax < self.nCase:
            # Remove from the mask
            m[imax:] = False
        # Loop through equality constraints.
        for c in EqCons:
            # Get the key (for instance if matching ``k%10``)
            k = re.split('[^a-zA-Z_]', c)[0]
            # Check for the key.
            if k not in self.keys:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Get the target value.
            x0 = getattr(self,k)[i0]
            # Form the constraint.
            con = 'self.%s == %s' % (c, x0)
            # Apply the constraint.
            m = np.logical_and(m, eval(con))
        # Loop through tolerance-based constraints.
        for c in TolCons:
            # Get the key (for instance if matching 'i%10', key is 'i')
            k = re.split('[^a-zA-Z_]', c)[0]
            # Check for the key.
            if k not in self.keys:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Get tolerance.
            tol = TolCons[c]
            # Get the target value.
            x0 = getattr(self,k)[i0]
            # Form the greater-than constraint.
            con = 'self.%s >= %s' % (c, x0-tol)
            # Apply the constraint.
            m = np.logical_and(m, eval(con))
            # Form the less-than constraint.
            con = 'self.%s <= %s' % (c, x0+tol)
            # Apply the constraint.
            m = np.logical_and(m, eval(con))
        # Initialize output.
        I = np.arange(self.nCase)
        # Apply the final mask.
        J = I[m]
        # Check for a sort variable.
        if xk is not None:
            # Sort based on that key.
            j = np.argsort(getattr(self,xk)[J])
            # Sort the indices.
            J = J[j]
        # Output
        return J
            
    # Function to get sweep based on constraints
    def GetCoSweep(self, x0, i0, **kw):
        """
        Return a list of indices meeting sweep constraints
        
        The sweep uses point *i0* of co-trajectory *x0* as the reference for the
        constraints.
        
        For example, using ``EqCons=['mach']`` will cause the method to return
        points with *x.mach* matching *x0.mach[i0]*.
        
        :Call:
            >>> I = x.GetSweep(x0, i0, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *x0*: :class:`cape.trajectory.Trajectory`
                Another instance of the pyCart trajectory class
            *i0*: :class:`int`
                Index of point in *x0* to use as reference for constraints
            *SortVar*: :class:`str`
                Variable by which to sort each sweep
            *EqCons*: :class:`list` (:class:`str`)
                List of trajectory keys which must match (exactly) the first
                point in the sweep
            *TolCons*: :class:`dict` (:class:`float`)
                Dictionary whose keys are trajectory keys which must match the
                first point in the sweep to a specified tolerance and whose
                values are the specified tolerances
            *IndexTol*: :class:`int`
                If specified, only trajectory points in the range
                ``[i0,i0+IndexTol]`` are considered for the sweep
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of *x* indices to consider in the sweep
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of *x* indices in the sweep
        :Versions:
            * 2015-06-03 ``@ddalle``: First version
        """
        # Check for list of indices
        I = kw.get('I')
        # Initial mask
        if I is not None and len(I) > 0:
            # Initialize mask
            m = np.arange(self.nCase) < 0
            # Consider cases in initial list
            m[I] = True
        else:
            # Initialize mask to ``True``.
            m = np.arange(self.nCase) > -1
        # Sort key.
        xk = kw.get('SortVar')
        # Get constraints
        EqCons  = kw.get('EqCons',  [])
        TolCons = kw.get('TolCons', {})
        # Ensure no NoneType
        if EqCons  is None: EqCons = []
        if TolCons is None: TolCons = {}
        # Check for an IndexTol.
        itol = kw.get('IndexTol', self.nCase)
        # Max index to consider.
        if type(itol).__name__.startswith('int'):
            # Possible maximum index
            imax = min(self.nCase, i0+itol)
        else:
            # Do not reject points based on index.
            imax = self.nCase
        # Filter if necessary.
        if imax < self.nCase:
            # Remove from the mask
            m[imax:] = False
        # Loop through equality constraints.
        for c in EqCons:
            # Get the key (for instance if matching ``k%10``)
            k = re.split('[^a-zA-Z_]', c)[0]
            # Check for the key.
            if k not in self.keys:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Get the target value.
            v0 = getattr(x0,k)[i0]
            # Form the constraint.
            con = 'self.%s == %s' % (c, v0)
            # Apply the constraint.
            m = np.logical_and(m, eval(con))
        # Loop through tolerance-based constraints.
        for c in TolCons:
            # Get the key (for instance if matching 'i%10', key is 'i')
            k = re.split('[^a-zA-Z_]', c)[0]
            # Check for the key.
            if k not in self.keys:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Get tolerance.
            tol = TolCons[c]
            # Get the target value.
            v0 = getattr(x0,k)[i0]
            # Form the greater-than constraint.
            con = 'self.%s >= %s' % (c, v0-tol)
            # Apply the constraint.
            m = np.logical_and(m, eval(con))
            # Form the less-than constraint.
            con = 'self.%s <= %s' % (c, v0+tol)
            # Apply the constraint.
            m = np.logical_and(m, eval(con))
        # Initialize output.
        I = np.arange(self.nCase)
        # Apply the final mask.
        J = I[m]
        # Check for a sort variable.
        if xk is not None:
            # Sort based on that key.
            j = np.argsort(getattr(self,xk)[J])
            # Sort the indices.
            J = J[j]
        # Output
        return J
        
    # Function to get set of sweeps based on criteria
    def GetSweeps(self, **kw):
        """
        Return a list of index sets in which each list contains cases that
        satisfy specified criteria.
        
        For example, using ``EqCons=['mach']`` will cause the method to return
        lists of points with the same Mach number.
        
        :Call:
            >>> J = x.GetSweeps(**kw)
        :Inputs:
            *cons*: :class:`list` (:class:`str`)
                List of global constraints; only points satisfying these
                constraints will be in one of the output sweeps
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices to restrict to
            *SortVar*: :class:`str`
                Variable by which to sort each sweep
            *EqCons*: :class:`list` (:class:`str`)
                List of trajectory keys which must match (exactly) the first
                point in the sweep
            *TolCons*: :class:`dict` (:class:`float`)
                Dictionary whose keys are trajectory keys which must match the
                first point in the sweep to a specified tolerance and whose
                values are the specified tolerances
            *IndexTol*: :class:`int`
                If specified, only trajectory points in the range
                ``[i0,i0+IndexTol]`` are considered for the sweep
        :Outputs:
            *J*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of trajectory point sweeps
        :Versions:
            * 2015-05-25 ``@ddalle``: First version
        """
        # Expand global index constraints.
        I0 = self.GetIndices(I=kw.get('I'), cons=kw.get('cons'))
        # Initialize mask (list of ``False`` with *nCase* entries)
        M = np.arange(self.nCase) < 0
        # Set the mask to ``True`` for any cases passing global constraints.
        M[I0] = True
        # Initialize output.
        J = []
        # Initialize target output
        JT = []
        # Safety check: no more than *nCase* sets.
        i = 0
        # Loop through cases.
        while np.any(M) and i<self.nCase:
            # Increase number of sweeps.
            i += 1
            # Get the current sweep.
            I = self.GetSweep(M, **kw)
            # Save the sweep.
            J.append(I)
            # Update the mask.
            M[I] = False
        # Output
        return J
        
        
    # Function to return the full folder names.
    def GetFullFolderNames(self, i=None, prefix=None):
        """
        List full folder names for each of the cases in a trajectory.
        
        The folder names will be of the form
    
            ``Grid/F_m2.0a0.0b-0.5/``
            
        if there are no trajectory keys that require separate grids or
        
            ``Grid_d1.0/F_m2.0a0.0b-0.5/``
            
        if there is a key called ``"delta"`` with abbreviation ``'d'`` that
        requires a separate mesh each time the value of that key changes.  All
        keys in the trajectory file are included in the folder name at one of
        the two levels.  The number of digits used will match the number of
        digits in the trajectory file.
        
        :Call:
            >>> dname = x.GetFullFolderNames()
            >>> dname = x.GetFullFolderNames(i=None, prefix="F")
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
            *prefix*: :class:`str`
                Header for name of each case folder
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
        """
        # Get the two components.
        glist = self.GetGroupFolderNames(i)
        flist = self.GetFolderNames(i, prefix)
        # Check for list or not.
        if type(glist) is list:
            # Return the list of combined strings.
            return [os.path.join(glist[i],flist[i]) for i in range(len(glist))]
        else:
            # Just join the one.
            return os.path.join(glist, flist)
            
    # Function to get the group index from the case index
    def GetGroupIndex(self, i):
        """Get group index from case index
        
        :Call:
            k = x.GetGroupIndex(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int`
                Index of case
        :Outputs:
            *j*: :class:`int`
                Index of group that contains case *i*
        :Versions:
            * 2014-09-27 ``@ddalle``: First versoin
        """
        # Check inputs.
        if not type(i).__name__.startswith('int'):
            raise TypeError("Input to :func:`Trajectory.GetGroupIndex` must"
                + " be :class:`int`.")
        # Get name of group for case *i*.
        grp = self.GetGroupFolderNames(i)
        # Get the list of all unique groups.
        grps = self.GetUniqueGroupFolderNames()
        # Find the index.
        j = np.where(grps == grp)[0][0]
        # Output
        return j
        
    # Get all keys by type
    def GetKeysByType(self, KeyType):
        """Get all keys by type
        
        :Call:
            >>> keys = x.GetKeysByType(KeyType)
            >>> keys = x.GetKeysByType(KeyTypes)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of pyCart trajectory class
            *KeyType*: :class:`str`
                Key type to search for
            *KeyTypes*: :class:`list` (:class:`str`)
                List of key types to search for
        :Outputs:
            *keys*: :class:`numpy.ndarray` (:class:`str`)
                List of keys such that ``x[key]['Type']`` matches *KeyType*
        :Versions:
            * 2014-10-07 ``@ddalle``: First version
        """
        # List of key types
        KT = np.array([self.defns[k]['Type'] for k in self.keys])
        # Class of input
        kt = type(KeyType).__name__
        # Depends on the type of what we are searching for
        if kt.startswith('str') or kt=='unicode':
            # Return matches
            return np.array(self.keys)[KT == KeyType]
        elif kt not in ['list', 'ndarray']:
            # Not usable
            raise TypeError("Cannot search for keys of type '%s'" % KeyType)
        # Initialize list of matches to all ``False``
        U = np.arange(len(self.keys)) < 0
        # Loop through types given as input.
        for k in KeyType:
            # Search for this kind of key.
            U = np.logical_or(U, KT == k)
        # Output.
        return np.array(self.keys)[np.where(U)[0]]
        
    # Get keys by type of its value
    def GetKeysByValue(self, val):
        """Get all keys with specified type of value
        
        :Call:
            >>> keys = x.GetKeysByValue(val)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of pyCart trajectory class
            *val*: :class:`str`
                Key value class to search for
        :Outputs:
            *keys*: :class:`numpy.ndarray` (:class:`str`)
                List of keys such that ``x[key]['Value']`` matches *val*
        :Versions:
            * 2014-10-07 ``@ddalle``: First version
        """
        # List of key types
        KV = np.array([self.defns[k]['Value'] for k in self.keys])
        # Return matches
        return np.array(self.keys)[KV == val]
    
    # Get Reynolds number
    def GetReynoldsNumber(self, i):
        """Get Reynolds number (per foot)
        
        :Call:
            >>> Re = x.GetReynoldsNumber(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *Re*: :class:`float`
                Reynolds number [1/inch | 1/ft]
        :Versions:
            * 2016-03-23 ``@ddalle``: First version
        """
        # Process the key types.
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for Reynolds number
        if 'Re' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('Re')[0]
            # Set the value.
            return getattr(self,k)[i]
        # If reached here, we need 'T' and "Mach' at least
        if 'Mach' not in KeyTypes:
            return None
        elif 'T' not in KeyTypes:
            return None
        # Get temperature and Mach keys
        kM = self.GetKeysByType('Mach')[0]
        kT = self.GetKeysByType('T')[0]
        # Get temperature and Mach values
        M = getattr(self,kM)[i]
        T = getattr(self,kT)[i]
        # Get units
        u = self.defns[kT].get('Units', 'fps')
        # Check for pressure specifier
        if 'p' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('p')[0]
            # Get the pressure
            p = getattr(self,k)[i]
            # Calculate Reynolds number
            if u.lower() in ['fps', 'r']:
                # Imperial units
                return convert.ReynoldsPerFoot(p, T, M)/12
            else:
                # MKS units
                return convert.ReynoldsPerMeter(p, T, M)
        elif 'q' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('q')[0]
            # Get the dynamic pressure
            q = getattr(self,k)[i]
            # Calculate static pressure
            p = q / (0.7*M*M)
            # Calculate Reynolds number
            if u.lower() in ['fps', 'r']:
                # Imperial units
                return convert.ReynoldsPerFoot(p, T, M)/12
            else:
                # MKS units
                return convert.ReynoldsPerMeter(p, T, M)
        # If we reach here, missing info.
        return None
        
    # Get Mach number
    def GetMach(self, i):
        """Get Mach number
        
        :Call:
            >>> M = x.GetMach(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *M*: :class:`float`
                Mach number
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for temperature
        if 'Mach' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('Mach')[0]
            # Return the value
            return getattr(self,k)[i]
        # If we reach this point... not doing this conversion
        return None
        
    # Get angle of attack
    def GetAlpha(self, i):
        """Get the angle of attack
        
        :Call:
            >>> alpha = x.GetAlpha(i)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *alpha*: :class:`float`
                Angle of attack in degrees
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for angle of attack
        if 'alpha' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('alpha')[0]
            # Return the value
            return getattr(self,k)[i]
        # Check for total angle of attack
        if 'alpha_t' in KeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha_t')[0]
            # Get the value
            av = getattr(self,k)[i]
            # Check for roll angle
            if 'phi' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('phi')[0]
                # Get the value
                rv = getattr(self,k)[i]
            else:
                # Default value
                rv = 0.0
            # Convert to aoa and aos
            a, b = convert.AlphaTPhi2AlphaBeta(av, rv)
            # Output
            return a
        # No info
        return None
        
    # Get total angle of attack
    def GetAlphaTotal(self, i):
        """Get the total angle of attack
        
        :Call:
            >>> av = x.GetAlphaTotal(i)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *av*: :class:`float`
                Total angle of attack in degrees
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for total angle of attack
        if 'alpha_t' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('alpha_t')[0]
            # Return the value
            return getattr(self,k)[i]
        # Check for angle of attack
        if 'alpha' in nKeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha')[0]
            # Get the value
            a = getattr(self,k)[i]
            # Check for sideslip
            if 'beta' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('beta')[0]
                # Get the value
                b = getattr(self,k)[i]
            else:
                # Default value
                b = 0.0
            # Convert to alpha total, phi
            av, rv = convert.AlphaBeta2AlphaTPhi(a, b)
            # Output
            return av
        # no info
        return None
        
    # Get angle of sideslip
    def GetBeta(self, i):
        """Get the sideslip angle
        
        :Call:
            >>> beta = x.GetBeta(i)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *beta*: :class:`float`
                Angle of sideslip in degrees
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for angle of attack
        if 'beta' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('beta')[0]
            # Return the value
            return getattr(self,k)[i]
        # Check for total angle of attack
        if 'alpha_t' in KeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha_t')[0]
            # Get the value
            av = getattr(self,k)[i]
            # Check for roll angle
            if 'phi' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('phi')[0]
                # Get the value
                rv = getattr(self,k)[i]
            else:
                # Default value
                rv = 0.0
            # Convert to aoa and aos
            a, b = convert.AlphaTPhi2AlphaBeta(av, rv)
            # Output
            return b
        # No info
        return None
        
    # Get velocity roll angle
    def GetPhi(self, i):
        """Get the velocity roll angle
        
        :Call:
            >>> phiv = x.GetPhi(i)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *phiv*: :class:`float`
                Velocity roll angle in degrees
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for total angle of attack
        if 'phi' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('alpha_t')[0]
            # Return the value
            return getattr(self,k)[i]
        # Check for angle of attack
        if 'alpha' in nKeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha')[0]
            # Get the value
            a = getattr(self,k)[i]
            # Check for sideslip
            if 'beta' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('beta')[0]
                # Get the value
                b = getattr(self,k)[i]
            else:
                # Default value
                b = 0.0
            # Convert to alpha total, phi
            av, rv = convert.AlphaBeta2AlphaTPhi(a, b)
            # Output
            return rv
        # no info
        return None
        
    # Get freestream temperature
    def GetTemperature(self, i):
        """Get static freestream temperature
        
        :Call:
            >>> T = x.GetTemperature(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *T*: :class:`float`
                Static temperature [R | K]
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for temperature
        if 'T' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('T')[0]
            # Return the value
            return getattr(self,k)[i]
        # If we reach this point... not doing this conversion
        return None
        
    # Get freestream pressure
    def GetPressure(self, i):
        """Get static freestream pressure (in psf or Pa)
        
        :Call:
            >>> p = x.GetPressure(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *p*: :class:`float`
                Static pressure [psf | Pa]
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for static pressure
        if 'p' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('p')[0]
            # Output
            return getattr(self,k)[i]
        # If we reach this point, we need at least Mach and temperature
        # Get temperature and Mach keys
        kM = self.GetKeysByType('Mach')[0]
        kT = self.GetKeysByType('T')[0]
        # Get temperature and Mach values
        M = getattr(self,kM)[i]
        T = getattr(self,kT)[i]
        # Get units
        u = self.defns[kT].get('Units', 'fps')
        # Check for pressure specifier
        if 'Re' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('Re')[0]
            # Get the Reynolds number
            Re = getattr(self,k)[i]
            # Calculate Reynolds number
            if u.lower() in ['fps', 'r']:
                # Imperial units
                return convert.PressureFPSFromRe(Re*12, T, M)
            else:
                # MKS units
                return convert.PressureMKSFromRe(Re, T, M)
        elif 'q' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('q')[0]
            # Get the dynamic pressure
            q = getattr(self,k)[i]
            # Calculate static pressure
            return q / (0.7*M*M)
        # If we reach here, missing info.
        return None
        
    # Get ratio of specific heats
    def GetGamma(self, i):
        """Get freestream ratio of specific heats
        
        :Call:
            >>> gam = x.GetGamma(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *gam*: :class:`float`
                Ratio of specific heats
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for ratio of specific heats
        if 'gamma' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('gamma')[0]
            # Use the value of that key
            return getattr(self,k)[i]
        else:
            # Default value
            return 1.4
    
    # Get freestream pressure
    def GetDynamicPressure(self, i):
        """Get dynamic freestream pressure (in psf or Pa)
        
        :Call:
            >>> q = x.GetDynamicPressure(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *q*: :class:`float`
                Dynamic pressure [psf | Pa]
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for dynamic pressure
        if 'q' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('q')[0]
            # Output
            return getattr(self,k)[i]
        # If we reach this point, we need at least Mach and temperature
        # Get temperature and Mach keys
        kM = self.GetKeysByType('Mach')[0]
        kT = self.GetKeysByType('T')[0]
        # Get temperature and Mach values
        M = getattr(self,kM)[i]
        T = getattr(self,kT)[i]
        # Get units
        u = self.defns[kT].get('Units', 'fps')
        # Check for pressure specifier
        if 'Re' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('Re')[0]
            # Get the Reynolds number
            Re = getattr(self,k)[i]
            # Check units
            if u.lower() in ['fps', 'r']:
                # Imperial units; get pressure
                p = convert.PressureFPSFromRe(Re*12, T, M)
            else:
                # MKS units
                p = convert.PressureMKSFromRe(Re, T, M)
            # Dynamic pressure
            return 0.7*p*M*M
        elif 'p' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('p')[0]
            # Get the static pressure
            p = getattr(self,k)[i]
            # Calculate dynamic pressure
            return p * (0.7*M*M)
        # If we reach here, missing info.
        return None
        
    # Get thrust for SurfCT input
    def GetSurfCT_Thrust(self, i, key=None):
        """Get thrust input for surface *CT* key
        
        :Call:
            >>> CT = x.GetSurfCT_Thrust(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *CT*: :class:`float`
                Thrust parameter, either thrust or coefficient
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = sefl.GetKeysByType('SurfCT')[0]
        # Get the thrust parameter
        ot = self.defns[key].get('Thrust')
        # Type
        tt = tpye(ot).__name__
        # Process the option
        if ot is None:
            # Use the value of this key
            return getattr(self,key)[i]
        elif tt in ['str', 'unicode']:
            # Use this as a key
            kT = ot
            # Use the value of that key
            return getattr(self,kT)[i]
        else:
            # Return the fixed value
            return ot
            
    # Get reference dynamic pressure
    def GetSurfCT_RefDynamicPressure(self, k, key=None):
        """Get reference dynamic pressure for surface *CT* key
        
        :Call:
            >>> qinf = x.GetSurfCT_RefDynamicPressure(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
        :Outputs:
            *qinf*: :class:`float`
                Reference dynamic pressure to use, this divides the *CT* value
        :Versions:
            * 2016-04-12 ``@ddalle``: First version
        """
        # Process the key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Get the pressure parameter
        op = self.defns[key].get('RefDynamicPressure')
        # Type
        tp = type(op).__name__
        # Process the option
        if op is None:
            # Use the freestream value
            return 0.5*self.GetGamma(i)*self.GetMach(i)**2
        elif tp in ['str', 'unicode']:
            # Check for special names
            if op.lower() in ['freestream', 'inf']:
                # Use the freestream value
                return self.GetDynamicPressure(i)
            else:
                # Use this as a key
                kP = op
                # Use the value of that key
                return getattr(self,kP)[i]
        else:
            # Use the fixed value
            return op
            
    # Get total temperature
    def GetSurfCT_RefPressure(self, i, key=None):
        """Get reference pressure input for surface *CT* total pressure
        
        :Call:
            >>> Tref = x.GetSurfCT_RefPressure(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *pref*: :class:`float`
                Reference pressure for normalizing *T0*
        :Versions:
            * 2016-04-13 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Call the SurfBC equivalent
        return self.GetSurfBC_RefTemperature(i, key)
            
    # Get pressure calibration factor
    def GetSurfCT_PressureCalibration(self, i, key=None):
        """Get pressure calibration factor for *CT* key
        
        :Call:
            >>> fp = x.GetSurfCT_PressureCalibration(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *fp*: {``1.0``} | :class:`float`
                Pressure calibration factor
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Call the SurfBC equivalent
        return self.GetSurfBC_PressureCalibration(i, key)
            
    # Get total temperature
    def GetSurfCT_TotalTemperature(self, i, key=None):
        """Get total temperature input for surface *CT* key
        
        :Call:
            >>> T0 = x.GetSurfCT_TotalTemperature(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *T0*: :class:`float`
                Total temperature of thrust conditions
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Call the SurfBC equivalent
        return self.GetSurfBC_TotalTemperature(i, key)
            
    # Get total temperature
    def GetSurfCT_RefTemperature(self, i, key=None):
        """Get reference temperature input for surface *CT* total temperature
        
        :Call:
            >>> Tref = x.GetSurfCT_RefTemperature(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *Tref*: :class:`float`
                Reference temperature for normalizing *T0*
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Call the SurfBC equivalent
        return self.GetSurfBC_RefTemperature(i, key)
    
    # Get Mach number
    def GetSurfCT_Mach(self, i, key=None):
        """Get Mach number input for surface *CT* key
        
        :Call:
            >>> M = x.GetSurfCT_TotalTemperature(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *T0*: :class:`float`
                Total temperature of thrust conditions
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Call the SurfBC equivalent
        return self.GetSurfBC_Mach(i, key)
    
    # Get exit Mach number input for SurfCT input
    def GetSurfCT_ExitMach(self, i, key=None):
        """Get Mach number input for surface *CT* key
        
        :Call:
            >>> M2 = x.GetSurfCT_ExitMach(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
        :Outputs:
            *M2*: :class:`float`
                Nozzle exit Mach number
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfBC' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Get the pressure parameter
        om = self.defns[key].get('ExitMach')
        # Type
        tm = type(om).__name__
        # Process the option
        if om is None:
            # Not using this key
            return None
        elif tm in ['str', 'unicode']:
            # Use the value of that key
            return getattr(self,om)[i]
        else:
            # Use the fixed value
            return om
    
    # Get area ratio
    def GetSurfCT_AreaRatio(self, i, key=None):
        """Get area ratio for surface *CT* key
        
        :Call:
            >>> AR = x.GetSurfCT_AreaRatio(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *AR*: :class:`float`
                Area ratio
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Get the area ratio parameter
        oa = self.defns[key].get('AreaRatio')
        # Type
        ta = type(oa).__name__
        # Process the option
        if oa is None:
            # Use the value of this
            return None
        elif ta in ['str', 'unicode']:
            # Use this value as a key
            ka = oa
            # Use the value of that key.
            return getattr(self,ka)[i]
        else:
            # Use the fixed value
            return oa
            
    # Get reference area
    def GetSurfCT_RefArea(self, i, key=None):
        """Get reference area for surface *CT* key, this divides *CT* value
        
        If this is ``None``, it defaults to the vehicle reference area
        
        :Call:
            >>> Aref = x.GetSurfCT_RefArea(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *ARef*: {``None``} | :class:`float`
                Reference area; if ``None``, use the vehicle area 
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Get the area ratio parameter
        oa = self.defns[key].get('RefArea')
        # Type
        ta = type(oa).__name__
        # Process the option
        if oa is None:
            # Flag to use the vehicle value from *cntl.opts*
            return None
        elif ta in ['str', 'unicode']:
            # Use this value as a key
            ka = oa
            # Use the value of that key.
            return getattr(self,ka)[i]
        else:
            # Use the fixed value
            return oa
        
    
    # Get component ID(s) for input SurfCT key
    def GetSurfCT_CompID(self, i, key=None):
        """Get component ID input for surface *CT* key
        
        :Call:
            >>> compID = x.GetSurfCT_CompID(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *compID*: :class:`list` (:class:`int` | :class:`str`)
                Surface boundary condition Mach number
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Call the SurfBC equivalent
        return self.GetSurfBC_CompID(i, key)
        
    # Get ratio of specific heats for SurfCT key
    def GetSurfCT_Gamma(self, i, key=None):
        """Get ratio of specific heats input for surface *CT* key
        
        :Call:
            >>> gam = x.GetSurfCT_Gamma(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
        :Outputs:
            *gam*: {``1.4``} | :class:`float`
                Ratio of specific heats
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfCT' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfCT')[0]
        # Call the SurfBC equivalent
        return self.GetSurfBC_Gamma(i, key)
        
    # Get stagnation pressure input for SurfBC input
    def GetSurfBC_TotalPressure(self, i, key=None):
        """Get stagnation pressure input for surface BC key
        
        :Call:
            >>> p0 = x.GetSurfBC_TotalPressure(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
        :Outputs:
            *p0*: :class:`float`
                Stagnation pressure parameter, usually *p0/pinf*
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfBC' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfBC')[0]
        # Get the pressure parameter
        op0 = self.defns[key].get('TotalPressure')
        # Type
        tp0 = type(op0).__name__
        # Process the option
        if op0 is None:
            # Use the value of this key
            return getattr(self,key)[i]
        elif tp0 in ['str', 'unicode']:
            # Use this as a key
            kP = op0
            # Use the value of that key
            return getattr(self,kP)[i]
        else:
            # Use the fixed value
            return op0
            
    # Get reference pressure
    def GetSurfBC_RefPressure(self, i, key=None):
        """Get reference pressure for surface BC key
        
        :Call:
            >>> pinf = x.GetSurfBC_RefPressure(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
        :Outputs:
            *pinf*: :class:`float`
                Reference pressure to use, this divides the *p0* value
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Process the key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfBC' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfBC')[0]
        # Get the pressure parameter
        op = self.defns[key].get('RefPressure')
        # Type
        tp = type(op).__name__
        # Process the option
        if op is None:
            # Use the value of this key
            return 1.0
        elif tp in ['str', 'unicode']:
            # Check for special keys
            if op.lower() in ['freestream', 'inf']:
                # Use the freestream value
                return self.GetPressure(i)
            else:
                # Use this as a key
                kP = op
                # Use the value of that key
                return getattr(self,kP)[i]
        else:
            # Use the fixed value
            return op
            
    # Get pressure scaling
    def GetSurfBC_PressureCalibration(self, i, key=None):
        """Get total pressure scaling factor used for calibration
        
        :Call:
            >>> fp = x.GetSurfBC_PressureCalibration(i, key=None)
        :Inputs:
            *x*: :Class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defautls to first ``SurfBC`` key
        :Outputs:
            *fp*: {``1.0``} | :class:`float`
                Pressure calibration factor
        :Versions:
            * 2016-04-12 ``@ddalle``: First version
        """
        # Process the key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfBC' not in KeyTypes: return 1.0
            # Get first key
            key = self.GetKeysByType('SurfBC')[0]
        # Get the pressure parameter
        op = self.defns[key].get('PressureCalibration', 1.0)
        # Type
        tp = type(op).__name__
        # Process the option
        if op is None:
            # Use the value of this key
            return 1.0
        elif tp in ['str', 'unicode']:
            # Use this as a key
            kP = op
            # Use the value of that key
            return getattr(self,kP)[i]
        else:
            # Use the fixed value
            return op
        
    # Get stagnation temperature input for SurfBC input
    def GetSurfBC_TotalTemperature(self, i, key=None):
        """Get stagnation pressure input for surface BC key
        
        :Call:
            >>> T0 = x.GetSurfBC_TotalTemperature(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
        :Outputs:
            *T0*: :class:`float`
                Stagnation temperature parameter, usually *T0/Tinf*
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfBC' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfBC')[0]
        # Get the pressure parameter
        ot0 = self.defns[key].get('TotalTemperature')
        # Type
        tt0 = type(ot0).__name__
        # Process the option
        if ot0 is None:
            # Use the value of this key
            return getattr(self,key)[i]
        elif tt0 in ['str', 'unicode']:
            # Use the value of that key
            return getattr(self,ot0)[i]
        else:
            # Use the fixed value
            return ot0
            
    # Get reference temperature
    def GetSurfBC_RefTemperature(self, i, key=None):
        """Get reference temperature for surface BC key
        
        :Call:
            >>> Tinf = x.GetSurfBC_RefTemperature(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
        :Outputs:
            *Tinf*: :class:`float`
                Reference temperature to use, this divides the *T0* value
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Process the key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfBC' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfBC')[0]
        # Get the pressure parameter
        ot = self.defns[key].get('RefTemperature')
        # Type
        tt = type(ot).__name__
        # Process the option
        if ot is None:
            # Use the freestream value
            return 1.0
        elif tt in ['str', 'unicode']:
            # Check for special keys
            if ot.lower() in ['freestream', 'inf']:
                # Use frestream value
                return self.GetTemperature(i)
            else:
                # Use this as a key
                return getattr(self,ot)[i]
        else:
            # Use the fixed value
            return ot
            
    # Get Mach number input for SurfBC input
    def GetSurfBC_Mach(self, i, key=None):
        """Get Mach number input for surface BC key
        
        :Call:
            >>> M = x.GetSurfBC_Mach(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
        :Outputs:
            *M*: :class:`float`
                Surface boundary condition Mach number
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfBC' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfBC')[0]
        # Get the pressure parameter
        om = self.defns[key].get('Mach', 1.0)
        # Type
        tm = type(om).__name__
        # Process the option
        if om is None:
            # Use the value of this key
            return getattr(self,key)[i]
        elif tm in ['str', 'unicode']:
            # Use the value of that key
            return getattr(self,om)[i]
        else:
            # Use the fixed value
            return om
            
    # Get ratio of specific heats input for SurfBC key
    def GetSurfBC_Gamma(self, i, key=None):
        """Get ratio of specific heats for surface BC key
        
        :Call:
            >>> gam = x.GetSurfBC_Gamma(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
        :Outputs:
            *gam*: :class:`float`
                Surface boundary condition ratio of specific heats
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfBC' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfBC')[0]
        # Get the pressure parameter
        og = self.defns[key].get('Gamma', 1.4)
        # Type
        tg = type(og).__name__
        # Process the option
        if og is None:
            # Use the value of this key
            return self.GetGamma(i)
        elif tg in ['str', 'unicode']:
            # Use the value of that key
            return getattr(self,og)[i]
        else:
            # Use the fixed value
            return og
            
    # Get component ID(s) input for SurfBC key
    def GetSurfBC_CompID(self, i, key=None):
        """Get component ID input for surface BC key
        
        :Call:
            >>> compID = x.GetSurfBC_CompID(i, key=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
        :Outputs:
            *compID*: :class:`list` (:class:`int` | :class:`str`)
                Surface boundary condition Mach number
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if 'SurfBC' not in KeyTypes: return
            # Get first key
            key = self.GetKeysByType('SurfBC')[0]
        # Get the pressure parameter
        oc = self.defns[key].get('CompID')
        # Type
        tc = type(oc).__name__
        # Process the option
        if oc is None:
            # Use the value of this key
            return getattr(self,key)[i]
        elif tc in ['str', 'unicode']:
            # Use the value of that key
            return getattr(self,oc)[i]
        else:
            # Use the fixed value
            return oc
    
    # Function to assemble a folder name based on a list of keys and an index
    def _AssembleName(self, keys, prefix, i):
        """
        Assemble names using common code.
        
        :Call:
            >>> dname = x._AssembleName(keys, prefix, i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *keys*: :type:`list` (:class:`str`)
                List of keys to use for this folder name
            *prefix*: :class:`str`
                Header for name of each case folder
            *i*: :class:`int`
                Index of case to process
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Name containing value for each key in *keys*
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
            * 2014-10-03 ``@ddalle``: Added suffixes
        """
        # Process the key types.
        types = [self.defns[k].get("Type","") for k in keys]
        # Check for a prefix.
        if "Config" in types:
            # Figure out which key it is
            j = types.index("Config")
            # Get the specified prefix.
            fpre = getattr(self,keys[j])[i]
            # Initialize the name.
            if fpre:
                # Use the specified prefix/config
                dname = str(fpre)
            else:
                # Use the input/default prefix/config
                dname = str(prefix)
            # Add underscore if more keys remaining.
            if len(types) > 1: dname += "_"
        elif prefix:
            # The prefix is likely to be the whole name.
            dname = str(prefix)
            # Add underscore if there are keys.
            if (keys is not None) and (len(keys)>0): dname += "_"
        else:
            # Initialize an empty string.
            dname = ""
        # Append based on the keys.
        for k in keys:
            # Skip text
            if self.defns[k]["Value"] == "str": continue
            # Check for unlabeled values
            if (not self.defns[k].get("Label", True)): continue
            # Skip unentered values
            if (i>=len(self.text[k])) or (not self.text[k][i]): continue
            # Check for "SkipZero" flag
            if self.defns[k].get("SkipIfZero", False): continue
            # Make the string of what's going to be printed.
            # This is something like ``'%.2f' % x.alpha[i]``.
            lbl = self.defns[k]["Format"] % getattr(self,k)[i]
            # Append the text in the trajectory file.
            dname += self.abbrv[k] + lbl 
        # Check for suffix keys.
        for k in keys:
            # Only look for labels.
            if self.defns[k].get("Type") != "Label": continue
            # Check the value.
            if (i < len(self.text[k])) and self.text[k][i].strip():
                # Add underscore if necessary.
                if dname: dname += "_"
                # Add the label itself
                dname += (self.abbrv[k] + self.text[k][i])
        # Return the result.
        return dname
        
    # Function to write a JSON file with the trajectory variables.
    def WriteConditionsJSON(self, i, fname="conditions.json"):
        """Write a simple JSON file with exact trajectory variables
        
        :Call:
            >>> x.WriteConditionsJSON(i, fname="conditions.json")
        :Inputs:
            *i*: :class:`int`
                Index of the run case to print
            *fname*: :class:`str`
                Name of file to create
        :Versions:
            * 2014-11-18 ``@ddalle``: First version
        """
        # Open the file.
        f = open(fname, 'w')
        # Create the header.
        f.write('{\n')
        # Number of keys
        n = len(self.keys)
        # Loop through the keys.
        for j in range(n):
            # Name of the key.
            k = self.keys[j]
            # If it's a string, add quotes.
            if self.defns[k]["Value"] in ['str', 'char']:
                # Use quotes.
                q = '"'
            else:
                # No quotes.
                q = ''
            # Test if a comma is needed.
            if j >= n-1:
                # No comma for last line.
                c = ''
            else:
                # Yes, a comma is needed.
                c = ','
            # Get the value.
            v = getattr(self,k)[i]
            # Initial portion of line.
            line = ' "%s": %s%s%s%s\n' % (k, q, v, q, c)
            # Write the line.
            f.write(line)
        # Close out the JSON object.
        f.write('}\n')
        # Close the file.
        f.close()
# class Trajectory


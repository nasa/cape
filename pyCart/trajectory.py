"""
Cart3D run case list module: :mod:`pyCart.trajectory`
=====================================================

This module handles classes and methods that are specific to the list of run
cases (i.e., the "trajectory").
"""

# Basic numerics
import numpy as np
# More powerful text processing
import re
# File system and operating system management
import os


# Trajectory class
class Trajectory:
    """
    Read a simple list of configuration variables
    
    :Call:
        >>> x = pyCart.Trajectory(**traj)
        >>> x = pyCart.Trajectory(File=fname, Keys=keys)
    :Inputs:
        *traj*: :class:`dict`
            Dictionary of options from ``cart3d.Options["Trajectory"]``
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
        *x*: :class:`pyCart.trajectory.Trajectory`
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
        # Create the numeric versions.
        for key in keys:
            # Check the key type.
            if self.defns[key]['Value'] == 'float':
                # Normal numeric value
                setattr(self, key,
                    np.array([float(v) for v in self.text[key]]))
            elif self.defns[key]['Value'] == 'int':
                # Normal numeric value
                setattr(self, key,
                    np.array([int(v) for v in self.text[key]]))
            else:
                # Assume string
                setattr(self, key, np.array(self.text[key]))
        # Save the number of cases.
        self.nCase = len(self.text[key])
        # Process the groups (conditions in a group can use same grid).
        self.ProcessGroups()
        
    # Function to display things
    def __repr__(self):
        """
        Return the string representation of a trajectory.
        
        This looks like ``<pyCart.Trajectory(nCase=N, keys=['Mach','alpha'])>``
        """
        # Return a string.
        return '<pyCart.Trajectory(nCase=%i, keys=%s)>' % (self.nCase,
            self.keys)
        
    # Function to read a file
    def ReadTrajectoryFile(self, fname):
        """Read trajectory variable values from file
        
        :Call:
            >>> x.ReadTrajectoryFile(fname)
        :Inputs:
            *x*: :class:`pyCart.trajectory.Trajectory`
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
            *x*: :class:`pyCart.trajectory.Trajectory`
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
                    "Type": "Prefix",
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
            *x*: :class:`pyCart.trajectory.Trajectory`
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
            *T*: :class:`pyCart.trajectory.Trajectory`
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
            *x*: :class:`pyCart.trajectory.Trajectory`
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
            *x*: :class:`pyCart.trajectory.Trajectory`
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
            *x*: :class:`pyCart.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *cons*: :class:`list` (:class:`str`)
                List of constraints
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                Index of group(s) to process
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
        # Output.
        return np.where(i)[0]
        
    # Function to expand indices
    def ExpandIndices(self, itxt):
        """Expand string of subscripts into a list of indices
        
        :Call:
            >>> I = x.ExpandIndices(itxt)
        :Inputs:
            *x*: :class:`pyCart.trajectory.Trajectory`
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
        if type(itxt).__name__ not in ['str', 'unicode']:
            return []
        # Get the full list of indices.
        I0 = range(self.nCase)
        # Initialize output
        I = []
        # Split the input by semicolons.
        for i in itxt.split(';'):
            # Ignore []
            i = i.lstrip('[').rstrip(']')
            try:
                # Check for a ':'
                if ':' in i:
                    # Add a range.
                    I += eval('I0[%s]' % i)
                else:
                    # List
                    I += list(eval(i))
            except Exception:
                # Status update.
                print("Index specification '%s' failed to evaluate." % i)
        # Return the matches.
        return I
        
    # Get indices
    def GetIndices(self, **kw):
        """Get indices from either list or constraints or both
        
        :Call:
            >>> I = x.GetIndices()
            >>> I = x.GetIndices(I=I)
            >>> I = x.GetIndices(I=I, cons=cons)
            >>> I = x.GetIndices(cons=cons)
        :Inputs:
            *x*: :class:`pyCart.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *I*: :class:`numpy.ndarray` or :class:`list`
                Array of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                Array of indices
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Check for list.
        if kw.get("I") is not None:
            # Just a list, use it.
            I = np.array(kw['I'])
            # Check for constraints.
            if kw.get("cons") not in [None, []]:
                # Apply the constraints, too.
                return self.Filter(kw['cons'], I)
            else:
                # Return the directly-specified indices
                return I
        elif kw.get("cons") not in [None, []]:
            # Apply the constraints filter.
            return self.Filter(kw['cons'])
        else:
            # Return all the indices
            return np.arange(self.nCase)
        
        
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
            *x*: :class:`pyCart.trajectory.Trajectory`
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
            *x*: :class:`pyCart.trajectory.Trajectory`
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
        :Inputs:
            *x*: :class:`pyCart.trajectory.Trajectory`
                Instance of pyCart trajectory class
            *KeyType*: :class:`str`
                Key type to search for
        :Outputs:
            *keys*: :class:`numpy.ndarray`(:class:`str`)
                List of keys such that ``x[key]['Type']`` matches *KeyType*
        :Versions:
            * 2014-10-07 ``@ddalle``: First version
        """
        # List of key types
        KT = np.array([self.defns[k]['Type'] for k in self.keys])
        # Return matches
        return np.array(self.keys)[KT == KeyType]
        
    # Get keys by type of its value
    def GetKeysByValue(self, val):
        """Get all keys with specified type of value
        
        :Call:
            >>> keys = x.GetKeysByValue(val)
        :Inputs:
            *x*: :class:`pyCart.trajectory.Trajectory`
                Instance of pyCart trajectory class
            *val*: :class:`str`
                Key value class to search for
        :Outputs:
            *keys*: :class:`numpy.ndarray`(:class:`str`)
                List of keys such that ``x[key]['Value']`` matches *val*
        :Versions:
            * 2014-10-07 ``@ddalle``: First version
        """
        # List of key types
        KV = np.array([self.defns[k]['Value'] for k in self.keys])
        # Return matches
        return np.array(self.keys)[KV == val]
            
    # Function to assemble a folder name based on a list of keys and an index
    def _AssembleName(self, keys, prefix, i):
        """
        Assemble names using common code.
        
        :Call:
            >>> dname = x._AssembleName(keys, prefix, i)
        :Inputs:
            *x*: :class:`pyCart.trajectory.Trajectory`
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
        if "Prefix" in types:
            # Figure out which key it is
            j = types.index("Prefix")
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
            # Skip unentered values
            if (not self.text[k][i]): continue
            # Check for unlabeled values
            if (not self.defns[k].get("Label", True)): continue
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
            if self.text[k][i]:
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
    

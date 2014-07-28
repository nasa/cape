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
        >>> T = pyCart.Trajectory(**traj)
        >>> T = pyCart.Trajectory(File=fname, Keys=keys)
    
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
        *GridPrefix*: :class:`str`
            Prefix to be used for each grid folder name
        *Definitions*: :class:`dict`
            Dictionary of definitions for each key
    
    :Outputs:
        *T*: :class:`pyCart.trajectory.Trajectory`
            Instance of the trajectory class
            
    :Data members:
        *T.nCase*: :class:`int`
            Number of cases in the trajectory
        *T.prefix*: :class:`str`
            Prefix to be used in folder names for each case in trajectory
        *T.GridPrefix*: :class:`str`
            Prefix to be used for each grid folder name
        *T.keys*: :class:`list`, *dtype=str*
            List of variable names used
        *T.text*: :class:`dict`, *dtype=list*
            Lists of variable values taken from trajectory file
        *T.Mach*: :class:`numpy.ndarray`, *dtype=float*
            Vector of Mach numbers in trajectory
        ``getattr(T, key)``: :class:`numpy.ndarray`, *dtype=float*
            Vector of values of each variable specified in *keys*
    """
    
    # Initialization method
    def __init__(self, **kwargs):
        """Initialization method"""
        # Versions:
        #  2014.05.28 @ddalle  : First version
        #  2014.06.05 @ddalle  : Generalized for user-defined keys
        
        # Process the inputs.
        fname = kwargs.get('File', None)
        keys = kwargs.get('Keys', ['Mach', 'alpha', 'beta'])
        prefix = kwargs.get('Prefix', "F")
        gridPrefix = kwargs.get('GridPrefix', "Grid")
        # Process the definitions.
        defns = kwargs.get('Definitions', {})
        # Number of variables
        nVar = len(keys)
        # Save properties.
        self.keys = keys
        self.prefix = prefix
        self.GridPrefix = gridPrefix
        # Process the key definitions.
        self.ProcessKeyDefinitions(defns)
        
        # Open the file.
        f = open(fname)
        # Loop through the lines.
        for line in f.readlines():
            # Strip the line.
            line = line.strip()
            # Check for empty line or comment
            if line.startswith('#') or len(line)==0:
                continue
            # Separate by commas and/or white space
            v = re.split("[\s\,]+", line)
            # Check the number of entities.
            if len(v) != nVar: continue
            # Save the strings.
            for k in range(nVar):
                self.text[keys[k]].append(v[k])
        # Close the file.
        f.close()
        # Create the numeric versions.
        for key in keys:
            setattr(self, key, np.array([float(v) for v in self.text[key]]))
        # Save the number of cases.
        self.nCase = len(self.text[key])
        # Process the groups (conditions in a group can use same grid).
        self.ProcessGroups()
        # Output the conditions.
        return None
        
    # Function to display things
    def __repr__(self):
        """
        Return the string representation of a trajectory.
        
        This looks like ``<pyCart.Trajectory(nCase=N, keys=['Mach','alpha'])>``
        """
        # Return a string.
        return '<pyCart.Trajectory(nCase=%i, keys=%s)>' % (self.nCase,
            self.keys)
        
    # Function to process the role that each key name plays.
    def ProcessKeyDefinitions(self, defns):
        """
        Process definitions for the function of each trajectory variable
        
        Many variables have default definitions, such as ``'Mach'``,
        ``'alpha'``, etc.  For user-defined trajectory keywords, defaults will
        be used for aspects of the definition that are missing from the inputs.
        
        :Call:
            >>> T.ProcessKeyDefinitions(defns)
        
        :Inputs:
            *T*: :class:`pyCart.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *defns*: :class:`dict`
                Dictionary of keyword definitions or partial definitions
        
        :Effects:
            *T.text*: :class:`dict`
                Text for each variable and each break point is initialized
            *T.defns*: :class:`dict`
                Definition dictionary is created after processing defaults
            *T.abbrv*: :class:`dict`
                Dictionary of abbreviations for each trajectory key
        """
        # Versions:
        #  2014.06.05 @ddalle  : First version
        #  2014.06.17 @ddalle  : Overhauled to read from ``defns`` dict
        
        # Overall default key
        odefkey = defns.get('Default', {})
        # Process the mandatory fields.
        odefkey.setdefault('Group', True)
        odefkey.setdefault('Type', "Group")
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
                    "Abbreviation": "m"
                }
            elif key in ['Alpha', 'alpha', 'aoa']:
                # Angle of attack; non group
                defkey = {
                    "Group": False,
                    "Type": "alpha",
                    "Abbreviation": "a"
                }
            elif key in ['Beta', 'beta', 'aos']:
                # Sideslip angle; non group
                defkey = {
                    "Group": False,
                    "Type": "beta",
                    "Abbreviation": "b"
                }
            elif key.lower() in ['alpha_t', 'alpha_total']:
                # Total angle of attack; non group
                defkey = {
                    "Group": False,
                    "Type": "alpha_t",
                    "Abbreviation": "a"
                }
            elif key.lower() in ['phi', 'phiv']:
                # Total roll angle; non group
                defkey = {
                    "Group": False,
                    "Type": "phi",
                    "Abbreviation": "r"
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
            >>> T.ProcessGroups()
            
        :Inputs:
            *T*: :class:`pyCart.trajectory.Trajectory`
                Instance of the pyCart trajectory class
                
        :Effects:
            Creates fields that save the properties of the groups.  These fields
            are called *T.GroupKeys*, *T.GroupX*, *T.GroupID*.
        """
        # Versions:
        #  2014.06.05 @ddalle  : First version
        
        # Initialize matrix of group-generating key values.
        x = []
        # Initialize list of group variables.
        gk = []
        ngk = []
        # Loop through the keys to check for groups.
        for key in self.keys:
            # Check the definition for the grouping status.
            if self.defns[key]['Group']:
                # Append the values to the list.
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
            
        if the prefix is ``'F```, or
        
            ``m2.0a0.0b-0.5/``
            
        if the prefix is empty.
            
        Trajectory keys that require separate meshes for each value of the key
        will not be part of the folder name.  The number of digits used will
        match the number of digits in the trajectory file.
        
        :Call:
            >>> dname = T.GetFolderNames()
            >>> dname = T.GetFolderNames(i=None, prefix="F")
        
        :Inputs:
            *T*: :class:`pyCart.cntl.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
            *prefix*: :class:`str`
                Header for name of each folder
                
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        #  2014.06.05 @ddalle  : Refined to variables that can use same grid
        
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
    def GetGridFolderNames(self, i=None):
        """
        Get names of folders that require separate meshes
        
        :Call:
            >>> T.GetGridFolderNames()
            >>> T.GetGridFolderNames(i)
        
        :Inputs:
            *T*: :class:`pyCart.cntl.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
        
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        """
        # Versions:
        #  2014.06.05 @ddalle  : First version
        
        # Set the prefix.
        prefix = self.GridPrefix
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
            >>> dname = T.GetFullFolderNames()
            >>> dname = T.GetFullFolderNames(i=None, prefix="F")
        
        :Inputs:
            *T*: :class:`pyCart.cntl.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
            *prefix*: :class:`str`
                Header for name of each case folder
                
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        """
        # Versions:
        #  2014.06.05 @ddalle  : First version
        
        # Get the two components.
        glist = self.GetGridFolderNames(i)
        flist = self.GetFolderNames(i, prefix)
        # Check for list or not.
        if type(glist) is list:
            # Return the list of combined strings.
            return [os.path.join(glist[i],flist[i]) for i in range(len(glist))]
        else:
            # Just join the one.
            return os.path.join(glist, flist)
            
    # Function to assemble a folder name based on a list of keys and an index
    def _AssembleName(self, keys, prefix, i):
        """
        Assemble names using common code.
        
        :Call:
            >>> dname = T._AssembleName(keys, prefix, i)
            
        :Inptus:
            *T*: :class:`pyCart.cntl.Trajectory`
                Instance of the pyCart trajectory class
            *keys*: :type:`list` (:class:`str`)
                List of keys to use for this folder name
            *prefix*: :class:`str`
                Header for name of each case folder
            *i*: :class:`int` or :class:`list`
                Index(es) of case(s) to process; if ``None``, all cases
                
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Name containing value for each key in *keys*
        """
        # Versions:
        #  2014.06.05 @ddalle  : First version
        
        # Initialize folder name.
        if prefix and keys:
            # Use a prefix if it's any non-empty thing.
            dname = str(prefix) + "_"
        elif prefix:
            # The prefix is likely to be the whole name.
            dname = str(prefix)
        else:
            # Initialize an empty string.
            dname = ""
        # Append based on the keys.
        for k in keys:
            # Append the text in the trajectory file.
            dname += self.abbrv[k] + self.text[k][i]
        # Return the result.
        return dname
        
    # Function to make the directories
    def CreateFolders(self, prefix=None):
        """
        Make directories for each of the cases in a trajectory.
        
        The folder names will be of the form::
        
            ``F_m2.0a0.0b-0.5/``
            
        using the abbreviations for all of the keys specified in the trajectory
        file.  The amount of digits used will match the number of digits in the
        trajectory file.
        
        :Call:
            >>> T.CreateFolders()
            >>> T.CreateFolders(prefix="F")
        
        :Inputs:
            *T*: :class:`pyCart.cntl.Trajectory`
                Instance of the pyCart trajectory class
            *prefix*: :class:`str`
                Header for name of each folder
                
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.27 @ddalle  : First version
        
        # Process the prefix
        if prefix is None: prefix = self.prefix
        # Get the grid folder and case folder lists.
        glist = self.GetGridFolderNames()
        dlist = self.GetFolderNames(prefix=prefix)
        # Loop through the conditions.
        for i in range(len(dlist)):
            # Check if the "Grid" folder exists.
            if not os.path.isdir(glist[i]):
                # Create the folder, and say so.
                print("Creating common-grid folder: %s" % glist[i])
                os.mkdir(glist[i], 0750)
            # Join the "Grid" prefix.
            dname = os.path.join(glist[i], dlist[i])
            # Check if the folder exists.
            if not os.path.isdir(dname):
                # Create the folder, and say so.
                print("  Creating folder %i: %s." % (i+1, dname))
                os.mkdir(dname, 0750)
        return None
        
    # Method to write a file for a single condition
    def WriteConditionsFile(self, fname=None, i=0):
        """
        Write a JSON file containing the conditions for a single case.
        
        :Call:
            >>> T.WriteConditionsFile(fname, i)
        
        :Inputs:
            *T*: :class:`pyCart.cntl.Trajectory`
                Instance of the pyCart trajectory class
            *fname*: :class:`str`
                Name of JSON file to write
            *i*: :class:`int`
                Index of conditions to write
        
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Process default input file name.
        if fname is None:
            fname = os.path.join(self.GetFullFulderNames(i), "Conditions.json")
        # Create a conditions file.
        f = open(fname, 'w')
        # Write the header lines.
        f.write('{\n')
        f.write('    "Conditions: {\n')
        # Loop through the keys.
        for k in self.keys:
            # Write the value.
            f.write('        "%s": %.8f,\n' % (k, getattr(self,k)[i])) 
        # Write the case number.
        f.write('        "CaseNumber": %i\n' % (i+1))
        # Write the end matter.
        f.write('    }\n')
        f.write('}\n')
        f.close()
        
    # Method to write a file for a single group
    def WriteGridConditionsFile(self, fname=None, i=0):
        """
        Write a JSON file containing the collective conditions for a group
        
        :Call:
            >>> T.WriteGridConditionsFile(fname, i)
        
        :Inputs:
            *T*: :class:`pyCart.cntl.Trajectory`
                Instance of the pyCart trajectory class
            *fname*: :class:`str`
                Name of JSON file to write
            *i*: :class:`int`
                Index of group to write
        
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Process default input file name.
        if fname is None:
            # Get the unique groups.
            glist = np.unique(self.GetGridFolderNames())
            # Put the file in that folder.
            fname = os.path.join(glist[i], "Conditions.json")
        # Get the case index for the first case in the group.
        j = np.nonzero(self.GroupID == i)[0][0]
        # Create a conditions file.
        f = open(fname, 'w')
        # Write the header lines.
        f.write('{\n')
        f.write('    "Conditions: {\n')
        # Loop through the keys.
        for k in self.GroupKeys:
            # Write the value.
            f.write('        "%s": %.8f,\n' % (k, getattr(self,k)[j])) 
        # Write the case number.
        f.write('        "GroupNumber": %i\n' % (i+1))
        # Write the end matter.
        f.write('    }\n')
        f.write('}\n')
        f.close()
        
    

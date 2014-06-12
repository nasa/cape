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
        >>> T = pyCart.Trajectory(File=fname, Keys=keys, Prefix=prefix)
    
    :Inputs:
        *traj*: :class:`dict`
            Dictionary of options from ``cart3d.Options["Trajectory"]``
        *fname*: :class:`str`
            Name of file to read, defaults to ``'Trajectory.dat'``
        *keys*: :class:`list` of :class:`str` items
            List of variable names, defaults to ``['Mach','alpha','beta']``
        *prefix*: :class:`str`
            Prefix to be used in folder names for each case in trajectory
    
    :Outputs:
        *T*: :class:`pyCart.cntl.Trajectory`
            Instance of the trajectory class
            
    :Data members:
        *T.nCase*: :class:`int`
            Number of cases in the trajectory
        *T.prefix*: :class:`str`
            Prefix to be used in folder names for each case in trajectory
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
        # Number of variables
        nVar = len(keys)
        # Save properties.
        self.keys = keys
        self.prefix = prefix
        # Process the key definitions
        self.ProcessKeyDefinitions(**kwargs)
        
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
    def ProcessKeyDefinitions(self, **kwargs):
        """
        Process defaults for the function of each trajectory variable
        
        :Call:
            >>> T.ProcessKeyDefinitions(**kwargs)
        
        :Inputs:
            *T*: :class:`pyCart.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *kwargs*: :class:`dict`
                Keyword arguments passed to :mod:`pyCart.trajectory` constructor
        
        :Effects:
            Creates fields *T.text* and *T.defs* containing definitions for each
            variable.
        """
        # Versions:
        #  2014.06.05 @ddalle  : First version
        
        # Define the default new key.
        defkey = {
            "Values": [0.0],
            "Group": True,
            "Type": "Group"
        }
        # Actually, the default key properties can be specified, too.'
        defkey = kwargs.get('Default', defkey)
        # Initialize the dictionaries.
        self.text = {}
        self.defs = {}
        # Initialize the fields.
        for key in self.keys:
            # Initialize the text for this key.
            self.text[key] = []
            # Check for a variable definition.
            optkey = kwargs.get(key)
            # Process defaults.
            if type(optkey) is not dict:
                if key in ['Mach', 'M', 'mach']:
                    # Mach number; non group
                    optkey = {
                        "Values": optkey,
                        "Group": False,
                        "Type": "Mach",
                        "Abbreviation": "M"
                    }
                elif key in ['Alpha', 'alpha', 'aoa']:
                    # Angle of attack; non group
                    optkey = {
                        "Values": optkey,
                        "Group": False,
                        "Type": "alpha",
                        "Abbreviation": "a"
                    }
                elif key in ['Beta', 'beta', 'aos']:
                    # Sideslip angle; non group
                    optkey = {
                        "Values": optkey,
                        "Group": False,
                        "Type": "beta",
                        "Abbreviation": "b"
                    }
                elif key.lower() in ['alpha_t', 'alpha_total']:
                    # Total angle of attack; non group
                    optkey = {
                        "Values": optkey,
                        "Group": False,
                        "Type": "alpha_t",
                        "Abbreviation": "at"
                    }
                elif key in ['phi', 'Phi']:
                    # Total roll angle; non group
                    optkey = {
                        "Values": optkey,
                        "Group": False,
                        "Type": "phi",
                        "Abbreviation": "ph"
                    }
                elif optkey is None:
                    # Use defaults entirely.
                    optkey = defkey
                else:
                    # Unrecognized/undescribed variable
                    # Just values were given.  Use defaults
                    subkey = defkey
                    # Restore the values
                    subkey["Values"] = optkey
                    optkey = subkey
            # Save the definitions
            self.defs[key] = optkey
        
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
            if self.defs[key]['Group']:
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
    
            ``F_Mach_2.0_alpha_0.0_beta_-0.5/``
            
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
        prefix = "Grid"
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
    
            ``Grid/F_Mach_2.0_alpha_0.0_beta_-0.5/``
            
        if there are no trajectory keys that require separate grids or
        
            ``Grid_delta_1.0/F_Mach_2.0_alpha_0.0_beta_-0.5/``
            
        if there is a key called ``"delta"`` that requires a separate mesh each time
        the value of that key changes.  All keys in the trajectory file are included
        in the folder name at one of the two levels.  The number of digits used will
        match the number of digits in the trajectory file.
        
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
            *keys*: :type:`list`
                List of keys to use for this folder name
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
            *prefix*: :class:`str`
                Header for name of each case folder
                
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Name containing value for each key in *keys*
        """
        # Versions:
        #  2014.06.05 @ddalle  : First version
        
        # Initialize folder name.
        dname = prefix
        # Append based on the keys.
        for k in keys:
            # Append the text in the trajectory file.
            dname += "_" + k + "_" + self.text[k][i]
        # Return the result.
        return dname
        
    # Function to make the directories
    def CreateFolders(self, prefix=None):
        """
        Make directories for each of the cases in a trajectory.
        
        The folder names will be of the form::
        
            ``F_Mach_2.0_alpha_0.0_beta_-0.5/``
            
        using all of the keys specified in the trajectory file.  The amount of
        digits used will match the number of digits in the trajectory file.
        
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
                print("Creating common-grid folder: %s" % glis[i])
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
        
        # Process input file.
        if fname is None: fname = "Conditions.json"
        # Create a conditions file
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
        
        
    

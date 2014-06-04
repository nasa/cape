"""
Trajectory Module
=================

This module handles classes and methods that are specific to the list of run
cases (i.e., the trajectory)
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
    Class to handle trajectories in pyCart
    """
    
    # Initialization method
    def __init__(self, fname='Trajectory.dat', 
        keys=['Mach','alpha','beta'], prefix="F"):
        """
        Read a simple list of configuration variables
        
        :Call:
            >>> T = pyCart.Trajectory(fname)
            >>> T = pyCart.Trajectory(fname, keys, prefix)
        
        :Inputs:
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
            *nCase*: :class:`int`
                Number of cases in the trajectory
            *prefix*: :class:`str`
                Prefix to be used in folder names for each case in trajectory
            *keys*: :class:`list`, *dtype=str*
                List of variable names used
            *text*: :class:`dict`, *dtype=list*
                Lists of variable values taken from trajectory file
            *Mach*: :class:`numpy.ndarray`, *dtype=float*
                Vector of Mach numbers in trajectory
            *'key'*: :class:`numpy.ndarray`, *dtype=float*
                vector of values of each variable specified in *keys*
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Number of variables
        nVar = len(keys)
        # Initialize the dictionary.
        self.keys = keys
        self.text = {}
        # Initialize the fields.
        for key in keys:
            self.text[key] = []
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
        # Save the prefix.
        self.prefix = prefix
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
        
    # Function to list directory names
    def GetFolderNames(self, i=None, prefix=None):
        """
        List folder names for each of the cases in a trajectory.
        
        The folder names will be of the form
        
            "F_Mach_2.0_alpha_0.0_beta_-0.5/"
            
        using all of the keys specified in the trajectory file.  The amount of
        digits used will match the number of digits in the trajectory file.
        
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
        
        # Process the prefix.
        if prefix is None: prefix = self.prefix
        # Process the index list.
        if i is None: i = range(self.nCase)
        # Get the variable names.
        keys = self.keys
        # Check for a list.
        if np.isscalar(i):
            # Initialize the output.
            dlist = prefix
            # Append based on the keys.
            for k in keys:
                # Append the text in the trajectory file.
                dlist += "_" + k + "_" + self.text[k][i]
        else:
            # Initialize the list.
            dlist = []
            # Loop through the conditions.
            for j in i:
                # Initialize folder name.
                dname = prefix
                # Append based on the keys.
                for k in keys:
                    # Append the text in the trajectory file.
                    dname += "_" + k + "_" + self.text[k][j]
                # Append to the list.
                dlist.append(dname)
        return dlist
        
    # Function to make the directories
    def CreateFolders(self, prefix=None):
        """
        Make directories for each of the cases in a trajectory.
        
        The folder names will be of the form
        
            "F_Mach_2.0_alpha_0.0_beta_-0.5/"
            
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
        # Get the folder list.
        dlist = self.GetFolderNames(prefix=prefix)
        # Check if the "Grid" folder exists.
        if not os.path.isdir("Grid"):
            os.mkdir("Grid", 0750)
        # Loop through the conditions.
        for i in range(len(dlist)):
            # Join the "Grid" prefix.
            dname = os.path.join("Grid", dlist[i])
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
        
        
    

"""
Cart3D setup module: :mod:`pyCart.cntl`
=======================================

This module provides tools to quickly setup basic Cart3D runs from a small set
of input files.  Alternatively, the methods and classes can be used to help
setup a problem that is too complex or customized to conform to standardized
script libraries.
"""

# Basic numerics
import numpy as np
# Configuration file processor
import json
# File system and operating system management
import os, shutil
# More powerful text processing
import re


# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyCartFolder = os.path.split(_fname)[0]
TemplateFodler = os.path.join(PyCartFolder, "templates")

# Change the umask to a reasonable value.
os.umask(0027)


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
    def _GetFolderNames(self, prefix=None):
        """
        List folder names for each of the cases in a trajectory.
        
        The folder names will be of the form
        
            "F_Mach_2.0_alpha_0.0_beta_-0.5/"
            
        using all of the keys specified in the trajectory file.  The amount of
        digits used will match the number of digits in the trajectory file.
        
        :Call:
            >>> dname = T._GetFolderNames()
            >>> dname = T._GetFolderNames(prefix="F")
        
        :Inputs:
            *T*: :class:`pyCart.cntl.Trajectory`
                Instance of the pyCart trajectory class
            *prefix*: :class:`str`
                Header for name of each folder
                
        :Outputs:
            *dname*: :class:`list`, *dtype=str*
                List of folder names
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Process the prefix
        if prefix is None: prefix = self.prefix
        # Get the variable names.
        keys = self.keys
        # Initialize the list.
        dlist = []
        # Loop through the conditions.
        for i in range(self.nCase):
            # Initialize folder name.
            dname = prefix
            # Append based on the keys.
            for k in keys:
                # Append the text in the trajectory file.
                dname += "_" + k + "_" + self.text[k][i]
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
        dlist = self._GetFolderNames(prefix)
        # Check if the "Grid" folder exists.
        if not os.path.isdir("Grid"):
            os.mkdir("Grid", 0750)
        # Loop through the conditions.
        for i in range(len(dlist)):
            # Join the "Grid" prefix.
            dname = os.path.join("Grid", dlist[i])
            # Print the name of the folder.
            print("Condition name %i: %s" % (i+1, dname))
            # Check if the folder exists.
            if os.path.isdir(dname):
                # Say so
                print("  Folder exists!")
            else:
                # Create the folder, and say so.
                print("  Creating folder.")
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
        


# Function to read conditions file.
def ReadTrajectoryFile(fname='Trajectory.dat', keys=['Mach','alpha','beta'],
    prefix="F"):
    """
    Read a simple list of configuration variables
    
    :Call:
        >>> T = pyCart.ReadTrajectoryFile(fname)
        >>> T = pyCart.ReadTrajectoryFile(fname, keys)
    
    :Inputs:
        *fname*: :class:`str`
            Name of file to read, defaults to ``'Trajectory.dat'``
        *keys*: :class:`list` of :class:`str` items
            List of variable names, defaults to ``['Mach','alpha','beta']``
        *prefix*: :class:`str`
            Header for name of each folder
    
    :Outputs:
        *T*: :class:`pyCart.cntl.Trajectory`
            Instance of the pyCart trajectory class
    
    """
    # Versions:
    # 2014.05.27 @ddalle  : First version
    return Trajectory(fname, keys, prefix)
    
    
# Function to make the directories
def CreateFolders(T, prefix="F"):
    """
    Make directories for each of the cases in a trajectory.
    
    The folder names will be of the form
    
        "F_Mach_2.0_alpha_0.0_beta_-0.5/"
        
    using all of the keys specified in the trajectory file.  The amount of
    digits used will match the number of digits in the trajectory file.
    
    :Call:
        >>> pyCart.CreateFolders(T, prefix="F")
    
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
    T.CreateFolders(prefix)
    return None
    
    
    
# Class to read input files
class Cntl:
    """
    Class for handling global options to be used for Cart3D cases
    """
    
    # Initialization method
    def __init__(self, fname="pyCart.json"):
        """
        Class for handling global options for Cart3D run cases.
        
        This class is intended to handle all settings used to describe a group
        of Cart3D cases.  For situations where it is not sufficiently
        customized, it can be used partially, e.g., to set up a Mach/alpha sweep
        for each single control variable setting.
        
        The settings are read from a JSON file, which is robust and simple to
        read, but has the disadvantage that there is no support for comments.
        Hopefully the various names are descriptive enough not to require
        explanation.
        
        Defaults are read from the file "$PYCART/settings/pyCart.default.json".
        
        :Call:
            >>> cntl = pyCart.Cntl(fname="pyCart.json")
            
        :Inputs:
            *fname*: :class:`str`
                Name of pyCart input file
                
        :Outputs:
            *cntl*: :class:`pyCart.cntl.Cntl`
                Instance of the pyCart control class
        
        :Data members:
            *Grid*: :class:`dict`
                Dictionary containing grid-related parameters
            *RunOptions*: :class:`dict`
                Dictionary containing run-related parameters
            *Trajectory*: :class:`pyCart.cntl.Trajectory`
                Trajectory description read from file
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Read the default input file.
        f = open(os.path.join(PyCartFolder, 
            "..", "settings", "pyCart.default.json"))
        lines = f.read()
        f.close()
        # Process the default input file.
        defs = json.loads(lines)
        
        # Read the specified input file.
        f = open(fname)
        lines = f.read()
        f.close()
        # Process the actual input file.
        opts = json.loads(lines)
        
        # Initialize the grid parameters.
        self.Grid = opts.get('Grid', {})
        # Process the defaults.
        for k in defs['Grid'].keys():
            # Use the setdefault() method.
            self.Grid.setdefault(k, defs['Grid'][k])
        
        # Initialize the run parameters. 
        self.RunOptions = opts.get('RunOptions', {})
        # Process the defaults.
        for k in defs['RunOptions'].keys():
            # Use the setdefault() method.
            self.RunOptions.setdefault(k, defs['RunOptions'][k])
        
        # Process the trajectory dict if it exists.
        oTraj = opts.get('Trajectory', {})
        # Get the name of the trajectory file.
        tfile = oTraj.get('File', defs['Trajectory']['File'])
        # Get the key (variable) names.
        tkeys = oTraj.get('Keys', defs['Trajectory']['Keys'])
        # Get the prefix.
        tpre = oTraj.get('Keys', defs['Trajectory']['Prefix'])
        # Read the trajectory file.
        self.Trajectory = Trajectory(tfile, tkeys)
        
        
    # Output representation
    def __repr__(self):
        """
        Output representation for the class.
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Display basic information from all three areas.
        return "<pyCart.Cntl(nCase=%i, nIter=%i, tri='%s'>" % (
            self.Trajectory.nCase, self.RunOptions['nIter'],
            self.Grid['TriFile'])
        
        
    # Method to create the folders
    def CreateFolders(self):
        """
        Create the folders based on the trajectory.
        
        :Call:
            >>> cntl.CreateFolders()
        
        :Inputs:
            *cntl*: :class:`pyCart.cntl.Cntl`
                Global pyCart settings object instance
        
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Use the trajectory method.
        self.Trajectory.CreateFolders()
        return None
        
        
    # Method to set up the grid
    def CreateMesh(self):
        """ 
        Create the common mesh based on self-contained parameters.
        
        :Call:
            >>> cntl.CreateMesh()
        
        :Inputs:
            *cntl*: :class:`pyCart.cntl.Cntl`
                Instance of control class containing relevant parameters
        
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Check if the "Grid" folder exists.
        if not os.path.isdir("Grid"):
            os.mkdir("Grid", 750)
        # Extract the grid parameters.
        Grid = self.Grid
        # Get the name of the tri file(s).
        ftri = os.path.split(Grid['TriFile'])[-1]
        # Copy the tri file there if necessary.
        shutil.copyfile(Grid['TriFile'], os.path.join('Grid', ftri))
        # Get the component list.
        fxml = Grid['ComponentFile']
        if os.path.isfile(dxml):
            # Copy
            shutil.copyfile(fxml, os.path.join('Grid', 'Config.xml'))
        # Change to the Grid folder.
        os.chdir('Grid')
        # Start by running autoInputs
        cmd = 'autoInputs -r %i -t %s' % (Grid['MeshRadius'], ftri)
        os.system(cmd)
        # Create a "Components.i.tri" file if necessary
        if ftri != "Components.i.tri":
            os.system('ln -sf %s Components.i.tri' % ftri)
        # Run cubes
        cmd = 'cubes -maxR %i -pre preSpec.c3d.cntl -reorder' % \
            Grid['nRefinements']
        os.system(cmd)
        # Multigrid setup
        cmd = 'mgPrep -n %i' % Grid['nMultiGrid']
        os.system(cmd)
        # Return to previous folder.
        os.chdir('..')
        # End.
        return None
        
        
    # Function to copy/link files
    def CopyFiles(self):
        """
        Copy or link the relevant files to the Grid folders.
        
        :Call:
            >>> cntl.CopyFiles()
            
        :Inputs:
            *cntl*: :class:`pyCart.cntl.Cntl`
                Instance of control class containing relevant parameters
        
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Check the 'Grid' directory.
        if not os.path.isdir('Grid'):
            raise IOError('Folder "Grid" not found.')
        # Get the folder names.
        dlist = self.Trajectory._GetFolderNames()
        # Change to the 'Grid' folder.
        os.chdir('Grid')
        # Check if the grid has been created.
        if not os.path.isfile('Mesh.R.c3d'):
            raise IOError('It appears the mesh has not been created.')
        # Convenient storage of 'Grid' plus filesep
        fg = 'Grid' + os.sep
        # List of files to copy
        f_copy = ['input.c3d', 'Config.xml',
            'Mesh.c3d.Info', 'preSpec.c3d.cntl']
        # List of files to link
        f_link = ['Components.i.tri', 'Mesh.R.c3d', 'Mesh.mg.c3d']
        # Loop through the case folders.
        for i in range(len(dlist)):
            # Get the folder name.
            d = dlist[i]
            # Check status.
            print("Case %i: %s" % (i+1, d))
            if not os.path.isdir(d):
                # Does not exist, skipping
                print("  Does not exist, skipping")
                continue
            # Folder exists
            print("  Copying files")
            # Loop through files to copy.
            for f in f_copy:
                # Check the file
                if os.path.isfile(d + os.sep + f):
                    # File exists
                    print("    File '%s' exists!" % f)
                    continue
                elif not os.path.isfile(f):
                    # File missing
                    print("    File '%s' is missing." % f)
                    continue
                # File does not exist.
                print("    Copying file '%s'." % f)
                shutil.copyfile(f, os.path.join(d,f))
            # Change to the folder.
            os.chdir(d)
            # Loop through files to link.
            for f in f_link:
                # Check the file
                if os.path.isfile(f):
                    # File exists
                    print("    File '%s' exists!" % f)
                    continue
                elif not os.path.isfile(os.path.join('..',f)):
                    # File missing
                    print("    File '%s' is missing." % f)
                    continue
                # File does not exist.
                print("    Linking file '%s'." % f)
                os.system('ln -sf %s %s' % (os.path.join('..',f), f))
            # Return to previous folder.
            os.chdir('..')
        # Return to previous folder.
        os.chdir('..')
        # Done
        return None
        
    # Function to setup the run cases.
    def PrepareRuns(self):
        """
        Create run scripts for each case according to the pyCart settings.
        
        :Call:
            >>> cntl.PrepareRuns()
        
        :Inputs:
            *cntl*: :class:`pyCart.cntl.Cntl`
                Instance of control class containing relevant parameters
        
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Check the 'Grid directory.
        if not os.path.isdir('Grid'):
            raise IOError('Folder "Grid" not found.')
        # Get the name of the .cntl file.
        fname = self.RunOptions['CntlFile']
        # Check if it exists.
        if not os.path.isfile(fname):
            raise IOError('Input CNTL File "%s" not found.' % fname)
        # Copy the file to the grid folder.
        shutil.copyfile(fname, os.path.join('Grid', 'input.cntl'))
        # Move to the Grid folder.
        os.chdir('Grid')
        # Extract the trajectory.
        T = self.Trajectory
        # Read the input.cntl file.
        lines = open('input.cntl').readlines()
        # Get the folder names
        dlist = T._GetFolderNames()
        # Create a file to run all the cases.
        fname_sh = 'run_cases.sh'
        fname_i = 'run_case.sh'
        fsh = open(fname_sh, 'w')
        # Print the first-line magic
        fsh.write('#!/bin/bash\n\n')
        # Extract the options.
        opts = self.RunOptions
        # Loop through the conditions.
        for i in range(len(dlist)):
            # Print a status update
            print("Preparing case %i: Mach=%.2f, alpha=%-.2f, beta=%-.2f" 
                % (i, T.Mach[i], T.alpha[i], T.beta[i]))
            # Create the input file.
            f = open(os.path.join(dlist[i], 'input.cntl'), 'w')
            # Loop through the lines of the input.cntl template.
            for line in lines:
                # Replace placeholders
                line = line.replace('Mach_TMP',  '%.8f'%T.Mach[i])
                line = line.replace('alpha_TMP', '%.8f'%T.alpha[i])
                line = line.replace('beta_TMP',  '%.8f'%T.beta[i])
                # Write the line.
                f.write(line)
            # Close the input.cntl file.
            f.close()
            # Create a conditions file
            T.WriteConditionsFile(os.path.join(dlist[i], 'Conditions.json'), i)
            # Create the run script.
            f = open(os.path.join(dlist[i], fname_i), 'w')
            f.write('#!/bin/bash\n\n')
            # Set the number of processors.
            f.write('export OMP_NUM_THREADS=%i\n' % opts['nThreads'])
            # Create the command to do the work.
            f.write('flowCart -N %i -v -mg %i -his -clic\n' %
                (opts['nIter'], self.Grid['nMultiGrid']))
            # Close the file.
            f.close()
            # Make it executable.
            os.chmod(os.path.join(dlist[i], fname_i), 0750)
            # Append to the global script.
            fsh.write('# Case %i\n' % i)
            fsh.write('cd %s\n' % dlist[i])
            fsh.write('./%s\n' % fname_i)
            fsh.write('cd ..\n\n')
        # Close the global script file.
        fsh.close()
        # Make it executable
        os.chmod(fname_sh, 0750)
        # Change back to original folder.
        os.chdir('..')
        # End
        return None
        
    

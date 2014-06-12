"""
Cart3D setup module: :mod:`pyCart.cart3d`
=========================================

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

# Import the trajectory class
from trajectory import Trajectory
# Read "loadsXX.dat" files
from post import LoadsDat

# Import specific file control classes
from InputCntl import InputCntl
from AeroCsh   import AeroCsh


# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyCartFolder = os.path.split(_fname)[0]
TemplateFodler = os.path.join(PyCartFolder, "templates")

# Change the umask to a reasonable value.
os.umask(0027)



# Function to get the defaults
def _getPyCartDefaults():
    """
    Get default pyCart JSON settings.  To change default pyCart settings, edit
    the 'pyCart.default.json' file in the settings directory.
    """
    # Versions:
    #  2014.06.03 @ddalle  : First version
    
    # Read the default input file.
    lines = open(os.path.join(PyCartFolder, 
            "..", "settings", "pyCart.default.json")).readlines()
    # Strip comments and join list into a single string.
    lines = stripComments(lines, '#')
    # Process the default input file.
    return json.loads(lines)
    
    
# Class to read input files
class Cart3d:
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
    
    Defaults are read from the file ``$PYCART/settings/pyCart.default.json``.
    
    :Call:
        >>> cart3d = pyCart.Cart3d(fname="pyCart.json")
        
    :Inputs:
        *fname*: :class:`str`
            Name of pyCart input file
            
    :Outputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Instance of the pyCart control class
    
    :Data members:
        *Grid*: :class:`dict`
            Dictionary containing grid-related parameters
        *RunOptions*: :class:`dict`
            Dictionary containing run-related parameters
        *Trajectory*: :class:`pyCart.trajectory.Trajectory`
            Trajectory description read from file
    """
    
    # Initialization method
    def __init__(self, fname="pyCart.json"):
        """Initialization method for :mod:`pyCart.cart3d.Cart3d`"""
        # Versions:
        #  2014.05.28 @ddalle  : First version
        #  2014.06.03 @ddalle  : Renamed class 'Cntl' --> 'Cart3d'
        
        # Process the default input file.
        defs = _getPyCartDefaults()
        
        # Read the specified input file.
        lines = open(fname).readlines()
        # Strip comments.
        lines = stripComments(lines, '#')
        # Process the actual input file.
        opts = json.loads(lines)
        
        # Loop through the keys in the defaults.
        for k in defs.keys():
            # Set the defaults.
            opts.setdefault(k, defs[k])
            # Check for a dict
            if not(type(defs[k]) is dict and type(opts[k]) is dict):
                continue
            # Loop though the sub-keys
            for j in defs[k].keys():
                # Set the defaults.
                opts[k].setdefault(j, defs[k][j])
        
        # Save the major keys.
        self.RunOptions = opts["RunOptions"]
        self.Mesh = opts["Mesh"]
        self.Config = opts["Config"]
        
        # Process the trajectory.
        self.Trajectory = Trajectory(**opts['Trajectory'])
        
        # Save all the options as a reference.
        self.Options = opts        
        
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Display basic information from all three areas.
        return "<pyCart.Cart3d(nCase=%i, nIter=%i, tri='%s')>" % (
            self.Trajectory.nCase, self.RunOptions['nIter'],
            self.Mesh['TriFile'])
        
    # Trajectory's folder name method
    def GetFolderNames(self, i=None, prefix=None):
        """
        List folder names for each of the cases in a trajectory.
        
        The folder names will be of the form
    
            ``Grid/F_Mach_2.0_alpha_0.0_beta_-0.5/``
            
        if there are no trajectory keys that require separate grids or
        
            ``Grid_delta_1.0/F_Mach_2.0_alpha_0.0_beta_-0.5/``
            
        if there is a key called ``"delta"`` that requires a separate mesh each time
        the value of that key changes.  All keys in the trajectory file are included
        in the folder name at one of the two levels.  The number of digits used will
        match the number of digits in the trajectory file.
        
        :Call:
            >>> dname = cart3d.GetFolderNames()
            >>> dname = cart3d.GetFolderNames(i=None, prefix="F")
        
        :Inputs:
            *cart3d*: :class:`pyCart.trajectory.Trajectory`
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
        #  2014.06.02 @ddalle  : First version
        
        # Run the trajectory's method
        return self.Trajectory.GetFolderNames(i, prefix)
    
    # Method to create the folders
    def CreateFolders(self):
        """
        Create the folders based on the trajectory.
        
        :Call:
            >>> cart3d.CreateFolders()
        
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
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
            >>> cart3d.CreateMesh()
        
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        #  2014.06.06 @ddalle  : Multiple grid folders
        
        # Extract the grid parameters.
        Mesh = self.Mesh
        # Get the name of the tri file(s).
        ftri = os.path.split(Mesh['TriFile'])[-1]
        # Get the names of the grid foloders
        glist = self.Trajectory.GetGridFolderNames()
        # Loop through the grids.
        for g in np.unique(glist):
            # Copy the tri file there if necessary.
            shutil.copyfile(Mesh['TriFile'], 
                os.path.join(g, 'Components.i.tri'))
            # Get the component list.
            fxml = self.Config['File']
            if os.path.isfile(fxml):
                # Copy
                shutil.copyfile(fxml, os.path.join(g, 'Config.xml'))
            # Change to the Grid folder.
            os.chdir(g)
            # Start by running autoInputs.
            if Mesh['AutoInputs']:
                cmd = 'autoInputs -r %i -t Components.i.tri' % Mesh['MeshRadius']
                os.system(cmd)
            # Run cubes
            cmd = 'cubes -maxR %i -pre preSpec.c3d.cntl -reorder' % \
                Mesh['nRefinements']
            os.system(cmd)
            # Multigrid setup
            cmd = 'mgPrep -n %i' % Mesh['nMultiGrid']
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
            >>> cart3d.CopyFiles()
            
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        
        # Check the 'Grid' directory.
        if not os.path.isdir('Grid'):
            raise IOError('Folder "Grid" not found.')
        # Name of tri file
        ftri = os.path.split(self.Mesh['TriFile'])[-1]
        # Get the folder names.
        glist = self.Trajectory.GetGridFolderNames()
        dlist = self.Trajectory.GetFolderNames()
        # List of files to copy
        f_copy = ['input.c3d', 'Config.xml',
            'Mesh.c3d.Info', 'preSpec.c3d.cntl']
        # List of files to link
        f_link = ['Components.i.tri', 'Mesh.R.c3d']
        # Check if a link will break things.
        if self.RunOptions["nAdapt"] > 0:
            # Adapt: copy the mesh.
            f_copy.append('Mesh.mg.c3d')
        else:
            # No adaptations: link the mesh.
            f_link.append('Mesh.mg.c3d')
        # Loop through the cases.
        for i in range(len(dlist)):
            # Extract folders.
            g = glist[i]
            d = dlist[i]
            # Change to the 'Grid' folder.
            os.chdir(g)
            # Check if the grid has been created.
            if not os.path.isfile('Mesh.R.c3d'):
                raise IOError('It appears the mesh has not been created.')
            # Convenient storage of 'Grid' plus filesep
            fg = g + os.sep
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
            # Return to root folder.
            os.chdir('..' + os.sep + '..')
        # Done
        return None
        
    # Function to setup the run cases.
    def PrepareRuns(self):
        """
        Create run scripts for each case according to the pyCart settings.
        
        :Call:
            >>> cart3d.PrepareRuns()
        
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.28 @ddalle  : First version
        #  2014.05.30 @ddalle  : Moved input.cntl filtering to separate func
        #  2014.06.04 @ddalle  : Moved input.cntl handling entirely to func
        
        # Prepare the "input.cntl" files.
        self.PrepareInputCntl()
        # Prepare the "aero.csh" files
        self.PrepareAeroCsh()
        # Prepare the run scripts.
        self.CreateRunScripts()
        # Get the trajectory.
        T = self.Trajectory
        # Get the folder names.
        dlist = T.GetFullFolderNames()
        # Loop through the conditions.
        for i in range(len(dlist)):
            # Get the folder name for this case.
            d = dlist[i]
            # Create a conditions file
            T.WriteConditionsFile(os.path.join(d, 'Conditions.json'), i)
        # End
        return None
        
    # Function to create run scripts
    def CreateRunScripts(self):
        """
        Create all run scripts
        
        :Call:
            >>> cart3d.CreateRunScripts()
            
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        #  2014.06.06 @ddalle  : Added support for multiple grid folders
        
        # Global script name
        fname_all = 'run_all.sh'
        # Grid script name
        fname_grid = 'run_cases.sh'
        # Local script name.
        fname_i = 'run_case.sh'
        # Get the grid folder names
        glist = self.Trajectory.GetGridFolderNames()
        # Create the global run script.
        fa = open(fname_all, 'w')
        # Print the first-line magic
        fa.write('#!/bin/bash\n\n')
        # Loop through the grid folder names.
        for g in glist:
            # As of now there's only one grid; run it and return.
            fa.write('cd %s\n' % g)
            fa.write('./%s\n' % fname_grid)
            fa.write('cd ..\n')
            # Create the grid-level run script.
            fg = open(os.path.join(g, fname_grid), 'w')
            # Initialize it.
            fg.write('#!/bin/bash\n\n')
            # Close it for now.
            fg.close()
            # Make it executable now.
            os.chmod(os.path.join(g, fname_grid), 0750)
        # Close the global script.
        fa.close()
        # Make the script executable.
        os.chmod(fname_all, 0750)
        # Get the folder names.
        dlist = self.Trajectory.GetFolderNames()
        # Loop through the folders.
        for i in range(len(dlist)):
            # Change to the appropriate grid folder.
            os.chdir(glist[i])
            # Open the (existing) run script for appending.
            fg = open(fname_grid, 'a')
            # Append to the grid script.
            fg.write('# Case %i\n' % i)
            fg.write('cd %s\n' % dlist[i])
            fg.write('./%s\n' % fname_i)
            fg.write('cd ..\n\n')
            # Close it (temporarily).
            fg.close()
            # Go back to the main folder.
            os.chdir('..')
            # Create the local run script.
            self.CreateCaseRunScript(i)
        # Done
        return None
        
    # Function to prepare "input.cntl" files
    def PrepareInputCntl(self):
        """
        Read template "input.cntl" file, customize it for each case, and write
        it to trajectory folders.
        
        :Call:
            >>> cart3d.PrepareInputCntl()
        
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
                
        :Effects:
            *  Reads 'input.cntl' file from the destination specified in
               *cart3d.RunOptions* and copies it to each case folder after
               processing appropriate options.
               
            *  Creates *cart3d.InputCntl* data member
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        #  2014.06.06 @ddalle  : Low-level functionality for grid folders
        
        # Get the name of the .cntl file.
        fname = self.Options['InputCntl']
        # Read it.
        self.InputCntl = InputCntl(fname)
        # Process global options...
        self.InputCntl.SetCFL(self.RunOptions['CFL'])
        # Extract the trajectory.
        T = self.Trajectory
        # Get grid folders.
        glist = T.GetGridFolderNames()
        # Write to each "Grid" folder.
        for g in glist:
            self.InputCntl.Write(os.path.join(g, 'input.cntl'))
        # Get the folder names.
        dlist = T.GetFolderNames()
        # Loop through the conditions.
        for i in range(len(dlist)):
            # Print a status update
            print("  Preparing 'input.cntl' for case %i" % i)
            # Set the flight conditions.
            self.InputCntl.SetMach(T.Mach[i])
            self.InputCntl.SetAlpha(T.alpha[i])
            self.InputCntl.SetBeta(T.beta[i])
            # Destination file name
            fout = os.path.join(glist[i], dlist[i], 'input.cntl')
            # Write the input file.
            self.InputCntl.Write(fout)
        # Done
        return None
        
    # Function prepare the aero.csh files
    def PrepareAeroCsh(self):
        """
        Read template "aero.csh" file, customize it according to pyCart
        settings, and write it to trajectory folders.
        
        :Call:
            >>>car3d.PrepareAeroCsh()
        
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        
        :Effects:
            * Read 'aero.csh' file from the destination specified in
              *cart3d.RunOptions* and copies it to each case folder after
              processing appropriate options.
            
            * Creates *cart3d.AeroCsh* data member
        """
        # Versions:
        #  2014.06.10 @ddalle  : First version
        
        # Get the name of the file.
        fname = self.Options['AeroCsh']
        # Check for the file.
        if not os.path.isfile(fname): return None
        # Read it.
        self.AeroCsh = AeroCsh(fname)
        # Extract run options
        opts = self.RunOptions
        # Process global options
        self.AeroCsh.SetCFL(opts['CFL'])
        self.AeroCsh.SetnIter(opts['nIter'])
        self.AeroCsh.SetnAdapt(opts['nAdapt'])
        self.AeroCsh.SetnRefinements(self.Mesh['nRefinements'])
        self.AeroCsh.SetnMultiGrid(self.Mesh['nMultiGrid'])
        # Extract the trajectory.
        T = self.Trajectory
        # Get grid folders.
        glist = T.GetGridFolderNames()
        # Write to each "Grid" folder.
        for g in glist:
            self.AeroCsh.Write(os.path.join(g, 'aero.csh'))
        # Get the folder names.
        dlist = T.GetFolderNames()
        # Loop through the conditions.
        for i in range(len(dlist)):
            # Destination file name
            fout = os.path.join(glist[i], dlist[i], 'aero.csh')
            # Write the input file.
            self.AeroCsh.Write(fout)
        # Done
        return None
        
    # Function to create the flowCart run script
    def CreateCaseRunScript(self, i):
        """
        Write the "run_case.sh" script to run a given case.
        
        :Call:
            >>> cart3d.CreateCaseRunScript(i)
        
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *i*: :class:`int`
                Trajectory case number
                
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.30 @ddalle  : First version
        
        # Get the folder name.
        dname = self.Trajectory.GetFullFolderNames(i=i)
        # File name
        fout = os.path.join(dname, 'run_case.sh')
        # Create the file.
        f = open(fout, 'w')
        # Get the options.
        opts = self.RunOptions
        # Write the shell magic.
        f.write('#!/bin/bash\n\n')
        # Set the number of processors.
        f.write('export OMP_NUM_THREADS=%i\n\n' % opts['nThreads'])
        # Check for an adaptive case.
        if opts['nAdapt'] > 0:
            # Create the aero.csh command to do the work.
            f.write('./aero.csh jumpstart | tee aero.out\n')
        else:
            # Create the flowCart command to do the work.
            f.write('flowCart -N %i -v -mg %i -his -clic | tee flowCart.out\n' %
                (opts['nIter'], self.Mesh['nMultiGrid']))
        # Close the file.
        f.close()
        # Make it executable.
        os.chmod(fout, 0750)
        
    # Function to read "loadsCC.dat" files
    def GetLoadsCC(self):
        """
        Read all available 'loadsCC.dat' files.
        
        :Call:
            >>> cart3d.GetLoadsCC()
            
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
                
        :Effects:
            Creates *cart3d.LoadsCC* instance
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Call the constructor.
        self.LoadsCC = LoadsDat(self, fname="loadsCC.dat")
        return None
        
    # Function to write "loadsCC.csv"
    def WriteLoadsCC(self):
        """
        Write gathered loads to CSV file to "loadsCC.csv"
        
        :Call:
            >>> cart3d.WriteLoadsCC()
            
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Check for the attribute.
        if not hasattr(self, 'LoadsCC'):
            self.GetLoadsCC()
        # Write.
        self.LoadsCC.Write(self.Trajectory)
        return None
        
    # Function to read "loadsCC.dat" files
    def GetLoadsTRI(self):
        """
        Read all available 'loadsTRI.dat' files.
        
        :Call:
            >>> cart3d.GetLoadsTRI()
            
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
                
        :Effects:
            Creates *cart3d.LoadsCC* instance
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Call the constructor.
        self.LoadsTRI = LoadsDat(self, fname="loadsTRI.dat")
        return None
        
    # Function to write "loadsCC.csv"
    def WriteLoadsTRI(self):
        """
        Write gathered loads to CSV file to "loadsTRI.csv"
        
        :Call:
            >>> cart3d.WriteLoadsTRI()
            
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Check for the attribute.
        if not hasattr(self, 'LoadsTRI'):
            self.GetLoadsTRI()
        # Write.
        self.LoadsTRI.Write(self.Trajectory)
        return None
        
        
# Function to delete comment lines.
def stripComments(lines, char='#'):
    """
    Delete lines that begin with a certain comment character.
    
    :Call:
        >>> txt = stripComments(lines, char='#')
    
    :Inputs:
        *lines*: :class:`list` of :class:`str`
            List of lines
        *char*: :class:`str`
            String that represents start of a comment
        
    :Outputs:
        *txt*: :class:`str`
            Lines joined into a single string but with comments removed
    """
    # Versions:
    #  2014.06.03 @ddalle  : First version
    
    # Start with the first line.
    i = 0
    # Loop until last line
    while i < len(lines):
        # Get the line and strip leading and trailing white space.
        line = lines[i].strip()
        # Check it.
        if line.startswith(char):
            # Remove it.
            lines.__delitem__(i)
        else:
            # Move to the next line.
            i += 1
    # Return the remaining lines.
    return "".join(lines)
    


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
        *T*: :class:`pyCart.trajectory.Trajectory`
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
    
        ``Grid/F_Mach_2.0_alpha_0.0_beta_-0.5/``
        
    if there are no trajectory keys that require separate grids or
    
        ``Grid_delta_1.0/F_Mach_2.0_alpha_0.0_beta_-0.5/``
        
    if there is a key called ``"delta"`` that requires a separate mesh each time
    the value of that key changes.  All keys in the trajectory file are included
    in the folder name at one of the two levels.  The number of digits used will
    match the number of digits in the trajectory file.
    
    :Call:
        >>> pyCart.CreateFolders(T, prefix="F")
    
    :Inputs:
        *T*: :class:`pyCart.trajectory.Trajectory`
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

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

# Import the trajectory class
from trajectory import Trajectory


# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyCartFolder = os.path.split(_fname)[0]
TemplateFodler = os.path.join(PyCartFolder, "templates")

# Change the umask to a reasonable value.
os.umask(0027)



        


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
        # Save all the options.
        self.JSON = opts
        
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
        
    # Trajectory's folder name method
    def GetFolderNames(self, i=None, prefix=None):
        """
        List folder names for each of the cases in a trajectory.
        
        The folder names will be of the form
        
            "F_Mach_2.0_alpha_0.0_beta_-0.5/"
            
        using all of the keys specified in the trajectory file.  The amount of
        digits used will match the number of digits in the trajectory file.
        
        :Call:
            >>> dname = cntl.GetFolderNames()
            >>> dname = cntl.GetFolderNames(i=None, prefix="F")
        
        :Inputs:
            *cntl*: :class:`pyCart.cntl.Trajectory`
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
            os.mkdir("Grid", 0750)
        # Extract the grid parameters.
        Grid = self.Grid
        # Get the name of the tri file(s).
        ftri = os.path.split(Grid['TriFile'])[-1]
        # Copy the tri file there if necessary.
        shutil.copyfile(Grid['TriFile'], 
            os.path.join('Grid', 'Components.i.tri'))
        # Get the component list.
        fxml = Grid['ComponentFile']
        if os.path.isfile(fxml):
            # Copy
            shutil.copyfile(fxml, os.path.join('Grid', 'Config.xml'))
        # Change to the Grid folder.
        os.chdir('Grid')
        # Start by running autoInputs
        cmd = 'autoInputs -r %i -t Components.i.tri' % Grid['MeshRadius']
        os.system(cmd)
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
        # Name of tri file
        ftri = os.path.split(self.Grid['TriFile'])[-1]
        # Get the folder names.
        dlist = self.Trajectory.GetFolderNames()
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
        f_link = ['Components.i.tri', 'Mesh.R.c3d']
        # Check if a link will break things.
        if self.RunOptions["nAdapt"] > 0:
            # Adapt: copy the mesh.
            f_copy.append('Mesh.mg.c3d')
        else:
            # No adaptations: link the mesh.
            f_link.append('Mesh.mg.c3d')
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
        #  2014.05.30 @ddalle  : Moved input.cntl filtering to separate func
        
        # Check the 'Grid directory.
        if not os.path.isdir('Grid'):
            raise IOError('Folder "Grid" not found.')
        # Get the name of the .cntl file.
        fname = self.RunOptions['CntlFile']
        # Number of adaptations
        nAdapt = self.RunOptions['nAdapt']
        # Check if it exists.
        if not os.path.isfile(fname):
            raise IOError('Input CNTL File "%s" not found.' % fname)
        # Copy the file to the grid folder.
        shutil.copyfile(fname, os.path.join('Grid', 'input.cntl'))
        # Copy the aero.csh file
        if nAdapt > 0:
            shutil.copyfile('aero.csh', os.path.join('Grid', 'aero.csh'))
        # Move to the Grid folder.
        os.chdir('Grid')
        # Extract the trajectory.
        T = self.Trajectory
        # Read the input.cntl file.
        lines = open('input.cntl').readlines()
        # Get the folder names
        dlist = T.GetFolderNames()
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
            # Prepare the input file.
            self.PrepareCntlFile('input.cntl', i)
            # Create a conditions file
            T.WriteConditionsFile(os.path.join(dlist[i], 'Conditions.json'), i)
            # Check for adaptation.
            if nAdapt > 0:
                # Create the aero.csh instance.
                self.PrepareAeroCsh('aero.csh', i)
            # Create the run script.
            self.CreateCaseRunScript(i)
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
        
    # Function to filter/replace input.cntl files
    def PrepareCntlFile(self, fin, i):
        """
        Create a specific instance of an 'input.cntl' file.
        
        This function will create a new 'input.cntl' file and replace various
        ``*_TMP`` placeholders with specific values.
        
        This function must be called from the 'Grid' folder, unlike the other
        higher-level functions, which are called from the parent folder.
        
        :Call:
            >>> cntl.PrepareCntlFile(fin, i)
            
        :Inputs:
            *cntl*: :class:`pyCart.cntl.Cntl`
                Instance of global pyCart settings object
            *fin*: :class:`str`
                Name of template input file
            *i*: :class:`int`
                Trajectory case number
                
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.30 @ddalle  : First version
        
        # Read the lines from the template file.
        lines = open(fin).readlines()
        # Get the trajectory.
        T = self.Trajectory
        # Get the folder name.
        dname = T.GetFolderNames(i=i)
        # Create the output file name.
        fout = os.path.join(dname, 'input.cntl')
        # Create the specific input file.
        f = open(fout, 'w')
        # Loop through the lines of the template.
        for line in lines:
            line = line.replace('Mach_TMP',  '%.8f'%T.Mach[i])
            line = line.replace('alpha_TMP', '%.8f'%T.alpha[i])
            line = line.replace('beta_TMP',  '%.8f'%T.beta[i])
            # Write the line.
            f.write(line)
        # Close the input.cntl file.
        f.close()
        # End
        return None
        
        
    # Function to filer/replace aero.csh files
    def PrepareAeroCsh(self, fin, i):
        """
        Create a specific instance of an 'aero.csh' file.
        
        This function will create a new 'input.cntl' file and replace various
        ``set * =`` placeholders with specific values.
        
        This function must be called from the 'Grid' folder, unlike the other
        higher-level functions, which are called from the parent folder.
        
        :Call:
            >>> cntl.PrepareCntlFile(fin, i)
            
        :Inputs:
            *cntl*: :class:`pyCart.cntl.Cntl`
                Instance of global pyCart settings object
            *fin*: :class:`str`
                Name of template input file/script
            *i*: :class:`int`
                Trajectory case number
                
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.30 @ddalle  : First version
        
        # Read the lines from the template file/script.
        lines = open(fin).readlines()
        # Get the trajectory.
        T = self.Trajectory
        # Get the folder name.
        dname = T.GetFolderNames(i=i)
        # Create the output file name.
        fout = os.path.join(dname, 'aero.csh')
        # Create the specific input file/script.
        f = open(fout, 'w')
        # Loop through the lines of the template.
        for line in lines:
            # Do stuff...
            # Write the line.
            f.write(line)
        # Close the aero.csh file.
        f.close()
        # Make it executable.
        os.chmod(fout, 0750)
        # End.
        return None
        
    # Function to create the flowCart run script
    def CreateCaseRunScript(self, i):
        """
        Write the "run_case.sh" script to run a given case.
        
        :Call:
            >>> cntl.CreateCaseRunScript(i)
        
        :Inputs:
            *cntl*: :class:`pyCart.cntl.Cntl`
                Instance of global pyCart settings object
            *i*: :class:`int`
                Trajectory case number
                
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.30 @ddalle  : First version
        
        # Get the folder name.
        dname = self.Trajectory.GetFolderNames(i=i)
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
            f.write('flowCart -N %i -v -mg %i -his -clic | tee flowCart.txt\n' %
                (opts['nIter'], self.Grid['nMultiGrid']))
        # Close the file.
        f.close()
        # Make it executable.
        os.chmod(fout, 0750)
        
        
    

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

# pyCart settings class
import options


# Import the trajectory class
from trajectory import Trajectory
# Read "loadsXX.dat" files
from post import LoadsDat

# Import specific file control classes
from inputCntl import InputCntl
from aeroCsh   import AeroCsh

# Import triangulation
from tri import Tri


# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyCartFolder = os.path.split(_fname)[0]
TemplateFodler = os.path.join(PyCartFolder, "templates")


# Function to automate minor changes to docstrings to make them pyCart.Cart3d
def _upgradeDocString(docstr, fromclass):
    """
    Upgrade docstrings from a certain subclass to make them look like
    :class:`pyCart.cart3d.Cart3d` docstrings.
    
    :Call:
        >>> doc3d = _upgradDocString(docstr, fromclass)
        
    :Inputs:
        *docstr*: :class:`str`
            Docstring (e.g. ``x.__doc__``) from some other method
        *fromclass*: :class:`str`
            Name of class of the original docstring (e.g. ``type(x).__name__``)
            
    :Outputs:
        *doc3d*: :class:`str`
            Docstring with certain substitutions, e.g. ``x.`` --> ``cart3d.``
            
    :Versions:
        * 2014.07.28 ``@ddalle``: First version
    """
    # Check the input class.
    if fromclass in ['Trajectory']:
        # Replacements in the :Call: area
        docstr = docstr.replace(">>> x.", ">>> cart3d.")
        docstr = docstr.replace("= x.", "= cart3d.")
        # Replacements in variable names
        docstr = docstr.replace('*x*', '*cart3d*')
        # Class name
        docstr = docstr.replace('trajectory.Trajectory', 'cart3d.Cart3d')
        docstr = docstr.replace('trajectory class', 'control class')
    # Output
    return docstr

#<!--
# ---------------------------------
# I consider this portion temporary

# Change the umask to a reasonable value.
os.umask(0027)

# ---------------------------------
#-->
    
    
# Class to read input files
class Cart3d(object):
    """
    Class for handling global options and setup for Cart3D.
    
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
        *cart3d.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cart3d.x*: :class:`pyCart.trajectory.Trajectory`
            Values and definitions for variables in the run matrix
        *cart3d.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2014.05.28 ``@ddalle``  : First version
        * 2014.06.03 ``@ddalle``  : Renamed class `Cntl` --> `Cart3d`
        * 2014.06.30 ``@ddalle``  : Reduced number of data members
        * 2014.07.27 ``@ddalle``  : `cart3d.Trajectory` --> `cart3d.x`
    """
    
    # Initialization method
    def __init__(self, fname="pyCart.json"):
        """Initialization method for :mod:`pyCart.cart3d.Cart3d`"""
        
        # Apply missing settings from defaults.
        opts = options.Options(fname=fname)
        
        # Process the trajectory.
        self.x = Trajectory(**opts['Trajectory'])
        
        # Save all the options as a reference.
        self.opts = opts
        
        # Read the triangulation file(s)
        self.ReadTri()
        
        # Save the current directory as the root.
        self.RootDir = os.path.split(os.path.abspath(fname))[0]
        
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<pyCart.Cart3d(nCase=%i, tri='%s')>" % (
            self.x.nCase,
            self.opts.get_TriFile()
        
    # Function to prepare the triangulation for each grid folder
    def ReadTri(self):
        """Read initial triangulation file(s)
        
        :Call:
            >>> cart3d.ReadTri(v=True)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *v*: :class:`bool`
                If ``True``, displays output on command line.
        :Versions:
            * 2014.08.30 ``@ddalle``: First version
        """
        # Get the list of tri files.
        ftri = self.opts.get_TriFile()
        # Status update.
        print("Reading tri file(s) from root directory.")
        # Read them.
        if type(ftri).__name__ == 'list':
            # Read the initial triangulation.
            tri = Tri(ftri[0])
            # Loop through the remaining tri files.
            for f in ftri[1:]:
                # Append the file.
                tri.Add(Tri(f))
        else:
            # Just read the triangulation file.
            tri = Tri(ftri)
        # Save it.
        self.tri = tri
    
    # Check for root location
    def CheckRootDir(self):
        """Check if the current directory is the case root directory
        
        Suppose the directory structure for a case is as follows.
        
        * `/nobackup/uuser/plane/`
            * `Grid_d0.0/`
                * `m1.20/`
                * `m1.40/`
            * `Grid_d1.0/`
                * `m1.20/`
                * `m1.40/`
                
        Then this function will return ``True`` if and only if the current
        working directory is ``'/nobackup/uuser/plane/'``.
        
        :Call:
            >>> q = cart3d.CheckRootDir()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Global pyCart settings object instance
        :Outputs:
            *q*: :class:`bool`
                True if current working directory is the case root directory
        :Versions:
            * 2014.06.30 ``@ddalle``: First version
        """
        # Compare the working directory and the stored root directory.
        return os.path.abspath('.') == self.RootDir
    
    # Check for group location
    def CheckGroupDir(self):
        """
        Check if the current directory is a group-level directory
        
        Suppose the directory structure for a case is as follows.
        
        * `/nobackup/uuser/plane/`
            * `Grid_d0.0/`
                * `m1.20/`
                * `m1.40/`
            * `Grid_d1.0/`
                * `m1.20/`
                * `m1.40/`
                
        Then this function will return ``True`` if and only if the current
        working directory is ``'/nobackup/uuser/plane/Grid_d0.0'`` or 
        ``'/nobackup/uuser/plane/Grid_d1.0'``.
        
        :Call:
            >>> q = cart3d.CheckGroupDir()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Global pyCart settings object instance
        :Outputs:
            *q*: :class:`bool`
                True if current working directory is a group-level directory
        :Versions:
            * 2014.06.30 ``@ddalle``: First version
        """
        # Compare the working directory and the stored root directory.
        return os.path.abspath('..') == self.RootDir
    
    # Check for group location
    def CheckCaseDir(self):
        """
        Check if the current directory is a case-level directory
        
        Suppose the directory structure for a case is as follows.
        
        * `/nobackup/uuser/plane/`
            * `Grid_d0.0/`
                * `m1.20/`
                * `m1.40/`
            * `Grid_d1.0/`
                * `m1.20/`
                * `m1.40/`
                
        Then this function will return ``True`` if the current working 
        directory is either of the ``'m1.20'`` or ``'m1.40'`` folders.
        
        :Call:
            >>> q = cart3d.CheckCaseDir()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Global pyCart settings object instance
        :Outputs:
            *q*: :class:`bool`
                True if current working directory is a group-level directory
        :Versions:
            * 2014.06.30 ``@ddalle``: First version
        """
        # Compare the working directory and the stored root directory.
        return os.path.abspath(os.path.join('..','..')) == self.RootDir
        
        
    # Method to create the folders
    def CreateFolders(self):
        # Use the trajectory's method.
        return self.x.CreateFolders()
    # Copy the docstring.
    CreateFolders.__doc__ = _upgradeDocString(
        Trajectory.CreateFolders.__doc__, 'Trajectory')
        
        
    # Method to set up the grid
    def CreateMesh(self):
        """Create the common mesh based on self-contained parameters.
        
        :Call:
            >>> cart3d.CreateMesh()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        :Versions:
            * 2014.05.28 ``@ddalle``: First version
            * 2014.06.06 ``@ddalle``: Multiple group folders
        """
        # Write conditions files.
        self.Grids_WriteConditionsFiles()
        # Copy Config.xml files.
        self.Grids_CopyConfigFile()
        # Prepare the tri files.
        self.Grids_PrepareTri()
        # Run autoInputs (if necessary).
        self.Grids_autoInputs()
        # Bounding box control.....
        # Run cubes
        self.Grids_cubes()
        # Prepare multigrid
        self.Grids_mgPrep()
        # End.
        return None
        
    # Write conditions files.
    def Grids_WriteConditionsFiles(self):
        """Write conditions files for each group
        
        :Call:
            >>> cart3d.Grids_WriteConditionsFiles()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        :Versions:
            * 2014.06.23 ``@ddalle``: First version
        """
        # Loop through groups.
        for i in range(len(self.x.GroupX)):
            # Write the conditions file.
            self.x.WriteGridConditionsFile(i=i)
        
    # Method to copy 'Config.xml' to all grid folders.
    def Grids_CopyConfigFile(self, fxml=None):
        """Copy configuration file (usually :file:`Config.xml`) to grid folders
        
        :Call:
            >>> cart3d.Grids_CopyConfigFile(fxml=None)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *fxml*: :class:`str`
                Name of configuration file to copy
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Get the xml file names.
        if fxml is None:
            conf = self.opts.get('Config', {})
            fxml = conf.get('File', 'Config.xml')
        # Check if the file exists.
        if not os.path.isfile(fxml):
            return None
        # Get grid folders.
        glist = self.x.GetGroupFolderNames()
        # Loop through the grids.
        for g in glist:
            # Copy the file.
            shutil.copyfile(fxml, os.path.join(g, 'Config.xml'))
            
    # Function to prepare the triangulation for each grid folder
    def Grids_PrepareTri(self, v=True):
        """Prepare and copy triangulation file to each grid folder.  If
        ``cart3d.Mesh['TriFile']`` is a list, calling this function will
        include appending the triangulation files in that list before writing
        it to each grid folder.  Recognized translations and rotations are
        performed as well.
        
        :Call:
            >>> cart3d.Grids_PrepareTri(v=True)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *v*: :class:`bool`
                If ``True``, displays output on command line.
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Get grid folders.
        glist = np.unique(self.x.GetGroupFolderNames())
        # Announce.
        print("Writing 'Components.i.tri' for case:")
        # Loop through the grids.
        for g in glist:
            # Perform recognized rotations and translations...
            # Status update.
            print("  %s" % g)
            # Write the new .tri file.
            self.tri.Write(os.path.join(g, 'Components.i.tri'))
            
        
    # Method to run 'autoInputs' in the current folder.
    def autoInputs(self, r=None, ftri=None, v=True):
        """Run :file:`autoInputs` in the current working directory.  Writes
        output too :file:`autoInputs.out`.
        
        :Call:
            >>> cart3d.autoInputs(r=None, ftri=None, v=True)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *r*: :class:`int`
                Mesh radius, defaults to ``cart3d.Mesh['MeshRadius']``
            *ftri*: :class:`str`
                Name of triangulation file to use
            *v*: :class:`bool`
                If ``True``, displays output on command line.
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Get the mesh radius.
        if r is None:
            r = self.opts.get_r()
        # Exit if set to quit.
        if not r: return None
        # Get the triangulation file.
        if ftri is None: ftri = 'Components.i.tri'
        # Check for source file.
        if not os.path.isfile(ftri):
            raise IOError("No surface file '%s' found." % ftri)
        # Form the command.
        cmd = 'autoInputs -r %i -t Components.i.tri' % r
        # Check verbosity.
        if v:
            # Run command and display output.
            os.system(cmd + " | tee autoInputs.out")
        else:
            # Hide the output.
            os.system(cmd + " > autoInputs.out")
        
    # Method to run 'autoInput' in all grid folders.
    def Grids_autoInputs(self):
        """Run :file:`autoInputs` in each grid folder.
        
        :Call:
            >>> cart3d.Grids_autoInputs()
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Check if autoInputs should be run.
        if not self.opts['Mesh']['autoInputs']['r']:
            return None
        # Get grid folders.
        glist = np.unique(self.x.GetGroupFolderNames())
        # Common announcement.
        print("  Running 'autoInputs' for grid:")
        # Loop through the grids.
        for g in glist:
            # Change to that directory.
            os.chdir(g)
            # Announce.
            print("    %s" % g)
            # Run.
            self.autoInputs(v=False)
            # Change back to home directory.
            os.chdir('..')
    
    # Method to run 'bues' in the current folder.
    def cubes(self, maxR=None, v=True):
        """Run :file:`cubes` in the current working directory.  Writes output
        too :file:`cubes.out`.
        
        :Call:
            >>> cart3d.cubes(maxR=None, v=True)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *r*: :class:`int`
                Refinements, defaults to ``cart3d.Mesh['nRefinements']``
            *v*: :class:`bool`
                If ``True``, displays output on command line.
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Get the mesh radius.
        if maxR is None: maxR = self.opts.get_maxR()
        # Check for input files.
        if not os.path.isfile('input.c3d'):
            raise IOError("No input file 'input.c3d' found.")
        if not os.path.isfile('preSpec.c3d.cntl'):
            raise IOError("No input file 'preSpec.c3d.cntl' found.")
        # Form the command.
        cmd = 'cubes -maxR %i -pre preSpec.c3d.cntl -reorder' % maxR
        # Check verbosity.
        if v:
            # Run command and display output.
            os.system(cmd + " | tee cubes.out")
        else:
            # Hide the output.
            os.system(cmd + " > cubes.out")
        
    # Method to run 'cubes' in all grid folders.
    def Grids_cubes(self):
        """Run :file:`cubes` in each grid folder.
        
        :Call:
            >>> cart3d.Grids_cubes()
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Get grid folders.
        glist = np.unique(self.x.GetGroupFolderNames())
        # Common announcement.
        print("  Running 'cubes' for grid:")
        # Loop through the grids.
        for g in glist:
            # Change to that directory.
            os.chdir(g)
            # Announce.
            print("    %s" % g)
            # Run.
            self.cubes(v=False)
            # Change back to home directory.
            os.chdir('..')
            
    
    # Method to run 'bues' in the current folder.
    def mgPrep(self, mg=None, v=True):
        """Run :file:`mgPrep` in the current working directory.  Writes output
        too :file:`mgPrep.out`.
        
        :Call:
            >>> cart3d.mgPrep(mg=None, v=True)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *mg*: :class:`int`
                Number of multigrid levels, ``cart3d.Mesh['nMultiGrid']``
            *v*: :class:`bool`
                If ``True``, displays output on command line.
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Get the mesh radius.
        if mg is None: mg = self.opts.get_mg()
        # Check for input files.
        if not os.path.isfile('Mesh.R.c3d'):
            raise IOError("No mesh file 'Mesh.R.c3d' found.")
        # Form the command.
        cmd = 'mgPrep -n %i' % mg
        # Check verbosity.
        if v:
            # Run command and display output.
            os.system(cmd + " | tee mgPrep.out")
        else:
            # Hide the output.
            os.system(cmd + " > mgPrep.out")
        
    # Method to run 'cubes' in all grid folders.
    def Grids_mgPrep(self):
        """Run :file:`mgPrep` in each grid folder.
        
        :Call:
            >>> cart3d.Grids_autoInputs()
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Get grid folders.
        glist = np.unique(self.x.GetGroupFolderNames())
        # Get n multigrid levels
        mg = self.opts.get_mg()
        # Quit if appropriate.
        if not mg:
            return None
        # Common announcement.
        print("  Running 'mgPrep' for grid:")
        # Loop through the grids.
        for g in glist:
            # Change to that directory.
            os.chdir(g)
            # Announce.
            print("    %s" % g)
            # Run.
            self.mgPrep(mg=mg, v=False)
            # Change back to home directory.
            os.chdir('..')
    
    # Function to copy/link files
    def CopyFiles(self):
        """Copy or link the relevant files to the Grid folders.
        
        :Call:
            >>> cart3d.CopyFiles()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        :Versions:
            * 2014.05.28 ``@ddalle``: First version
        """
        # Get the folder names.
        glist = self.x.GetGroupFolderNames()
        dlist = self.x.GetFolderNames()
        # List of files to copy
        f_copy = ['input.c3d', 'Config.xml',
            'Mesh.c3d.Info', 'preSpec.c3d.cntl']
        # List of files to link
        f_link = ['Components.i.tri', 'Mesh.R.c3d']
        # Check if a link will break things.
        if self.opts["Adaptation"]["n_adapt_cycles"] > 0:
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
            # Check the 'Grid' directory.
            if not os.path.isdir(g):
                raise IOError('Folder "%s" not found.' % g)
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
        """Create run scripts for each case according to the pyCart settings.
        
        :Call:
            >>> cart3d.PrepareRuns()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        :Versions:
            * 2014.05.28 ``@ddalle``: First version
            * 2014.05.30 ``@ddalle``: Moved input.cntl filter to separate func
            * 2014.06.04 ``@ddalle``: Moved input.cntl prep entirely to func
        """
        # Prepare the "input.cntl" files.
        self.PrepareInputCntl()
        # Prepare the "aero.csh" files
        self.PrepareAeroCsh()
        # Prepare the run scripts.
        self.CreateRunScripts()
        # Get the trajectory.
        x = self.x
        # Get the folder names.
        dlist = x.GetFullFolderNames()
        # Loop through the conditions.
        for i in range(len(dlist)):
            # Get the folder name for this case.
            d = dlist[i]
            # Create a conditions file
            x.WriteConditionsFile(os.path.join(d, 'Conditions.json'), i)
        # End
        return None
        
    # Function to create run scripts
    def CreateRunScripts(self):
        """Create all run scripts
        
        :Call:
            >>> cart3d.CreateRunScripts()
            
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        :Versions:
            * 2014.06.04 ``@ddalle``: First version
            * 2014.06.06 ``@ddalle``: Added support for multiple grid folders
        """
        # Global script name
        fname_all = 'run_all.sh'
        # Grid script name
        fname_grid = 'run_cases.sh'
        # Local script name.
        fname_i = 'run_case.sh'
        # Get the grid folder names
        glist = self.x.GetGroupFolderNames()
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
        dlist = self.x.GetFolderNames()
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
            
        :Versions:
            * 2014.06.04 ``@ddalle``: First version
            * 2014.06.06 ``@ddalle``: Low-level functionality for grid folders
        """
        # Get the name of the .cntl file.
        fname = self.opts.get('InputCntl', 'input.cntl')
        # Get the configuration settings.
        conf = self.opts.get('Config', {})
        # Read it.
        self.InputCntl = InputCntl(fname)
        # Process global options...
        self.InputCntl.SetCFL(self.opts.get_cfl())
        # Extract the trajectory.
        x = self.x
        # Get grid folders.
        glist = x.GetGroupFolderNames()
        # Write to each "Grid" folder.
        for g in glist:
            self.InputCntl.Write(os.path.join(g, 'input.cntl'))
        # Get the folder names.
        dlist = x.GetFolderNames()
        # Loop through the conditions.
        for i in range(len(dlist)):
            # Print a status update
            print("  Preparing 'input.cntl' for case %i" % i)
            # Set the flight conditions.
            self.InputCntl.SetMach(x.Mach[i])
            self.InputCntl.SetAlpha(x.alpha[i])
            self.InputCntl.SetBeta(x.beta[i])
            # Set the Reference values
            self.InputCntl.SetReferenceArea(conf.get('ReferenceArea'))
            self.InputCntl.SetReferenceLength(conf.get('ReferenceLength'))
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
            
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        # Get the name of the file.
        fname = self.opts['AeroCsh']
        # Check for the file.
        if not os.path.isfile(fname): return None
        # Read it.
        self.AeroCsh = AeroCsh(fname)
        # Process global options
        self.AeroCsh.SetCFL(self.opts.get_cfl())
        self.AeroCsh.SetnIter(self.opts.get_it_fc())
        self.AeroCsh.SetnAdapt(self.opts.get_n_adapt_cycles())
        self.AeroCsh.SetnRefinements(self.opts.get_maxR())
        self.AeroCsh.SetnMultiGrid(self.opts.get_mg())
        # Extract the trajectory.
        x = self.x
        # Get grid folders.
        glist = x.GetGroupFolderNames()
        # Get the folder names.
        dlist = x.GetFolderNames()
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
        """Write the "run_case.sh" script to run a given case.
        
        :Call:
            >>> cart3d.CreateCaseRunScript(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *i*: :class:`int`
                Trajectory case number
        :Versions:
            * 2014.05.30 ``@ddalle``: First version
        """
        # Get the folder name.
        dname = self.x.GetFullFolderNames(i=i)
        # File name
        fout = os.path.join(dname, 'run_case.sh')
        # Create the file.
        f = open(fout, 'w')
        # Run options
        opts_fc = self.opts.get('flowCart', {})
        opts_ad = self.opts.get('Adaptation', {})
        # Get the global options.
        nThreads = self.opts.get_OMP_NUM_THREADS()
        nIter    = self.opts.get_it_fc()
        nAdapt   = self.opts.get_n_adapt_cycles()
        # Run options
        mg  = self.opts.get_mg_fc()
        tm  = self.opts.get_tm()
        cfl = self.opts.get_cfl()
        # Write the shell magic.
        f.write('#!/bin/bash\n\n')
        # Set the number of processors.
        f.write('export OMP_NUM_THREADS=%i\n\n' % nThreads)
        # Check for an adaptive case.
        if nAdapt > 0:
            # Create the aero.csh command to do the work.
            f.write('./aero.csh jumpstart | tee aero.out\n')
        else:
            # Create the flowCart command to do the work.
            f.write(('flowCart -N %i -v -mg %i -cfl %f -his -clic -tm %i ' +
                '| tee flowCart.out\n') % (nIter, mg, cfl, tm))
        # Close the file.
        f.close()
        # Make it executable.
        os.chmod(fout, 0750)
        
    # Function to read "loadsCC.dat" files
    def GetLoadsCC(self):
        """Read all available 'loadsCC.dat' files.
        
        :Call:
            >>> cart3d.GetLoadsCC()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        :Effects:
            Creates *cart3d.LoadsCC* instance
        :Versions:
            * 2014.06.05 ``@ddalle``: First version
        """
        # Call the constructor.
        self.LoadsCC = LoadsDat(self, fname="loadsCC.dat")
        return None
        
    # Function to write "loadsCC.csv"
    def WriteLoadsCC(self):
        """Write gathered loads to CSV file to "loadsCC.csv"
        
        :Call:
            >>> cart3d.WriteLoadsCC()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        :Versions:
            * 2014.06.04 ``@ddalle``: First version
        """
        # Check for the attribute.
        if not hasattr(self, 'LoadsCC'):
            self.GetLoadsCC()
        # Write.
        self.LoadsCC.Write(self.x)
        return None
        
    # Function to read "loadsCC.dat" files
    def GetLoadsTRI(self):
        """Read all available 'loadsTRI.dat' files.
        
        :Call:
            >>> cart3d.GetLoadsTRI()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        :Effects:
            Creates *cart3d.LoadsCC* instance
        :Versions:
            * 2014.06.04 ``@ddalle``: First version
        """
        # Call the constructor.
        self.LoadsTRI = LoadsDat(self, fname="loadsTRI.dat")
        return None
        
    # Function to write "loadsCC.csv"
    def WriteLoadsTRI(self):
        """Write gathered loads to CSV file to "loadsTRI.csv"
        
        :Call:
            >>> cart3d.WriteLoadsTRI()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        :Versions:
            * 2014.06.04 ``@ddalle``: First version
        """
        # Check for the attribute.
        if not hasattr(self, 'LoadsTRI'):
            self.GetLoadsTRI()
        # Write.
        self.LoadsTRI.Write(self.x)
        return None
    


# Function to read conditions file.
def ReadTrajectoryFile(fname='Trajectory.dat', keys=['Mach','alpha','beta'],
    prefix="F"):
    """Read a simple list of configuration variables
    
    :Call:
        >>> x = pyCart.ReadTrajectoryFile(fname)
        >>> x = pyCart.ReadTrajectoryFile(fname, keys)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read, defaults to ``'Trajectory.dat'``
        *keys*: :class:`list` of :class:`str` items
            List of variable names, defaults to ``['Mach','alpha','beta']``
        *prefix*: :class:`str`
            Header for name of each folder
    :Outputs:
        *x*: :class:`pyCart.trajectory.Trajectory`
            Instance of the pyCart trajectory class
    :Versions:
        * 2014.05.27 ``@ddalle``: First version
    """
    return Trajectory(fname, keys, prefix)
    


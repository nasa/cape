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
import subprocess as sp

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

# Cart3D binary interfaces
from . import bin

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

# Get the umask value.
umask = 0027
# Get the folder permissions.
fmask = 0777 - umask
dmask = 0777 - umask

# Change the umask to a reasonable value.
os.umask(umask)

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
        
        # Read the input files.
        self.InputCntl = InputCntl(self.opts.get_InputCntl())
        self.AeroCsh   = AeroCsh(self.opts.get_AeroCsh())
        
        # Save the current directory as the root.
        self.RootDir = os.path.split(os.path.abspath(fname))[0]
        
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<pyCart.Cart3d(nCase=%i, tri='%s')>" % (
            self.x.nCase,
            self.opts.get_TriFile())
        
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
        if self.opts.get_r(): self.autoInputs()
        # Bounding box control.....
        # Run cubes
        self.cubes()
        # Prepare multigrid
        self.mgPrep()
        # End.
        return None
        
    # Interface for ``cubes``
    def cubes(self, i=None):
        """Run ``cubes`` for all groups or a specific group *i*
        
        :Call:
            >>> cart3d.cubes()
            >>> cart3d.cubes(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int` or :class:`list`(:class:`int`)
                Group index to prepare
        :Versions:
            * 2014.08.31 ``@ddalle``: First version
            * 2014.09.02 ``@ddalle``: Uses absolute paths
        """
        # Store the current location.
        fpwd = os.getcwd()
        # Check for index filter
        if i and np.isscalar(i): i = [i]
        # Get group names
        glist = self.x.GetUniqueGroupFolderNames(i=i)
        # Loop through them.
        for fdir in glist:
            # Go to the root folder.
            os.chdir(self.RootDir)
            # Go to the grid folder.
            os.chdir(fdir)
            # Run cubes.
            bin.cubes(self)
        # Return to original directory.
        os.chdir(fpwd)
        
    # Interface for ``autoInputs``
    def autoInputs(self, i=None):
        """Run ``autoInputs`` for all groups or a specific group *i*
        
        :Call:
            >>> cart3d.autoInputs()
            >>> cart3d.autoInputs(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int` or :class:`list`(:class:`int`)
                Group index to prepare
        :Versions:
            * 2014.09.02 ``@ddalle``: First version
        """
        # Store the current location.
        fpwd = os.getcwd()
        # Check for index filter
        if i and np.isscalar(i): i = [i]
        # Get group names
        glist = self.x.GetUniqueGroupFolderNames(i=i)
        # Loop through them.
        for fdir in glist:
            # Go to the root folder.
            os.chdir(self.RootDir)
            # Go to the grid folder.
            os.chdir(fdir)
            # Run cubes.
            bin.autoInputs(self)
        # Return to original directory.
        os.chdir(fpwd)
        
    # Interface for ``autoInputs``
    def mgPrep(self, i=None):
        """Run ``mgPrep`` for all groups or a specific group *i*
        
        :Call:
            >>> cart3d.mgPrep()
            >>> cart3d.mgPrep(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int` or :class:`list`(:class:`int`)
                Group index to prepare
        :Versions:
            * 2014.09.14 ``@ddalle``: First version
        """
        # Store the current location.
        fpwd = os.getcwd()
        # Check for index filter
        if i and np.isscalar(i): i = [i]
        # Get group names
        glist = self.x.GetUniqueGroupFolderNames(i=i)
        # Loop through them.
        for fdir in glist:
            # Go to the root folder.
            os.chdir(self.RootDir)
            # Go to the grid folder.
            os.chdir(fdir)
            # Run cubes.
            bin.mgPrep(self)
        # Return to original directory.
        os.chdir(fpwd)
        
        
    # Function to check if the mesh for case i exists
    def CheckMesh(self, i):
        """Check if the mesh for case *i* is prepared.
        
        :Call:
            >>> q = cart3d.check_mesh(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *q*: :class:`bool`
                Whether or not the mesh for case *i* is prepared
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        # Check input.
        if type(i).__name__ != "int":
            raise TypeError(
                "Input to :func:`Cart3d.check_mesh()` must be :class:`int`.")
        # Get the group name.
        fgrp = self.x.GetGroupFolderNames(i)
        # Initialize with "pass" setting.
        q = True
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if (not os.path.isdir(fgrp)): q = False
        # Check that test.
        if q:
            # Go to the group folder.
            os.chdir(fgrp)
            # Check for the surface file.
            if not os.path.isfile('Components.i.tri'): q = False
            # Check for which mesh file to look for.
            if q and self.opts.get_mg() > 0:
                # Look for the multigrid mesh
                if not os.path.isfile('Mesh.mg.c3d'): q = False
            else:
                # Look for the original mesh
                if not os.path.isfile('Mesh.c3d'): q = False
        # Return to original folder.
        os.chdir(fpwd)
        # Output.
        return q
        
    # Prepare the mesh for case i (if necessary)
    def PrepareMesh(self, i):
        """Prepare the mesh for case *i* if necessary.
        
        :Call:
            >>> q = cart3d.PrepareMesh(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Display it.
        print(frun)
        print("  Checking status...")
        # Check the mesh.
        if self.CheckMesh(i):
            # Display.
            print("  Mesh already prepared")
            return None
        # Get name of group.
        fgrp = self.x.GetGroupFolderNames(i)
        # Get the group index.
        j = self.x.GetGroupIndex(i)
        # Status update
        print("  Group name: '%s' (index %i)" % (fgrp,j))
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check for the group folder and make it if necessary.
        if not os.path.isdir(fgrp):
            os.mkdir(fgrp, fmask)
        # Go there.
        os.chdir(fgrp)
        # Get the name of the configuration file.
        fxml = os.path.join(self.RootDir, self.opts.get_ConfigFile())
        # Test if the file exists.
        if os.path.isfile(fxml):
            # Copy it, to a fixed file name in this folder.
            shutil.copyfile(fxml, 'Config.xml')
        # Status update
        print("  Preparing surface triangulation...")
        # Apply rotations, etc.
        
        # Write the tri file.
        self.tri.Write('Components.i.tri')
        # Run autoInputs if necessary.
        if self.opts.get_r(): self.autoInputs(j)
        # Bounding box control...
        
        # Run cubes.
        self.cubes(j)
        # Run mgPrep
        self.mgPrep(j)
        # Return to original folder.
        os.chdir(fpwd)
        
    # Check a case.
    def CheckCase(self, i):
        """Check current status of run *i*
        
        :Call:
            >>> n = cart3d.CheckCase(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *n*: :class:`int` or ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2014.09.27 ``@ddalle``: First version
        """
         # Check input.
        if type(i).__name__ != "int":
            raise TypeError(
                "Input to :func:`Cart3d.check_mesh()` must be :class:`int`.")
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Initialize iteration number.
        n = 0
        # Check if the folder exists.
        if (not os.path.isdir(frun)): n = None
        # Check that test.
        if n is not None:
            # Go to the group folder.
            os.chdir(frun)
            # Check for the surface file.
            if not os.path.isfile('Components.i.tri'): n = None
            # Input file.
            if not os.path.isfile('input.00.cntl'): n=None
            # Check for which mesh file to look for.
            if self.opts.get_mg() > 0:
                # Look for the multigrid mesh
                if not os.path.isfile('Mesh.mg.c3d'): n = None
            else:
                # Look for the original mesh
                if not os.path.isfile('Mesh.c3d'): n = None
        # Output if None
        if n is None:
            # Go back to starting point.
            os.chdir(fpwd)
            # Quit.
            return None
        # Count iterations....
        if os.path.isfile('history.dat'):
            # Get the last line of the history file.
            txt = sp.Popen(['tail', '-1', 'history.dat'],
                stdout=sp.PIPE).communicate()[0]
            # Check if it's a comment.
            if txt.startswith('#') or len(txt)<2:
                # No iterations yet.
                n = 0
            else:
                # Iterations
                n = int(txt.split()[0])
        else:
            # No history; zero iterations.
            n = 0
        # Return to original folder.
        os.chdir(fpwd)
        # Output.
        return n
        
    # Prepare a case.
    def PrepareCase(self, i):
        """Prepare case for running if necessary
        
        :Call:
            >>> n = cart3d.PrepareCase(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to analyze
        :Versions:
            * 2014.09.30 ``@ddalle``: First version
        """
        # Prepare the mesh.
        self.PrepareMesh(i)
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if prepared.
        if n is not None: return None
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Save current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check for the run directory.
        if not os.path.isdir(frun): os.mkdir(frun, dmask)
        # Go there.
        os.chdir(frun)
        # Copy the required files.
        for fname in ['input.c3d', 'Mesh.c3d.Info', 'Config.xml']:
            # Source path.
            fsrc = os.path.join('..', fname)
            # Check for the file.
            if os.path.isfile(fsrc):
                # Copy it.
                shutil.copy(fsrc, fname)
        # Create links that are available.
        for fname in ['Components.i.tri', 'Mesh.c3d', 'Mesh.mg.c3d',
                'Mesh.R.c3d']:
            # Source path.
            fsrc = os.path.join(os.path.abspath('..'), fname)
            # Remove the file if it's present.
            if os.path.isfile(fname):
                os.remove(fname)
            # Check for the file.
            if os.path.isfile(fsrc):
                # Create a symlink.
                os.symlink(fsrc, fname)
        # Write 
        # Write the input.cntl file.
        self.PrepareInputCntl(i)
        # Write a JSON file with the flowCart settings.
        f = open('case.json', 'w')
        # Dump the flowCart settings.
        json.dump(self.opts['flowCart'], f, indent=1)
        # Close the file.
        f.close()
        # Write the PBS script.
        self.WritePBS(i)
        # Return to original location.
        os.chdir(fpwd)
        
    # Get PBS name
    def GetPBSName(self, i):
        """Get PBS name for a given case
        
        :Call:
            >>> lbl = cart3d.GetPBSName(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014.09.30 ``@ddalle``: First version
        """
        # Extract the trajectory.
        x = self.x
        # Get the first key (special because allowed two decimals)
        k0 = x.keys[0]
        # Initialize label.
        lbl = '%s%.2f' % (x.abbrv[k0], getattr(x,k0)[i])
        # Loop through keys.
        for k in x.keys[1:]:
            # Append to the label
            lbl += ('%s%.1f' % (x.abbrv[k], getattr(x,k)[i]))
        # Check length.
        if len(lbl) > 16:
            # 16-char limit (or is it 15?)
            lbl = lbl[:15]
        # Output
        return lbl
        
        
    # Write the PBS script.
    def WritePBS(self, i):
        """Write the PBS script for a given case
        
        :Call:
            >>> cart3d.WritePBS(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014.09.30 ``@ddalle``: First version
        """
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Remember current location.
        fpwd = os.getcwd()
        # Go to the root directory.
        os.chdir(self.RootDir)
        # Make folder if necessary.
        if not os.path.isdir(frun): os.mkdir(frun, dmask)
        # Go to the folder.
        os.chdir(frun)
        
        # Initialize the PBS script.
        f = open('run_cart3d.pbs', 'w')
        # Get the shell path (must be bash)
        sh = self.opts.get_PBS_S()
        # Write to script both ways.
        f.write('#!%s\n' % sh)
        f.write('#PBS -S %s\n' % sh)
        # Get the shell name.
        lbl = self.GetPBSName(i)
        # Write it to the script
        f.write('#PBS -N %s\n' % lbl)
        # Get the rerun status.
        PBS_r = self.opts.get_PBS_r()
        # Write if specified.
        if PBS_r: f.write('#PBS -r %s\n' % PBS_r)
        # Get the option for combining STDIO/STDOUT
        PBS_j = self.opts.get_PBS_j()
        # Write if specified.
        if PBS_j: f.write('#PBS -j %s\n' % PBS_j)
        # Get the number of nodes, etc.
        nnode = self.opts.get_PBS_select()
        ncpus = self.opts.get_PBS_ncpus()
        nmpis = self.opts.get_PBS_mpiprocs()
        smodl = self.opts.get_PBS_model()
        # Form the -l line.
        line = '#PBS -l select=%i:ncpus=%i' % (nnode, ncpus)
        # Add other settings
        if nmpis: line += (':mpiprocs=%i' % nmpis)
        if smodl: line += (':model=%s' % smodl)
        # Write the line.
        f.write(line + '\n')
        # Get the walltime.
        t = self.opts.get_PBS_walltime()
        # Write it.
        f.write('#PBS -l walltime=%s\n' % t)
        # Check for a group list.
        PBS_W = self.opts.get_PBS_W()
        # Write if specified.
        if PBS_W: f.write('#PBS -W %s\n' % PBS_W)
        # Get the queue.
        PBS_q = self.opts.get_PBS_q()
        # Write it.
        if PBS_q: f.write('#PBS -q %s\n\n' % PBS_q)
        
        # Go to the working directory.
        f.write('# Go to the working directory.\n')
        f.write('cd %s\n\n' % os.getcwd())
        
        # Write a header for the shell commands.
        f.write('# Additional shell commands\n')
        # Loop through the shell commands.
        for line in self.opts.get_ShellCmds():
            # Write it.
            f.write('%s\n' % line)
        
        # Simply call the advanced interface.
        f.write('\n# Call the flow_cart/mpi_flowCart/aero.csh interface.\n')
        f.write('run_flowCart.py')
        
        # Close the file.
        f.close()
        # Return.
        os.chdir(fpwd)
        
        
        
    # Interface for getting ``flowCart`` commands
    def flowCartCmd(self, k, i=None):
        """Get ``flowCart`` commands for one case, either all runs or run *i*
        
        :Call:
            >>> cmdk = cart3d.flowCartCmd(k, i=None)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *k*: :class:`int`
                Index of case to analyze
            *i*: :class:`int` or :class:`list` (:class:`int`)
                Run index or indices to apply
        :Outputs:
            *cmdk*: :class:`list` (:class:`list` (:class:`str`))
                List of commands for each run index in case *k*
        :Versions:
            * 2014.09.14 ``@ddalle``: First version
            * 2014.09.27 ``@ddalle``: Added "Cmd" to the name
        """
        # Save current directory.
        fpwd = os.getcwd()
        # Get the case.
        fdir = self.x.GetFullFolderNames(k)
        # Go there.
        os.chdir(self.RootDir)
        os.chdir(fdir)
        # Process the run index.
        if i is None:
            # Get the iteration break points or total num of iterations.
            it_fc = self.opts.get_it_fc()
            # Check for a list or scalar.
            if np.isscalar(it_fc):
                # Single run
                i = [0]
            else:
                # Multple runs.
                i = range(len(it_fc))
        elif np.isscalar(i):
            # Make it a list.
            i = [i]
        # Initialize commands.
        cmdk = []
        # Loop through the runs.
        for j in i:
            # Run the case.
            cmdk.append(bin.cmd.flowCart(self, i=j))
        # Return to original directory.
        os.chdir(fpwd)
        # Output
        return cmdk
        
    # Direct interface for ``flowCart``
    def flowCart(self, k, i=None):
        """Run ``flowCart`` for one case, either all runs or run *i*
        
        :Call:
            >>> cmdk = cart3d.flowCartCmd(k, i=None)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *k*: :class:`int`
                Index of case to analyze
            *i*: :class:`int` or :class:`list` (:class:`int`)
                Run index or indices to apply
        :Versions:
            * 2014.09.27 ``@ddalle``: First version
        """
        # Get the commands.
        cmdk = self.flowCartCmd(k, i)
        # Loop through the commands.
        for cmdi in cmdk:
            # Run the command.
            bin.callf(cmdi)
        
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
    def PrepareInputCntl(self, i):
        """
        Write :file:`input.cntl` in for run case *i* in the appropriate folder
        and with the appropriate settings.
        
        :Call:
            >>> cart3d.PrepareInputCntl(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014.06.04 ``@ddalle``: First version
            * 2014.06.06 ``@ddalle``: Low-level functionality for grid folders
            * 2014.09.30 ``@ddalle``: Changed to write only a single case
        """
        # Set the options.
        self.InputCntl.SetCFL(self.opts.get_cfl())
        # Set the flight conditions.
        self.InputCntl.SetMach(self.x.Mach[i])
        self.InputCntl.SetAlpha(self.x.alpha[i])
        self.InputCntl.SetBeta(self.x.beta[i])
        # Set reference values.
        self.InputCntl.SetReferenceArea(self.opts.get_RefArea())
        self.InputCntl.SetReferenceLength(self.opts.get_RefLength())
        # Go safely to root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        # Make folder if necessary.
        if not os.path.isdir(frun): os.mkdir(frun, dmask)
        # Loop through the runs.
        for j in range(self.opts.get_nSeq()):
            # Get the first-order status.
            fo = self.opts.get_first_order(j)
            # Set the status.
            if fo:
                # Run `flowCart` in first-order mode (everywhere)
                self.InputCntl.SetFirstOrder()
            else:
                # Run `flowCart` in second-order mode (cut cells are separate)
                self.InputCntl.SetSecondOrder()
            # Name of output file.
            fout = os.path.join(frun, 'input.%02i.cntl' % j)
            # Write the input file.
            self.InputCntl.Write(fout)
        # Return to original path.
        os.chdir(fpwd)
        
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
    


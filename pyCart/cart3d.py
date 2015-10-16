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
# Date processing
from datetime import datetime
# File system and operating system management
import os, shutil
import subprocess as sp

# Import template class
from cape.cntl import Cntl

# pyCart settings class
from . import options
# pyCart queue itnerface
from . import queue
# Cart3D binary interfaces
from . import bin
# Run directory module
from . import case
# Run directory management
from . import manage
# Alpha-beta / alpha_t-phi business
from . import convert
# Data book and plotting
from . import dataBook

# Functions and classes from other modules
from trajectory import Trajectory

# Import specific file control classes
from inputCntl   import InputCntl
from aeroCsh     import AeroCsh
from preSpecCntl import PreSpecCntl
from config      import Config

# Import triangulation
from tri       import Tri
from cape.geom import RotatePoints

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
        * 2014-07-28 ``@ddalle``: First version
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

    
# Class to read input files
class Cart3d(Cntl):
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
        * 2014-05-28 ``@ddalle``  : First version
        * 2014-06-03 ``@ddalle``  : Renamed class `Cntl` --> `Cart3d`
        * 2014-06-30 ``@ddalle``  : Reduced number of data members
        * 2014-07-27 ``@ddalle``  : `cart3d.Trajectory` --> `cart3d.x`
    """ 
    # Initialization method
    def __init__(self, fname="pyCart.json"):
        """Initialization method for :mod:`cape.cntl.Cntl`"""
        
        # Read settings
        self.opts = options.Options(fname=fname)
        
        #Save the current directory as the root
        self.RootDir = os.getcwd()
        
        # Import modules
        self.ImportModules()
        
        # Process the trajectory.
        self.x = Trajectory(**self.opts['Trajectory'])

        # Job list
        self.jobs = {}
        
        # Set umask
        os.umask(self.opts.get_umask())
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<pyCart.Cart3d(nCase=%i, tri='%s')>" % (
            self.x.nCase,
            self.opts.get_TriFile())
        
    # Function to read the databook.
    def ReadDataBook(self):
        """Read the current data book
        
        :Call:
            >>> cart3d.ReadDataBook()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        :Versions:
            * 2014-12-28 ``@ddalle``: First version
        """
        # Test for an existing data book.
        try:
            self.DataBook
            return
        except AttributeError:
            pass
        # Go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the data book.
        self.DataBook = dataBook.DataBook(self.x, self.opts)
        # Return to original folder.
        os.chdir(fpwd)
        
        
    # Function to collect statistics
    def Aero(self, **kw):
        """Collect force and moment data
        
        :Call:
            >>> cart3d.Aero(comp=None, cons=[], **kw)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *comp*: :class:`str`
                Optional name of class to plot
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Outputs:
            *d*: :class:`dict` (:class:`numpy.ndarray` (:class:`float`))
                Dictionary of mean, min, max, std for each coefficient
        :Versions:
            * 2014-12-12 ``@ddalle``: First version
            * 2014-12-22 ``@ddalle``: Completely rewrote with DataBook class
        """
        # Save current location.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the existing data book.
        self.ReadDataBook()
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.Delete(I)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I)
        # Write the data book to file.
        self.DataBook.Write()
        # Return to original location.
        os.chdir(fpwd)
    
    # Call the correct :mod:`case` module
    def CaseStartCase(self):
        """Start a case by either submitting it or running it
        
        This function relies on :mod:`pyCart.case`, and so it is customized for
        the Cart3D solver only in that it calles the correct *case* module.
        
        :Call:
            >>> pbs = cart3d.CaseStartCase()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        return case.StartCase()
        
    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentIter(self):
        """Get the current iteration number from the appropriate module
        
        This function utilizes the :mod:`cape.case` module, and so it must be
        copied to the definition for each solver's control class
        
        :Call:
            >>> n = cart3d.CaseGetCurrentIter()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *n*: :class:`int` or ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        return case.GetCurrentIter()
            
    # Function to check if the mesh for case i exists
    def CheckMesh(self, i):
        """Check if the mesh for case *i* is prepared.
        
        :Call:
            >>> q = cart3d.CheckMesh(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *q*: :class:`bool`
                Whether or not the mesh for case *i* is prepared
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        # Check input.
        if not type(i).__name__.startswith("int"):
            raise TypeError(
                "Input to :func:`Cart3d.CheckMesh()` must be :class:`int`.")
        # Get the group name.
        fgrp = self.x.GetGroupFolderNames(i)
        # Initialize with "pass" setting.
        q = True
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if (not os.path.isdir(fgrp)):
            os.chdir(fpwd)
            return False
        # Go to the group folder.
        os.chdir(fgrp)
        # Extract options
        opts = self.opts
        # Check for group mesh.
        if not opts.get_GroupMesh():
            # Get the case name.
            frun = self.x.GetFolderNames(i)
            # Check if it's there.
            if (not os.path.isdir(frun)):
                os.chdir(fpwd)
                return False
            # Go to the folder.
            os.chdir(frun)
        # Go to working folder. ('.' or 'adapt??/')
        os.chdir(case.GetWorkingFolder())
        # Check for a mesh file?
        if not opts.get_use_aero_csh(0) or opts.get_jumpstart(0):
            # Intersected mesh file.
            if not os.path.isfile('Components.i.tri'): q = False
            # Mesh file.
            if q and opts.get_mg() > 0:
                # Look for multigrid mesh
                if not os.path.isfile('Mesh.mg.c3d'): q = False
            elif q:
                # Look for original mesh
                if not os.path.isfile('Mesh.c3d'): q = False
        elif opts.get_intersect():
            # Pre-intersect surface files.
            if not os.path.isfile('Components.c.tri'): q = False
            if q and not os.path.isfile('Components.tri'): q = False
        else:
            # Intersected file
            if not os.path.isfile('Components.i.tri'): q = False
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
            * 2014-09-29 ``@ddalle``: First version
        """
        # ---------
        # Case info
        # ---------
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Get name of group.
        fgrp = self.x.GetGroupFolderNames(i)
        # Check the mesh.
        if self.CheckMesh(i):
            return None
        # ------------------
        # Folder preparation
        # ------------------
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check for the group folder and make it if necessary.
        if not os.path.isdir(fgrp):
            self.mkdir(fgrp)
        # Check for groups with common meshes
        if self.opts.get_GroupMesh():
            # Get the group index.
            j = self.x.GetGroupIndex(i)
            # Status update
            print("  Group name: '%s' (index %i)" % (fgrp,j))
            # Go there.
            os.chdir(fgrp)
        else:
            # Check if the run folder exists.
            if not os.path.isdir(frun):
                self.mkdir(frun)
            # Status update.
            print("  Case name: '%s' (index %i)" % (frun,i))
            # Go there.
            os.chdir(frun)
        # ----------
        # Copy files
        # ----------
        # Get the name of the configuration file.
        fxml = os.path.join(self.RootDir, self.opts.get_ConfigFile())
        fpre = os.path.join(self.RootDir, self.opts.get_preSpecCntl())
        fc3d = os.path.join(self.RootDir, self.opts.get_inputC3d())
        # Copy the config file.
        if os.path.isfile(fxml):
            shutil.copyfile(fxml, 'Config.xml')
        # Copy the preSpec file.
        if os.path.isfile(fpre):
            shutil.copyfile(fpre, 'preSpec.c3d.cntl')
        # Copy the cubes input file.
        if os.path.isfile(fc3d):
            shutil.copyfile(fc3d, 'input.c3d')
        # ------------------
        # Triangulation prep
        # ------------------
        # Status update
        print("  Preparing surface triangulation...")
        # Read the mesh.
        self.ReadTri()
        # Revert to initial surface.
        self.tri = self.tri0.Copy()
        # Apply rotations, translations, etc.
        self.PrepareTri(i)
        # Check intersection status.
        if self.opts.get_intersect():
            # Write the tri file as non-intersected; each volume is one CompID
            self.tri.WriteVolTri('Components.tri')
            # Write the existing triangulation with existing CompIDs.
            self.tri.Write('Components.c.tri')
        else:
            # Write the tri file.
            self.tri.Write('Components.i.tri')
        # --------------------
        # Volume mesh creation
        # --------------------
        # Run autoInputs if necessary.
        if self.opts.get_r():
            # Run autoInputs
            bin.autoInputs(self)
            # Fix the name of the triangulation in the 'input.c3d' file
            if self.opts.get_intersect():
                # Read the intersect file.
                lines = open('input.c3d').readlines()
                # Change the triangulation file
                lines[7] = '  Components.i.tri\n'
                # Write the corrected file.
                open('input.c3d', 'w').writelines(lines)
        # Read the resulting preSpec.c3d.cntl file
        self.PreSpecCntl = PreSpecCntl('preSpec.c3d.cntl')
        # Bounding box control...
        self.PreparePreSpecCntl()
        # Check for jumpstart.
        if not self.opts.get_use_aero_csh(0) or self.opts.get_jumpstart(0):
            # Check for intersect step.
            if self.opts.get_intersect():
                # Run intersect.
                bin.intersect('Components.tri', 'Components.o.tri')
                # Read the original triangulation.
                tric = Tri('Components.c.tri')
                # Read the intersected triangulation.
                trii = Tri('Components.o.tri')
                # Read the pre-intersection triangulation.
                tri0 = Tri('Components.tri')
                # Map the Component IDs.
                trii.MapCompID(tric, tri0)
                # Write the triangulation.
                trii.Write('Components.i.tri')
            # Check for verify step.
            if self.opts.get_verify():
                # Run verify.
                bin.verify('Components.i.tri')
            # Run cubes.
            bin.cubes(self)
            # Run mgPrep
            bin.mgPrep(self)
        # Return to original folder.
        os.chdir(fpwd)
        
    # Function to apply special triangulation modification keys
    def PrepareTri(self, i):
        """Rotate/translate/etc. triangulation for given case
        
        :Call:
            >>> cart3d.PrepareTri(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2014-12-01 ``@ddalle``: First version
        """
        # Get function for rotations, etc.
        keys = self.x.GetKeysByType(['translation', 'rotation', 'TriFunction'])
        # Loop through keys.
        for key in keys:
            # Type
            kt = self.x.defns[key]['Type']
            # Filter on which type of triangulation modification it is.
            if kt == "TriFunction":
                # Special triangulation function
                self.PrepareTriFunction(key, i)
            elif kt.lower() == "translation":
                # Component(s) translation
                self.PrepareTriTranslation(key, i)
            elif kt.lower() == "rotation":
                # Component(s) rotation
                self.PrepareTriRotation(key, i)
            
    # Apply a special triangulation function
    def PrepareTriFunction(self, key, i):
        """Apply special triangulation modification function for a case
        
        :Call:
            >>> cart3d.PrepareTriFunction(key, i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: First version
        """
        # Get the function for this *TriFunction*
        func = self.x.defns[key]['Function']
        # Apply it.
        exec("%s(self,%s,i=%i)" % (func, getattr(self.x,key)[i], i))
        
    # Apply a triangulation translation
    def PrepareTriTranslation(self, key, i):
        """Apply a translation to a component or components
        
        :Call:
            >>> cart3d.PrepareTriTranslation(key, i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: First version
        """
        # Get the options for this key.
        kopts = self.x.defns[key]
        # Get the components to translate.
        compID  = self.tri.GetCompID(kopts.get('CompID'))
        # Components to translate in opposite direction
        compIDR = self.tri.GetCompID(kopts.get('CompIDSymmetric', []))
        # Check for a direction
        if 'Vector' not in kopts:
            raise IOError(
                "Rotation key '%s' does not have a 'Vector'." % key)
        # Get the direction and its type
        vec = kopts['Vector']
        tvec = type(vec).__name__
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists.
        if type(pts).__name__  != 'list': pts  = list(pts)
        if type(ptsR).__name__ != 'list': ptsR = list(ptsR)
        # Check the type
        if tvec in ['list', 'ndarray']:
            # Specified directly.
            u = np.array(vec)
        else:
            # Named vector
            u = np.array(self.opts.get_Point(vec))
        # Form the translation vector
        v = u * getattr(self.x,key)[i]
        # Translate the triangulation
        self.tri.Translate(v, i=compID)
        self.tri.Translate(-v, i=compIDR)
        # Loop through translation points.
        for pt in pts:
            # Get point
            x = self.opts.get_Point(pt)
            # Apply transformation.
            self.opts.set_Point(x+v, pt)
        # Loop through translation points.
        for pt in ptsR:
            # Get point
            x = self.opts.get_Point(pt)
            # Apply transformation.
            self.opts.set_Point(x-v, pt)
            
    # Apply a triangulation rotation
    def PrepareTriRotation(self, key, i):
        """Apply a rotation to a component or components
        
        :Call:
            >>> cart3d.PrepareTriRotation(key, i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: First version
        """
        # Get the options for this key.
        kopts = self.x.defns[key]
        # Get the components to translate.
        compID = self.tri.GetCompID(kopts.get('CompID'))
        # Components to translate in opposite direction
        compIDR = self.tri.GetCompID(kopts.get('CompIDSymmetric', []))
        # Symmetry applied to rotation vector.
        kv = kopts.get('VectorSymmetry', [1.0, 1.0, 1.0])
        ka = kopts.get('AngleSymmetry', -1.0)
        # Convert list -> numpy.ndarray
        if type(kv).__name__ == "list": kv = np.array(kv)
        # Check for a direction
        if 'Vector' not in kopts:
            raise KeyError(
                "Rotation key '%s' does not have a 'Vector'." % key)
        # Get the direction and its type.
        vec = kopts['Vector']
        # Check type
        if len(vec) != 2:
            raise KeyError(
                "Rotation key '%s' vector must be exactly two points." % key)
        # Get start and end points of rotation vector.
        v0 = np.array(self.opts.get_Point(kopts['Vector'][0]))
        v1 = np.array(self.opts.get_Point(kopts['Vector'][1]))
        # Symmetry rotation vectors.
        v0R = kv*v0
        v1R = kv*v1
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists.
        if type(pts).__name__  != 'list': pts  = list(pts)
        if type(ptsR).__name__ != 'list': ptsR = list(ptsR)
        # Rotation angle
        theta = getattr(self.x,key)[i]
        # Rotate the triangulation.
        self.tri.Rotate(v0,  v1,  theta,  i=compID)
        self.tri.Rotate(v0R, v1R, ka*theta, i=compIDR)
        # Points to be rotated
        X  = np.array([self.opts.get_Point(pt) for pt in pts])
        XR = np.array([self.opts.get_Point(pt) for pt in ptsR])
        # Apply transformation
        Y  = RotatePoints(X,  v0,  v1,  theta)
        YR = RotatePoints(XR, v0R, v1R, ka*theta)
        # Save the points.
        for j in range(len(pts)):
            # Set the new value.
            self.opts.set_Point(Y[j], pts[j])
        # Save the symmetric points.
        for j in range(len(ptsR)):
            # Set the new value.
            self.opts.set_Point(YR[j], ptsR[j])
    
    
        
    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self):
        """Check if case *i* has the necessary files to run
        
        :Versions:
            * 2015-09-27 ``@ddalle``: First version
        """
        # Check for the surface file.
        if not (os.path.isfile('Components.i.tri')
                or os.path.isfile('Components.tri')):
            n = None
        # Input file.
        if not os.path.isfile('input.00.cntl'): return True
        # Settings file.
        if not os.path.isfile('case.json'): return True
        # Read the settings.
        fc = case.ReadCaseJSON()
        # Check for which mesh file to look for.
        if fc.get_use_aero_csh(0):
            # Mesh file is gone or will be created during aero.csh
            pass
        elif self.opts.get_mg() > 0:
            # Look for the multigrid mesh
            if not os.path.isfile('Mesh.mg.c3d'): return True
        else:
            # Look for the original mesh
            if not os.path.isfile('Mesh.c3d'): return True
        # Apparently no issues.
        return False
        
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
            * 2014-09-30 ``@ddalle``: First version
        """
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if prepared.
        if n is not None: return None
        # Prepare the mesh.
        self.PrepareMesh(i)
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Save current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check for the run directory.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Go there.
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Different processes for GroupMesh and CaseMesh
        if self.opts.get_GroupMesh():
            # Copy the required files.
            for fname in ['input.c3d', 'preSpec.c3d.cntl', 
                    'Mesh.c3d.Info', 'Config.xml']:
                # Source path.
                fsrc = os.path.join('..', fname)
                # Check for the file.
                if os.path.isfile(fsrc):
                    # Copy it.
                    shutil.copy(fsrc, fname)
            # Create links that are available.
            for fname in ['Mesh.c3d', 'Mesh.mg.c3d', 'Mesh.R.c3d',
                    'Components.i.tri', 'Components.tri', 'Components.c.tri']:
                # Source path.
                fsrc = os.path.join(os.path.abspath('..'), fname)
                # Remove the file if it's present.
                if os.path.isfile(fname):
                    os.remove(fname)
                # Check for the file.
                if os.path.isfile(fsrc):
                    # Create a symlink.
                    os.symlink(fsrc, fname)
        else:
            # Get the name of the configuration and input files.
            fxml = os.path.join(self.RootDir, self.opts.get_ConfigFile())
            fc3d = os.path.join(self.RootDir, self.opts.get_inputC3d())
            # Copy the config file.
            if os.path.isfile(fxml):
                shutil.copy(fxml, 'Config.xml')
            # Copy the input.c3d file.
            if os.path.isfile(fc3d):
                shutil.copy(fc3d, 'input.c3d')
        # Get function for setting boundary conditions, etc.
        keys = self.x.GetKeysByType('CaseFunction')
        # Get the list of functions.
        funcs = [self.x.defns[key]['Function'] for key in keys] 
        # Reread the input file(s).
        self.ReadInputCntl()
        self.ReadAeroCsh()
        # Loop through the functions.
        for (key, func) in zip(keys, funcs):
            # Apply it.
            exec("%s(self,%s,i=%i)" % (func, getattr(self.x,key)[i], i))
        # Write the input.cntl and aero.csh file(s).
        self.PrepareInputCntl(i)
        self.PrepareAeroCsh(i)
        # Write a JSON files with flowCart and plot settings.
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Return to original location.
        os.chdir(fpwd)
        
    # Write flowCart options to JSON file
    def WriteCaseJSON(self, i):
        """Write JSON file with `flowCart` and related settings for case *i*
        
        :Call:
            >>> cart3d.WriteCaseJSON(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-12-08 ``@ddalle``: First version
        """
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Check if it exists.
        if not os.path.isdir(frun):
            # Go back and quit.
            os.chdir(fpwd)
            return
        # Go to the folder.
        os.chdir(frun)
        # Write folder.
        f = open('case.json', 'w')
        # Dump the flowCart settings.
        json.dump(self.opts['flowCart'], f, indent=1)
        # Close the file.
        f.close()
        # Return to original location
        os.chdir(fpwd)
        
    # Get last iter
    def GetLastIter(self, i):
        """Get minimum required iteration for a given run to be completed
        
        :Call:
            >>> nIter = cart3d.GetLastIter(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *nIter*: :class:`int`
                Number of iterations required for case *i*
        :Versions:
            * 2014-10-03 ``@ddalle``: First version
        """
        # Check the case
        if self.CheckCase(i) is None:
            return None
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Go there.
        os.chdir(frun)
        # Read the local case.json file.
        fc = case.ReadCaseJSON()
        # Return to original location.
        os.chdir(fpwd)
        # Output
        return fc.get_LastIter()
        
    
        
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
            * 2014-09-30 ``@ddalle``: First version
        """
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Remember current location.
        fpwd = os.getcwd()
        # Go to the root directory.
        os.chdir(self.RootDir)
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Go to the folder.
        os.chdir(frun)
        # Determine number of unique PBS scripts.
        if self.opts.get_nPBS() > 1:
            # If more than one, use unique PBS script for each run.
            nPBS = self.opts.get_nSeq()
        else:
            # Otherwise use a single PBS script.
            nPBS = 1
        
        # Loop through the runs.
        for j in range(nPBS):
            # PBS script name.
            if nPBS > 1:
                # Put PBS number in file name.
                fpbs = 'run_cart3d.%02i.pbs' % j
            else:
                # Use single PBS script with plain name.
                fpbs = 'run_cart3d.pbs'
            # Initialize the PBS script.
            f = open(fpbs, 'w')
            # Write the header.
            self.WritePBSHeader(f, i, j)
            
            # Initialize options to `run_flowCart.py`
            flgs = ''
            # Check for potential need of preprocessing.
            qflg = j==0 and (self.opts.get_use_aero_csh(0) 
                and not self.opts.get_jumpstart(0))
            # Check for `verify` call.
            if qflg and self.opts.get_verify():
                # Call `verify`
                flgs += ' --verify'
            # Check for `intersect` call.
            if qflg and self.opts.get_intersect():
                # Call `intersect`
                flgs += ' --intersect'

            # Simply call the advanced interface.
            f.write('\n# Call the flow_cart/mpi_flowCart/aero.csh interface.\n')
            f.write('run_flowCart.py' + flgs)
            
            # Close the file.
            f.close()
        # Return.
        os.chdir(fpwd)
        
    # Function to prepare "input.cntl" files
    def PreparePreSpecCntl(self):
        """
        Prepare and write :file:`preSpec.c3d.cntl` according to the current
        settings and in the current folder.
        
        :Call:
            >>> cart3d.PreparePreSpecCntl()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Loop through BBoxes
        for BBox in self.opts.get_BBox():
            # Safely get number of refinements.
            n = BBox.get("n", 7)
            # Filter the type.
            if "compID" in BBox:
                # Bounding box specified relative to a component.
                xlim = self.tri.GetCompBBox(**BBox)
            else:
                # Bounding box coordinates given.
                xlim = BBox.get("xlim")
            # Check for degeneracy.
            if (not n) or (xlim is None): continue
            # Add the bounding box.
            self.PreSpecCntl.AddBBox(n, xlim)
        # Loop through the XLevs
        for XLev in self.opts.get_XLev():
            # Safely extract info from the XLev.
            n = XLev.get("n", 0)
            compID = XLev.get("compID", [])
            # Process it into a list of integers (if not already).
            compID = self.tri.config.GetCompID(compID)
            # Check for degeneracy.
            if (not n) or (not compID): continue
            # Add an XLev line.
            self.PreSpecCntl.AddXLev(n, compID)
        # Write the file.
        self.PreSpecCntl.Write('preSpec.c3d.cntl')
        
    # Function to create a PNG for the 3-view of each component.
    def ExplodeTri(self):
        """Create a 3-view of each named or numbered component using TecPlot
        
        This will create a folder called ``subtri/`` in the master directory for
        this *cart3d* object, and it will contain a triangulation for each named
        component inf :file:`Config.xml` along with a three-view plot of each
        component created using TecPlot if possible.
        
        :Call:
            >>> cart3d.ExplodeTri()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Go to root folder safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the triangulation if necessary.
        self.ReadTri()
        # Folder name to hold subtriangulations and 3-view plots
        fdir = "subtri"
        # Create the folder if necessary.
        if not os.path.isdir(fdir): self.mkdir(fdir)
        # Go to the folder.
        os.chdir(fdir)
        # Be safe.
        try:
            # Start creating the figures and subtris.
            self.tri.TecPlotExplode()
        except:
            pass
        # Go to original location.
        os.chdir(fpwd)
        
        
    # Function to unarchive 'adaptXX/' folders (except for newest)
    def UntarAdapt(self, **kw):
        """Tar ``adaptNN/`` folders except for most recent one
        
        :Call:
            >>> cart3d.UntarAdapt()
            >>> cart3d.UntarAdapt(cons=[])
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints
        :Versions:
            * 2015-04-12 ``@ddalle``: First version
        """
        # Get the format.
        fmt=self.opts.get_TarAdapt()
        # Check for directive not to archive.
        if not fmt: return
        # Save current path.
        fpwd = os.getcwd()
        # Get list of indices.
        i = self.x.GetIndices(**kw)
        # Get folder names.
        fruns = self.x.GetFullFolderNames(i)
        # Loop through folders.
        for frun in fruns:
            # Go home.
            os.chdir(self.RootDir)
            # Check for folder.
            if not os.path.isdir(frun): continue
            # Status update
            print(frun)
            # Go to the folder
            os.chdir(frun)
            # Manage the directory.
            manage.ExpandAdapt(fmt)
        # Go back to original directory.
        os.chdir(fpwd)    
        
    # Function to archive 'adaptXX/' folders (except for newest)
    def TarAdapt(self, **kw):
        """Tar ``adaptNN/`` folders except for most recent one
        
        :Call:
            >>> cart3d.TarAdapt()
            >>> cart3d.TarAdapt(cons=[])
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints
        :Versions:
            * 2014-11-14 ``@ddalle``: First version
            * 2014-12-10 ``@ddalle``: Added constraints
        """
        # Get the format.
        fmt=self.opts.get_TarAdapt()
        # Check for directive not to archive.
        if not fmt: return
        # Save current path.
        fpwd = os.getcwd()
        # Get list of indices.
        i = self.x.GetIndices(**kw)
        # Get folder names.
        fruns = self.x.GetFullFolderNames(i)
        # Loop through folders.
        for frun in fruns:
            # Go home.
            os.chdir(self.RootDir)
            # Check for folder.
            if not os.path.isdir(frun): continue
            # Status update
            print(frun)
            # Go to the folder
            os.chdir(frun)
            # Manage the directory.
            manage.TarAdapt(fmt)
        # Go back to original directory.
        os.chdir(fpwd)
        
    # Function to archive 'adaptXX/' folders (except for newest)
    def TarViz(self, **kw):
        """Tar ``adaptNN/`` folders except for most recent one
        
        :Call:
            >>> cart3d.TarViz()
            >>> cart3d.TarViz(cons=[], **kw)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints
        :Versions:
            * 2014-12-18 ``@ddalle``: First version
        """
        # Get the format.
        fmt=self.opts.get_TarViz()
        # Check for directive not to archive.
        if not fmt: return
        # Save current path.
        fpwd = os.getcwd()
        # Loop through folders.
        for i in self.x.GetIndices(**kw):
            # Get folder name.
            frun = self.x.GetFullFolderNames(i)
            # Go home.
            os.chdir(self.RootDir)
            # Check for folder.
            if not os.path.isdir(frun): continue
            # Status update
            print(frun)
            # Go to the folder
            os.chdir(frun)
            # Read the options.
            fc = case.ReadCaseJSON()
            # Check if it's unsteady.
            if not fc.get_unsteady(-1): continue
            # Manage the directory.
            manage.TarViz(fmt)
        # Go back to original directory.
        os.chdir(fpwd)

        
    # Function to archive 'adaptXX/' folders (except for newest)
    def ArchiveCases(self, **kw):
        """Archive completed cases and clean them up if specified
        
        :Call:
            >>> cart3d.ArchiveCases()
            >>> cart3d.ArchiveCases(cons=[], **kw)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints
        :Versions:
            * 2015-01-11 ``@ddalle``: First version
        """
        # Get the format.
        fmt = self.opts.get_ArchiveAction()
        # Check for directive not to archive.
        if not fmt or not self.opts.get_ArchiveFolder(): return
        # Save current path.
        fpwd = os.getcwd()
        # Loop through folders.
        for i in self.x.GetIndices(**kw):
            # Go to root folder.
            os.chdir(self.RootDir)
            # Get folder name.
            frun = self.x.GetFullFolderNames(i)
            # Status update
            print(frun)
            # Check if the case is ready.
            if not os.path.isdir(frun):
                print("  Folder does not exist.")
                continue
            elif self.CheckCaseStatus(i) != 'PASS':
                print("  Case is not marked PASS.")
                continue
            # Go to the folder.
            os.chdir(frun)
            # Archive.
            manage.ArchiveFolder(self.opts)
        # Go back to original directory.
        os.chdir(fpwd)
        
    # Function to apply settings from a specific JSON file
    def ApplyFlowCartSettings(self, **kw):
        """Apply settings from *cart3d.opts* to a set of cases
        
        This rewrites the :file:`case.json` file in the specified directories.
        
        :Call:
            >>> cart3d.ApplyFlowCartSettings(cons=[])
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints
        :Versions:
            * 2014-12-11 ``@ddalle``: First version
        """
        # Apply filter.
        I = self.x.GetIndices(**kw)
        # Loop through cases.
        for i in I:
            # Write the JSON file.
            self.WriteCaseJSON(i)

    # Function to read the "input.cntl" file
    def ReadInputCntl(self):
        """Read the :file:`input.cntl` file

        :Call:
            >>> cart3d.ReadInputCntl()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
        :Versions:
            * 2015-06-13 ``@ddalle``: First version
        """
        # Change to root safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the file.
        self.InputCntl = InputCntl(self.opts.get_InputCntl())
        # Go back to original location
        os.chdir(fpwd)
        
    # Function re read "aero.csh" files
    def ReadAeroCsh(self):
        """Read the :file:`aero.csh` file

        :Call:
            >>> cart3d.ReadAeroCsh()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
        :Versions:
            * 2015-10-14 ``@ddalle``: Revived from deletion
        """
        # Check for adaptation.
        if not np.any(self.opts.get_use_aero_csh()): return
        # Change to root safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the file.
        self.AeroCsh = AeroCsh(self.opts.get_AeroCsh())
        # Go back to original location.
        os.chdir(fpwd)
    
    # Function to prepare "input.cntl" files
    def PrepareInputCntl(self, i):
        """
        Write :file:`input.cntl` for run case *i* in the appropriate folder
        and with the appropriate settings.
        
        :Call:
            >>> cart3d.PrepareInputCntl(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
            * 2014-06-06 ``@ddalle``: Low-level functionality for grid folders
            * 2014-09-30 ``@ddalle``: Changed to write only a single case
        """
        # Extract trajectory.
        x = self.x
        # Process the key types.
        KeyTypes = [x.defns[k]['Type'] for k in x.keys]
        # Go safely to root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        
        # Set the flight conditions.
        # Mach number
        for k in x.GetKeysByType('Mach'):
            self.InputCntl.SetMach(getattr(x,k)[i])
        # Angle of attack
        if 'alpha' in KeyTypes:
            # Find the key.
            k = x.GetKeysByType('alpha')[0]
            # Set the value.
            self.InputCntl.SetAlpha(getattr(x,k)[i])
        # Sideslip angle
        if 'beta' in KeyTypes:
            # Find the key.
            k = x.GetKeysByType('beta')[0]
            # Set the value.
            self.InputCntl.SetBeta(getattr(x,k)[i])
        # Check for total angle of attack.
        if 'alpha_t' in KeyTypes:
            # Find out which key it is.
            k = x.GetKeysByType('alpha_t')[0]
            # Get the value.
            av = getattr(x,k)[i]
            # Check for roll angle.
            if 'phi' in KeyTypes:
                # Kind the ky.
                k = x.GetKeysByType('phi')[0]
                # Get the value.
                rv = getattr(x,k)[i]
            else:
                # Set roll to zero.
                rv = 0.0
            # Convert the values to aoa and aos.
            a, b = convert.AlphaTPhi2AlphaBeta(av, rv)
            # Set them.
            self.InputCntl.SetAlpha(a)
            self.InputCntl.SetBeta(b)
        # Specify list of forces to track with `clic`
        self.InputCntl.RequestForce(self.opts.get_ClicForces())
        # Set reference values.
        self.InputCntl.SetReferenceArea(self.opts.get_RefArea())
        self.InputCntl.SetReferenceLength(self.opts.get_RefLength())
        self.InputCntl.SetMomentPoint(self.opts.get_RefPoint())
        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Get the cut planes.
        XSlices = self.opts.get_Xslices()
        YSlices = self.opts.get_Yslices()
        ZSlices = self.opts.get_Zslices()
        # Process cut planes
        if XSlices: self.InputCntl.SetXSlices(XSlices)
        if YSlices: self.InputCntl.SetYSlices(YSlices)
        if ZSlices: self.InputCntl.SetZSlices(ZSlices)
        # Get the sensors
        PS = self.opts.get_PointSensors()
        LS = self.opts.get_LineSensors()
        # Process sensors
        if PS: self.InputCntl.SetPointSensors(PS)
        if LS: self.InputCntl.SetLineSensors(LS)
        # Loop through the output functional 'optForce's
        for Name, kw in self.opts.get_optForces().items():
            # Set the force.
            self.InputCntl.SetOutputForce(Name, **kw)
        # Loop through the output functional 'optSensor's
        for Name, kw in self.opts.get_optSensors().items():
            # Set the sensor.
            self.InputCntl.SetOutputSensor(Name, **kw)
        # Loop through the output functional 'optMoment's
        for Name, kw in self.opts.get_optMoments().items():
            # Set the sensor.
            self.InputCntl.SetOutputMoment(Name, **kw)
        
        # Loop through the runs.
        for j in range(self.opts.get_nSeq()):
            # Set up the Runge-Kutta coefficients.
            self.InputCntl.SetRungeKutta(self.opts.get_RKScheme(j))
            # Set the CFL number
            self.InputCntl.SetCFL(self.opts.get_cfl(j))
            # Write the number of orders of magnitude for early convergence.
            self.InputCntl.SetNOrders(self.opts.get_nOrders(j))
            # Get the first-order status.
            fo = self.opts.get_first_order(j)
            # Set the status.
            if fo:
                # Run `flowCart` in first-order mode (everywhere)
                self.InputCntl.SetFirstOrder()
            # Get robust mode.
            if self.opts.get_robust_mode(j):
                # Set robust mode.
                self.InputCntl.SetRobustMode()
            # Name of output file.
            fout = os.path.join(frun, 'input.%02i.cntl' % j)
            # Write the input file.
            self.InputCntl.Write(fout)
        # Return to original path.
        os.chdir(fpwd)
        
    # Function prepare the aero.csh files
    def PrepareAeroCsh(self, i):
        """
        Write :file:`aero.csh` for run case *i* in the appropriate folder and
        with the appropriate settings.
        
        :Call:
            >>>car3d.PrepareAeroCsh(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of global pyCart settings object
            *i*: :class:`int`
                Run idnex
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
            * 2014-10-03 ``@ddalle``: Version 2.0
        """
        # Test if it's present (not required)
        try:
            self.AeroCsh
        except Exception:
            return
        # Safely go to the root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Loop through the run sequence.
        for j in range(self.opts.get_nSeq()):
            # Only write aero.csh for adaptive cases.
            if not self.opts.get_use_aero_csh(j): continue
            # Process global options
            self.AeroCsh.SetErrorTolerance(self.opts.get_etol(j))
            self.AeroCsh.SetCFL(self.opts.get_cfl(j))
            self.AeroCsh.SetCFLMin(self.opts.get_cflmin(j))
            self.AeroCsh.SetnIter(self.opts.get_it_fc(j))
            self.AeroCsh.SetnIterAdjoint(self.opts.get_it_ad(j))
            self.AeroCsh.SetnAdapt(self.opts.get_n_adapt_cycles(j))
            self.AeroCsh.SetnRefinements(self.opts.get_maxR(j))
            self.AeroCsh.SetFlowCartMG(self.opts.get_mg_fc(j))
            self.AeroCsh.SetAdjointCartMG(self.opts.get_mg_ad(j))
            self.AeroCsh.SetFMG(self.opts.get_fmg(j))
            self.AeroCsh.SetPMG(self.opts.get_pmg(j))
            self.AeroCsh.SetTM(self.opts.get_tm(j))
            self.AeroCsh.SetAdjFirstOrder(self.opts.get_adj_first_order(j))
            self.AeroCsh.SetLimiter(self.opts.get_limiter(j))
            self.AeroCsh.SetYIsSpanwise(self.opts.get_y_is_spanwise(j))
            self.AeroCsh.SetABuffer(self.opts.get_abuff(j))
            self.AeroCsh.SetFinalMeshXRef(self.opts.get_final_mesh_xref(j))
            self.AeroCsh.SetBinaryIO(self.opts.get_binaryIO(j))
            # Initial mesh inputs; may not be used.
            self.AeroCsh.SetCubesA(self.opts.get_cubes_a(0))
            self.AeroCsh.SetCubesB(self.opts.get_cubes_b(0))
            self.AeroCsh.SetMaxR(self.opts.get_maxR(0))
            self.AeroCsh.SetPreSpec(True)
            # Process the adaptation-specific lists.
            self.AeroCsh.SetAPC(self.opts.get_apc())
            self.AeroCsh.SetMeshGrowth(self.opts.get_mesh_growth())
            self.AeroCsh.SetnIterList(self.opts.get_ws_it())
            # Destination file name
            fout = os.path.join(frun, 'aero.%02i.csh' % j)
            # Write the input file.
            self.AeroCsh.Write(fout)
            # Make it executable.
            os.chmod(fout, 0777 - self.opts.get_umask())
        # Go back home.
        os.chdir(fpwd)
        # Done
        return None
        
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
            * 2014-06-05 ``@ddalle``: First version
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
            * 2014-06-04 ``@ddalle``: First version
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
            * 2014-06-04 ``@ddalle``: First version
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
            * 2014-06-04 ``@ddalle``: First version
        """
        # Check for the attribute.
        if not hasattr(self, 'LoadsTRI'):
            self.GetLoadsTRI()
        # Write.
        self.LoadsTRI.Write(self.x)
        return None
        





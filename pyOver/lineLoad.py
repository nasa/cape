"""
OVERFLOW Sectional Loads Module: :mod:`pyOver.lineLoad`
========================================================

This module contains functions for reading and processing sectional loads.  It
is a submodule of :mod:`pyFun.dataBook`.

:Versions:
    * 2016-12-20 ``@ddalle``: Started
"""

# File interface
import os, glob, shutil
# Basic numerics
import numpy as np
# Date processing
from datetime import datetime

# Utilities or advanced statistics
from . import util
from . import case
from . import config
from cape import tar
# Line load template
import cape.lineLoad


# Data book of line loads
class DBLineLoad(cape.lineLoad.DBLineLoad):
    """Line load (sectional load) data book for one group
    
    :Call:
        >>> DBL = DBLineLoad(x, opts. comp, conf=None, RootDir=None)
    :Inputs:
        *x*: :class:`cape.trajectory.Trajectory`
            Trajectory/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *comp*: :class:`str`
            Name of line load component
        *conf*: {``"None"``} | :class:`cape.config.Config`
            Surface configuration interface
        *RootDir*: {``"None"``} | :class:`str`
            Root directory for the configuration
    :Outputs:
        *DBL*: :class:`pyOver.lineLoad.DBLineLoad`
            Instance of line load data book
        *DBL.nCut*: :class:`int`
            Number of *x*-cuts to make, based on options in *cart3d*
        *DBL.RefL*: :class:`float`
            Reference length
        *DBL.MRP*: :class:`numpy.ndarray` shape=(3,)
            Moment reference center
        *DBL.x*: :class:`numpy.ndarray` shape=(*nCut*,)
            Locations of *x*-cuts
        *DBL.CA*: :class:`numpy.ndarray` shape=(*nCut*,)
            Axial force sectional load, d(CA)/d(x/RefL))
    :Versions:
        * 2015-09-16 ``@ddalle``: First version
    """
    
    # Get component ID numbers
    def GetCompID(self):
        """Create list of component IDs
        
        :Call:
            >>> DBL.GetCompID()
        :Inputs:
            *DBL*: :class:`cape.lineLoad.DBLineLoad`
                Instance of line load data book
        :Versions:
            * 2016-12-22 ``@ddalle``: First version, extracted from __init__
        """
        # Figure out reference component
        self.CompID = self.opts.get_DataBookCompID(self.comp)
        # Get input files
        fmixsur  = self.opts.get_DataBook_mixsur(self.comp)
        fsplitmq = self.opts.get_DataBook_splitmq(self.comp)
        # Get absolute file paths
        if not os.path.isabs(fmixsur):
            fmixsur = os.path.join(self.RootDir, fmixsur)
        if not os.path.isabs(fsplitmq):
            fsplitmq = os.path.join(self.RootDir, fsplitmq)
        # Save files
        self.mixsur  = fmixsur
        self.splitmq = fsplitmq
        # Get Q/X files
        self.fqi = self.opts.get_DataBook_QIn(self.comp)
        self.fxi = self.opts.get_DataBook_XIn(self.comp)
        self.fqo = self.opts.get_DataBook_QOut(self.comp)
        self.fxo = self.opts.get_DataBook_XOut(self.comp)
        # Read MapBC
        try:
            # Name of the MapBC file (from the input, not a case)
            fmapbc = os.path.join(self.RootDir, self.opts.get_MapBCFile())
            # Read the MapBC
            self.MapBC = mapbc.MapBC(fmapbc)
        except Exception:
            pass
        # Make sure it's not a list
        if type(self.CompID).__name__ == 'list':
            # Take the first component
            self.RefComp = self.CompID[0]
        else:
            # One component listed; use it
            self.RefComp = self.CompID
        # Try to read the configuration
        try:
            # Read the MIXSUR.I file
            self.conf = config.ConfigMIXSUR(self.mixsur)
        except Exception:
            pass
        # Try to get all components
        try:
            # Use the configuration interface
            self.CompID = self.conf.GetCompID(self.CompID)
        except Exception:
            pass
    
    # Get file
    def GetTriqFile(self):
        """Get most recent ``triq`` file and its associated iterations
        
        :Call:
            >>> qpre, fq, n, i0, i1 = DBL.GetTriqFile()
        :Inputs:
            *DBL*: :class:`pyCart.lineLoad.DBLineLoad`
                Instance of line load data book
        :Outputs:
            *qpre*: {``False``}
                Whether or not to convert file from other format
            *fq*: :class:`str`
                Name of ``triq`` file
            *n*: :class:`int`
                Number of iterations included
            *i0*: :class:`int`
                First iteration in the averaging
            *i1*: :class:`int`
                Last iteration in the averaging
        :Versions:
            * 2016-12-19 ``@ddalle``: Added to the module
        """
        # Get properties of triq file
        fq, n, i0, i1 = GetQFile(self.fqi)
        # Get the corresponding .triq file name
        ftriq = os.path.join('lineload', 'grid.i.triq')
        # Check for 'q.strt'
        if os.path.isfile(fq):
            # Source file exists
            fsrc = os.path.realpath(fq)
        else:
            # No source just yet
            fsrc = None
        # Check if the TRIQ file exists
        if os.path.isfile(ftriq) and os.path.isfile(fsrc):
            # Check modification dates
            if os.path.getmtime(ftriq) > os.path.getmtime(fsrc):
                # 'grid.i.triq' exists, but Q file is newer
                qpre = True
            else:
                # Triq file exists and is up-to-date
                qpre = False
        else:
            # Need to run ``overint`` to get triq file
            qpre = True
        # Output
        return qpre, fq, n, i0, i1
    
    # Preprocess triq file (convert from PLT)
    def PreprocessTriq(self, fq, **kw):
        """Perform any necessary preprocessing to create ``triq`` file
        
        :Call:
            >>> ftriq = DBL.PreprocessTriq(fq, qpbs=False, f=None)
        :Inputs:
            *DBL*: :class:`pyFun.lineLoad.DBLineLoad`
                Line load data book
            *ftriq*: :class:`str`
                Name of q file
            *qpbs*: ``True`` | {``False``}
                Whether or not to create a script and submit it
            *f*: {``None``} | :class:`file`
                File handle if writing PBS script
        :Versions:
            * 2016-12-20 ``@ddalle``: First version
            * 2016-12-21 ``@ddalle``: Added PBS
        """
        # Do the SPLITMQ and MIXSUR files exist?
        qsplitm = os.path.isfile(self.splitmq)
        qmixsur = os.path.isfile(self.mixsur)
        # Local names for input files
        fsplitmq = 'splitmq.%s.i' % self.comp
        fsplitmx = 'splitmx.%s.i' % self.comp
        fmixsur  = 'mixsur.%s.i' % self.comp
        # Source q file is in parent folder
        fsrc = os.path.join('..', fq)
        # Iteration number from volume and surface files
        tvol = case.checkqt(fsrc)
        # If this file does not exist, nothing is going to work.
        # Cannot have x.save file if we need to create triq file
        if os.path.isfile("x.save"): os.remove("x.save")
        if os.path.isfile("q.save"): os.remove("q.save")
        # ----------------------------
        # Prepare SPLITMQ if necessary
        # ----------------------------
        if not qsplitm: splitmq = False
        # Use this while loop as a method to use ``break``
        while qsplitm:
            # Check for "q.srf" file
            if os.path.isfile("q.srf"):
                # Get iteration number
                tsrf = case.checkqt(tsrf)
                # Check if it's up to date
                if tsrf < tvol:
                    # Exists but out-of-date
                    os.remove("q.srf")
                else:
                    # Up-to-date; get out of here
                    qspltimq = False
                    break
            # Name of parent "q.srf" file from parent directory
            fqo = self.fqo
            # Expand "q.srf" file
            if fqo is None:
                # No target "q.srf" file to link
                fqsrc = None
            else:
                # Expand to parent folder
                fqsrc = os.path.join('..', fqo)
            break
            # Expand "x.srf" file
            if fxo is None:
                # No target "x.srf" file to link
                fxsrc = None
            else:
                # Expand to parent folder
                fxsrc = os.path.join('..', fxo)
            # If "q.srf" file exists, get iteration number
            if fqo and os.path.isfile(fqsrf):
                # Get actual iteration number
                tsrf = case.checkqt(fqsrc)
            else:
                # No iteration number
                tsrf = 0
            # Check for existing SRF file
            if fqo is None:
                # No q.srf file to link
                qsplitmq = True
            elif tsrf < tvol:
                # q.srf file may exist, but it is out of date
                qsplitmq = False
            else:
                # Link target surface solution so OVERINT uses it
                os.symlink(fqsrc, "q.save")
                qsplitmq = False
            # Check for existing x.srf file
            if fxo is None:
                # No x.srf target
                qsplitmx = True
            elif 
                # Path to target x.srf file in parent
                fxsrc = os.path.join('..', fxo)
                # Check if file exists in parent folder
                if os.path.isfile(fxsrc):
                    # Link it to "x.save" so OVERINT uses it
                    os.symlink(fxsrc, "x.save")
                    qsplitmx = False
                else:
                    # Still need to run SPLITMX because target did not exist
                    qsplitmx = True
        else:
            # Copy the source file to this folder
            os.symlink(fsrc, 'x.save')
        # If these files exist, copy to this folder
        if qsplitq: shutil.copy(self.splitmq, fsplitmq)
        if qmixsur: shutil.copy(self.mixsur,  fmixsur)
        # Check for PBS script
        if kw.get('qpbs', False):
            # Get the file handle
            f = kw.get('f')
            # Check for open file
            if f is None:
                raise ValueError(
                    "No open file handle for preprocessing TRIQ file")
            # Check for ``splitmq``
            if qsplitq:
                f.write("\n# Extract surface and L=2 from solution\n")
                f.write("splitmq < %s > splitmq.%s.o\n" %
                    (fsplitmq, self.comp))
            # Check for ``mixsur``
            if qmixsur:
                f.write("\n# Use mixsur to create triangulation\n")
                f.write("mixsur < %s > mixsur.%s.o\n" % (fmixsur, self.comp))
        else:
            # Check for ``splitmq``
            if qsplitq:
                # Command to run splitmq
                cmd = "splitmq < %s > spltimq.%s.o" % (fsplitmq, self.comp)
                # Status update
                print("    %s" % cmd)
                # Run ``splitmq``
                ierr = os.system(cmd)
                # Check for errors
                if ierr:
                    raise SystemError("Failure while running ``splitmq``")
            # Check for ``mixsur``
            if qmixsur:
                # Command to mixsur
                cmd = "mixsur < %s > mixsur.%s.o" % (fmixsur, self.comp)
                # Status update
                print("    %s" % cmd)
                # Run ``mixsur``
                ierr = os.system(cmd)
                # Check for errors
                if ierr:
                    raise SystemError("Failure while running ``mixsur``")
        
# class DBLineLoad
    

# Line loads
class CaseLL(cape.lineLoad.CaseLL):
    """Individual class line load class
    
    :Call:
        >>> LL = CaseLL(cart3d, i, comp)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Master pyCart interface
        *i*: :class:`int`
            Case index
        *comp*: :class:`str`
            Name of line load group
    :Outputs:
        *LL*: :class:`pyCart.lineLoad.CaseLL`
            Instance of individual case line load interface
        *LL.nCut*: :class:`int`
            Number of *x*-cuts to make, based on options in *cart3d*
        *LL.nIter*: :class:`int`
            Last iteration in line load file
        *LL.nStats*: :class:`int`
            Number of iterations in line load file
        *LL.RefL*: :class:`float`
            Reference length
        *LL.MRP*: :class:`numpy.ndarray` shape=(3,)
            Moment reference center
        *LL.x*: :class:`numpy.ndarray` shape=(*nCut*,)
            Locations of *x*-cuts
        *LL.CA*: :class:`numpy.ndarray` shape=(*nCut*,)
            Axial force sectional load, d(CA)/d(x/RefL))
    :Versions:
        * 2015-09-16 ``@ddalle``: First version
        * 2016-06-07 ``@ddalle``: Subclassed
    """
    pass
# class CaseLL

# Class for seam curves
class CaseSeam(cape.lineLoad.CaseSeam):
    """Seam curve interface
    
    :Call:
        >>> S = CaseSeam(fname, comp='entire', proj='LineLoad')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *comp*: :class:`str`
            Name of the component
    :Outputs:
        *S* :class:`cape.lineLoad.CaseSeam`
            Seam curve interface
        *S.ax*: ``"x"`` | ``"y"`` | ``"z"``
            Name of coordinate being held constant
        *S.x*: :class:`float` | {:class:`list` (:class:`np.ndarray`)}
            x-coordinate or list of seam x-coordinate vectors
        *S.y*: :class:`float` | {:class:`list` (:class:`np.ndarray`)}
            y-coordinate or list of seam y-coordinate vectors
        *S.z*: {:class:`float`} | :class:`list` (:class:`np.ndarray`)
            z-coordinate or list of seam z-coordinate vectors
    :Versions:
        * 2016-06-09 ``@ddalle``: First version
    """
    pass
# class CaseSeam


# Function to determine newest triangulation file
def GetQFile(fqi="q.pyover.p3d"):
    """Get most recent OVERFLOW ``q`` file and its associated iterations
    
    Averaged solution files, such as ``q.avg`` take precedence.
    
    :Call:
        >>> fq, n, i0, i1 = GetQFile(fqi="q.pyover.p3d")
    :Inputs:
        *fqi*: {q.pyover.p3d} | q.pyover.avg | q.pyover.vol | :class:`str`
            Target Overflow solution file after linking most recent files
    :Outputs:
        *fq*: :class:`str`
            Name of ``q`` file
        *n*: :class:`int`
            Number of iterations included
        *i0*: :class:`int`
            First iteration in the averaging
        *i1*: :class:`int`
            Last iteration in the averaging
    :Versions:
        * 2016-12-30 ``@ddalle``: First version
    """
    # Link grid and solution files
    case.LinkQ()
    case.LinkX()
    # Check for the input file
    if os.path.isfile(fqi):
        # Use the file (may be a link, in fact it usually is)
        fq = fqi
    else:
        # Best Q file available (usually "q.avg" or "q.save")
        fq = case.GetQ()
    # Check for q.avg iteration count
    n = case.checkqavg(fq)
    # Read the current "time" parameter
    i1 = case.checkqt(fq)
    # Get start parameter
    if (n is not None) and (i1 is not None):
        # Calculate start iteration
        i0 = i1 - n + 1
    else:
        # Cannot determine start iteration
        i0 = None
    # Output
    return fq, n, i0, i1
# def GetQFile
            

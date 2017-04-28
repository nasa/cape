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
from cape import config
# Line load template
import cape.lineLoad


# Create grid.itriq
def PreprocessTriqOverflow(DB, fq, fdir="lineload"):
    """Perform any necessary preprocessing to create ``triq`` file
    
    :Call:
        >>> PreprocessTriqOverflow(DB, fq)
    :Inputs:
        *DB*: :class:`pyOver.dataBook.DBTriqFM` | :class:`pyOver.lineLoad.DBLineLoad`
            TriqFM or line load data book
        *q*: :class:`str`
            Name of q file
    :Versions:
        * 2016-12-20 ``@ddalle``: First version
        * 2016-12-21 ``@ddalle``: Added PBS
        * 2017-04-13 ``@ddalle``: Wrote single version for LL and TriqFM
    """
   # -------
   # Options
   # -------
    # Get input files
    fusurp   = DB.opts.get_DataBook_usurp(DB.comp)
    fmixsur  = DB.opts.get_DataBook_mixsur(DB.comp)
    fsplitmq = DB.opts.get_DataBook_splitmq(DB.comp)
    ffomo    = DB.opts.get_DataBook_fomo(DB.comp)
    # Get absolute file paths
    if (fusurp) and (not os.path.isabs(fusurp)):
        fusurp = os.path.join(DB.RootDir, fusurp)
    if (fmixsur) and (not os.path.isabs(fmixsur)):
        fmixsur = os.path.join(DB.RootDir, fmixsur)
    if (fsplitmq) and (not os.path.isabs(fsplitmq)):
        fsplitmq = os.path.join(DB.RootDir, fsplitmq)
    if (ffomo) and (not os.path.isabs(ffomo)):
        ffomo = os.path.join(DB.RootDir, ffomo)
    # Check for the files
    qfusurp  = (fusurp!=None)   and os.path.isfile(fusurp)
    qfmixsur = (fmixsur!=None)  and os.path.isfile(fmixsur)
    qfsplitm = (fsplitmq!=None) and os.path.isfile(fsplitmq)
    # Check for a folder we can copy MIXSUR/USURP files from 
    qfomo = (ffomo!=None) and os.path.isdir(ffomo)
    # Get Q/X files
    fqi = DB.opts.get_DataBook_QIn(DB.comp)
    fxi = DB.opts.get_DataBook_XIn(DB.comp)
    fqo = DB.opts.get_DataBook_QOut(DB.comp)
    fxo = DB.opts.get_DataBook_XOut(DB.comp)
    # If there's no mixsur file, there's nothing we can do
    if not (qfmixsur or qfusurp):
        raise RuntimeError(
            ("No 'mixsur' or 'overint' or 'usurp' input file found ") +
            ("for component '%s'" % DB.comp))
    # Local names for input files
    lsplitmq = 'splitmq.%s.i' % DB.comp
    lsplitmx = 'splitmx.%s.i' % DB.comp
    lmixsur  = 'mixsur.%s.i' % DB.comp
    # Source *q* file is in parent folder
    fqvol = fq
    # Source *x* file if needed
    fxvol = os.path.join('..', "x.pyover.p3d")
    # If this file does not exist, nothing is going to work.
    if not os.path.isfile(fqvol):
        os.chdir('..')
        return
    ## If we're in PreprocessTriq, all x/q files are out-of-date
    #for f in ["grid.in", "x.srf", "x.vol", "q.save", "q.srf", "q.vol"]:
    #    # Check if file esists
    #    if os.path.isfile(f): os.remove(f)
   # -------------------------------------
   # Determine MIXSUR output folder status
   # -------------------------------------
    # Check status of self.fomodir folder
    if qfomo:
        # List of required mixsur files
        fmo = [
            "grid.i.tri", "grid.bnd", "grid.ib",  "grid.ibi",
            "mixsur.fmp", "grid.map", "grid.nsf", "grid.ptv"
        ]
        # Initialize a flag that all these files exist
        qmixsur = True
        qusurp = True
        # Loop through files
        for f in fmo:
            # Check if the file exists
            if not os.path.isfile(os.path.join(ffomo, f)):
                # Missing file
                print("Label 0021: missing mixsur file '%s'" % f)
                qmixsur = False
                break
        # List of required usurp files
        fus = ["grid.i.tri", "panel_weights.dat", "usurp.map"]
        # Loop through ``usurp`` files
        for f in fus:
            # Check if the file exists
            if not os.path.isfile(os.path.join(ffomo, f)):
                # Missing file
                print("Label 0023: missing usurp file '%s'" % f)
                qusurp = False
                break
        print("Label 0024: qmixsur=%s, qusurp=%s" % (qmixsur, qusurp))
    else:
        # Must run mixsur or usurp
        qmixsur = False
        qusurp = False
    # Copy files if ``mixsur`` output found
    if (qmixsur):
        # Loop through files
        for f in fmo:
            # If file exists in `lineload/` folder, delete it
            if os.path.isfile(f): os.remove(f)
            # Link file
            fsrc = os.path.join(ffomo, f)
            os.symlink(fsrc, f)
    # Copy files if ``usurp`` output found
    if (qusurp):
        # Loop through files
        for f in fus:
            # If file exists in `lineload/` folder, delete it
            if os.path.isfile(f): os.remove(f)
            # Link file
            fsrc = os.path.join(ffomo, f)
            os.symlink(fsrc, f)
   # ------------------------
   # Determine SPLITMQ status
   # ------------------------
    print("Label 0028: qfsplitm=%s" % qfsplitm)
    # Use this while loop as a method to use ``break``
    if qfsplitm:
        # Source file option(s)
        fqo = DB.opts.get_DataBook_QSurf(DB.comp)
        fxo = DB.opts.get_DataBook_XSurf(DB.comp)
        
        # Get absolute path
        if fqo is None:
            # No source file
            fqsrf = os.path.join('..', 'q.pyover.srf')
        else:
            # Get path to parent folder
            fqsrf = os.path.join('..', fqo)
        if fxo is None:
            # No target file
            fxsrf = os.path.join('..', 'x.pyover.srf')
        else:
            # Get path to parent folder
            fxsrf = os.path.join('..', fxo)
        # Check for existing split surface file in lineload/
        if not os.path.isfile(fxsrf) and os.path.isfile("grid.in"):
            # Use the existing grid file in the lineload/ folder
            fxsrf = "grid.in"
        # Check for existing split surface file in lineload/
        if not os.path.isfile(fqsrf) and os.path.isfile("q.save"):
            # Use the existing grid file in the lineload/ folder
            fqsrf = "q.save"
        print("Label 0029: fxsrf='%s' (%s)" % (fxsrf, os.path.isfile(fxsrf)))
        print("Label 0030: fqsrf='%s' (%s)" % (fqsrf, os.path.isfile(fqsrf)))
        # Check for "q.srf" file
        if fqsrf and os.path.isfile(fqsrf):
            # Get iteration number
            tvol = case.checkqt(fqvol)
            tsrf = case.checkqt(fqsrf)
            print("Label 0041: tvol=%s, tsrf=%s" % (tvol, tsrf))
            print("Label 0042: fqsrf='%s', fxsrf='%s'" % (fqsrf,fxsrf))
            # Check if it's up to date
            if tsrf < tvol:
                # Exists but out-of-date
                qsplitmq = True
                qsplitmx = True
            elif fxsrf and os.path.isfile(fxsrf):
                # Up-to-date, and surface grid good too
                qsplitmq = False
                qsplitmx = False
            else:
                # Up-to-date; but need to create 'x.srf'
                qspltimq = False
                qsplitmx = True
        else:
            # No candidate "q.srf" file from parent directory
            qsplitmq = True
            qsplitmx = True
    else:
        # Do not run splitmq
        qsplitmq = False
        qsplitmx = False
   # ---------------------
   # Prepare SPLITMQ files
   # ---------------------
    # Whether or not to split
    qsplitq = qsplitmq or qsplitmx
    # Copy "splitmq"/"splitmx" input template
    if qsplitq: shutil.copy(fsplitmq, "splitmq.i")
    # Copy "mixsur"/"overint" input file
    shutil.copy(fmixsur, lmixsur)
    shutil.copy(fmixsur, "mixsur.i")
    # Prepare files for ``splitmq``
    if qsplitmq:
        # Link parent Q volume
        os.symlink(fqvol, "q.vol")
        # Edit the SPLITMQ input file
        case.EditSplitmqI("splitmq.i", lsplitmq, "q.vol", "q.save")
        # Command to run splitmq
        cmd = "splitmq < %s >& splitmq.%s.o" % (lsplitmq, DB.comp)
        # Status update
        print("    %s" % cmd)
        # Run ``splitmq``
        ierr = os.system(cmd)
        # Check for errors
        if ierr:
            raise SystemError("Failure while running ``splitmq``")
    elif qfsplitm:
        # Link parent *q.srf* to "q.save" so OVERINT uses it
        os.symlink(fqsrf, "q.save")
    else:
        # Use volume grid
        os.symlink(fqvol, "q.vol")
    # Prepare files for ``splitmx``
    if qsplitmx:
        # Link parent X volume
        os.symlink(fxvol, "x.vol")
        # Edit the SPLITMX input file
        case.EditSplitmqI("splitmq.i", lsplitmx, "x.vol", "grid.in")
        # Command to run splitmx
        cmd = "splitmx < %s >& splitmx.%s.o" % (lsplitmx, DB.comp)
        # Status update
        print("    %s" % cmd)
        # Run ``splitmx``
        ierr = os.system(cmd)
        # Check for errors
        if ierr:
            raise SystemError("Failure while running ``splitmx``")
    elif qfsplitm:
        # Link parent *x.srf* to "x.save" so OVERINT uses it
        os.symlink(fxsrf, "grid.in")
    else:
        # Link parent volume grid
        os.symlink(fxvol, "grid.in")
   # ----------------------
   # Prepare ``grid.i.tri``
   # ----------------------
    # Check for ``mixsur`` or ``usurp``
    print("Label 0081: qfusurp=%s, qusurp=%s, qmixsur=%s"
        % (qfusurp, qusurp, qmixsur))
    if qfusurp or (not qusurp):
        # Command to usurp
        cmd = ("usurp -v --watertight --disjoin=yes < %s >& usurp.%s.o"
            % (fmixsur, DB.comp))
        # Status update
        print("    %s" % cmd)
        # Run ``usurp``
        ierr = os.system(cmd)
        # Check for errors
        if ierr:
            raise SystemError("Failure while running ``usurp``")
    elif (not qfusurp) and (not qusurp) and (not qmixsur):
        # Command to mixsur
        cmd = "mixsur < %s >& mixsur.%s.o" % (fmixsur, DB.comp)
        # Status update
        print("    %s" % cmd)
        # Run ``mixsur``
        ierr = os.system(cmd)
        # Check for errors
        if ierr:
            raise SystemError("Failure while running ``mixsur``")
   # -----------------------
   # Prepare ``grid.i.triq``
   # -----------------------
    # Check for ``mixsur`` or ``usurp``
    if qfusurp or qusurp:
        # Command to usurp
        cmd = ("usurp -v --use-map < %s >& usurp.%s.o"
            % (lmixsur, DB.comp))
        # Status update
        print("    %s" % cmd)
        # Run ``usurp
        ierr = os.system(cmd)
        # Go back up
        os.chdir("..")
        # Check for errors
        if ierr:
            raise SystemError("Failure while running ``usurp``")
    else:
        # Command to overint
        cmd = "overint < %s >& overint.%s.o" % (lmixsur, DB.comp)
        # Status update
        print("    %s" % cmd)
        # Run ``overint``
        ierr = os.system(cmd)
        # Go back up to run directory
        os.chdir("..")
        # Check for errors
        if ierr:
            raise SystemError("Failure while running ``overint``")
# def PreprocessTriq



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
        fusurp   = self.opts.get_DataBook_usurp(self.comp)
        fsplitmq = self.opts.get_DataBook_splitmq(self.comp)
        ffomo    = self.opts.get_DataBook_fomo(self.comp)
        # Get absolute file paths
        if (fmixsur) and (not os.path.isabs(fmixsur)):
            fmixsur = os.path.join(self.RootDir, fmixsur)
        if (fusurp) and (not os.path.isabs(fusurp)):
            fusurp = os.path.join(self.RootDir, fusurp)
        if (fsplitmq) and (not os.path.isabs(fsplitmq)):
            fsplitmq = os.path.join(self.RootDir, fsplitmq)
        if (ffomo) and (not os.path.isabs(ffomo)):
            ffomo = os.path.join(self.RootDir, ffomo)
        # Save files
        self.mixsur  = fmixsur
        self.usurp   = fusurp
        self.splitmq = fsplitmq
        self.fomodir = ffomo
        # Get Q/X files
        self.fqi = self.opts.get_DataBook_QIn(self.comp)
        self.fxi = self.opts.get_DataBook_XIn(self.comp)
        self.fqo = self.opts.get_DataBook_QOut(self.comp)
        self.fxo = self.opts.get_DataBook_XOut(self.comp)
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
            self.conf = config.ConfigMIXSUR(self.usurp)
        except Exception:
            # Try the MIXSUR input file
            try:
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
        fq, n, i0, i1 = case.GetQFile(self.fqi)
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
        if os.path.isfile(ftriq) and fsrc and os.path.isfile(fsrc):
            # Check modification dates
            if os.path.getmtime(ftriq) < os.path.getmtime(fsrc):
                # 'grid.i.triq' exists, but Q file is newer
                qpre = True
            else:
                # Triq file exists and is up-to-date
                qpre = False
        else:
            # Need to run ``overint`` to get triq file
            qpre = True
        # If "grid.i.triq" file is up-to-date, check if it has the component
        if not qpre:
            # Read the existing "mixsur.i" file
            fmixsur = os.path.join("lineload", "mixsur.i")
            # Open the file
            if os.path.isfile(fmixsur):
                # Read that file
                cfg = config.ConfigMIXSUR(fmixsur)
                compID = self.opts.get_DataBookCompID(self.comp)
                # Check if the component is present
                if compID not in cfg.faces:
                    # The "grid.i.triq" does not include the component we need
                    qpre = True
            else:
                # No knowledge of components; must run overint 
                qpre = True
        # Output
        return qpre, fq, n, i0, i1
        
    # Write triload.i input file
    def WriteTriloadInput(self, ftriq, i, **kw):
        """Write ``triload.i`` input file for ``triloadCmd``
        
        This versions uses a fixed input solution/grid file, ``"grid.i.triq"``
        
        :Call:
            >>> DBL.WriteTriloadInput(ftriq, i, **kw)
        :Inputs:
            *DBL*: :class:`pyOver.lineLoad.DBLineLoad`
                Line load data book
            *ftriq*: :class:`str`
                Name of the ``triq`` file to analyze
            *i*: :class:`int`
                Case number
        :Keyword arguments:
            *mach*: :class:`float`
                Override Mach number
            *Re*: :class:`float`
                Override Reynolds number input
            *gamma*: :class:`float`
                Override ratio of specific heats
            *MRP*: :class:`float`
                Override the moment reference point from the JSON input file
        :Versions:
            * 2017-01-11 ``@ddalle``: First separate version
        """
        # Point to a fixed "grid.i.triq" file
        self.WriteTriloadInputBase("grid.i.triq", i, **kw)
    
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
            *f*: {``None``} | :class:`file`
                File handle if writing PBS script
        :Versions:
            * 2016-12-20 ``@ddalle``: First version
            * 2016-12-21 ``@ddalle``: Added PBS
            * 2017-04-06 ``@ddalle``: Support ``usurp``, remove PBS
        """
        # Call local function
        PreprocessTriqOverflow(self, fq)
        
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
            

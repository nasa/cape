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
        self.mixsur  = os.path.join(self.RootDir, fmixsur)
        self.splitmq = os.path.join(self.RootDir, fsplitmq)
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
            >>> qtriq, ftriq, n, i0, i1 = DBL.GetTriqFile()
        :Inputs:
            *DBL*: :class:`pyCart.lineLoad.DBLineLoad`
                Instance of line load data book
        :Outputs:
            *qtriq*: {``False``}
                Whether or not to convert file from other format
            *ftriq*: :class:`str`
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
        fplt, n, i0, i1 = GetPltFile()
        # Get the corresponding .triq file name
        ftriq = fplt.rstrip('.plt') + '.triq'
        # Check if the TRIQ file exists
        if os.path.isfile(ftriq):
            # No conversion needed
            qtriq = False
        else:
            # Need to convert PLT file to TRIQ
            qtriq = True
        # Output
        return qtriq, ftriq, n, i0, i1
    
    # Preprocess triq file (convert from PLT)
    def PreprocessTriq(self, ftriq, **kw):
        """Perform any necessary preprocessing to create ``triq`` file
        
        :Call:
            >>> ftriq = DBL.PreprocessTriq(ftriq, qpbs=False, f=None)
        :Inputs:
            *DBL*: :class:`pyFun.lineLoad.DBLineLoad`
                Line load data book
            *ftriq*: :class:`str`
                Name of triq file
            *qpbs*: ``True`` | {``False``}
                Whether or not to create a script and submit it
            *f*: {``None``} | :class:`file`
                File handle if writing PBS script
        :Versions:
            * 2016-12-20 ``@ddalle``: First version
            * 2016-12-21 ``@ddalle``: Added PBS
        """
        # Do the SPLITMQ and MIXSUR files exist?
        qsplitq = os.path.isfile(self.splitmq)
        qmixsur = os.path.isfile(self.mixsur)
        # Local names for input files
        fsplitmq = 'splitmq.%s.i' % self.comp
        fmixsur  = 'mixsur.%s.i' % self.comp
        # If these files exist, copy to this folder
        if qsplitq: shutil.copy(self.splitmq, ftplitmq)
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
            # Run a script
            f.write("\n# Convert PLT file to TRIQ\n")
            f.write("pf_Plt2Triq.py %s --mach %s\n" % (fplt, self.mach))
        else:
            # Read the plt information
            plt.Plt2Triq(fplt, ftriq, mach=self.mach)
        
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
def GetQFile():
    """Get most ``q.*`` solution file and its associated iterations
    
    :Call:
        >>> fq, n, i0, i1 = GetQFile()
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
        * 2016-12-20 ``@ddalle``: First version
    """
    # Read *rc* options to figure out iteration values
    rc = case.ReadCaseJSON()
    # Get current phase number
    j = case.GetPhaseNumber(rc)
    # Read the namelist to get prefix and iteration options
    nml = case.GetNamelist(rc, j)
    # =============
    # Best PLT File
    # =============
    # Prefix
    proj = case.GetProjectRootname(nml=nml)
    # Create glob to search for
    fglb = '%s_tec_boundary_timestep[1-9]*.plt' % proj
    # Check in working directory?
    if rc.get_Dual():
        # Look in the 'Flow/' folder
        fglb = os.path.join('Flow', fglb)
    # Get file
    fplt = case.GetFromGlob(fglb)
    # Get the iteration number
    nplt = int(fplt.rstrip('.plt').split('timestep')[-1])
    # ============================
    # Actual Iterations after Runs
    # ============================
    # Glob of ``run.%02i.%i`` files
    fgrun = case.glob.glob('run.[0-9][0-9].[1-9]*')
    # Form dictionary of iterations
    nrun = []
    drun = {}
    # Loop through files
    for frun in fgrun:
        # Get iteration number
        ni = int(frun.split('.')[2])
        # Get phase number
        ji = int(frun.split('.')[1])
        # Save
        nrun.append(ni)
        drun[ni] = ji
    # Sort on iteration number
    nrun.sort()
    nrun = np.array(nrun)
    # Determine the last run that terminated before this PLT file was created
    krun = np.where(nplt > nrun)[0]
    # If no 'run.%02i.%i' before *nplt*, then use 0
    if len(krun) == 0:
        # Use current phase as reported
        nprev = 0
        nstrt = 1
        jstrt = j
    else:
        # Get the phase from the last run that finished before *nplt*
        kprev = krun[-1]
        nprev = nrun[kprev]
        jprev = drun[nprev]
        # Have we moved to the next phase?
        if nprev >= rc.get_PhaseIters(jprev):
            # We have *nplt* from the next phase
            mprev = rc.get_PhaseSequence().index(jprev)
            jstrt = rc.get_PhaseSequence(mprev+1)
        else:
            # Still running phase *jprev* to create *fplt*
            jstrt = jprev
        # First iteration included in PLT file
        nstrt = nprev + 1
    # Make sure we have the right namelist
    if j != jstrt:
        # Read the new namelist
        j = jstrt
        nml = case.GetNamelist(rc, j)
    # ====================
    # Iteration Statistics
    # ====================
    # Check for averaging
    qavg = nml.GetVar('time_avg_params', 'itime_avg')
    # Number of iterations
    if qavg:
        # Time averaging included
        nStats = nplt - nprev
    else:
        # One iteration
        nStats = 1
        nstrt = nplt
    # ======
    # Output
    # ======
    return fplt, nStats, nstrt, nplt
# def GetPltFile
            

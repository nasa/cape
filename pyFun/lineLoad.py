"""
Sectional Loads Module: :mod:`pyCart.lineLoad`
==============================================

This module contains functions for reading and processing sectional loads.  It
is a submodule of :mod:`pyCart.dataBook`.

:Versions:
    * 2015-09-15 ``@ddalle``: Started
"""

# File interface
import os, glob
# Basic numerics
import numpy as np
# Date processing
from datetime import datetime

# Utilities or advanced statistics
from . import util
from . import case
from . import plt
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
        *DBL*: :class:`pyCart.lineLoad.DBLineLoad`
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
    def PreprocessTriq(self, ftriq):
        """Perform any necessary preprocessing to create ``triq`` file
        
        :Call:
            >>> ftriq = DBL.PreprocessTriq(ftriq)
        :Inputs:
            *DBL*: :class:`pyFun.lineLoad.DBLineLoad`
                Line load data book
            *ftriq*: :class:`str`
                Name of triq file
        :Versions:
            * 2016-12-20 ``@ddalle``: First version
        """
        # Get name of plt file
        fplt = ftriq.rstrip('triq') + 'plt'
        # Read the plt information
        plt.Plt2Triq(fplt, ftriq)
        
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
def GetPltFile():
    """Get most recent boundary ``plt`` file and its associated iterations
    
    :Call:
        >>> fplt, n, i0, i1 = GetPltFile()
    :Outputs:
        *fplt*: :class:`str`
            Name of ``plt`` file
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
            

r"""
:mod:`cape.pyfun.lineLoad`: Sectional loads module
==================================================

This module contains functions for reading and processing sectional
loads. It is a version of :mod:`cape.cfdx.lineLoad` that is closely
tied to :mod:`cape.pyfun.dataBook`.

It provides the primary class :class:`DBLineLoad`, which
is a subclass of :class:`cape.cfdx.dataBook.DBBase`.  This class is an
interface to all line load data for a specific surface component.

For reading the sectional load for a single solution on one component
(which includes multiple coefficients), there is another class
:class:`CaseLL`.

Finally, reading seam curves from individual cases utilizes the class
:class:`CaseSeam`.

:See also:
    * :mod:`cape.cfdx.lineLoad`
    * :mod:`cape.cfdx.dataBook`
    * :mod:`cape.pyfun.dataBook`
"""

# Standard library modules
import os
import fnmatch
import glob
import shutil

# Third-party modules
import numpy as np

# Local modules
from . import case
from . import mapbc
from . import plt as pltfile
from ..cfdx import lineLoad


# Data book of line loads
class DBLineLoad(lineLoad.DBLineLoad):
    r"""Line load (sectional load) data book for one group
    
    :Call:
        >>> DBL = DBLineLoad(x, opts. comp, conf=None, RootDir=None)
    :Inputs:
        *x*: :class:`cape.runmatrix.RunMatrix`
            RunMatrix/run matrix interface
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
    def GetNamelist(self):
        r"""Read FUN3D namelist from parent folder, if possible

        :Call:
            >>> nml = db.GetNamelist()
        :Inputs:
            *db*: :class:`DBLineLoad`
                Line load component databook instance
        :Outputs:
            *nml*: :class:`cape.pyfun.namelist.Namelist` | ``None``
                Active FUN3D namelist interface
        :Versions:
            * 2023-02-03 ``@ddalle``: v1.0
        """
        # Check location
        fpwd = os.getcwd()
        # Go to parent folder (usual use case)
        os.chdir("..")
        # Read namelist
        nml = case.GetNamelist()
        # Return to original location
        os.chdir(fpwd)
        # Output
        return nml

    # Get reference area
    def GetRefArea(self):
        r"""Get reference area, reading from namelist if needed

        :Call:
            >>> Aref = db.GetRefArea()
        :Inputs:
            *db*: :class:`DBLineLoad`
                Line load component databook instance
        :Outputs:
            *Aref*: :class:`float` | ``None``
                Reference area
        :Versions:
            * 2023-02-03 ``@ddalle``: v1.0
        """
        # Return main value if defined
        if self.RefA is not None:
            return self.RefA
        # Read namelist
        nml = self.GetNamelist()
        # Exit if no namelist
        if nml is None:
            return
        # Try to find component
        icomp = nml.find_comp_index(self.RefComp)
        # If one was found, use that index (icomp=None -> None)
        Aref = nml.GetVar("component_parameters", "component_sref", icomp)
        # Save and return
        self.RefA = Aref
        return Aref

    # Get reference length
    def GetRefLength(self):
        r"""Get reference length, reading from namelist if needed

        :Call:
            >>> Lref = db.GetRefLength()
        :Inputs:
            *db*: :class:`DBLineLoad`
                Line load component databook instance
        :Outputs:
            *Lref*: :class:`float` | ``None``
                Reference length
        :Versions:
            * 2023-02-03 ``@ddalle``: v1.0
        """
        # Return main value if defined
        if self.RefL is not None:
            return self.RefL
        # Read namelist
        nml = self.GetNamelist()
        # Exit if no namelist
        if nml is None:
            return
        # Try to find component
        icomp = nml.find_comp_index(self.RefComp)
        # If one was found, use that index (icomp=None -> None)
        Lref = nml.GetVar("component_parameters", "component_cref", icomp)
        # Save and return
        self.RefL = Lref
        return Lref

    # Get reference length
    def GetMRP(self):
        r"""Get moment reference point, reading from namelist if needed

        :Call:
            >>> MRP = db.GetMRP()
        :Inputs:
            *db*: :class:`DBLineLoad`
                Line load component databook instance
        :Outputs:
            *MRP*: :class:`np.ndarray`\ [:class:`float`] | ``None``
                Reference length
        :Versions:
            * 2023-02-03 ``@ddalle``: v1.0
        """
        # Return main value if defined
        if self.MRP is not None:
            return self.MRP
        # Read namelist
        nml = self.GetNamelist()
        # Exit if no namelist
        if nml is None:
            return
        # Try to find component
        icomp = nml.find_comp_index(self.RefComp)
        # Exit if no match
        if icomp is None:
            return
        # If one was found, use that index (icomp=None -> None)
        xmrp = nml.GetVar("component_parameters", "component_xmc", icomp)
        ymrp = nml.GetVar("component_parameters", "component_ymc", icomp)
        zmrp = nml.GetVar("component_parameters", "component_zmc", icomp)
        # Save and return
        self.MRP = np.array([xmrp, ymrp, zmrp])
        return self.MRP
    
    # Get component ID numbers
    def GetCompID(self):
        r"""Create list of component IDs
        
        :Call:
            >>> DBL.GetCompID()
        :Inputs:
            *DBL*: :class:`lineLoad.DBLineLoad`
                Instance of line load data book
        :Versions:
            * 2016-12-22 ``@ddalle``: v1.0, extracted from __init__()
        """
        # Figure out reference component
        self.CompID = self.opts.get_DataBookCompID(self.comp)
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
        # Try to get all components
        try:
            # Use the configuration interface
            self.CompID = self.conf.GetCompID(self.CompID)
            return
        except Exception:
            pass
        # Convert to MapBC numbers, since that's how the PLT file numbers them
        try:
            # Convert component IDs to surface IDs
            self.CompID = [
                self.MapBC.GetSurfID(compID) for compID in self.CompID
            ]
        except Exception:
            pass
    
    # Get file
    def GetTriqFile(self):
        r"""Get most recent ``triq`` file and its associated iterations
        
        :Call:
            >>> qtriq, ftriq, n, i0, i1 = DBL.GetTriqFile()
        :Inputs:
            *DBL*: :class:`pyfun.lineLoad.DBLineLoad`
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
        fplt, n, i0, i1 = case.GetPltFile()
        # Exit if nothing to do
        if fplt is None:
            return False, None, None, None, None
        # Check for iteration resets
        nh, ns = case.GetHistoryIter()
        # Add in the last iteration number before restart
        if nh is not None:
            i0 += nh
            i1 += nh
        # Get the corresponding .triq file name
        ftriq = fplt.rstrip('.plt') + '.triq'
        # Check if the TRIQ file exists
        if os.path.isfile(ftriq):
            # Check if it's new enough
            if os.path.getmtime(ftriq) > os.path.getmtime(fplt):
                # No conversion needed
                qtriq = False
            else:
                # Out-of-date file with correct name
                qtriq = True
        else:
            # Need to convert PLT file to TRIQ
            qtriq = True
        # Output
        return qtriq, ftriq, n, i0, i1
    
    # Preprocess triq file (convert from PLT)
    def PreprocessTriq(self, ftriq, **kw):
        r"""Perform any necessary preprocessing to create ``triq`` file
        
        :Call:
            >>> ftriq = DBL.PreprocessTriq(ftriq, qpbs=False, f=None)
        :Inputs:
            *DBL*: :class:`pyfun.lineLoad.DBLineLoad`
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
        # Get name of plt file
        fplt = ftriq.rstrip('triq') + 'plt'
        # Output format
        fmt = self.opts.get_DataBookTriqFormat(self.comp)
        # Get case index
        i = kw.get('i')
        # Copy the mapbc files
        fmapbc = glob.glob(os.path.join('..', '*.mapbc'))
        # Copy them
        for fm in fmapbc:
            shutil.copy(fm, '.')

        # Read Mach number
        if i is None:
            # Read from :file:`conditions.json`
            mach = case.ReadConditions('mach')
        else:
            # Get from trajectory
            mach = self.x.GetMach(i)
        # Convert the plt file
        pltfile.Plt2Triq(fplt, ftriq, mach=mach, fmt=fmt)
        
# class DBLineLoad
    

# Line loads
class CaseLL(lineLoad.CaseLL):
    r"""Individual class line load class
    
    :Call:
        >>> LL = CaseLL(cart3d, i, comp)
    :Inputs:
        *cart3d*: :class:`cape.pycart.cntl.Cntl`
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
class CaseSeam(lineLoad.CaseSeam):
    r"""Seam curve interface
    
    :Call:
        >>> S = CaseSeam(fname, comp='entire', proj='LineLoad')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *comp*: :class:`str`
            Name of the component
    :Outputs:
        *S* :class:`CaseSeam`
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
    r"""Get most recent boundary ``plt`` file and associated iterations
    
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
    fout = "%s_tec_boundary.plt" % proj
    fglb = '%s_tec_boundary_timestep[1-9]*.plt' % proj
    # Check in working directory?
    if rc.get_Dual():
        # Look in the 'Flow/' folder
        fglb = os.path.join('Flow', fglb)
    # Get file
    fplt = case.GetFromGlob(fglb, fout)
    # Get the iteration number in file if possible
    if fnmatch.fnmatch(fplt, fglb):
        # Iteration number listed
        nplt = int(fplt.rstrip('.plt').split('timestep')[-1])
    else:
        # No iteration number contained
        nplt = case.GetRestartIter()
    # ============================
    # Actual Iterations after Runs
    # ============================
    # Glob of ``run.%02i.%i`` files
    fgrun = glob.glob('run.[0-9][0-9].[1-9]*')
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
            

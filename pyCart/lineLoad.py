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
    pass
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
def GetTriqFile():
    """Get most recent ``triq`` file and its associated iterations
    
    :Call:
        >>> ftriq, n, i0, i1 = GetTriqFile()
    :Outputs:
        *ftriq*: :class:`str`
            Name of ``triq`` file
        *n*: :class:`int`
            Number of iterations included
        *i0*: :class:`int`
            First iteration in the averaging
        *i1*: :class:`int`
            Last iteration in the averaging
    :Versions:
        * 2015-09-16 ``@ddalle``: First version
    """
    # Get the working directory.
    fwrk = case.GetWorkingFolder()
    # Go there.
    fpwd = os.getcwd()
    os.chdir(fwrk)
    # Get the glob of numbered files.
    fglob3 = glob.glob('Components.*.*.*.triq')
    fglob2 = glob.glob('Components.*.*.triq')
    fglob1 = glob.glob('Components.[0-9]*.triq')
    # Check it.
    if len(fglob3) > 0:
        # Get last iterations
        I0 = [int(f.split('.')[3]) for f in fglob3]
        # Index of best iteration
        j = np.argmax(I0)
        # Iterations there.
        i1 = I0[j]
        i0 = int(fglob3[j].split('.')[2])
        # Count
        n = int(fglob3[j].split('.')[1])
        # File name
        ftriq = fglob3[j]
    elif len(fglob2) > 0:
        # Get last iterations
        I0 = [int(f.split('.')[2]) for f in fglob2]
        # Index of best iteration
        j = np.argmax(I0)
        # Iterations there.
        i1 = I0[j]
        i0 = int(fglob2[j].split('.')[1])
        # File name
        ftriq = fglob2[j]
    # Check it.
    elif len(fglob1) > 0:
        # Get last iterations
        I0 = [int(f.split('.')[1]) for f in fglob1]
        # Index of best iteration
        j = np.argmax(I0)
        # Iterations there.
        i1 = I0[j]
        i0 = I0[j]
        # Count
        n = i1 - i0 + 1
        # File name
        ftriq = fglob1[j]
    # Plain file
    elif os.path.isfile('Components.i.triq'):
        # Iteration counts: assume it's most recent iteration
        i1 = case.GetCurrentIter()
        i0 = i1
        # Count
        n = 1
        # file name
        ftriq = 'Components.i.triq'
    else:
        # No iterations
        i1 = None
        i0 = None
        n = None
        ftriq = None
    # Return to original location
    os.chdir(fpwd)
    # Prepend name of folder if appropriate
    if fwrk != '.': ftriq = os.path.join(fwrk, ftriq)
    # Output
    return ftriq, n, i0, i1
            

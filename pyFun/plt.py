"""
Tecplot PLT File Interface for Fun3D
====================================


"""

# System interface
import glob, os
# Basic numerics
import numpy as np
# Useful tool for more complex binary I/O methods
import cape.io
import cape.plt
import cape.tri
# Local tool for MapBC file interface
import pyFun.mapbc

# Convert a PLT to TRIQ
def Plt2Triq(fplt, ftriq=None, **kw):
    """Convert a Tecplot PLT file to a Cart3D annotated triangulation (TRIQ)
    
    :Call:
        >>> Plt2Triq(fplt, ftriq=None, **kw)
    :Inputs:
        *fplt*: :class:`str`
            Name of Tecplot PLT file
        *ftriq*: {``None``} | :class:`str`
            Name of output file (default: replace extension with ``.triq``)
        *mach*: {``1.0``} | positive :class:`float`
            Freestream Mach number for skin friction coeff conversion
        *triload*: {``True``} | ``False``
            Whether or not to write a triq tailored for ``triloadCmd``
        *avg*: {``True``} | ``False``
            Use time-averaged states if available
        *rms*: ``True`` | {``False``}
            Use root-mean-square variation instead of nominal value
    :Versions:
        * 2016-12-20 ``@ddalle``: First version
    """
    # Output file name
    if ftriq is None:
        # Default: strip .plt and add .triq
        ftriq = fplt.rstrip('plt').rstrip('dat') + 'triq'
    # TRIQ settings
    ll  = kw.get('triload', True)
    avg = kw.get('avg', True)
    rms = kw.get('rms', False)
    # Read the PLT file
    plt = Plt(fplt)
    # Check for mapbc file
    fglob = glob.glob("*.mapbc")
    # Check for more than one
    if len(fglob) > 0:
        # Make a crude attempt at sorting
        fglob.sort()
        # Import the alphabetically last one (should be the same anyway)
        kw["mapbc"] = pyFun.mapbc.MapBC(fglob[0])
    # Attempt to get *cp_tavg* state
    if "mach" in kw:
        plt.GetCpTAvg(kw["mach"])
    # Create the TRIQ interface
    triq = plt.CreateTriq(**kw)
    # Get output file extension
    ext = triq.GetOutputFileType(**kw)
    # Write triangulation
    triq.Write(ftriq, **kw)

# Tecplot class
class Plt(cape.plt.Plt):
    """Interface for Tecplot PLT files
    
    :Call:
        >>> plt = pyFun.plt.Plt(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *plt*: :class:`pyFun.plt.Plt`
            Tecplot PLT interface
        *plt.nVar*: :class:`int`
            Number of variables
        *plt.Vars*: :class:`list` (:class:`str`)
            List of of variable names
        *plt.nZone*: :class:`int`
            Number of zones
        *plt.Zone*: :class:`int`
            Name of each zone
        *plt.nPt*: :class:`np.ndarray` (:class:`int`, *nZone*)
            Number of points in each zone
        *plt.nElem*: :class:`np.ndarray` (:class:`int`, *nZone*)
            Number of elements in each zone
        *plt.Tris*: :class:`list` (:class:`np.ndarray` (*N*,4))
            List of triangle node indices for each zone
    :Versions:
        * 2016-11-22 ``@ddalle``: First version
        * 2017-03-30 ``@ddalle``: Subclassed to :class:`cape.plt.Plt`
    """
    # Calculate cp_tavg
    def GetCpTAvg(self, mach, gam=1.4):
        """Calculate *cp_tavg* if *p_tavg* exists
        
        :Call:
            >>> plt.GetCpTAvg(mach, gam=1.4)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
            *mach*: :class:`float`
                Freestream Mach number
            *gam*: {``1.4``} | :class:`float`
                Ratio of specific heats
        :Versions:
            * 2017-05-16 ``@ddalle``: First version
        """
        # Check if already present
        if 'cp_tavg' in self.Vars:
            # Already completed
            return
        elif 'p_tavg' not in self.Vars:
            # No time-averaging information
            return
        # Append to variables
        self.Vars.append('cp_tavg')
        self.nVar += 1
        # Append column to *qmin* and *qmax*
        self.qmin = np.hstack((self.qmin, np.zeros((self.nVar,1))))
        self.qmax = np.hstack((self.qmax, np.zeros((self.nVar,1))))
        # Use time-averaged pressure
        k = self.Vars.index('p_tavg')
        # Loop through states
        for n in range(self.nZone):
            # Get *cp*
            cp = (self.q[n][:,k] - 1/gam)/(0.5*mach*mach)
            # Append state
            self.q[n] = np.hstack((self.q[n], np.transpose([cn])))
            # Update min/max
            self.qmin[n,-1] = np.min(cp)
            self.qmax[n,-1] = np.max(cp)
            # Append the other random info
            self.VarLocs[n] = np.append(self.VarLocs[n], self.VarLocs[n][k])
            self.fmt[n]     = np.append(self.fmt[n], self.fmt[n][k])
            

# class Plt


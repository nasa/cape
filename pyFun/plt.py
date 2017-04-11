"""
Tecplot PLT File Interface for Fun3D
====================================


"""

# System interface
import glob
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
    pass

# class Plt


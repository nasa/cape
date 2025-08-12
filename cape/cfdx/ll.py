
# Standard library
import os
from typing import Optional

# Third-party
import numpy as np

# Local imports
from ..dkit.rdb import DataKit


# Line load from one case
class CaseLineLoad(DataKit):
    r"""Interface to individual sectional load from one case

    :Call:
        >>> ll = CaseLineLoad(comp, proj='LineLoad', sec='slds', **kw)
    :Inputs:
        *comp*: :class:`str`
            Name of component
        *proj*: {``"LineLoad"``} | :class:`str`
            Prefix for sectional load output files
        *sec*: ``"clds"`` | {``"dlds"``} | ``"slds"``
            Cut type, cumulative, derivative, or sectional
        *dirname*: {``"lineload"``} | ``None`` | :class:`str`
            Name of sub folder to use
    :Outputs:
        *LL*: :class:`cape.cfdx.lineload.CaseLL`
            Individual line load for one component from one case
    :Versions:
        * 2015-09-16 ``@ddalle``: v1.0
        * 2016-06-07 ``@ddalle``: v2.0; CFDX version
        * 2025-08-11 ``@ddalle``: v3.0; fork old ``CaseLL``
    """
  # --- Config ---
    # Initialization method
    def __init__(
            self,
            comp: str,
            proj: str = 'LineLoad',
            sec: str = 'dlds',
            dirname: Optional[str] = "lineload", **kw):
        """Initialization method"""
        # Save input options
        self.comp = comp
        self.proj = proj
        self.sec = sec
        self.cols = []
        # Options
        self.seam = kw.get('seam', False)
        # File name
        flds = f"{proj}_{comp}.{sec}"
        fname = flds if dirname is None else os.path.join(dirname, flds)
        # Check for file
        if os.path.isfile(fname):
            # Read the file
            self.read_lds(fname)
        else:
            # Create empty line loads
            for col in LL_COLS:
                self.save_col(col, np.zeros(0))
        # Read the seams
        if self.seam:
            self.ReadSeamCurves()

  # --- I/O ---
    # Function to read a file
    def read_lds(self, fname: str):
        r"""Read a sectional loads ``*.?lds`` file from `triloadCmd`

        :Call:
            >>> ll.read_lds(fname)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineload.CaseLineLoads`
                Single-case, single component, line load interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2015-09-15 ``@ddalle``: v1.0
            * 2025-08-11 ``@ddalle``: v2.0; was ``ReadLDS()``
        """
        # Open the file
        with open(fname, 'r') as fp:
            # Read lines until it is not a comment.
            line = '#'
            while (line.lstrip().startswith('#')) and len(line):
                # Read the next line
                line = fp.readline()
            # Exit if empty.
            if len(line) == 0:
                raise ValueError("Empty triload file '%s'" % fname)
            # Number of columns
            ncol = len(line.split())
            # Go backwards one line from current position.
            fp.seek(fp.tell() - len(line))
            # Read the rest of the file.
            data = np.fromfile(fp, count=-1, sep=' ')
            # Reshape to a matrix
            data = data.reshape((data.size//ncol, ncol))
            # Save the columns
            for j, col in enumerate(LL_COLS):
                self.save_col(col, data[:, j])

    # Read the seam curves
    def ReadSeamCurves(self):
        """Read seam curves from a data book directory

        :Call:
            >>> LL.ReadSeamCurves()
        :Inputs:
            *LL*: :class:`pyCart.lineload.CaseLL`
                Instance of data book line load interface
        :Versions:
            * 2015-09-17 ``@ddalle``: v1.0
        """
        # Seam file names
        if self.fdir is None:
            # Folder
            fpre = '%s_%s' % (self.proj, self.comp)
        else:
            # Include subfolder
            fpre = os.path.join(self.fdir, '%s_%s' % (self.proj, self.comp))
        # Name of output files.
        fsmx = '%s.smx' % fpre
        fsmy = '%s.smy' % fpre
        fsmz = '%s.smz' % fpre
        # Read the seam curves.
        self.smx = CaseSeam(fsmx)
        self.smy = CaseSeam(fsmy)
        self.smz = CaseSeam(fsmz)

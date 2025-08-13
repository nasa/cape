
# Standard library
import os
from typing import Optional

# Third-party
import numpy as np

# Local imports
from ..dkit.rdb import DataKit


# Typical line load columns
LL_COLS = ("x", "CA", "CY", "CN", "CLL", "CLM", "CLN")


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
            self.read_seam_cures()

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
    def read_seam_cures(self):
        r"""Read seam curves from a data book directory

        :Call:
            >>> ll.read_seam_curves()
        :Inputs:
            *ll*: :class:`cape.cfdx.ll.CaseLineLoad`
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


# Class for seam curves
class CaseSeam(object):
    r"""Seam curve interface

    :Call:
        >>> S = CaseSeam(fname, comp='entire', proj='LineLoad')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *comp*: :class:`str`
            Name of the component
    :Outputs:
        *S* :class:`cape.cfdx.lineload.CaseSeam`
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
        * 2016-06-09 ``@ddalle``: v1.0
    """
    # Initialization method
    def __init__(
            self,
            fname: str,
            comp: str = 'entire',
            proj: str = 'LineLoad'):
        r"""Initialization method

        :Versions:
            * 2016-06-09 ``@ddalle``: v1.0
        """
        # Save file
        self.fname = fname
        # Save prefix and component name
        self.proj = proj
        self.comp = comp
        # Read file
        self.read_sm()

    # Representation method
    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2016-06-09 ``@ddalle``: v1.0
        """
        return "<CaseSeam '%s', n=%s>" % (
            os.path.split(self.fname)[-1], self.n)

    # Function to read a seam file
    def read_sm(self, fname: Optional[str] = None):
        r"""Read a seam  ``*.sm[yz]`` file

        :Call:
            >>> seam.Read(fname=None)
        :Inputs:
            *seam* :class:`cape.cfdx.lineload.CaseSeam`
                Seam curve interface
            *fname*: {``None``} | :class:`str`
                Name of file to read
        :Attributes:
            *seam.n*: :class:`int`
                Number of points in vector entries
            *seam.x*: :class:`list` (:class:`numpy.ndarray`)
                List of *x* coordinates of seam curves
            *seam.y*: :class:`float` or :class:`list` (:class:`numpy.ndarray`)
                Fixed *y* coordinate or list of seam curve *y* coordinates
            *seam.z*: :class:`float` or :class:`list` (:class:`numpy.ndarray`)
                Fixed *z* coordinate or list of seam curve *z* coordinates
        :Versions:
            * 2015-09-17 ``@ddalle``: v1.0
            * 2016-06-09 ``@ddalle``: Added possibility of x-cuts
        """
        # Default file name
        fname = self.fname if fname is None else fname
        # Initialize seam count
        self.n = 0
        # Initialize seams
        self.x = []
        self.y = []
        self.z = []
        self.ax = 'y'
        # Check for the file
        if not os.path.isfile(fname):
            return
        # Open the file.
        f = open(fname, 'r')
        # Read first line.
        line = f.readline()
        # Get the axis and value
        txt = line.split()[-2]
        ax  = txt.split('=')[0]
        val = float(txt.split('=')[1])
        # Name of cut axis
        self.ax = ax
        # Save the value
        setattr(self, ax, val)
        # Read two lines.
        f.readline()
        f.readline()
        # Loop through curves.
        while line != '':
            # Get data
            D = np.fromfile(f, count=-1, sep=" ")
            # Check size.
            m = int(np.floor(D.size/2) * 2)
            # Save the data.
            if ax == 'x':
                # x-cut
                self.y.append(D[0:m:2])
                self.z.append(D[1:m:2])
            elif ax == 'y':
                # y-cut
                self.x.append(D[0:m:2])
                self.z.append(D[1:m:2])
            else:
                # z-cut
                self.x.append(D[0:m:2])
                self.y.append(D[1:m:2])
            # Segment count
            self.n += 1
            # Read two lines.
            line = f.readline()
            line = f.readline()
        # Cleanup
        f.close()

    # Function to write a seam file
    def write_sm(self, fname=None):
        r"""Write a seam curve file

        :Call:
            >>> seam.write_sm(fname)
        :Inputs:
            *seam* :class:`cape.cfdx.lineload.CaseSeam`
                Seam curve interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2015-09-17 ``@ddalle``: v1.0
            * 2016-06-09 ``@ddalle``: v1.1; cut direction
            * 2016-06-09 ``@ddalle``: v1.2; Moved to seam class
        """
        # Default file name
        if fname is None:
            fname = '%s_%s.sm%s' % (self.proj, self.comp, self.ax)
        # Check if there's anything to write.
        if self.n < 1:
            return
        # Check axis
        if self.ax == 'x':
            # x-cuts
            x1 = 'y'
            x2 = 'z'
        elif self.ax == 'z':
            # z-cuts
            x1 = 'x'
            x2 = 'y'
        else:
            # y-cuts
            x1 = 'x'
            x2 = 'z'
        # Save axis
        ax = self.ax
        # Open the file
        with open(fname, 'w') as fp:
            # Write the header line
            fp.write(
                ' #Seam curves for %s=%s plane\n'
                % (self.ax, getattr(self, ax)))
            # Loop through seems
            for i in range(self.n):
                # Header
                fp.write(' #Seam curve %11i\n' % i)
                # Extract coordinates
                x = getattr(self, x1)[i]
                y = getattr(self, x2)[i]
                # Write contents
                for j in np.arange(len(x)):
                    fp.write(" %11.6f %11.6f\n" % (x[j], y[j]))

r"""
:mod:`cape.pylava.dataiterfile`: Interface to LAVA ``data.iters`` files
-----------------------------------------------------------------------

This class provdes the class :class:`DataIterFile`, which can read data
from the ``data.iter`` file written by LAVA. This file records the
residual, force & moment, and potentially more while the LAVA solver is
running.

The :class:`DataIterFile` is configured so that it can read just
metadata about the file (including the last iteration number) or all of
the data in the file.
"""

# Standard library
import os
from io import IOBase
from typing import Optional

# Local imports
from ..capeio import fromfile_lb4_i, fromfile_lb8_f


# Constants
DEFAULT_STRLEN = 50


# Primary class
class DataIterFile(dict):
    r"""LAVA ``data.iter`` file reader

    :Call:
        >>> db = DataIterFile(fname=None, meta=False)
    :Inputs:
        *fname*: {``"data.iter"``} | None`` | :class:`str`
            Name of file to read
        *meta*: ``True`` | {``False``}
            Option to just read column names and last iteration
    :Outputs:
        *db*: :class:`DataIterFile`
            Instance of ``data.iter`` interface
    :Versions:
        * 2024-10-11 ``@ddalle``: v1.0
    """
    # Class attributes
    __slots__ = (
        "cols",
        "filename",
        "filesize",
        "n",
        "l2conv",
        "strsize",
        "t",
    )

    # Initialization
    def __init__(self, fname: Optional[str] = "data.iter", **kw):
        r"""Initialization method

        :Versions:
            * 2024-10-11 ``@ddalle``: v1.0
        """
        # Initialize column list and other attributes
        self.cols = []
        self.filename = None
        self.filesize = 0
        self.n = 0
        self.l2conv = 1.0
        self.strsize = DEFAULT_STRLEN
        self.t = 0.0
        # Check for a file
        if fname is None:
            return
        # Save base file name
        self.filename = os.path.basename(fname)
        # Read the file
        self.read(fname, **kw)

    # String
    def __str__(self) -> str:
        r"""String method

        :Versions:
            >>> 2024-10-11 ``@ddalle``: v1.0
        """
        # Initialize with class name
        txt = f"<{self.__class__.__name__}"
        # Check for file name
        if self.filename:
            txt += f" '{self.filename}' "
        # Get last iteration
        if self.n is not None:
            txt += f" n={self.n:d}"
        # Close the bracket
        return txt + '>'

    # Data reader
    def read(self, fname: str, **kw):
        r"""Read a ``data.iter`` file

        :Call:
            >>> db.read(fname)
        :Inputs:
            *db*: :class:`DataIterFile`
                Instance of ``data.iter`` interface
            *fname*: :class:`str`
                Name of file, usually ``data.iter``
            *meta*: ``True`` | {``False``}
                Option to just read column names and last iteration
        :Versions:
            * 2024-10-11 ``@ddalle``: v1.0
        """
        with open(fname, 'rb') as fp:
            self._read(fp, **kw)

    # Data reader driver
    def _read(self, fp: IOBase, **kw):
        # Get file size
        fsize = fp.seek(0, 2)
        # Reset position
        fp.seek(0)
        # Read number of variables
        ncol, = fromfile_lb4_i(fp, 1)
        # Read the number of chars in a string
        strsize, = fromfile_lb4_i(fp, 1)
        # Intialize columns
        cols = []
        # Read column names
        for j in range(ncol):
            # Check for problems
            try:

                colj = _read_strn(fp, strsize)
            except Exception:
                raise ValueError(
                    f"Could not decode col {j} (0-based) in '{self.fname}'")
            # Save the column
            cols.append(colj)
        # Save the information
        self.filesize = fsize
        self.cols = cols
        self.strsize = strsize
        # Record current position
        pos = fp.tell()
        # Number of lines/records
        nrec = (fsize - pos) // (ncol * 8)
        # Report "iteration" number
        if "iter" in self.cols:
            icol = "iter"
        elif "nt" in self.cols:
            icol = "nt"
        else:
            icol = None
        # Get iterations
        if icol and (icol in self.cols):
            # Index of "iters" column
            jcol = self.cols.index(icol)
            # Head to the last record
            fp.seek((nrec - 1)*ncol*8 + 8*jcol, 1)
            # Next entry is most recently reported "iteration"
            n, = fromfile_lb8_f(fp, 1)
            # Selectively switch to 1-based iteration
            self.n = n + 1 if (icol == "iter") else n
        # Report "iteration" number
        if "ctu" in self.cols:
            tcol = "ctu"
        else:
            tcol = None
        # Get iterations
        if tcol and (tcol in self.cols):
            # Index of "iters" column
            jcol = self.cols.index(tcol)
            # Head to the last record
            fp.seek(pos)
            fp.seek((nrec - 1)*ncol*8 + 8*jcol, 1)
            # Next entry is most recently reported "iteration"
            t, = fromfile_lb8_f(fp, 1)
            # Selectively switch to 1-based iteration
            self.t = t
        # Report residual drop
        if "flowres" in self.cols:
            icol = "flowres"
        else:
            icol = None
        # Get residual drop
        if icol and (icol in self.cols):
            # Index of "flowres" column
            jcol = self.cols.index(icol)
            # Read original and final L2 residual
            fp.seek(pos)
            fp.seek(8*jcol, 1)
            resid0, = fromfile_lb8_f(fp, 1)
            fp.seek((nrec-1)*ncol*8 - 8, 1)
            resid1, = fromfile_lb8_f(fp, 1)
            # Save the convergence
            self.l2conv = resid1 / resid0
        # Check for "meta" option
        if kw.get("meta", False):
            return
        # Return to beginning of data
        fp.seek(pos)
        # Read all the data
        data = fromfile_lb8_f(fp, nrec*ncol)
        # Save to each column (collated)
        for j, col in enumerate(cols):
            self[col] = data[j::ncol]


# Fixed-length strings
def _read_strn(fp: IOBase, n: int = DEFAULT_STRLEN):
    r"""Read string from next *n* bytes

    :Call:
        >>> txt = _read_str(fp, n=32)
    :Inputs:
        *fp*: :class:`file`
            File handle open "rb"
        *n*: {``50``} | :class:`int` > 0
            Number of bytes to read
    :Outputs:
        *txt*: :class:`str`
            String decoded from *n* bytes w/ null chars trimmed
    """
    # Read *n* bytes
    buf = fp.read(n)
    # Encode
    return buf.decode("utf-8").strip()

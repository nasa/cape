r"""
:mod:`gruvoc.dataformats`: Common data format identification tools
===================================================================

This module provides tools for recognizing which format data is
written in, either by the data itself or by determining from a file
name.

This is concerned with questions like

*   big-endian vs little-endian,
*   record markers (FORTRAN unformatted) vs not (C stream), and
*   single- or double-precision

and not on identifying various conventions on what data is written. For
example, it does not have any tools to distinguish between an ``.avm``
file and a ``.ugrid`` file.
"""


# Standard library
import re
from io import IOBase

# Third-party
import numpy as np

# Local imports
from ._vendor.capeio import (
    read_record_lr4_i, read_record_r4_i,
    read_record_lr4_f, read_record_r4_f,
    read_record_lr8_f, read_record_r8_f,
    fromfile_lb4_i, fromfile_b4_i,
    fromfile_lb4_f, fromfile_b4_f,
    fromfile_lb8_f, fromfile_b8_f,
    tofile_lb4_i, tofile_b4_i,
    tofile_lb4_f, tofile_b4_f,
    tofile_lb8_f, tofile_b8_f,
    write_record_lr4_i, write_record_r4_i,
    write_record_lr4_f, write_record_r4_f,
    write_record_lr8_f, write_record_r8_f)


# Regex pattern to identify file formats
PATTERN_GRID_FMT = "(?P<end>l)?(?P<mark>[rb])(?P<prec>[48])(?P<long>l)?"

# Compiled regex for processing
REGEX_GRID_FMT = re.compile(PATTERN_GRID_FMT)


def read_ascii_i(fp: IOBase, n: int):
    return np.fromfile(fp, sep=" ", count=n, dtype="int")


def read_ascii_f(fp: IOBase, n: int):
    return np.fromfile(fp, sep=" ", count=n, dtype="float")


def write_ascii(fp: IOBase, x: np.ndarray):
    # Check dimension
    if x.ndim == 1:
        # Write one int per line
        x.tofile(fp, sep="\n")
        # Terminate line
        fp.write(b"\n")
    else:
        # Write each element on one line
        for xj in x:
            # Write one row of *x* as a row in file
            xj.tofile(fp, sep=" ")
            # Terminate line
            fp.write(b"\n")


def fromfile_lr4_i(fp: IOBase, n: int):
    return read_record_lr4_i(fp)


def fromfile_lr4_f(fp: IOBase, n: int):
    return read_record_lr4_f(fp)


def fromfile_lr8_f(fp: IOBase, n: int):
    return read_record_lr8_f(fp)


def fromfile_r4_i(fp: IOBase, n: int):
    return read_record_r4_i(fp)


def fromfile_r4_f(fp: IOBase, n: int):
    return read_record_r4_f(fp)


def fromfile_r8_f(fp: IOBase, n: int):
    return read_record_r8_f(fp)


# Standard dictionary of read/write functions
READ_FUNCS = {
    "ascii": (read_ascii_i, read_ascii_f),
    "lb4": (fromfile_lb4_i, fromfile_lb4_f),
    "lb8": (fromfile_lb4_i, fromfile_lb8_f),
    "b4": (fromfile_b4_i, fromfile_b4_f),
    "b8": (fromfile_b4_i, fromfile_b8_f),
    "lr4": (fromfile_lr4_i, fromfile_lr4_f),
    "lr8": (fromfile_lr4_i, fromfile_lr8_f),
    "r4": (fromfile_r4_i, fromfile_r4_f),
    "r8": (fromfile_r4_i, fromfile_r8_f),
}
WRITE_FUNCS = {
    "ascii": (write_ascii, write_ascii),
    "lb4": (tofile_lb4_i, tofile_lb4_f),
    "lb8": (tofile_lb4_i, tofile_lb8_f),
    "b4": (tofile_b4_i, tofile_b4_f),
    "b8": (tofile_b4_i, tofile_b8_f),
    "lr4": (write_record_lr4_i, write_record_lr4_f),
    "lr8": (write_record_lr4_i, write_record_lr8_f),
    "r4": (write_record_r4_i, write_record_r4_f),
    "r8": (write_record_r4_i, write_record_r8_f),
}

r"""
:class:`cape.cfdx.tarcmd`: Common tar/untar utilities
======================================================

This module provides simple functions like :func:`tar` and :func:`untar`
that either call the command-line executable (if available) or the
built-in :mod:`tarfile` module.
"""

# Standard library
import glob
import shutil
import tarfile
import zipfile
from subprocess import check_call, PIPE


# Create an archive
def tar(ftar: str, fmt: str = '', *a):
    # Check if ``tar`` is available
    if shutil.which("tar") is None:
        _tar_py(ftar, fmt, *a)  # pragma: no cover
    else:
        _tar_cli(ftar, fmt, *a)


# Untar a file
def untar(ftar: str, fmt: str = ''):
    # Check if ``tar`` is available
    if shutil.which("tar") is None:
        # Use :mod:`tarfile`
        _untar_py(ftar, fmt)  # pragma: no cover
    else:
        # Use CLI (should be faster)
        _untar_cli(ftar, fmt)


def _tar_cli(ftar: str, fmt: str = '', *a):
    # Convert format to flags
    fmtcmd = _fmt2cmd(fmt)
    # Command
    if fmtcmd[0] == "zip":
        # Zimpler zip command
        cmdlist = ["zip", ftar, "-r"]
    else:
        # Full tar command
        cmdlist = fmtcmd + ['-c', '-f', ftar]
    # Add files to include
    cmdlist.extend(multiglob(*a))
    # Run the command
    check_call(cmdlist, stdout=PIPE, stderr=PIPE,)


def _tar_py(ftar: str, fmt: str = '', *a):
    # Get mode
    fmtmode = _fmt2mode(fmt)
    mode = f"w:{fmtmode}"
    # Open the file
    if mode == "zip":
        # Use a ZIP archive
        tar = zipfile.ZipFile(ftar, mode='w')
    else:
        # Usually using tar-compatible
        tar = tarfile.open(ftar, mode)
    # Loop through args
    for fname in multiglob(*a):
        tar.write(fname)


def _untar_cli(ftar: str, fmt: str = ''):
    # Convert format to flags
    fmtcmd = _fmt2cmd(fmt)
    # Command
    if fmtcmd[0] == "zip":
        # unzip command
        cmdlist = ["unzip", ftar]
    else:
        # tar command
        cmdlist = fmtcmd + ['-x', '-f', ftar]
    # Run the command
    check_call(cmdlist, stdout=PIPE, stderr=PIPE)


def _untar_py(ftar: str, fmt: str = ''):
    # Get mode
    fmtmode = _fmt2mode(fmt)
    mode = f"r:{fmtmode}"
    # Open the file
    if mode == "zip":
        # Use a ZIP archive
        tar = zipfile.ZipFile(ftar, mode='r')
    else:
        # Usually using tar-compatible
        tar = tarfile.open(ftar, mode)
    # Untar everything
    tar.extractall(".")


def _fmt2cmd(fmt: str = '') -> list:
    # Convert flag
    if fmt == "":
        # No compression
        fmtcmd = ["tar"]
    elif fmt in ("j", "bz2", "bzip2"):
        # bzip2
        fmtcmd = ["tar", "-j"]
    elif fmt in ("z", "gz", "gzip"):
        # gzip
        fmtcmd = ["tar", "-z"]
    elif fmt in ("J", "xz"):
        # xz
        fmtcmd = ["tar", "-J"]
    elif fmt in ("lzma",):
        # lzma
        fmtcmd = ["tar", "--lzma"]
    elif fmt in ("zst", "zstd"):
        # zstd
        assert shutil.which("zstd") is not None
        fmtcmd = ["tar", "--zstd"]
    elif fmt in ("zip",):
        # zip
        fmtcmd = ["zip"]
    else:
        raise ValueError(f"Unrecognized tar format '{fmt}'")
    # Output
    return fmtcmd


def _fmt2mode(fmt: str = '') -> str:
    # Convert flag
    if fmt == "":
        # No compression
        mode = ''
    elif fmt in ("j", "bz2", "bzip2"):
        # bzip2
        mode = 'bz2'
    elif fmt in ("z", "gz", "gzip"):
        # gzip
        mode = 'gz'
    elif fmt in ("J", "xz"):
        # xz
        mode = 'xz'
    elif fmt in ("lzma",):
        # lzma
        mode = 'lzma'
    elif fmt in ("zip",):
        # ZIP
        mode = "zip"
    else:
        raise ValueError(f"Unrecognized tar format '{fmt}'")
    # Output
    return mode


def multiglob(*a) -> list:
    # Initialize matches
    mtchs = set()
    # Loop through patterns
    for pat in a:
        mtchs.update(glob.glob(pat))
    # Return as sorted list
    return sorted(list(mtchs))

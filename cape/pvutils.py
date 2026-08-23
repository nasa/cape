r"""
:mod:`cape.pvutils`: Utilities for using PyVista with CAPE
====================================================================

This module provides common utilities to create PyVista flow
visualization figures in combination with CAPE. It includes a
:func:`main_template` that can be used as the main interface to
PyVista for individual customized templates.

This includes the class :class:`PVArgs` to process various command-line
options, which can either make a single frame or a video. Videos can be
made using multiple processes working in parallel (each process working
a specific frame).
"""

# Standard library
import os
import sys
import time
from typing import Callable, Optional, Union
from subprocess import call

# Third-party
import numpy as np
import pyvista as pv
from cape.argread import ArgReader
from cape.pylava.casecntl import CaseRunner
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()


# Constants
DEFAULT_SLEEPTIME = 0.2
WIDTH = 1920
HEIGHT = 1080


# CLI args
class PVArgs(ArgReader):
    r"""Argument and CLI parser for PyVista utilities"""
    __slots__ = ()

    _optlist = (
        "height",
        "help",
        "n",
        "nmax",
        "nmin",
        "nproc",
        "vid",
        "width",
    )

    _optmap = {
        "H": "height",
        "W": "width",
        "h": "help",
        "maxiter": "nmax",
        "miniter": "nmin",
        "proc": "nproc",
        "video": "vid",
        "w": "width",
    }

    _optlist_noval = (
        "help",
        "vid",
    )

    _opttypes = {
        "height": int,
        "help": bool,
        "n": int,
        "nmax": int,
        "nmin": int,
        "nproc": int,
        "vid": bool,
        "width": int,
    }

    _optconverters = {
        "height": int,
        "n": int,
        "nmax": int,
        "nmin": int,
        "nproc": int,
        "width": int,
    }

    _arglist = (
        "n",
    )

    _rc = {
        "height": HEIGHT,
        "nproc": 8,
        "vid": False,
        "width": WIDTH,
    }


# Output an image
def savefig(pl: pv.Plotter, outs: list, w: int = WIDTH, h: int = HEIGHT):
    r"""Save an image from a PyVista plotter

    :Call:
        >>> savefig(pl, outs, w, h)
    :Inputs:
        *pl*: :class:`pyvista.Plotter`
            Active PyVista plotter instance
        *outs*: :class:`list`\ [:class:`str`]
            List of output file names; uses first entry and pops it
        *w*: {``1920``} | :class:`int`
            Width of output image in pixels
        *h*: {``1080``} | :class:`int`
            Height of output image in pixels
    """
    # Output
    pl.screenshot(outs.pop(0), window_size=[w, h])


# Create file names
def genr8_filenames(
        OUTS: list,
        n: Optional[int] = None,
        j: Optional[int] = None) -> list:
    r"""Create list of file names, which may have frame number

    The inputs are simple prefixes. If `n` is ``None``, the outputs
    will simply add ``".png"`` to each template:

    .. code-block:: pycon

        >>> genr8_filenames(["y0-far", "y0-near"])
        ["y0-far.png", "y0-near.png"]

    If an iteration is given, the output will have a frame number

    .. code-block:: pycon

        >>> genr8_filenames(["y0-far", "y0-near"], 1000)
        ["y0-far.001000.png", "y0-near.001000.png"]
        >>> genr8_filenames(["y0-far", "y0-near"], 1000, 20)
        ["y0-far.000020.png", "y0-near.000020.png"]

    :Call:
        >>> outfiles = genr8_filenames(outs, n=None, j=None)
    :Inputs:
        *outs*: :class:`list`\ [:class:`str`]
            List of prefixes for file names
        *n*: {``None``} | :class:`int`
            Iteration number to plot (or use latest if ``None``)
        *j*: {``None``} | :class:`int`
            Output frame number (or use `n` if ``None``)
    :Outputs:
        *outfiles*: :class:`list`\ [:class:`str`]
            List of output files
    """
    # Output file names
    if n is None:
        # Direct output
        return [f"{out}.png" for out in OUTS]
    else:
        # Make output folder
        if not os.path.isdir("img"):
            try:
                os.mkdir("img")
            except FileExistsError:
                pass
        # Frame number
        k = n if (j is None) else j
        # Otherwise include iteration in the name
        outs = [os.path.join("img", f"{out}.{k:06d}.png") for out in OUTS]
        # Check for output
        for outj in outs:
            if not os.path.isfile(outj):
                return outs
        else:
            return []


# Video maker
def make_video(
        runner: CaseRunner,
        outs: list,
        pat: Union[str, int],
        frame_func: Callable,
        parser: PVArgs,
        cutplanes: Optional[list] = None,
        surfaces: Optional[list] = None) -> int:
    r"""Template to assemble PyVista frames into a video

    :Call:
        >>> make_video(runner, outs, pat, frame_func, parser)
    :Inputs:
        *runner*: :class:`cape.cfdx.casecntl.CaseRunner`
            Single-case control instance
        *outs*: :class:`list`\ [:class:`str`]
            List of prefixes for file names
        *pat*: :class:`str`
            Template file name to search for iterations of
        *frame_func*: :class:`callable`
            Function to process a single frame, takes args as follows:
            ``frame_func(runner, parser, n=None, j=None)``
        *parser*: :class:`PVArgs`
            CLI args parsed using ``PVAgs``
    """
    # Default lists
    cutplanes = [] if cutplanes is None else list(cutplanes)
    surfaces = [] if surfaces is None else list(surfaces)
    # Infer *pat* input
    if isinstance(pat, str):
        # Split into folder and file name
        dirname, basename = os.path.split(pat)
        # Check folder
        if dirname == "surface":
            # Use three digits
            surfaces.append(int(basename[4:7]) + 1)
        else:
            # Use two digits
            cutplanes.append(int(basename[5:6]) + 1)
    elif pat > 0:
        cutplanes.append(pat)
    else:
        surfaces.append(max(1, -pat))
    # Find available iterations
    l1 = [runner.find_surfdata_iters(nsurf) for nsurf in surfaces]
    l2 = [runner.find_cutplane_iters(nsurf) for nsurf in cutplanes]
    # Combine everything
    n1 = np.zeros(0, dtype="int32") if len(l1) == 0 else np.hstack(l1)
    n2 = np.zeros(0, dtype="int32") if len(l2) == 0 else np.hstack(l2)
    # Use overlap
    if n2.size == 0:
        # Use just surfaces
        iters = np.unique(n1)
    elif n1.size == 0:
        # Use just cut planes
        iters = np.unique(n2)
    else:
        # Use overlap
        iters = np.intersect1d(n1, n2)
    # Number of matches
    m = iters.size
    # Initialize list of subprocesses
    workers = []
    # Get options
    nmin = parser.get_opt("nmin")
    nmax = parser.get_opt("nmax")
    nproc = parser.get_opt("nproc")
    # Check for start frame
    nmin = 0 if nmin is None else nmin
    # Conversions for *nmin* and *nmax* to frame numbers
    jmin = None
    # Number of frames
    nframe = 0
    # Loop through same
    for j, n in enumerate(iters):
        # Check if we want to skip this frame
        if n < nmin:
            continue
        # Check for a max frame
        if (nmax is not None) and (n > nmax):
            continue
        # Process minimum frame
        jmin = (j + 1) if (jmin is None) else jmin
        # Update frame counter
        nframe += 1
        # Wait until worker count is subsided
        while len(workers) >= nproc:
            # Wait
            time.sleep(DEFAULT_SLEEPTIME)
            # Check them all
            _update_workers(workers)
        # Call the fork
        pid = os.fork()
        # Check parent/child
        if pid != 0:
            # Save the PID
            workers.append(pid)
            # Check if we want to skip this frame
            continue
        # Status update
        sys.stdout.write("\r%*s\riter %i (%i/%i)" % (60, '', n, j+1, m))
        sys.stdout.flush()
        # Process it
        frame_func(runner, parser, n=n, j=j)
        # Exit this shell
        os._exit(0)
    # Wait until the workers are complete
    # Clean up prompt
    print("")
    # Make lists of videos
    outs_png = [f"{out}.%06d.png" for out in outs]
    outs_mp4 = [f"{out}.mp4" for out in outs]
    # Create options for number of frames
    for pid in workers:
        os.waitpid(pid, 0)
    # Initialzie error code
    ierr = 0
    # Create videos
    for out_png, out_mp4 in zip(outs_png, outs_mp4):
        # Create video
        ierrj = call(
            [
                "ffmpeg", "-y",
                "-framerate", "30",
                "-start_number", str(nmin),
                "-i", os.path.join("img", out_png),
                "-frames:v", str(nframe),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                out_mp4
            ])
        # Update error code
        ierr = ierr | ierrj
    # Return code
    return ierr


def _update_workers(workers: list):
    # Loop through workers
    for pid in list(workers):
        # Check if it's active
        try:
            # Check on the requested process
            outpid, ierr = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            print(f"  PID {pid} is um....")
            continue
        # Check if it's running
        if outpid != 0:
            # Check error code
            if ierr:
                raise SystemError(f"Worker {pid} returned code {ierr}")
            # Already done
            workers.remove(pid)


# Main function
def main_template(frame_func: Callable, outs: list, pat: str) -> int:
    r"""Template function for customized PyVista scripts

    :Call:
        >>> ierr = main_template(frame_func, outs, pat)
    :Inputs:
        *frame_func*: :class:`callable`
            Function to process a single frame, takes args as follows:
            ``frame_func(runner, parser, n=None, j=None)``
        *outs*: :class:`list`\ [:class:`str`]
            List of prefixes for file names
        *pat*: :class:`str`
            Template file name to search for iterations of
    :Outputs:
        *ierr*: :class:`int`
            Return code
    """
    # Read arguments
    parser = PVArgs()
    parser.parse()
    # Check for help message
    if parser.get_opt("help"):
        print(parser.genr8_help())
        return 0
    # Read runner
    runner = CaseRunner()
    # Get video option
    vid = parser.get_opt("vid")
    # Check if video
    if vid:
        # Make video
        ierr = make_video(runner, outs, pat, frame_func, parser)
    else:
        # Check for an integer
        n = parser.get_opt("n")
        # Do latest frame
        ierr = frame_func(runner, parser, n=n)
    # Output code
    return 0 if ierr is None else ierr

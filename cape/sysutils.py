r"""
:mod:`cape.sysutils`: System utilities for using CAPE
======================================================

This module provides various "system" utilities such as providing a
universal Python method to open a PDF file for viewing.
"""

# Standard library
import glob
import os
import shutil
import sys
import tarfile
from subprocess import Popen, run, DEVNULL, PIPE
from typing import Optional, Tuple

# Local imports
from . import capeconfig
from .errors import CapeFileNotFoundError, CapeRuntimeError, CapeValueError

# Cache file name
CACHE_FILE = "tmp.tar.gz"


# Post file(s)
def post_file(pat1: str, *pats, v: bool = False) -> Tuple[int, list]:
    r"""Collect file(s) in tar ball; send to *RemoteHost* if necessary

    :Call:
        >>> ierr, filenames = post_file(pat1, *pats)
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *filenames*: :class:`list`\ [:class:`str`]
            List of posted file names
    """
    # Get [and create] cache dir
    cachedir = capeconfig.get_cape_cachedir()
    # Path to tar file
    tarpath = os.path.join(cachedir, CACHE_FILE)
    # Combine all patterns
    patterns = [pat1] + list(pats)
    # Create tar.gz file
    try:
        # Create tar file
        with tarfile.open(tarpath, "w:gz") as tar:
            # Full list of file names
            filenames = []
            # Loop through file name patterns
            for pattern in patterns:
                # Expand glob pattern
                matches = glob.glob(pattern)
                # Check if pattern matched anything
                if not matches:
                    print(f"Warning: pattern '{pattern}' matched no files")
                    continue
                # Add each matched file to archive
                for filepath in matches:
                    if v:
                        print(f"Adding: {filepath}")
                    # Add file to tarball
                    tar.add(filepath)
                    # Add to output
                    filenames.append(filepath)
        # Send the file
        _send_file(tarpath)
        # Return code
        return 0, filenames
    except Exception as e:
        print(f"Failed to post file(s) '{pat1} {' '.join(pats)}': {e}")
        return 1, []


# Receive file(s)
def receive_file() -> list:
    r"""Receive the current cached tarball

    :Call:
        >>> filenames = receive_file()
    :Outputs:
        *filenames*: :class:`list`\ [:class:`str`]
            List of unpacked files
    """
    # Get cache dir
    cachedir = capeconfig.get_cape_cachedir()
    # Path to tar file
    tarpath = os.path.join(cachedir, CACHE_FILE)
    # Receive file if necessary
    if capeconfig.check_cape_local():
        # Get remote host and config commands
        remote_host = capeconfig.get_cape_opt("RemoteHost")
        remote_cmds = capeconfig.get_cape_opt("RemoteLoginCommands")
        # Validate
        if remote_host is None:
            raise CapeValueError("No CAPE 'RemoteHost' setting found")
        # Format environment prep commands
        if remote_cmds:
            remote_env = '; '.join(remote_cmds) + '; '
        else:
            remote_env = ''
        # Remote command
        remote_cmd = (
            f'{remote_env}D=$(cape get-config CacheDir); '
            f'cat "$D/{CACHE_FILE}"')
        # Open local file to pipe into STDIN
        with open(tarpath, 'wb') as fp:
            # Run SSH and receive file there
            proc = run(['ssh', remote_host, remote_cmd], stdout=fp)
        # Check status
        if proc.returncode:
            raise CapeFileNotFoundError(
                f"Failed to receive {CACHE_FILE} from {remote_host}")
    # Check if tar file exists
    if not os.path.isfile(tarpath):
        return []
    # List to hold extracted file names
    extracted_files = []
    # Open and extract tar file
    with tarfile.open(tarpath, "r:gz") as tar:
        # Get list of members
        members = tar.getmembers()
        # Extract all files to cache dir
        tar.extractall(path=os.getcwd(), filter='data')
        # Collect file names
        extracted_files = [member.name for member in members]
    # Return list of extracted files
    return extracted_files


# Send file to remote host
def _send_file(localfile: Optional[str] = None):
    # Check if already remote
    if capeconfig.check_cape_remote():
        return
    # Default file name is the CACHE file
    if localfile is None:
        localfile = os.path.join(capeconfig.get_cape_cachedir(), CACHE_FILE)
    # Check for local file
    if not os.path.isfile(localfile):
        raise CapeFileNotFoundError(f"No file '{localfile}'")
    # Get remote host and config commands
    remote_host = capeconfig.get_cape_opt("RemoteHost")
    remote_cmds = capeconfig.get_cape_opt("RemoteLoginCommands")
    # Validate
    if remote_host is None:
        raise CapeValueError("No CAPE 'RemoteHost' setting found")
    # Format environment prep commands
    if remote_cmds:
        remote_env = '; '.join(remote_cmds) + '; '
    else:
        remote_env = ''
    # Base name of file (for destination to mimic)
    basename = os.path.basename(localfile)
    # Remote command
    remote_cmd = (
        f'{remote_env}D=$(cape get-config CacheDir); '
        f'cat > "$D/{basename}"')
    # Open local file to pipe into STDIN
    with open(localfile, 'rb') as fp:
        # Run SSH and receive file there
        run(['ssh', remote_host, remote_cmd], stdin=fp, check=True)


# Get preferred PDF viewer
def get_pdf_viewer() -> str:
    r"""Get the preferred PDF viewer application based on system

    :Call:
        >>> viewer = get_pdf_viewer()
    :Outputs:
        *viewer*: :class:`str`
            Name of application to open PDF
    :Versions:
        * 2026-08-07 ``@ddalle``: v1.0
    """
    return capeconfig.get_pdf_viewer()


# Get preferred PNG viewer
def get_png_viewer() -> str:
    r"""Get the preferred PNG viewer application based on system

    :Call:
        >>> viewer = get_png_viewer()
    :Outputs:
        *viewer*: :class:`str`
            Name of application to open PNG
    :Versions:
        * 2026-08-31 ``@ddalle``: v1.0
    """
    return capeconfig.get_png_viewer()


# Open an image as PNG
def open_img(
        fname: str,
        terminal: bool = True,
        dpi: int = 120,
        page: int = 0) -> str:
    r"""Attempt to open any image file, preferring in-terminal viewing

    :Call:
        >>> open_img(fname, wait=False)
    :Inputs:
        *fname*: :class:`str`
            Name of file to open
        *terminal*: {``True``} | ``False``
            Option to try showing image in terminal
        *dpi*: {``120``} | :class:`int`
            Resolution for converting PDFs to PNGs
        *page*: {``0``} | :class:`int`
            Zero-based page index of PDF to view in terminal
    :Output:
        *viewer*: :class:`str`
            Name of viewer application (or ``"terminal"`` or ``"pdf"``)
    """
    # Check for file
    if not os.path.isfile(fname):
        raise CapeFileNotFoundError(f"No file '{fname}'")
    # Get extension
    ext = os.path.splitext(fname)[1]
    ext = ext.lower()
    # Check for PDF
    if ext == ".pdf":
        # Convert to PNG
        fpng = pdftopng(fname, dpi=dpi, page=page)
        converted = True
    else:
        # Assume we can show the image in its current format
        fpng = fname
        converted = False
    # Try to show it in the terminal
    if terminal:
        # Show in terminal
        viewer = _open_png_terminal(fpng)
    else:
        viewer = None
    # Fall back
    if viewer is None:
        if ext == ".pdf":
            # Open the original PDF
            open_pdf(fname)
            viewer = "pdf"
        else:
            # Open as a PNG
            viewer = _open_png_local(fname)
    # Clean up converted file
    if converted:
        os.remove(fpng)
    # Output
    return viewer


# Open a PDF
def open_pdf(
        fname: str,
        remote: Optional[str] = None,
        wait: bool = False,
        pull: bool = False,
        local: Optional[bool] = None) -> Popen:
    r"""Open a PDF file if found

    :Call:
        >>> open_pdf(fname, wait=False)
    :Inputs:
        *fname*: :class:`str`
            Name of file to open
        *remote*: {``None``} | :class:`str`
            Absolute folder on remote host (or ``$PWD``)
        *wait*: ``True`` | {``False``}
            Option to wait until PDF is closed
        *pull*: ``True`` | {``False``}
            Pull PDF from *RemoteHost* and open
        *local*: {``None``} | ``True`` | ``False``
            Force "local" host instead of determining automatically
    :Output:
        *proc*: :class:`subprocess.Popen`
            Subprocess handle
    :Versions:
        * 2026-08-07 ``@ddalle``: v1.0
    """
    # Check for "pull" option
    if pull:
        return _open_pdf_pull(fname, remote, wait)
    # Check for file
    if not os.path.isfile(fname):
        raise CapeFileNotFoundError(f"No file '{fname}'")
    # Check local option
    if local or capeconfig.check_cape_local():
        return _open_pdf_local(fname, wait)
    # Get name of "local" host to push file to
    local_host = capeconfig.get_cape_opt("LocalHost")
    # Fall-back if not specified
    if local_host is None:
        return _open_pdf_local(fname, wait)
    # Check if SSH jump-host is needed
    jump_host = capeconfig.get_cape_jumphost()
    # Commands to initialzie environment there
    remote_cmds = capeconfig.get_cape_opt("RemoteLoginCommands")
    # Format environment prep commands
    if remote_cmds:
        remote_env = '; '.join(remote_cmds) + '; '
    else:
        remote_env = ''
    # Format jumphost command
    if jump_host:
        # Include `-J` step
        base_cmd = ["ssh", "-J", jump_host, local_host]
    else:
        # Direct SSH login
        base_cmd = ["ssh", local_host]
    # Base name of file (for destination to mimic)
    basename = os.path.basename(fname)
    # Special command to detach remote app from SSH
    setsid = "" if wait else "setsid "
    # Remote command
    remote_cmd = (
        f'{remote_env}D=$(cape get-config CacheDir); '
        'V=$(cape get-config PDFReader); '
        f'cat > "$D/{basename}"; '
        f'{setsid}$V "$D/{basename}" &'
    )
    # Open the PDF file locally so we can pipe it through STDIN
    with open(fname, 'rb') as fp:
        proc = Popen(
            base_cmd + [remote_cmd],
            stdin=fp,
            stdout=DEVNULL,
            stderr=DEVNULL)
    # Wait option
    if wait:
        proc.communicate()
    # Return subprocess handle
    return proc


def _open_pdf_pull(
        fname: str,
        remote: Optional[str],
        wait: bool = False) -> Popen:
    # Check for absolute path
    if os.path.isabs(fname):
        raise CapeValueError(
            "'cape open-pdf --pull' is meant to work with relative files; "
            f"'{fname}' is absolute")
    # Get remote host and config commands
    remote_host = capeconfig.get_cape_opt("RemoteHost")
    # Validate
    if remote_host is None:
        raise CapeValueError("No CAPE 'RemoteHost' setting found")
    # Get folder name
    dirname = os.path.dirname(fname)
    # Create subfolders if necessary
    os.makedirs(dirname, exist_ok=True)
    # Path to remote file
    fdir = os.getcwd() if remote is None else remote
    fabs = os.path.join(fdir, fname)
    fabs = fabs.replace(os.path.sep, '/')
    # Copyt the file
    proc = run(['scp', f"{remote_host}:{fabs}", dirname])
    # Check status
    if proc.returncode:
        raise CapeFileNotFoundError(
            f"Failed to receive {CACHE_FILE} from {remote_host}")
    # Open the (now-updated) local file
    return _open_pdf_local(fname, wait)


def _open_pdf_local(fname: str, wait: bool = False) -> Popen:
    # Get viewer
    viewer = get_pdf_viewer()
    # Command to open it
    proc = Popen([viewer, fname], stdout=PIPE, stderr=PIPE)
    # Wait option
    if wait:
        proc.wait()
    # Return subprocess handle
    return proc


def open_png(
        fname: str,
        terminal: bool = True,
        wait: bool = False) -> str:
    r"""Open a PNG file if found

    :Call:
        >>> open_png(fname, wait=False)
    :Inputs:
        *fname*: :class:`str`
            Name of file to open
        *terminal*: {``True``} | ``False``
            Option to try showing image in terminal
        *wait*: ``True`` | {``False``}
            Option to wait until PNG is closed
    :Output:
        *viewer*: :class:`str`
            Name of viewer used
    """
    # Check for file
    if not os.path.isfile(fname):
        raise CapeFileNotFoundError(f"No file '{fname}'")
    # Try to show it in the terminal
    if terminal:
        viewer = _open_png_terminal(fname)
        # Exit if successful
        if viewer is not None:
            return viewer
    return _open_png_local(fname, wait=wait)


def _open_png_terminal(fname: str) -> str | None:
    try:
        # Necessary imports
        from rich.console import Console
        from textual_image.renderable import Image, TGPImage, SixelImage
        # Check which class we got
        if Image not in (TGPImage, SixelImage):
            # Clean up prompt
            sys.__stdout__.write("\r\x1b[2K")
            sys.__stdout__.flush()
            return
        # Open the console
        console = Console()
    except Exception:
        return
    # Read image and show it
    console.print(Image(fname))
    # Blank line
    print("")
    # Output
    return "terminal"


def _open_png_local(fname: str, wait: bool = False) -> str:
    # Get viewer
    viewer = get_png_viewer()
    # Command to open it
    proc = Popen([viewer, fname], stdout=PIPE, stderr=PIPE)
    # Wait option
    if wait:
        proc.wait()
    # Return subprocess handle
    return viewer


# Convert a page of a PDF to a PNG image
def pdftopng(
        fpdf: str,
        fpng: Optional[str] = None,
        dpi: int = 120,
        page: int = 0) -> str:
    r"""Convert one page of a PDF to a PNG image

    Uses the first available system command:

    1.  ``pdftoppm`` (poppler)
    2.  ``gs``, ``gswin64c``, ``gswin32c`` (Ghostscript)
    3.  ``pdftocairo`` (poppler)
    4.  ``magick`` or ``convert`` (ImageMagick)

    :Call:
        >>> fpng = pdftopng(fpdf, fpng=None, dpi=200, page=0)
    :Inputs:
        *fpdf*: :class:`str`
            Name of PDF file to read
        *fpng*: {``None``} | :class:`str`
            Name of PNG file to write; uses *fpdf* with extension changed
            to ``.png`` if not given
        *dpi*: {``120``} | :class:`int`
            Resolution of output image, in dots per inch
        *page*: {``0``} | :class:`int`
            Index of PDF page to convert, first page is 0
    :Outputs:
        *fpng*: :class:`str`
            Name of PNG file created
    :Versions:
        * 2026-08-31 ``@ddalle``: v1.0
    """
    # Check for file
    if not os.path.isfile(fpdf):
        raise CapeFileNotFoundError(f"No file '{fpdf}'")
    # Default output file name
    if fpng is None:
        fpng = os.path.splitext(fpdf)[0] + ".png"
    # Fall back to system tools
    if _pdftopng_system(fpdf, fpng, dpi, page):
        return fpng
    # No converter available
    raise CapeRuntimeError("No PDF-to-PNG converter available")


# Convert PDF to PNG using first available system command
def _pdftopng_system(fpdf: str, fpng: str, dpi: int, page: int) -> bool:
    # Note poppler and Ghostscript use 1-based page numbers
    n = page + 1
    # Try pdftoppm (poppler)
    if shutil.which("pdftoppm"):
        prefix = os.path.splitext(fpng)[0]
        run([
            "pdftoppm",
            "-png", "-singlefile",
            "-r", str(dpi),
            "-f", str(n), "-l", str(n),
            fpdf, prefix
        ], check=True)
        if prefix + ".png" != fpng:
            os.replace(prefix + ".png", fpng)
        return True
    # Try Ghostscript ('gs' on Linux, console name on Windows)
    gs = (
        shutil.which("gs") or
        shutil.which("gswin64c") or
        shutil.which("gswin32c"))
    if gs:
        run([
            gs,
            "-dBATCH", "-dNOPAUSE", "-dQUIET", "-dSAFER",
            "-sDEVICE=png16m",
            f"-r{dpi}",
            f"-dFirstPage={n}", f"-dLastPage={n}",
            f"-sOutputFile={fpng}",
            fpdf
        ], check=True)
        return True
    # Try pdftocairo (poppler)
    if shutil.which("pdftocairo"):
        # Prefix for output; command appends ".png" when using -singlefile
        prefix = os.path.splitext(fpng)[0]
        # Convert single page
        run([
            "pdftocairo",
            "-png", "-singlefile",
            "-r", str(dpi),
            "-f", str(n), "-l", str(n),
            fpdf, prefix
        ], check=True)
        # Rename if *fpng* does not end in ".png"
        if prefix + ".png" != fpng:
            os.replace(prefix + ".png", fpng)
        return True
    # Try ImageMagick (0-based page selector)
    magick = shutil.which("magick") or shutil.which("convert")
    if magick:
        run([
            magick,
            "-density", str(dpi),
            f"{fpdf}[{page}]",
            "-quality", "95",
            fpng
        ], check=True)
        return True
    # No system tools found
    return False

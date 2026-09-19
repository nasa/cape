r"""
:mod:`cape.agent.tools.cntl`: CAPE API tools for :mod:`cape.agent`
=======================================================================

This module provides definitions and tool schemas for tool calls to most
of the low-level CLI functions defined in :mod:`cape.cfdx.cli`.
"""

# Standard library
import base64
import contextlib
import mimetypes
import os
from typing import Callable

# Local imports
from .toolutils import register_module_tools
from ... import sysutils
from ...cfdx import cli


# List of parameters common to **all** run-matrix commands
CAPE_PARAMS = {
    "f": {
        "description": (
            "Name of JSON file to use. CAPE will find the most appropriate "
            "file if left empty. If users specify a file, continue to use "
            "that file until user specifically requests a different one."
            "Synonyms: file, json."
        ),
        "type": ["string", "null"],
    },
    "I": {
        "description": (
            "Indices of cases to consider. Indexing follows Python syntax. "
            "This can be a single case "
            "like '14', a comma-separated list like '14,19,20', "
            "a range such as '14:20', or a combination like 14,17:20. "
            "Examples:\n* Case 8 -> '8'\n* Cases 5-10 -> '5:11'"
        ),
        "type": ["string", "null"],
    },
    "report": {
        "description": "Name of specific report to generate. Optional",
        "type": ["string", "null"]
    },
    "subfig": {
        "description": (
            "Name of the report subfigure to create and look at."
        ),
        "type": "string",
    },
    "force": {
        "description": (
            "Regenerate the subfigure image even if a cached version "
            "exists."
        ),
        "type": ["boolean", "null"],
    },
}


def get_subfigs(f: str | None = None, report: str | None = None) -> dict:
    # Read *cntl*
    cntl = cli.read_cntl_q(f)
    # List the subfigures
    return {
        "report": report,
        "subfigures": cntl.get_subfigs(report),
    }


def get_reports(f: str | None = None) -> dict:
    # Read *cntl*
    cntl = cli.read_cntl_q(f)
    # List the reports
    return {
        "reports": cntl.opts.get_ReportList(),
    }


# Read an image file and format it as a base64 data URL
def _image_data_url(fname: str, dpi: int = 120, page: int = 0) -> str:
    r"""Read an image file as a base64 ``data:`` URL for vision input

    :Call:
        >>> url = _image_data_url(fname, dpi=120, page=0)
    :Inputs:
        *fname*: :class:`str`
            Name of image file; PDFs are converted to PNG
        *dpi*: :class:`int`
            Resolution for converting PDFs to PNGs
        *page*: :class:`int`
            Zero-based page index of PDF to convert
    :Outputs:
        *url*: :class:`str`
            Data URL of the image, e.g. ``"data:image/png;base64,..."``
    """
    # Convert PDF to PNG
    if os.path.splitext(fname)[1].lower() == ".pdf":
        fpng = sysutils.pdftopng(fname, dpi=dpi, page=page)
        converted = True
    else:
        fpng = fname
        converted = False
    # Guess MIME type
    mime = mimetypes.guess_type(fpng)[0] or "image/png"
    # Read and encode
    with open(fpng, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode("ascii")
    # Clean up converted file
    if converted:
        os.remove(fpng)
    # Output
    return f"data:{mime};base64,{b64}"


# Create a subfigure and return its image(s) for the agent to look at
def view_subfig(
        subfig: str,
        f: str | None = None,
        I: str | None = None,
        force: bool | None = None,
        dpi: int | None = None,
        page: int | None = None) -> dict:
    r"""Create one report subfigure per case and return image(s) to agent

    This mirrors :func:`cape.cfdx.cli.cape_open_subfig` except that the
    cached image file(s) are base64-encoded and attached to the
    conversation for the agent to look at, rather than displayed in the
    terminal.

    :Call:
        >>> result = view_subfig(subfig, f=None, I=None)
    :Inputs:
        *subfig*: :class:`str`
            Name of the report subfigure to create
        *f*: {``None``} | :class:`str`
            Name of CAPE JSON file (or use most recent)
        *I*: {``None``} | :class:`str`
            Indices of cases to consider
        *force*: {``None``} | ``True`` | ``False``
            Regenerate the subfigure even if cached
        *dpi*: {``None``} | :class:`int`
            Resolution for converting PDFs to PNGs
        *page*: {``None``} | :class:`int`
            Zero-based page index of PDF to convert
    :Outputs:
        *result*: :class:`dict`
            Keys include *success*, *subfig*, *caselist*, *skipped*,
            *n_images*, and *images*; each *images* entry has *case*,
            *folder*, *file*, and *url* (a base64 data URL). The agent
            controller extracts *images* and attaches them to the
            conversation as multimodal content.
    """
    # Read *cntl*
    cntl = cli.read_cntl_q(f)
    # Build case-selector kwargs
    kw_find = {}
    if I:
        kw_find["I"] = I
    # Resolve cases quietly
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            inds = cntl.GetIndices(**kw_find)
            keeps = set(cntl.GetNonzeroIndices(**kw_find))
    # Loop through cases
    images = []
    caselist = []
    skipped = []
    for i in inds:
        # Get case name
        frun = cntl.x.GetFullFolderNames(i)
        # Check for nonzero iterations
        if i not in keeps:
            skipped.append({"case": int(i), "folder": frun})
            continue
        # Create subfigure and cache its image
        v = cntl.get_subfigure(subfig, I=[i], force=bool(force))
        # Encode cached image file(s) (usually just one)
        for fimg in v.get("cachefiles", ()):
            images.append({
                "case": int(i),
                "folder": frun,
                "file": fimg,
                "url": _image_data_url(
                    fimg,
                    dpi=int(dpi) if dpi else 120,
                    page=int(page) if page else 0,
                ),
            })
        # Save it
        caselist.append([int(i), frun])
    # Output
    return {
        "success": True,
        "subfig": subfig,
        "caselist": caselist,
        "skipped": skipped,
        "n_images": len(images),
        "images": images,
        "message": (
            "The subfigure image(s) are attached to the message "
            "immediately following this tool result; look at them to "
            "answer the user's question."
        ),
    }


# Simplified definitions not in OpenAPI format
TOOL_DICT = {
    "get_subfigs": {
        "description": (
            "List the subfigures, either of all reports if 'report' is not "
            "given or of a specific named report."
        ),
        "parameters": [
            "f",
            "report",
        ],
    },
    "get_reports": {
        "description": "Get list of reports available",
        "parameters": ["f"],
    },
    "view_subfig": {
        "description": (
            "Create one report subfigure for the selected cases and "
            "return its image(s) so you can look at them directly. Use "
            "this to answer questions about what a plot or figure looks "
            "like. Use get_reports/get_subfigs first if the subfigure "
            "name is unknown. The image(s) arrive in the message right "
            "after the tool result."
        ),
        "parameters": ["subfig", "f", "I", "force"],
        "required": ["subfig"],
    },
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Tool sets per capability
TOOL_SETS = {
    "none": [],
    "low": [
        "get_subfigs",
        "get_reports",
    ],
    "medium": [
        "get_subfigs",
        "get_reports",
        "view_subfig",
    ],
    "full": [
        "get_subfigs",
        "get_reports",
        "view_subfig",
    ],
}


# Function generator
def genr8_func(funcname: str) -> Callable:
    # Create a function
    def fn(*a, **kw):
        return {}
    # Return it
    return fn


# Register tools
register_module_tools(CAPE_PARAMS)

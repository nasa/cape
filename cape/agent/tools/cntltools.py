r"""
:mod:`cape.agent.tools.cntl`: CAPE API tools for :mod:`cape.agent`
=======================================================================

This module provides definitions and tool schemas for tool calls to most
of the low-level CLI functions defined in :mod:`cape.cfdx.cli`.
"""

# Standard library
import base64
import contextlib
import fnmatch
import json
import math
import mimetypes
import numbers
import os
from typing import Callable

# Local imports
from .toolutils import register_module_tools
from ..agentutils import _NPEncoder
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
    "i": {
        "description": "Index of a specific case",
        "type": ["integer"],
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
    "key": {
        "description": (
            "Optional shell-style pattern selecting run matrix key names, "
            "for example 'a*'. By default all keys are described."
        ),
        "type": ["string", "null"],
    },
    "detail": {
        "description": (
            "Level of effective key-definition detail to return: 'summary' "
            "for the most useful properties or 'full' for every normalized "
            "definition property. Default: 'summary'."
        ),
        "type": ["string", "null"],
        "enum": ["summary", "full", None],
    },
    "include_values": {
        "description": (
            "Include a compact summary of the values present in each run "
            "matrix column. Default: true."
        ),
        "type": ["boolean", "null"],
    },
}


def enter_case(i: int, f: str | None = None) -> dict:
    # Read *cntl*
    cntl = cli.read_cntl_q(f)
    # Get name of folder
    try:
        frun = cntl.x.GetFullFolderNames(i)
    except Exception:
        return {
            "ok": False,
            "reason": f"Case {i} not found",
        }
    # Absolute path
    fabs = os.path.join(cntl.RootDir, frun)
    # Check for the folder
    if not os.path.isdir(fabs):
        return {
            "ok": False,
            "reason": f"No folder for case {i} ({frun})",
        }
    # Enter the folder
    os.chdir(fabs)
    # Result
    return {
        "ok": True,
        "cwd": fabs,
        "case_name": frun,
        "case_number": i,
        "json_file": os.path.normpath(os.path.join(cntl.fdir, cntl.fname)),
    }


def get_subfigs(f: str | None = None, report: str | None = None) -> dict:
    # Read *cntl*
    cntl = cli.read_cntl_q(f)
    # List the subfigures
    return {
        "report": report,
        "subfigures": cntl.get_subfigs(report),
    }


def describe_run_matrix_keys(
        f: str | None = None,
        key: str | None = None,
        detail: str | None = "summary",
        include_values: bool | None = True) -> dict:
    r"""Describe effective definitions and values of run matrix keys

    :Call:
        >>> result = describe_run_matrix_keys(f=None)
    :Inputs:
        *f*: {``None``} | :class:`str`
            Name of CAPE JSON file (or find the most appropriate file)
        *key*: {``None``} | :class:`str`
            Optional shell-style pattern matching key names
        *detail*: {``"summary"``} | ``"full"``
            Return selected or all effective definition properties
        *include_values*: {``True``} | ``False``
            Include compact summaries of the matrix-column values
    :Outputs:
        *result*: :class:`dict`
            Project metadata and an ordered list of key descriptions
    """
    # Validate options before doing the comparatively expensive JSON read
    detail = detail or "summary"
    if detail not in ("summary", "full"):
        return {
            "success": False,
            "error": "'detail' must be either 'summary' or 'full'",
        }
    # Read the effective, normalized control instance
    cntl = cli.read_cntl_q(f)
    cols = cntl.opts.get_RunMatrixKeys()
    if key:
        cols = fnmatch.filter(cols, key)
    # Describe keys in run-matrix order
    keys = []
    for col in cols:
        defn = cntl.x.defns[col]
        item = {
            "name": col,
            "type": defn.get("Type", col),
            "value_type": defn.get("Value", "float"),
            "group": defn.get("Group", False),
            "label": defn.get("Label", True),
            "abbreviation": defn.get("Abbreviation", col),
            "format": defn.get("Format", "%s"),
        }
        if detail == "full":
            item["definition"] = _jsonify(defn)
        if include_values is not False:
            item["values"] = _summarize_values(cntl.x[col])
        keys.append(item)
    # Identify the files that contributed the semantic result
    json_file = os.path.normpath(os.path.join(cntl.fdir, cntl.fname))
    matrix_file = cntl.x.fname
    return {
        "success": True,
        "json_file": json_file,
        "matrix_file": matrix_file,
        "case_count": int(cntl.x.nCase),
        "key_count": len(keys),
        "keys": keys,
    }


def get_keys(f: str | None = None) -> dict:
    r"""Compatibility wrapper for :func:`describe_run_matrix_keys`"""
    return describe_run_matrix_keys(f=f)


def _jsonify(v):
    r"""Convert a CAPE or NumPy value to JSON-compatible objects"""
    return json.loads(json.dumps(v, cls=_NPEncoder, default=str))


def _summarize_values(values, max_examples: int = 8) -> dict:
    r"""Create a bounded, JSON-compatible summary of one matrix column"""
    count = 0
    examples = []
    seen = set()
    vmin = None
    vmax = None
    for v in values:
        val = _jsonify(v)
        count += 1
        # Use canonical JSON as a hashable identity for scalar and structured
        # values alike. Keep only a bounded list of examples in memory.
        token = json.dumps(val, sort_keys=True)
        if token not in seen:
            seen.add(token)
            if len(examples) < max_examples:
                examples.append(val)
        # Booleans are numbers in Python, but numeric ranges are not useful
        # for a boolean-valued column.
        if _is_finite_real(val):
            vmin = val if vmin is None else min(vmin, val)
            vmax = val if vmax is None else max(vmax, val)
    result = {
        "count": count,
        "unique_count": len(seen),
    }
    if len(seen) <= max_examples:
        result["unique"] = examples
    else:
        result["examples"] = examples
        result["examples_truncated"] = True
    if vmin is not None:
        result["min"] = vmin
        result["max"] = vmax
    return result


def _is_finite_real(v) -> bool:
    r"""Return whether *v* is a finite, non-boolean real number"""
    return (isinstance(v, numbers.Real) and
            not isinstance(v, bool) and
            math.isfinite(v))


def get_reports(f: str | None = None) -> dict:
    # Read *cntl*
    cntl = cli.read_cntl_q(f)
    # List the reports
    return {
        "reports": cntl.opts.get_ReportList(),
    }


def return_to_root() -> dict:
    # Initialize result
    result = {"old": os.getcwd(), "ok": True}
    # Check for cache
    if len(cli.CNTL_CACHE) == 0:
        # Check if any JSON files were found
        json_files = cli.manage.find_json_solver()
        if len(json_files) == 0:
            # Nothing found
            result["ok"] = False
            result["reason"] = "No CAPE files found or in this session"
    else:
        # Use the first one
        for json_file, cntl in cli.CNTL_CACHE.items():
            break
        # Go to root folder forcibly
        os.chdir(cntl.RootDir)
        # Save the file used
        result["json_file"] = json_file
    # Save new path
    result["cwd"] = os.getcwd()
    # Output
    return result


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
    "enter_case": {
        "description": "Enter the working directory of a specific case.",
        "parameters": ["i", "f"],
        "required": ["i"],
    },
    "describe_run_matrix_keys": {
        "description": (
            "Describe the effective, normalized run matrix keys, including "
            "their CAPE type, value type, grouping and naming behavior, and "
            "a compact summary of column values. Use this for questions "
            "such as 'describe my run matrix keys'; use detail='full' for "
            "complete normalized definitions."
        ),
        "parameters": ["f", "key", "detail", "include_values"],
    },
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
    "return_to_root": {
        "description": "Return to top-level folder of project",
        "parameters": [],
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
        "describe_run_matrix_keys",
        "get_subfigs",
        "get_reports",
        "return_to_root",
    ],
    "medium": [
        "enter_case",
        "describe_run_matrix_keys",
        "get_subfigs",
        "get_reports",
        "return_to_root",
        "view_subfig",
    ],
    "full": [
        "enter_case",
        "describe_run_matrix_keys",
        "get_subfigs",
        "get_reports",
        "return_to_root",
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

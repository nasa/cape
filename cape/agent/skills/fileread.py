r"""
:mod:`cape.agent.skills.fileread`: LLM-driven reading of repo files
====================================================================

This module defines the built-in agent skill ``"file-reader"``, which
teaches the CAPE agent how to read any text file in the repo in which
the agent is launched, subject to a file-size limit of about 2 MB
(*MAX_READ_BYTES*).  Editing files is deliberately out of scope here;
it is restricted by the allow-list of the ``file-editor`` skill (see
:mod:`cape.agent.skills.fileedit`).

The skill provides one tool:

* :func:`read_file`: read a file with line numbers, truncating large
  files to their first and last lines

This module also hosts the path-resolution helpers
:func:`get_rootdir` and :func:`resolve_fname`, which are shared with
:mod:`cape.agent.skills.fileedit`.

Every tool resolves any symlinks in the target file's path and rejects
files outside the repo root folder.
"""

# Standard library
import os

# Local imports
from ..tools import toolutils


# Folder in which agent was launched; defaults to cwd
ROOT_DIR: str | None = None

# Maximum file size (bytes) accepted by read_file
MAX_READ_BYTES = 2 * 1024 * 1024

# Truncation limits for read_file output
MAX_READ_LINES = 2000
HEAD_LINES = 1500
TAIL_LINES = 500


# Parameter definitions for the tool schema
SKILL_PARAMS = {
    "fname": {
        "description": (
            "Name of file to read, either absolute or relative to the "
            "current folder. The file must be inside the repo root "
            "folder and smaller than 2 MB."
        ),
        "type": "string",
    },
}


# Get absolute path of repo root folder
def get_rootdir() -> str:
    # Substitute cwd if *ROOT_DIR* not set
    rootdir = os.getcwd() if ROOT_DIR is None else ROOT_DIR
    return os.path.realpath(rootdir)


# Resolve *fname* against the root folder and check containment
def resolve_fname(fname) -> tuple | dict:
    r"""Resolve a file name and check that it is inside the root folder

    Returns a tuple ``(fabs, relname)`` with the resolved absolute path
    and the POSIX-style name relative to the root folder, or an error
    :class:`dict` with ``"success": False``.
    """
    # Check type
    if not isinstance(fname, str) or not fname:
        return {
            "success": False,
            "error": "'fname' must be a nonempty string",
        }
    # Absolute path of repo root
    rootdir = get_rootdir()
    # Absolutize; relative names use the current folder (see ``chdir``)
    if os.path.isabs(fname):
        fabs = fname
    else:
        fabs = os.path.join(os.getcwd(), fname)
    # Resolve any links and ``.`` or ``..`` components
    freal = os.path.realpath(fabs)
    # Check that the resolved path is inside the root folder
    try:
        inside = os.path.commonpath([freal, rootdir]) == rootdir
    except ValueError:
        inside = False
    if not inside:
        return {
            "success": False,
            "error": f"File '{fname}' is outside the repo root folder",
        }
    # POSIX-style name relative to root folder
    relname = os.path.relpath(freal, rootdir).replace(os.sep, "/")
    # Output
    return freal, relname


# Read a file with line numbers
def read_file(fname: str) -> dict:
    r"""Read a repo file, prefixing each line with its number

    The file must be inside the repo root folder and no larger than
    *MAX_READ_BYTES*.  Large files are truncated to the first
    *HEAD_LINES* and last *TAIL_LINES* lines.

    :Call:
        >>> result = read_file(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read; must be inside the repo root folder
            and smaller than 2 MB
    :Outputs:
        *result*: :class:`dict`
            Keys include *success*, *file*, *n_lines*, *content*, and
            *truncated*
    """
    # Check whether the file may be accessed
    check = resolve_fname(fname)
    if isinstance(check, dict):
        return check
    freal, relname = check
    # Check for file
    if not os.path.isfile(freal):
        return {
            "success": False,
            "error": f"No such file: '{relname}'",
        }
    # Check file size
    try:
        nbytes = os.path.getsize(freal)
    except OSError as e:
        return {
            "success": False,
            "error": f"Could not read '{relname}': {e}",
        }
    if nbytes > MAX_READ_BYTES:
        mb = MAX_READ_BYTES/(1024*1024)
        return {
            "success": False,
            "error": (
                f"File '{relname}' is too large to read "
                f"({nbytes} bytes > {mb:.0f} MB limit)"),
        }
    # Read the file
    try:
        with open(freal) as fp:
            text = fp.read()
    except UnicodeDecodeError:
        return {
            "success": False,
            "error": f"File '{relname}' is not a valid text file",
        }
    except OSError as e:
        return {
            "success": False,
            "error": f"Could not read '{relname}': {e}",
        }
    # Split into lines
    lines = text.splitlines()
    n_lines = len(lines)
    # Truncate large files, keeping line numbering intact
    truncated = n_lines > MAX_READ_LINES
    if truncated:
        # Line numbers for the head and tail parts
        head = lines[:HEAD_LINES]
        tail = lines[-TAIL_LINES:]
        j0 = n_lines - TAIL_LINES
        # Assemble numbered content with a marker for the omitted part
        parts = [f"{j + 1}: {line}" for j, line in enumerate(head)]
        parts.append(
            f"... [{n_lines - HEAD_LINES - TAIL_LINES} lines omitted] ...")
        parts += [f"{j0 + k + 1}: {line}" for k, line in enumerate(tail)]
        content = "\n".join(parts)
    else:
        content = "\n".join(
            f"{j + 1}: {line}" for j, line in enumerate(lines))
    # Output
    return {
        "success": True,
        "file": relname,
        "n_lines": n_lines,
        "truncated": truncated,
        "content": content,
    }


# Full Markdown instructions provided to the agent via ``use_skill``
SKILL_CONTENT = r"""
# file-reader: reading files in this repo

Use this skill when you need to look at the contents of a file in this
repo, for example to inspect a CAPE JSON file, a script in the `tools/`
folder, or a set of notes. Reading is unrestricted within the repo, but
this skill cannot modify files.

## What you can read

* Any text file inside the repo root folder, up to about 2 MB.
* Symlinks are resolved, and files outside the repo root folder are
  always rejected.
* Binary files and files that are not valid UTF-8 text are rejected.

## Workflow

1. Call `read_file` with `fname`, the file name absolute or relative to
   the current folder. The content comes back with `"<line>: "`
   prefixes.
2. Large files are truncated to their first and last lines, with a
   marker showing how many lines were omitted; `n_lines` reports the
   true line count.
3. If the user asks you to *change* a file, this skill is not enough:
   call `use_skill('file-editor')` to activate the `edit_file` tool,
   which only accepts files matching its allow-list.

## Limits

* This skill cannot create, delete, or modify files.
* You cannot read files outside the repo root folder.
"""

# Simplified skill definition
SKILL_DICT = {
    "file-reader": {
        "description": (
            "Read any text file in this repo smaller than 2 MB, "
            "with line numbers. Use when the user asks you to look "
            "at, explain, or summarize a file. Does not allow edits."
        ),
        "content": SKILL_CONTENT,
        "tools": ["read_file"],
    },
}

# Simplified tool definitions not in OpenAPI format
TOOL_DICT = {
    "read_file": {
        "description": (
            "Read a repo file with line numbers; files must be "
            "smaller than 2 MB, and larger line counts are truncated "
            "to the first and last lines. Call "
            "use_skill('file-reader') for full instructions first."
        ),
        "parameters": ["fname"],
        "required": ["fname"],
    },
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Register tools
toolutils.register_module_tools(SKILL_PARAMS)

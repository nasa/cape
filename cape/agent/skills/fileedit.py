r"""
:mod:`cape.agent.skills.fileedit`: LLM-driven editing of allow-listed files
=============================================================================

This module defines the built-in agent skill ``"file-editor"``, which
teaches the CAPE agent how to edit a restricted set of files in the
repo in which the agent is launched.

The skill provides three tools:

* :func:`list_editable_files`: list the active allow-list patterns and
  the existing files that match them
* :func:`read_file`: read a file with line numbers (any repo file
  smaller than about 2 MB; provided by
  :mod:`cape.agent.skills.fileread`)
* :func:`edit_file`: apply an exact-match search-and-replace edit to an
  allow-listed file, returning a unified diff of the change

The *allow-list* restricts which files may be *edited*; it does not
apply to reading, which is limited only by the repo root folder and the
file-size limit of the ``file-reader`` skill (see
:mod:`cape.agent.skills.fileread`).

The set of editable files (the *allow-list*) comes from three sources:

* the top-level *EditAllowList* option of the ``cape-agent.json``
  file: a list of glob patterns relative to the repo root (see
  :mod:`cape.agent.options`)
* :func:`cape.cfdx.cntl.Cntl.get_edit_allowlist`, which allows the
  agent to edit the CAPE control JSON file itself
* :data:`cape.agent.agentcntl.EDIT_FILE_ALLOW_LIST`, a registry of
  file names that other skills (e.g. ``fix-json``) append to at run
  time; this is the only route for files that cannot be read into a
  *Cntl* instance, such as a JSON file with a syntax error

Note that :func:`fnmatch.fnmatch` is used for matching, so ``"*"``
matches folder separators as well; a pattern such as ``"tools/*.py"``
matches ``tools/a/b.py`` as well as ``tools/a.py``.

Every tool resolves any symlinks in the target file's path and rejects
files outside the repo root folder, regardless of the allow-list.
"""

# Standard library
import difflib
import fnmatch
import glob
import os

# Local imports
from . import fileread
from ..tools import toolutils
from ...cfdx import cli


# Glob patterns, relative to the root folder, of files the agent may edit
ALLOW_PATTERNS: list = []

# Truncation limit for the diff returned by edit_file
MAX_DIFF_CHARS = 12000

# Shared path helpers and read tool (allow-list applies only to edits)
_get_rootdir = fileread.get_rootdir
_resolve_fname = fileread.resolve_fname
read_file = fileread.read_file


# Parameter definitions for the tool schema
SKILL_PARAMS = {
    "fname": {
        "description": (
            "Name of file to read or edit, either absolute or relative "
            "to the current folder. Any text file in the repo smaller "
            "than 2 MB may be read; to edit a file it must match the "
            "file-editor skill's allow-list, so call "
            "list_editable_files to see which files are editable."
        ),
        "type": "string",
    },
    "old": {
        "description": (
            "Exact text to search for, including indentation. Must "
            "occur exactly once in the file; include more surrounding "
            "context if the first attempt is not unique."
        ),
        "type": "string",
    },
    "new": {
        "description": (
            "Replacement text that takes the place of 'old'. May be "
            "empty to delete the matched text."
        ),
        "type": "string",
    },
}


# Get the file names provided by the current ``Cntl`` instance
def _get_cntl_allowlist() -> list:
    # Read the most appropriate CAPE JSON file into a *Cntl*
    try:
        cntl = cli.read_cntl_cache(None, None)
        flist = cntl.get_edit_allowlist()
    except Exception:
        # No readable CAPE JSON file available; other sources only
        return []
    # Normalize to POSIX-style names relative to the root folder
    return [
        os.path.normpath(f).replace(os.sep, "/")
        for f in flist
        if isinstance(f, str)
    ]


# Get the file names registered by other skills during this session
def _get_skill_allowlist() -> list:
    # Local imports (avoid circular import at module load time)
    from .. import agentcntl
    # Normalize to POSIX-style names relative to the root folder
    return [
        os.path.normpath(f).replace(os.sep, "/")
        for f in agentcntl.EDIT_FILE_ALLOW_LIST
        if isinstance(f, str)
    ]


# Merge static patterns, ``Cntl`` file names, and skill registrations
def _genr8_allowlist() -> list:
    # Start with the static patterns
    patterns = list(ALLOW_PATTERNS)
    # Append file names from the CAPE control instance and other skills
    for f in _get_cntl_allowlist() + _get_skill_allowlist():
        if f not in patterns:
            patterns.append(f)
    # Output
    return patterns


# Resolve *fname* and check it against the allow-list
def _check_editable(fname) -> tuple | dict:
    r"""Resolve a file name and check that the agent may edit it

    Returns a tuple ``(fabs, relname)`` with the resolved absolute path
    and the POSIX-style name relative to the root folder, or an error
    :class:`dict` with ``"success": False``.
    """
    # Resolve and check containment in the root folder
    check = _resolve_fname(fname)
    if isinstance(check, dict):
        return check
    freal, relname = check
    # Check the allow-list
    patterns = _genr8_allowlist()
    if not any(fnmatch.fnmatchcase(relname, pat) for pat in patterns):
        return {
            "success": False,
            "error": f"File '{relname}' is not in the edit allow-list",
            "allowed_patterns": patterns,
        }
    # Output
    return freal, relname


# Register an in-root-folder file as editable for this session
def register_editable_file(fname: str) -> dict:
    r"""Add a file to the session edit allow-list, for use by skills

    This is how other skills, e.g. ``fix-json``, grant the agent edit
    access to a specific file that is not covered by the static
    *EditAllowList* patterns or by the ``Cntl``-provided allow-list
    (for example a JSON file that cannot be read due to a syntax
    error). The file must be inside the repo root folder.

    :Call:
        >>> result = register_editable_file(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to register, absolute or relative to the
            current folder
    :Outputs:
        *result*: :class:`dict`
            Keys include *success*, *path* (resolved absolute path),
            *file* (name relative to the root folder), and *added*
            (``False`` if already registered)
    """
    # Local imports (avoid circular import at module load time)
    from .. import agentcntl
    # Resolve and check containment in the root folder
    check = _resolve_fname(fname)
    if isinstance(check, dict):
        return check
    freal, relname = check
    # Append to the registry if not already present
    added = relname not in agentcntl.EDIT_FILE_ALLOW_LIST
    if added:
        agentcntl.EDIT_FILE_ALLOW_LIST.append(relname)
    # Output
    return {
        "success": True,
        "path": freal,
        "file": relname,
        "added": added,
    }


# List the allow-list patterns and matching files
def list_editable_files() -> dict:
    r"""List the edit allow-list and the existing files that match it

    The allow-list combines the *EditAllowList* glob patterns from the
    ``cape-agent.json`` file, the files returned by
    :func:`cape.cfdx.cntl.Cntl.get_edit_allowlist` for the current CAPE
    control instance (if any), and files registered by other skills via
    :data:`cape.agent.agentcntl.EDIT_FILE_ALLOW_LIST`.

    :Call:
        >>> result = list_editable_files()
    :Outputs:
        *result*: :class:`dict`
            Keys include *success*, *patterns*, *cntl_files*,
            *skill_files*, and *files* (existing files matching the
            patterns)
    """
    # Absolute path of repo root
    rootdir = _get_rootdir()
    # Get all three parts of the allow-list
    patterns = _genr8_allowlist()
    cntl_files = _get_cntl_allowlist()
    skill_files = _get_skill_allowlist()
    # Find existing files matching each pattern
    flist = set()
    for pat in patterns:
        # Glob, relative to root; ``**`` parts match any depth
        for fmatch in glob.glob(os.path.join(rootdir, pat), recursive=True):
            if os.path.isfile(fmatch):
                frel = os.path.relpath(fmatch, rootdir).replace(os.sep, "/")
                flist.add(frel)
    # Output
    return {
        "success": True,
        "rootdir": rootdir,
        "patterns": patterns,
        "cntl_files": cntl_files,
        "skill_files": skill_files,
        "files": sorted(flist),
    }


# Apply an exact-match search-and-replace edit to an allow-listed file
def edit_file(fname: str, old: str, new: str) -> dict:
    r"""Replace one occurrence of *old* with *new* in an allow-listed file

    The text *old* must occur exactly once in the file; otherwise no
    edit is made. On success the result includes a unified diff of the
    change, truncated to *MAX_DIFF_CHARS* characters.

    :Call:
        >>> result = edit_file(fname, old, new)
    :Inputs:
        *fname*: :class:`str`
            Name of file to edit; must match the edit allow-list
        *old*: :class:`str`
            Exact text to search for; must occur exactly once
        *new*: :class:`str`
            Replacement text; may be ``""`` to delete the match
    :Outputs:
        *result*: :class:`dict`
            Keys include *success*, *file*, and *diff*
    """
    # Check types of *old* and *new*
    if not isinstance(old, str) or not old:
        return {
            "success": False,
            "error": "'old' must be a nonempty string",
        }
    if not isinstance(new, str):
        return {
            "success": False,
            "error": "'new' must be a string",
        }
    # Check whether the file may be edited
    check = _check_editable(fname)
    if isinstance(check, dict):
        return check
    freal, relname = check
    # Check for file
    if not os.path.isfile(freal):
        return {
            "success": False,
            "error": f"No such file: '{relname}'",
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
    # Count occurrences of *old*
    nfound = text.count(old)
    if nfound == 0:
        return {
            "success": False,
            "error": (
                f"Search text not found in '{relname}'; read the file "
                f"again and check for exact indentation"),
        }
    elif nfound > 1:
        return {
            "success": False,
            "error": (
                f"Search text occurs {nfound} times in '{relname}'; "
                f"include more surrounding context to make it unique"),
        }
    # Apply the single replacement
    newtext = text.replace(old, new, 1)
    # Write the modified file
    try:
        with open(freal, "w") as fp:
            fp.write(newtext)
    except OSError as e:
        return {
            "success": False,
            "error": f"Could not write '{relname}': {e}",
        }
    # Create a unified diff of the change
    diff = "".join(difflib.unified_diff(
        text.splitlines(keepends=True),
        newtext.splitlines(keepends=True),
        fromfile=f"a/{relname}",
        tofile=f"b/{relname}",
    ))
    # Truncate very long diffs
    truncated = len(diff) > MAX_DIFF_CHARS
    if truncated:
        diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated]"
    # Output
    return {
        "success": True,
        "file": relname,
        "truncated": truncated,
        "diff": diff,
    }


# Full Markdown instructions provided to the agent via ``use_skill``
SKILL_CONTENT = r"""
# file-editor: editing allow-listed files

Use this skill when the user asks you to modify a file in this repo,
for example updating a CAPE JSON option, fixing a script in the
`tools/` folder, or editing notes. You may only *edit* files that match
the skill's *allow-list*; all other edit attempts are rejected.

Reading is not restricted by the allow-list: `read_file` accepts any
text file in the repo smaller than about 2 MB.

## Edit allow-list

The allow-list combines three sources:

1. The `EditAllowList` option in `cape-agent.json`: glob patterns
   relative to the repo root. Matching uses `fnmatch`, so `*` matches
   folder separators too: `"tools/*.py"` matches `tools/a/b/c.py` as
   well as `tools/a.py`.
2. The CAPE control JSON file for this repo, provided automatically by
   the run matrix `Cntl` instance.
3. Files registered for this session by other skills, e.g. `fix-json`
   registers the file the user asked to repair. These can be files
   that cannot be read into a `Cntl` instance, such as a JSON file
   with a syntax error.

Call `list_editable_files` to see the active patterns and the existing
files they match. Symlinks are resolved, and files outside the repo
root folder are always rejected, even if they match a pattern.

## Workflow

1. Call `list_editable_files` to confirm the file you need is editable.
   If it is not, tell the user which pattern to add to `EditAllowList`
   rather than attempting the edit.
2. Call `read_file` on the file. Always read a file before editing it,
   even if the user quoted its contents; you need exact indentation
   and context. Large files are truncated to their first and last
   lines, with a marker showing how many lines were omitted.
3. Call `edit_file` with:
   * `fname`: the file name, absolute or relative to the current folder
   * `old`: the exact text to replace, indented exactly as `read_file`
     shows (without the `"<line>: "` prefix). It must occur exactly
     once; if the call reports multiple occurrences, retry with more
     surrounding context.
   * `new`: the replacement text, or `""` to delete the matched text.

On success, the result includes a unified diff of the change. Read the
file again afterward if you need to confirm the result or make further
edits.

## Editing the CAPE JSON file

The repo's CAPE control JSON file is always editable. JSON is strict:
keep keys quoted, keep commas balanced, and preserve any `//` comment
style already present. Make the smallest change that satisfies the
request, and do not reformat the rest of the file.

## Limits

* You cannot create new files or delete files; there is no whole-file
  write tool in this skill.
* `read_file` accepts any text file in the repo smaller than about
  2 MB, but nothing outside the repo and no binary files.
* Never edit a file outside the allow-list. If the user asks for
  changes to such a file, ask them to add a pattern to `EditAllowList`
  in `cape-agent.json`.
"""

# Simplified skill definition
SKILL_DICT = {
    "file-editor": {
        "description": (
            "Edit files in this repo that match an allow-list, using "
            "exact-match search and replace; also provides read_file "
            "for any repo file under 2 MB. Use when the user asks for "
            "changes to allow-listed files such as the CAPE JSON file."
        ),
        "content": SKILL_CONTENT,
        "tools": ["list_editable_files", "read_file", "edit_file"],
    },
}

# Simplified tool definitions not in OpenAPI format
TOOL_DICT = {
    "list_editable_files": {
        "description": (
            "List the file-editor allow-list patterns and the "
            "existing files that match them. Call "
            "use_skill('file-editor') for full instructions first."
        ),
        "parameters": [],
        "required": [],
    },
    "read_file": {
        "description": (
            "Read a repo file with line numbers; any text file under "
            "2 MB is readable, and large files are truncated to the "
            "first and last lines. Call use_skill('file-editor') for "
            "full instructions first."
        ),
        "parameters": ["fname"],
        "required": ["fname"],
    },
    "edit_file": {
        "description": (
            "Replace one unique occurrence of an exact search string "
            "in an allow-listed file and return a unified diff. Call "
            "use_skill('file-editor') for full instructions first."
        ),
        "parameters": ["fname", "old", "new"],
        "required": ["fname", "old", "new"],
    },
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Register tools
toolutils.register_module_tools(SKILL_PARAMS)

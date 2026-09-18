r"""
:mod:`cape.agent.skills.fixjson`: Diagnose and repair invalid JSON files
=========================================================================

This module defines the built-in agent skill ``"fix-json"``, which
teaches the CAPE agent how to repair a JSON file that cannot be parsed,
typically a CAPE control JSON file with a syntax error.

Such files pose a special problem for the ``file-editor`` skill: a
JSON file with a syntax error cannot be read into a
:class:`cape.cfdx.cntl.Cntl` instance, so it is never part of the
``Cntl``-provided edit allow-list, and the agent would normally be
denied edit access to the very file the user asked it to fix. The
:func:`validate_json` tool therefore registers its target file with
:data:`cape.agent.agentcntl.EDIT_FILE_ALLOW_LIST`, which the
:mod:`cape.agent.skills.fileedit` module merges into its allow-list,
before reporting the first syntax error with numbered context lines.
"""

# Standard library
import json
import os

# Local imports
from . import fileedit
from ..tools import toolutils
from ...optdict import strip_comment


# Number of context lines to show before/after the error line
CONTEXT_BEFORE = 3
CONTEXT_AFTER = 2

# Short hints for common JSON parser messages
ERROR_HINTS = {
    "Expecting ',' delimiter": (
        "probably a missing comma, or a missing closing '}' or ']', "
        "at or before the marked line"),
    "Expecting property name enclosed in double quotes": (
        "probably a trailing comma or a key missing double quotes at "
        "the marked location"),
    "Expecting value": (
        "probably a bad or missing value, or a trailing comma, at the "
        "marked location"),
    "Unterminated string starting at": (
        "unbalanced double quote on the marked line"),
    "Extra data": (
        "content found after the end of the first complete JSON "
        "value"),
    "Invalid control character at": (
        "unescaped control character (e.g. a literal newline) inside "
        "a string; use an escape sequence"),
}


# Parameter definitions for the tool schema
SKILL_PARAMS = {
    "fname": {
        "description": (
            "Name of the JSON file to check, absolute or relative to "
            "the current folder. The file is also registered with the "
            "file-editor skill's allow-list so read_file and edit_file "
            "will accept it."
        ),
        "type": "string",
    },
}


# Format numbered context lines around an error line
def _format_context(lines: list, jerr: int) -> str:
    # One line above and below the window of interest
    j0 = max(jerr - CONTEXT_BEFORE, 0)
    j1 = min(jerr + CONTEXT_AFTER + 1, len(lines))
    # Assemble numbered lines, marking the error line with '-->'
    output = []
    for j in range(j0, j1):
        # Marker for the error line
        marker = "-->" if j == jerr else "   "
        # Numbered line, matching the format of fileedit.read_file
        output.append(f"{marker} {j + 1}: {lines[j]}")
    # Output
    return "\n".join(output)


# Prepare a short hint based on the parser's error message
def _get_hint(msg: str, eof: bool) -> str:
    # Check for end-of-file error (unclosed '{' or '[')
    if eof:
        return (
            "the parser reached the end of the file while expecting "
            "more input; the file is probably missing a closing '}' "
            "or ']' earlier in the file")
    # Look up a hint for the parser's message
    return ERROR_HINTS.get(
        msg, "fix the JSON syntax error at the marked location")


# Validate a JSON file and register it with the session allow-list
def validate_json(fname: str) -> dict:
    r"""Check a JSON file for syntax errors, reporting the first one

    The file must be inside the repo root folder and end with
    ``.json``. As a side effect, the file is registered with the
    ``file-editor`` skill's allow-list (via
    :data:`cape.agent.agentcntl.EDIT_FILE_ALLOW_LIST`) so that it can
    be repaired using that skill's ``read_file`` and ``edit_file``
    tools even though it cannot be read into a ``Cntl`` instance.

    CAPE-style ``//`` comments are stripped from each line (outside of
    quoted strings) before parsing, mirroring the way CAPE reads JSON
    files. Reported line numbers refer to the original file.

    :Call:
        >>> result = validate_json(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of JSON file to check
    :Outputs:
        *result*: :class:`dict`
            Keys include *success*, *file*, *registered* (whether the
            file was newly added to the allow-list), and *valid*; for
            invalid files also *error*, *lineno*, *colno*, *eof*,
            *hint*, and *context* (numbered lines around the error)
    """
    # Limit registration to JSON files (the purpose of this skill)
    if not isinstance(fname, str) or not fname.lower().endswith(".json"):
        return {
            "success": False,
            "error": f"File '{fname}' is not a '.json' file",
        }
    # Resolve and register file with the file-editor allow-list
    reg = fileedit.register_editable_file(fname)
    if not reg["success"]:
        return reg
    # Get resolved path and name relative to the root folder
    freal = reg.get("path")
    relname = reg["file"]
    registered = reg["added"]
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
    # Split into lines of the original file
    lines = text.splitlines()
    n_lines = len(lines)
    # Strip CAPE-style '//' comments from each line before parsing
    code = "\n".join(strip_comment(line) for line in lines)
    # Try to parse the comment-stripped contents
    try:
        json.loads(code)
    except json.JSONDecodeError as e:
        # Index (0-based) of reported error line, clamped to the file
        jerr = min(e.lineno - 1, n_lines - 1)
        # Check for error at end of file (usually an unclosed bracket);
        # the parser consumed all input before reporting the error
        eof = not e.doc[e.pos:].strip()
        # Output
        return {
            "success": True,
            "valid": False,
            "file": relname,
            "registered": registered,
            "n_lines": n_lines,
            "error": e.msg,
            "lineno": e.lineno,
            "colno": e.colno,
            "eof": eof,
            "hint": _get_hint(e.msg, eof),
            "context": _format_context(lines, jerr),
            "message": (
                f"'{relname}' has a JSON syntax error on line "
                f"{e.lineno}: {e.msg}"),
        }
    # File is valid
    return {
        "success": True,
        "valid": True,
        "file": relname,
        "registered": registered,
        "n_lines": n_lines,
        "message": f"'{relname}' is valid JSON",
    }


# Full Markdown instructions provided to the agent via ``use_skill``
SKILL_CONTENT = r"""
# fix-json: repairing invalid JSON files

Use this skill when the user asks you to fix a JSON file that cannot
be parsed, typically a CAPE control file such as `pyCart.json` with a
syntax error. Such files cannot be read into a run matrix `Cntl`
instance, so the file-editor skill's allow-list cannot include them;
this skill registers the file for editing as part of validating it.

## Workflow

1. Call `validate_json(fname)` with the file the user named. This does
   two things:
   * registers the file with the file-editor skill's allow-list for
     this session, so `read_file` and `edit_file` will accept it;
   * parses the file and reports `valid: true`, or the first syntax
     error with its line, column, a short `hint`, and numbered
     `context` lines with the error line marked by `-->`.
2. If you need to edit the file, first call
   `use_skill('file-editor')` to activate the `read_file` and
   `edit_file` tools, then `read_file` the file to see the problem
   area in context.
3. Repair the error with `edit_file`, making the smallest possible
   change. Fix syntax only: do not reformat, reorder, or rename
   anything, do not change any values, and preserve any `//` comments.
4. Call `validate_json` again. The JSON parser reports at most one
   error per pass, so repeat steps 2-4 until it returns
   `"valid": true`. Note that an unclosed `{` or `[` is reported at
   the END of the file; work upward from the last line to find where
   the closing bracket is missing.
5. When the file validates, summarize the edits you made for the user.

## Common errors and where to look

* `Expecting ',' delimiter`: missing comma or unclosed `}`/`]`, at or
  before the marked line.
* `Expecting property name enclosed in double quotes`: usually a
  trailing comma after the last item of an object, or an unquoted key.
* `Expecting value`: a bad or missing value, or a trailing comma in a
  list.
* `Unterminated string`: an unbalanced double quote on the marked line.
* Error at end of file (`eof` is true): a closing `}` or `]` is
  missing somewhere earlier in the file.

JSON rules: keys and strings use double quotes only, no trailing
commas, and every `{`/`[` needs a closing `}`/`]`. CAPE tolerates
`//` comments in its JSON files, so leave those alone.

If `edit_file` or `read_file` rejects the file with "not in the edit
allow-list", call `validate_json` on that file again to re-register
it.
"""

# Simplified skill definition
SKILL_DICT = {
    "fix-json": {
        "description": (
            "Diagnose and repair a JSON file that cannot be parsed "
            "(for example a CAPE control JSON file with a syntax "
            "error), using minimal syntax-only edits."
        ),
        "content": SKILL_CONTENT,
        "tools": ["validate_json"],
    },
}

# Simplified tool definitions not in OpenAPI format
TOOL_DICT = {
    "validate_json": {
        "description": (
            "Check a JSON file for syntax errors; reports the first "
            "error with line, column, hint, and numbered context "
            "lines, or confirms the file is valid. Also registers the "
            "file with the file-editor skill's allow-list so it can "
            "be repaired. Call use_skill('fix-json') for full "
            "instructions first."
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

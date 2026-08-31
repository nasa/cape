r"""
:mod:`cape.agent.tools.toolutils`: Common functions for CAPE tool usage
=======================================================================

This module provides a function :func:`wrap_cli` that places several
preprocessing, protection, and postprocessing steps around a tool call
eminated by an LLM.

This includes normalizing the arguments, e.g. replacing the string
``"true"`` with the Python value ``True``, removing any keyword
arguments set to ``None``, capturing STDOUT, and more.
"""

# Standard library
import contextlib
import io
import sys
from typing import Callable


def normalize_kwargs(kw: dict) -> dict:
    r"""Filter keyword args to tool, removing ``''`` and ``None``

    :Call:
        >>> kw_tool = normalize_kwargs(kw)
    :Inputs:
        *kw*: :class:`dict`
            Keyword args created by LLM tool caller
    :Outputs:
        *kw_tool*: :class:`dict`
            Modified *kw* as such:

            * String conversion: ``"true"`` -> ``True``
            * Remove keys set to ``None`` after normalization
    """
    # Filter out empty strings from kwargs
    kw_tool = {
        k: normalize_tool_arg(v) for k, v in kw.items()
        if v not in (None, "", "None", "null")
    }
    # Output
    return kw_tool


def normalize_tool_arg(v):
    r"""Normalize strings of special values to Pythonic values

    E.g. ``"null"`` -> ``None``
    """
    # Check for special cases
    if v in ("null", "None"):
        return None
    elif v in ("True", "true"):
        return True
    elif v in ("False", "false"):
        return False
    else:
        return v


def wrap_cli(func: Callable, *a, **kw) -> dict:
    r"""Run a CAPE command-line tool through :mod:`cape.cfdx.cli`

    This performs several preprocessing, error handling, and
    postprocessing steps. The output of this function includes the
    results of the underlying function and appends the returncode. It
    also captures STDOUT and appends that to the result

    :Call:
        >>> result = wrap_cli(func, *a, **kw)
    :Inputs:
        *func*: **callable**
            A function to be wrapped
        *a*: :class:`tuple`
            Args passed into *func* after preprocessing
        *kw*: :class:`dict`
            Keyword args passed into *func* after preprocessing
    :Outputs:
        *result*: :class:`dict`
            Results of code; includes keys *result*, *success*,
            *returncode*, and *stdout*
    """
    # Special flag to allow arbitarily long STDOUT
    long_ok = kw.pop("__long_stdout", False)
    # Normalize the args
    a_tool = (normalize_tool_arg(aj) for aj in a)
    # Normalize the kwargs
    kw_tool = normalize_kwargs(kw)
    # Redirect STDOUT while able
    buf = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buf)):
        # Run the tool
        try:
            # Call the nominal tool
            ierr, result = func(*a_tool, **kw_tool)
            # Output
            if isinstance(result, dict):
                # Add information
                result["success"] = True
                result["returncode"] = ierr
            else:
                # Combine information
                result = {
                    "result": result,
                    "success": True,
                    "returncode": ierr,
                }
        except Exception as e:
            result = {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
            }
    # Save the captured STDOUT
    stdout = buf.getvalue()
    # Tuncate it unless cape_c
    if long_ok:
        result["stdout"] = stdout
    else:
        result["stdout"] = _truncate_stdout(stdout)
    # Output
    return result


def _truncate_stdout(stdout: str) -> str:
    # Filter STDOUT if too long
    if len(stdout) > 2000:
        # Filter out lines that begin with a space
        lines = stdout.splitlines(keepends=True)
        lines = [line for line in lines if not line.startswith(' ')]
        stdout = ''.join(lines)
    # If still too long, trim beginning and end
    if len(stdout) > 1000:
        lines = stdout.splitlines(keepends=True)
        # Accumulate from beginning up to 500 chars
        head_lines = []
        head_len = 0
        for line in lines:
            line = line[:128]
            head_lines.append(line)
            head_len += len(line)
            if head_len >= 500:
                break
        # Accumulate from end up to 500 chars
        tail_lines = []
        tail_len = 0
        for line in reversed(lines):
            line = line[:128]
            tail_lines.append(line)
            tail_len += len(line)
            if tail_len >= 500:
                break
        tail_lines.reverse()
        # Combine, avoiding overlap
        if head_len + tail_len >= len(stdout):
            # Overlap: just use the whole thing
            stdout = ''.join(lines)
        else:
            stdout = ''.join(head_lines) + ''.join(tail_lines)
    return stdout


# Class to capture STDOUT and report it
class Tee:
    r"""Class to simultaneously capture and display STDOUT

    :Call:
        >>> tee = Tee(*streams)
    :Inputs:
        *streams*: :class:`tuple`\ [:class:`io.IOBase`]
            List of streams to capture/display
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


# Register tools from a submodule
def register_module_tools(params: dict):
    r"""Register ``TOOL_SCHEMAS`` and ``TOOLS`` to calling module

    :Call:
        >>> register_module_tools(params)
    :Inputs:
        *params*: :class:`dict`
            Dictionary of parameter definitiosn for use in this tool set
    """
    # Get parameters from calling module
    module_name = sys._getframe(1).f_globals["__name__"]
    mod = sys.modules[module_name]
    # Loop through tools defined in that module
    for name in mod.TOOL_DICT:
        _add_schema(mod, params, name)


def _add_schema(mod, params: dict, name: str):
    # Get options for that tool
    opts = mod.TOOL_DICT[name]
    # Get function name
    funcname = opts.get("function", name)
    # Get the function from this module
    func = getattr(mod, funcname, None)
    # Define the function
    if not callable(func):
        func = mod.genr8_func(funcname)
    # Initialize parameter properties
    tool_arg_props = {
        opt: params[opt] for opt in opts.get("parameters", [])
    }
    tool_args = {
        "type": "object",
        "properties": tool_arg_props,
        "required": opts.get("required", []),
    }
    # Initialize function schema
    tool_func = {
        "name": name,
        "description": opts["description"],
        "parameters": tool_args,
    }
    # Initialize tool schema
    tool_schema = {
        "type": "function",
        "function": tool_func,
    }
    # Append to overall schema
    mod.TOOL_SCHEMAS.append(tool_schema)
    # Append to list
    mod.TOOLS[name] = func

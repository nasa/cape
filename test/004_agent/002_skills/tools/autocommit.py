#!/usr/bin/env python


# Standard library
import importlib
import os
import re
import sys
from collections import defaultdict
from subprocess import call, check_output

# Third-party
from cape import argread
from cape import textutils
from cape.cfdx import manage
from cape.errors import CapeValueError
import numpy as np


# This folder and parent
TOOLS_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(TOOLS_DIR)

# Regex for "%05.1f_" -> "%05.1f"
REGEX_PRINTF = re.compile(r"(%-?[0-9]*(\.[1-9]?)?[fis])")

# Help message
CLI_HELP = r"""Auto-generate commit message for run matrix status

:Usage:

    .. code-block:: console

        ./tools/autocommit.py [-f FJSON] [OPTIONS]

:Options:

    -h, --help
        Display this help message and exit

    -f FJSON
        Use JSON file *FJSON* <pyFun.json>

    -m, --msg MSG
        Use *MSG* in first line of commit message instead of default

    --test
        Show commit message but do not commit
"""

# List of default JSON file name
DEFAULT_JSON_FILES = (
    "pyFun.json",
    "pyCart.json",
    "pyOver.json",
    "pyKes.json",
)


# Main command
def main():
    # Parse inputs
    a, kw = argread.readkeys(sys.argv)
    # Check for help message
    if kw.get("h") or kw.get("help"):
        print(textutils.markdown(CLI_HELP))
        return
    # Get options
    fname = kw.pop("f", None)
    solver = kw.pop("solver", None)
    # Get module name if necessary
    if fname is None:
        # Find valid JSON files
        json_files = manage.find_json_solver()
        # Check for match
        if len(json_files) == 0:
            raise CapeValueError("Found no CAPE JSON files in repo")
        # Filter by solver if needed
        if solver is None:
            # No filter
            solver, fname = json_files[0]
        else:
            # Loop through matches
            for sj, fj in json_files:
                # Check solver
                if sj == solver:
                    # Found match
                    fname = fj
                    break
            else:
                raise CapeValueError("Found no CAPE JSON files in repo")
        # Report which file we're using!
        print(f"Using {solver} JSON file: {fname}")
    elif solver is None:
        # Determine solver
        solver = manage.identify_solver(fname)
    # Name of module
    modname = f"cape.{solver}.cntl"
    # Import it
    cntlmod = importlib.import_module(modname)
    # Instantiate
    cntl = cntlmod.Cntl(fname)
    # Get run matrix file
    fmat = cntl.opts["RunMatrix"]["File"]
    # Check for unchanged file
    sts = check_output(["git", "status", "-s", fmat])
    if sts == b"":
        print("No changes to '%s'" % fmat)
        return
    # Get commit message
    msg = genr8_commit_msg(cntl, fmat, **kw)
    # Show message
    if kw.get("test"):
        print(msg)
        return msg
    # Do commit
    call(["git", "add", fmat])
    call(["git", "commit", "-m", msg])
    return msg


# Find a default JSON file if available
def find_json(**kw):
    # File name to read
    fjson = kw.get("f")
    # Find default if not present
    if fjson is None:
        # Loop through list of possibilities
        for candidate in DEFAULT_JSON_FILES:
            # Check if file exists
            if os.path.isfile(candidate):
                # Use that
                fjson = candidate
                break
        else:
            # None found
            print(textutils.markdown(__doc__))
            return 1
    # Check if it's a link
    if os.path.islink(fjson):
        # Get the target of the link
        fjson = os.path.realpath(fjson)
    # Output
    return fjson


# Get commit message from known matrix file
def genr8_commit_msg(cntl, fname, **kw):
    r"""Generate a commit message based from rum natrix file status

    :Call:
        >>> txt = genr8_commit_msg(cntl, fname):
    :Inputs:
        *fname*: :class:`str`
            Name of run matrix file
        *cols*: :class:`list`\ [:class:`str`]
            List of column names for run matrix
        *m*, *msg*: {``None``} | :class:`str`
            Optional text for first line of commit message
    :Outputs:
        *txt*: :class:`str`
            Text for commit message
    :Versions:
        * 2022-11-04 ``@ddalle``: v1.0
        * 2022-11-07 ``@ddalle``: v1.1; completion
        * 2023-03-28 ``@ddalle``: v2.0; generic *xcol*
    """
    # "Scheduling" key
    xcol = cntl.x.cols[0]
    # Get format used in run matix definition
    fmtx = cntl.x.defns[xcol].get("Format", "%.2f")
    # Extract only the string portion
    fmt = REGEX_PRINTF.search(fmtx).group(0)
    # Diff command
    cmd = ["git", "diff", "--", fname]
    # Get diff output
    txt = check_output(cmd).decode("utf-8")
    # Split to lines
    lines = txt.split("\n")[5:-1]
    # Differentiate into old/new
    oldlines = []
    newlines = []
    # Loop through lines
    for line in lines:
        # Check first char
        if line.startswith("-"):
            oldlines.append(line[1:])
        elif line.startswith("+"):
            newlines.append(line[1:])
    # Check for problems
    if len(oldlines) != len(newlines):
        sys.tracebacklimit = 0
        raise ValueError(
            ("Cannot process! Number of lines changed\n") +
            ("Found %i deletions and %i additions"
                % (len(oldlines), len(newlines))))
    # Initiate counters
    n = {
        "PASS": 0,
        "FAIL": 0,
        "UNPASS": 0,
        "UNFAIL": 0,
        "tag": 0,
        "arch": 0,
        "label": 0,
        "archmod": 0,
        "usermod": 0,
    }
    m = {
        "PASS": defaultdict(int),
        "FAIL": defaultdict(int),
        "UNPASS": defaultdict(int),
        "UNFAIL": defaultdict(int),
        "tag": defaultdict(_defaultdict),
        "arch": defaultdict(_defaultdict),
        "label": defaultdict(_defaultdict),
        "archmod": defaultdict(_defaultdict),
        "usermod": defaultdict(_defaultdict),
    }
    # Loop through changed lines
    for oldline, newline in zip(oldlines, newlines):
        # Get properties
        xold = _get_linedict(oldline, cntl)
        xnew = _get_linedict(newline, cntl)
        # Statuses
        oldmark = xold["MARK"]
        newmark = xnew["MARK"]
        # Architectures
        oldarch = xold.get("arch", '')
        newarch = xnew.get("arch", '')
        # Tags
        oldtag = xold.get("tag", '')
        newtag = xnew.get("tag", '')
        # User b4 and after
        olduser = xold.get("user", '')
        newuser = xnew.get("user", '')
        # Labels
        oldlbl = xold.get("Label", '')
        newlbl = xnew.get("Label", '')
        # Get Mach number or value of scheduling key
        xj = xnew[xcol]
        # Check for a MARK changes
        if oldmark in ("p", "P"):
            # Line *was* marked pass
            if newmark in ("", " "):
                # PASS flag removed
                n["UNPASS"] += 1
                m["UNPASS"][xj] += 1
            elif newmark in ("e", "E"):
                # PASS flag removed, FAIL instead
                n["UNPASS"] += 1
                n["FAIL"] += 1
                m["UNPASS"][xj] += 1
                m["FAIL"][xj] += 1
        elif oldmark in ("e", "E"):
            # Line *was* marked fail
            if newmark in ("", " "):
                # FAIL flag removed
                n["UNFAIL"] += 1
                m["UNFAIL"][xj] += 1
            elif newmark in ("p", "P"):
                # FAIL -> PASS
                n["UNFAIL"] += 1
                n["PASS"] += 1
                m["UNFAIL"][xj] += 1
                m["PASS"][xj] += 1
        else:
            # Line was not marked before
            if newmark in ("p", "P"):
                # New pass
                n["PASS"] += 1
                m["PASS"][xj] += 1
            elif newmark in ("e", "E"):
                # New fail
                n["FAIL"] += 1
                m["FAIL"][xj] += 1
        # Check for tags
        if newtag != oldtag:
            # Save it
            n["tag"] += 1
            m["tag"][newtag][xj] += 1
        # Check for new *arch*
        if newarch:
            # Check for old marker
            if oldarch == newarch:
                # Nothing to note
                pass
            elif oldarch == "":
                # New arch
                n["arch"] += 1
                m["arch"][newarch][xj] += 1
            else:
                # Arch modification
                arch = "%s->%s" % (oldarch, newarch)
                n["archmod"] += 1
                m["archmod"][arch][xj] += 1
        # Check for *user* change
        if newuser != olduser:
            # Counter for total
            n["usermod"] += 1
            # Overall tag
            user = f"{olduser} -> {newuser}"
            # Specific counter
            m["usermod"][user][xj] += 1
        # Check for *Label* change
        if newlbl != oldlbl:
            # Counter for total
            n["label"] += 1
            # Identifier
            lbl = f"X => {newlbl}"
            # Specific counter
            m["label"][lbl][xj] += 1
    # Start default headline and text
    headline = "Auto-commit %s:" % os.path.basename(fname)
    lines = []
    # Check for new passes
    if n["PASS"]:
        # Total summary
        headline += " PASS %i," % n["PASS"]
        lines.append("PASS +%i" % n["PASS"])
        # Display status by slice col
        _disp_by_xcol(xcol, fmt, lines, m["PASS"])
    # Check for unmarked PASSes
    if n["UNPASS"]:
        lines.append("PASS -%i" % n["UNPASS"])
        # By *xcol* value
        _disp_by_xcol(xcol, fmt, lines, m["UNPASS"])
    # Check for new failures
    if n["FAIL"]:
        headline += " FAIL %i," % n["FAIL"]
        lines.append("FAIL +%i" % n["FAIL"])
        # By *xcol* value
        _disp_by_xcol(xcol, fmt, lines, m["FAIL"])
    # Check for failures removed
    if n["UNFAIL"]:
        lines.append("FAIL -%i" % n["UNFAIL"])
        # By *xcol* value
        _disp_by_xcol(xcol, fmt, lines, m["UNFAIL"])
    # Check for labels bubmps
    if n["label"]:
        # Headers
        headline += " label +%i," % n["label"]
        lines.append("Changes to 'Label' settings: %i" % n["label"])
        # By *xcol* value
        _disp_by_v_xcol(xcol, fmt, lines, m["label"])
    # Check for new arch settings
    na = n["arch"]
    nam = n["archmod"]
    if na + nam:
        headline += " arch"
        if na and nam:
            headline += " +%i =%i," % (na, nam)
        elif na:
            headline += " +%i," % na
        else:
            headline += " =%i," % nam
    # New archs
    if na:
        lines.append("New 'arch' settings: %i" % na)
        _disp_by_v_xcol(xcol, fmt, lines, m["arch"])
    if nam:
        lines.append("Modified 'arch' settings: %i" % nam)
        _disp_by_v_xcol(xcol, fmt, lines, m["archmod"])
    # New tags
    ntag = n["tag"]
    if ntag:
        headline += " tag %i" % ntag
        lines.append("New/modified 'tag' settings: %i" % ntag)
        _disp_by_v_xcol(xcol, fmt, lines, m["tag"])
    # Check for user modifications
    num = n["usermod"]
    if num:
        headline += " user =%i," % num
        lines.append("Modified 'user' settings: %i" % num)
        _disp_by_v_xcol(xcol, fmt, lines, m["usermod"])
    # Mach numbers completed
    xcol_complete = []
    xcol_nopass = []
    xcol_partial = {}
    # Loop through status of each Mach
    for xj in np.unique(cntl.x[xcol]):
        # Get mask
        mask = np.where(cntl.x[xcol] == xj)[0]
        # Count
        nj = mask.size
        # Passes and FAILS
        npass = np.count_nonzero(cntl.x.PASS[mask])
        nerr = np.count_nonzero(cntl.x.ERROR[mask])
        # Total
        ntotalj = npass + nerr
        # Check status
        if ntotalj == nj:
            # Fully passed
            xcol_complete.append(fmt % xj)
        elif ntotalj > 0:
            # Save current status
            xcol_partial[xj] = (npass, nerr, nj)
        else:
            # No cases pased
            xcol_nopass.append(fmt % xj)
    # Completion counters
    ncomplete = len(xcol_complete)
    nnopass = len(xcol_nopass)
    # Overall status
    if ncomplete:
        lines.append("")
        lines.append(f"Completed {xcol} values:")
        # Number of rows
        nrow = (ncomplete + 4) // 5
        # Split into rows
        for irow in range(nrow):
            lines.append("  " + " ".join(xcol_complete[irow*5:irow*5 + 5]))
    # Partial status
    if len(xcol_partial):
        lines.append("")
        lines.append(f"Partially complete {xcol} values:")
    # Summary format line
    fmtline1 = f"  * {xcol}={fmt}: %i/%i"
    fmtline2 = f"  * {xcol}={fmt}: %i/%i (%iP,%iE)"
    # Loop through partially complete Mach numbers
    for xj, (npass, nerr, nj) in xcol_partial.items():
        # Check for errors
        if nerr:
            # Show pass/fail
            lines.append(fmtline2 % (xj, npass + nerr, nj, npass, nerr))
        else:
            # Just passes
            lines.append(fmtline1 % (xj, npass, nj))
    # Not-started status
    if nnopass:
        lines.append("")
        lines.append(f"{xcol} values w/ no PASS cases:")
        # Number of rows
        nrow = (nnopass + 4) // 5
        # Split into rows
        for irow in range(nrow):
            lines.append("  " + " ".join(xcol_nopass[irow*5:irow*5 + 5]))
    # Remove trailing comma added to header line
    headline = headline.rstrip(",")
    # Check for user-provided headline
    headline = kw.get("msg", kw.get('m', headline))
    # Initialize message
    msg = headline + "\n\n" + "\n".join(lines)
    return msg


def _disp_by_xcol(xcol, fmt, lines, m):
    # Create format
    fmtline = f"  * {xcol}={fmt}: %i"
    # Loop through slices
    for xj, nj in m.items():
        lines.append(fmtline % (xj, nj))


def _disp_by_v_xcol(xcol, fmt, lines, m):
    # Create main line format
    fmtline = f"    - {xcol}={fmt}: %i"
    # Loop through labels
    for lblj, mj in m.items():
        lines.append("  * %s" % lblj)
        for xj, nj in mj.items():
            lines.append(fmtline % (xj, nj))


def _get_linedict(line, cntl):
    # First col is status
    x = {"MARK": line[0]}
    # Get list of columns
    cols = cntl.x.cols
    # Loop through other cols
    for col, v in zip(cols, line[1:].split(",")):
        # Get type
        coltype = cntl.x.defns[col].get("Value", "str")
        # Remove white space
        v = v.strip()
        # Convert if appropriate
        if coltype == "float":
            x[col] = float(v)
        elif coltype == "int":
            x[col] = int(v)
        else:
            x[col] = v.strip()
    # Output
    return x


def _defaultdict():
    return defaultdict(int)


# Check if executed as script
if __name__ == "__main__":
    # Run main command
    main()


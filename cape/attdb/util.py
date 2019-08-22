#!/usr/bin/env python
"""
-----------------------------------------------------
Aero Task Team Database Utilities: :mod:`attdb.util`
-----------------------------------------------------

The utilities module stores the values for several SLS parameters along with
the paths to several folders.  The table below shows the values of numerical
SLS parameters.

    ============  =======================  =================================
    Variable      Symbol                   Value
    ============  =======================  =================================
    *Lref*        :math:`L_\mathit{ref}`   _Lref_ inches
    *Aref*        :math:`A_\mathit{ref}`   _Aref_ inches^2
    *xMRP*        :math:`x_\mathit{MRP}`   _xMRP_ inches
    *ySRB*        :math:`y_\mathit{SRB}`   _ySRB_ inches
    ============  =======================  =================================
    
This module also stores the text of the SBU - ITAR warnings in *sbu_text* and
*itar_text*.  To alter the contents of the restrictions worksheet of database
files, edit the text of these strings.  In addition, the yellow warning sheet
is pointed to using the variable *fpng*.

In addition, this module contains several flight condition conversion
functions, which are shown below.
"""

# System
import os
# Text processing
import fnmatch
import re
# Numerics
import numpy as np

# Typical SBU title
sbu_ttl = "SENSITIVE BUT UNCLASSIFIED (SBU) - ITAR"
import numpy as np

# Text for SBU definition
sbu_text = (
    "WARNING: This document is SENSITIVE BUT UNCLASSIFIED (SBU). It " +
    "contains information that may be exempt from public release under the " +
    "Freedom of information Act (5 U.S.C. 552) or other applicable laws or " +
    "restricted from disclosure based on NASA policy. It is to be " +
    "controlled, stored, handled, transmitted, distributed, and disposed " +
    "of in accordance with NASA policy relating to SBU information and is " +
    "not to be released to the public or other personnel who do not have a " +
    "valid \"need-to-know\" without prior approval of an authorized NASA " +
    "official. (See NPR 1600.1.)"
)
# Text for ITAR definition
itar_text = (
    "International Traffic in Arms Regulations (ITAR) Notice - This " +
    "document contains information which falls under the purview of the " +
    "U.S. Munitions List (USML), as defined in the International Traffic " +
    "in Arms Regulations (ITAR), 22 CFR 120-130, and is export controlled. " +
    "It shall not be transferred to foreign nationals in the U.S. or " +
    "abroad, without specific approval of a knowledgeable NASA export " +
    "control official, and/or unless an export license/license exemption " +
    "is obtained/available from the United States Department of State. " +
    "Violations of these regulations are punishable by fine, imprisonment, " +
    "or both."
)
# Name of yellow SBU description image
fpng = "NF1686.png"

# Reference quantities
Lref = 333.0
Aref = 87092.0
xMRP = 4759.68
xhat = xMRP/Lref

# y-coordinates of boosters
ySRB = 250.5

# Conversion
deg = np.pi / 180.0


# List modules
def tabulate_modules(fname, txt="", prefix="", maxdepth=2, tab=4, depth=0):
    """Create a bullet list of submodules starting from folder of *fname*
    
    :Call:
        >>> txt = tabulate_modules(fname, maxdepth=2, indent=4)
    :Inputs:
        *fname*: :class:`str`
            Name of (Python module) file from which list is started
        *maxdepth*: {``2``} | ``None`` | :class:`int` >= 0
            Maximum number of sublevels
        *indent*: {``4``} | :class:`int` > 0
            Number of white spaces at the beginning of line
        *tab*: {``4``} | :class:`int` > 0
            Number of additional spaces in an indentation
    :Outputs:
        *txt*: :class:`str`
            RST-formatted outline
    :Versions:
        * 2018-07-03 ``@ddalle``: First version
    """
    # Depth one level down
    if maxdepth is None:
        # Unlimited depth
        newdepth = None
    else:
        # Decrease depth counter
        newdepth = maxdepth - 1
    # Ensure full path of *fname*
    fname = os.path.abspath(fname)
    # Separate out folder part
    fdir  = os.path.dirname(fname)
    # Perform split
    fpth, fmod = os.path.split(fname)
    # Check for broken situation
    if (fpth == fdir) and (fmod not in ["__init__.py", "__init__.pyc"]):
        # Wrong type of module or not a module at all
        return txt
        
    # Name of the current module
    fmod = os.path.split(fdir)[-1]
    # Add to the prefix
    prefix = "%s%s." % (prefix, fmod)
    # Delimiter character for this depth
    if depth%2:
        # Odd depth: -
        bullet = '-'
    else:
        # Even depth: *
        bullet = '*'
    
    # Get folder
    fglob = os.listdir(fdir)
    # Sort them (ASCII order)
    fglob.sort()
    #import pdb
    #pdb.set_trace()
    # Check for folder without "__init__.py"
    if (fpth != fdir) and ("__init__.py" not in fglob):
        return txt
    
    
    # Loop through ".py" folders
    for fn in fglob:
        # Check if it is a "*.py"
        if not fn.endswith(".py"): continue
        # Check for "__init__.py" (handled as special case)
        if fn == "__init__.py": continue
        # Otherwise, get the name of the module
        fm = fn[:-3]
        # Add a new line
        txt += " "*((depth+1)*tab)
        txt += "%s :mod:`%s%s`\n" % (bullet, prefix, fm)
        
    # Loop through subfolders
    for fn in fglob:
        # Full name
        fa = os.path.join(fdir, fn)
        # Check if it's a folder
        if not os.path.isdir(fa): continue
        # Construct name of __init__.py definition
        fi = os.path.join(fa, "__init__.py")
        # Check if the folder contains an initialization
        if not os.path.isfile(fi): continue
        # Add a new line
        txt += " "*((depth+1)*tab)
        txt += "%s :mod:`%s%s`\n" % (bullet, prefix, fn)
        # Check depth limits
        if (maxdepth is None):
            # Unlimited depth
            if depth > 8:
                # Ok, 8 levels is too many
                continue
        elif depth+1 > maxdepth:
            # Hit depth limite
            continue
        # Recurse
        txt = tabulate_modules(fi, txt=txt, prefix=prefix,
            maxdepth=maxdepth, tab=4, depth=depth+1)
        
    # Output
    return txt

# Function to take out extensive markup for help messages
def markdown(doc):
    """Remove some extraneous markup for command-line help messages
    
    :Call:
        >>> txt = markdown(doc)
    :Inputs:
        *doc*: :class:`str`
            Docstring
    :Outputs:
        *txt*: :class:`str`
            Help message with some of the RST features removed for readability
    :Versions:
        * 2017-02-17 ``@ddalle``: First version
    """
    try:
        return markdown_try(doc)
    except Exception:
        print("[MARKDOWN]: Markdown process of docstring failed")
        return doc

# Function to take out extensive markup for help messages
def markdown_try(doc):
    """Remove some extraneous markup for command-line help messages
    
    :Call:
        >>> txt = markdown_try(doc)
    :Inputs:
        *doc*: :class:`str`
            Docstring
    :Outputs:
        *txt*: :class:`str`
            Help message with some of the RST features removed for readability
    :Versions:
        * 2017-02-17 ``@ddalle``: First version
    """
    # Replace section header
    def replsec(g):
        # Replace :Options: -> OPTIONS
        return g.group(1).upper() + "\n\n"
    # Replace ReST identifiers
    def replfn(g):
        # Get the modifier name
        fn = g.group(1)
        # Get the contents
        val = g.group(2)
        # Filter the modifier
        if fn == "func":
            # Add parentheses
            return "%s()" % val
        else:
            # Just use quotes
            return "'%s'" % val
    # Remove literals around usernames
    def repluid(g):
        # Strip "``"
        return g.group(1)
    # Generic literals
    def repllit(g):
        # Strip "``"
        return g.group(1)
    # Remove **emphasis**
    def replemph(g):
        # Strip "**"
        return g.group(1)
    # Split by lines
    lines = doc.split('\n')
    # Loop through lines
    i = 0
    while  i+1 < len(lines):
        # Get line
        line = lines[i]
        # Stripped
        txt = line.strip()
        # Skip empty lines
        if txt == "":
            # Leave it
            i += 1
            continue
        # Check repeats of first character
        if txt[0] in ['=', '-', '*']:
            # Get number of repeats
            nc = get_nstart(txt, txt[0])
            # Check section marker
            if nc > 3:
                # Delete the header line
                del lines[i]
                continue
        # Check for code block
        if txt.startswith(".. "):
            # Search for leading spaces
            nw = get_nstart(line, " ")
            # Delete the code-block line
            del lines[i]
            # Check the preceding line
            if (i>0) and (lines[i-1].strip()==""):
                # Delete the preceding line if empty
                del lines[i-1]
                i -=1
            # Check lines after code block
            while True:
                # Go to next line
                i += 1
                line = lines[i]
                # Check for empty line
                if i >= len(lines):
                    # We hit the end of the docstring somehow
                    break
                elif line.strip() == "":
                    # Leave empty line alone
                    lines[i] = (nw*" ")
                    continue
                # Get number of leading spaces
                ns = get_nstart(line, " ")
                # Check indent level
                if ns > nw:
                    # Shift left by whatever the additional indent is
                    lines[i] = line[ns-nw:]
                else:
                    # Back to original indent; done with block
                    break
        # Move to next line
        i += 1
    # Reform doc string
    txt = '\n'.join(lines)
    # Replace section headers
    txt = re.sub(":([\w _-]+):\s*\n", replsec, txt)
    # Replace modifiers, such as :mod:`pyCart`
    txt = re.sub(":([\w _-]+):`([^`\n]+)`", replfn, txt)
    # Simplify user names
    txt = re.sub("``(@\w+)``", repluid, txt)
    # Simplify bolds
    txt = re.sub("\*\*([\w ]+)\*\*", replemph, txt)
    # Mark string literals
    txt = re.sub("``([^`\n]+)``", repllit, txt)
    # Output
    return txt            

# Get number of incidences of character at beginning of string (incl. 0)
def get_nstart(line, c):
    """Count number of instances of character *c* at start of a line
    
    :Call:
        >>> nc = get_nstart(line, c)
    :Inputs:
        *line*: :class:`str` | :class:`unicode`
            String
        *c*: :class:`str`
            Character
    :Outputs:
        *nc*: nonnegative :class:`int`
            Number of times *c* occurs at beginning of string (can be 0)
    :Versions:
        * 2017-02-17 ``@ddalle``: First version
    """
    # Check if line starts with character
    if line.startswith(c):
        # Get number of instances
        nc = len(re.search("\\"+c+"+", line).group())
    else:
        # No instances
        nc = 0
    # Output
    return nc
# def get_nstart

# Utility function to get elements sanely
def getel(x, i=None):
    """Return element *i* of an array if possible
    
    :Call:
        >>> xi = getel(x)
        >>> xi = getel(x, i)
    :Inputs:
        *x*: number-like or list-like
            A number or list or NumPy vector
        *i*: :class:`int`
            Index.  If not specified, treated as though *i* is ``0``
    :Outputs:
        *xi*: scalar
            Equal to ``x[i]`` if possible, ``x[-1]`` if *i* is greater than the
            length of *x*, or ``x`` if *x* is not a :class:`list` or
            :class:`numpy.ndarray` instance
    :Examples:
        >>> getel('abc', 2)
        'abc'
        >>> getel(1.4, 0)
        1.4
        >>> getel([200, 100, 300], 1)
        100
        >>> getel([200, 100, 300], 15)
        300
        >>> getel([200, 100, 300])
        200
    :Versions:
        * 2014-07-29 ``@ddalle``: First version
    """
    # Check the type.
    if i is None:
        return x
    if type(x).__name__ in ['list', 'array', 'ndarray']:
        # Check for empty input.
        if len(x) == 0:
            return None
        # Array-like
        if i:
            # Check the length.
            if i >= len(x):
                # Take the last element.
                return x[-1]
            else:
                # Take the *i*th element.
                return x[i]
        else:
            # Use the first element.
            return x[0]
    else:
        # Scalar
        return x
# def getel

# Match list
def fnmatchlist(fglob, pat):
    """Return all entries from a list that match a glob-style pattern
    
    :Call:
        >>> fmatch = fnmatchlist(fglob, pat)
    :Inputs:
        *fglob*: :class:`list` (:class:`str` | :class:`unicode`)
            List of potential names/files
        *pat*: :class:`str`
            Pattern with wild-cards to match
    :Outputs:
        *fmatch*: :class:`list` (:class:`str` | :class:`unicode`)
            List of matching names/files
    :Versions:
        * 2017-08-14 ``@ddalle``: First version
    """
    # Initialize start/end indices
    ia = []
    ib = []
    # Initialize matches
    grps = []
    # Find matches again
    for g in re.finditer("{.*?}", pat):
        # Append start/end indices
        ia.append(g.start())
        ib.append(g.end())
        # Append group contents
        grps.append(g.group())
    # Number of such patterns
    ngrp = len(grps)
    # Check for trivial case
    if ngrp == 0:
        return fnmatchlist_single(fglob, pat)
    # Expansions
    pats = [grp[1:-1].split(",") for grp in grps]
    # Number of possibilities in each pattern
    npat = [len(p) for p in pats]
    # Total number of patterns
    n = np.prod(npat)
    N = np.hstack(([1], np.cumprod(npat)))
    # Initialize output
    fmatch = []
    # Loop through the total number of possible patterns
    for i in range(n):
        # Initialize pattern
        pati = pat[:ia[0]]
        # Loop through groups
        for j in range(ngrp):
            # Get pattern index
            k = i/N[j] % npat[j]
            # Add this expansion to the local pattern
            pati += pats[j][k]
            # Add the next pattern segment
            if j+1 == ngrp:
                # Last segment
                pati += pat[ib[j]:]
            else:
                # Next segment
                pati += pat[ib[j]:ia[j+1]]
        # Apply the pattern
        fmatchi = fnmatchlist_single(fglob, pati)
        # Loop through matches to avoid duplicates
        for fi in fmatchi:
            # Check if already present
            if fi not in fmatch:
                # Append to final list
                fmatch.append(fi)
    # Output
    return fmatch  
        
# Match a list to a single pattern
def fnmatchlist_single(fglob, pat):
    """Return all entries from a list that match a glob-style pattern
    
    :Call:
        >>> fmatch = fnmatchlist(fglob, pat)
    :Inputs:
        *fglob*: :class:`list` (:class:`str` | :class:`unicode`)
            List of potential names/files
        *pat*: :class:`str`
            Pattern with wild-cards to match
    :Outputs:
        *fmatch*: :class:`list` (:class:`str` | :class:`unicode`)
            List of matching names/files
    :Versions:
        * 2017-08-14 ``@ddalle``: First version
    """
    # Initialize matches
    fmatch = []
    # Loop through candidates
    for f in fglob:
        # Check for match
        if fnmatch.fnmatch(f, pat):
            # Save it
            fmatch.append(f)
    # Sort it
    fmatch.sort()
    # Output
    return fmatch
# def fnmatchlist_single

# Replace empty string with ``None``
def destring(s):
    """Replace empty string with ``None``
    
    :Call:
        >>> v = destring(s)
    :Inputs:
        *s*: :class:`str` | :class:`unicode`
            String to be interpreted
    :Outputs:
        *v*: ``None`` | :class:`float` | :class:`int` | :class:`str`
    :Versions:
        * 2017-08-04 ``@ddalle``: First version
    """
    # Check contients
    if s.__class__.__name__ not in ["str", "unicode"]:
        # No conversion
        return s
    elif s.strip() == "":
        # Empty string -> None
        return None
    else:
        # Try to interpret it
        try:
            return eval(s)
        except Exception:
            return s
# def destring

# Function to automatically get inclusive data limits.
def get_ylim(ha, ypad=0.05, **kw):
    """Calculate appropriate *y*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.
    
    :Call:
        >>> ymin, ymax = get_ylim(ha, ypad=0.05, ym=None, yp=None)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *ypad*: {``0.05``} | :class:`float`
            Extra padding to min and max values to plot
        *ym*: :class:`float`
            Padding on minimum side
        *yp*: :class:`float`
            Padding on maximum side
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
        * 2016-06-10 ``@ddalle``: Moved to :mod:`cape.util`
    """
    # Initialize limits.
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check for empty
            if len(h.get_xdata()) == 0: continue
            # Check the min and max data
            ymin = min(ymin, min(h.get_ydata()))
            ymax = max(ymax, max(h.get_ydata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Check length
            if P.vertices.size < 4: continue
            # Get the coordinates.
            ymin = min(ymin, min(P.vertices[:-1,1]))
            ymax = max(ymax, max(P.vertices[:-1,1]))
    # Process margins
    ym = kw.get('ym', ypad)
    yp = kw.get('yp', ypad)
    # Check for identical values
    if ymax - ymin <= 0.05*(ym+yp):
        # Expand by manual amount.
        ymax += yp*ymax
        ymin -= ym*ymin
    # Add padding.
    yminv = (1+ym)*ymin - ym*ymax
    ymaxv = (1+yp)*ymax - yp*ymin
    # Output
    return yminv, ymaxv
    
# Function to automatically get inclusive data limits.
def get_xlim(ha, xpad=0.05, **kw):
    """Calculate appropriate *x*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.
    
    :Call:
        >>> xmin, xmax = get_xlim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *xpad*: :class:`float`
            Extra padding to min and max values to plot.
        *xm*: :class:`float`
            Padding on minimum side
        *xp*: :class:`float`
            Padding on maximum side
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check for empty
            if len(h.get_xdata()) == 0: continue
            # Check the min and max data
            xmin = min(xmin, min(h.get_xdata()))
            xmax = max(xmax, max(h.get_xdata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Check length
            if P.vertices.size < 4: continue
            # Get the coordinates.
            xmin = min(xmin, min(P.vertices[:-1,0]))
            xmax = max(xmax, max(P.vertices[:-1,0]))
    # Process margins
    xm = kw.get('xm', xpad)
    xp = kw.get('xp', xpad)
    # Check for identical values
    if xmax - xmin <= 0.05*(xm+xp):
        # Expand by manual amount.
        xmax += xp*xmax
        xmin -= xm*xmin
    # Add padding.
    xminv = (1+xm)*xmin - xm*xmax
    xmaxv = (1+xp)*xmax - xp*xmin
    # Output
    return xminv, xmaxv
    
# Function to automatically get inclusive data limits.
def get_ylim_ax(ha, ypad=0.05, **kw):
    """Calculate appropriate *y*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.
    
    This version is specialized for equal-aspect ratio axes.
    
    :Call:
        >>> ymin, ymax = get_ylim_ax(ha, ypad=0.05, ym=None, yp=None)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *ypad*: {``0.05``} | :class:`float`
            Extra padding to min and max values to plot
        *ym*: :class:`float`
            Padding on minimum side
        *yp*: :class:`float`
            Padding on maximum side
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
        * 2016-06-10 ``@ddalle``: Moved to :mod:`cape.util`
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check for empty
            if len(h.get_xdata()) == 0: continue
            # Check the min and max data
            xmin = min(xmin, min(h.get_xdata()))
            xmax = max(xmax, max(h.get_xdata()))
            ymin = min(ymin, min(h.get_ydata()))
            ymax = max(ymax, max(h.get_ydata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Check length
            if P.vertices.size < 4: continue
            # Get the coordinates.
            xmin = min(xmin, min(P.vertices[:-1,0]))
            xmax = max(xmax, max(P.vertices[:-1,0]))
            ymin = min(ymin, min(P.vertices[:-1,1]))
            ymax = max(ymax, max(P.vertices[:-1,1]))
    # Process margins
    ym = kw.get('ym', ypad)
    yp = kw.get('yp', ypad)
    # Check for identical values
    if ymax - ymin <= 0.05*(ym+yp):
        # Expand by manual amount.
        ymax += yp*ymax
        ymin -= ym*ymin
    # Add padding.
    yminv = ymin - ym*max(xmax-xmin, ymax-ymin)
    ymaxv = ymax + yp*max(xmax-xmin, ymax-ymin)
    # Output
    return yminv, ymaxv
    
# Function to automatically get inclusive data limits.
def get_xlim_ax(ha, xpad=0.05, **kw):
    """Calculate appropriate *x*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.
    
    This version is specialized for equal-aspect ratio axes.
    
    :Call:
        >>> xmin, xmax = get_xlim_ax(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *xpad*: :class:`float`
            Extra padding to min and max values to plot.
        *xm*: :class:`float`
            Padding on minimum side
        *xp*: :class:`float`
            Padding on maximum side
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check for empty
            if len(h.get_xdata()) == 0: continue
            # Check the min and max data
            xmin = min(xmin, min(h.get_xdata()))
            xmax = max(xmax, max(h.get_xdata()))
            ymin = min(ymin, min(h.get_ydata()))
            ymax = max(ymax, max(h.get_ydata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Check length
            if P.vertices.size < 4: continue
            # Get the coordinates.
            xmin = min(xmin, min(P.vertices[:-1,0]))
            xmax = max(xmax, max(P.vertices[:-1,0]))
            ymin = min(ymin, min(P.vertices[:-1,1]))
            ymax = max(ymax, max(P.vertices[:-1,1]))
    # Process margins
    xm = kw.get('xm', xpad)
    xp = kw.get('xp', xpad)
    # Check for identical values
    if xmax - xmin <= 0.05*(xm+xp):
        # Expand by manual amount.
        xmax += xp*xmax
        xmin -= xm*xmin
    # Add padding.
    xminv = xmin - xm*max(xmax-xmin, ymax-ymin)
    xmaxv = xmax + xp*max(xmax-xmin, ymax-ymin)
    # Output
    return xminv, xmaxv
# def get_xlim_ax

# Create a simple dict+ class
class struct(dict):
    """Class to emulate struct and reduce characters needed for dictionary
    
    Names of parameters saved to *_keys*
    
    :Call:
        >>> s = struct(ka=a, kb=b, **kw)
    :Inputs:
        *a*: :class:`any`
            First key value
        *b*: :class:`any`
            Second key value
    :Attributes:
        *s.ka*: :class:`any`
            First key, equal to *a*
        *s.kb*: :class:`any`
            Second key, equal to *b*
        *s[ka]*: :class:`any`
            First key, equal to *a*
        *s[kb]*: :class:`any`
            Second key, equal to *b*
        *s._keys*: :class:`list` (:class:`str`)
            List of keys
    :Versions:
        * 2017-01-30 ``@ddalle``: First version
    """
    # Initialization method from dictionary
    def __init__(self, **kw):
        """Initialization method
        
        :Versions:
            * 2017-01-30 ``@ddalle``: First version
        """
        # Initialize key list
        self._keys = []
        # Loop through keys
        for k in kw:
            # Save the key
            setattr(self,k, kw[k])
            # Save the name of the key
            self._keys.append(k)
# class struct

# Make replacements to docstring
__doc__ = __doc__.replace("_Lref_", '%.1f'%Lref)
__doc__ = __doc__.replace("_Aref_", '%.2f'%Aref)
__doc__ = __doc__.replace("_xMRP_", '%.2f'%xMRP)
__doc__ = __doc__.replace("_ySRB_", '%.2f'%ySRB)


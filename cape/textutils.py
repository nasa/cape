r"""
:mod:`cape.text`: Text modification utils for CAPE
====================================================

This module provides several functions for modifying Python docstrings.
One of these, provided by the function :func:`markdown`, which removes
much of the reST syntax used to make this documentation and prints a
more readable message for command-line help messages.

There is another function :func:`setdocvals` which can be used to
rapidly substitute strings such as ``_atol_`` with a value for *atol*
taken from a :class:`dict`.
"""

# Regular expression tools
import math
import re
from typing import Union


# Suffixes
SI_SUFFIXES = {
    -3: "n",
    -2: "µ",
    -1: "m",
    0: "",
    1: "k",
    2: "M",
    3: "G",
    4: "T",
    5: "P",
    6: "E",
}


# Function to take out extensive markup for help messages
def markdown(doc):
    r"""Remove some extraneous markup for command-line help messages

    :Call:
        >>> txt = markdown(doc)
    :Inputs:
        *doc*: :class:`str`
            Docstring
    :Outputs:
        *txt*: :class:`str`
            Help message with some of the RST features removed for readability
    :Versions:
        * 2017-02-17 ``@ddalle``: v1.0
    """
    try:
        return markdown_try(doc)
    except Exception:
        print("[MARKDOWN]: Markdown process of docstring failed")
        return doc


# Function to take out extensive markup for help messages
def markdown_try(doc):
    r"""Remove some extraneous markup for command-line help messages

    :Call:
        >>> txt = markdown_try(doc)
    :Inputs:
        *doc*: :class:`str`
            Docstring
    :Outputs:
        *txt*: :class:`str`
            Help message with some of the RST features removed for readability
    :Versions:
        * 2017-02-17 ``@ddalle``: v1.0
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
    while i+1 < len(lines):
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
            if (i > 0) and (lines[i-1].strip() == ""):
                # Delete the preceding line if empty
                del lines[i-1]
                i -= 1
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
    txt = re.sub(r":([\w/ _-]+):\s*\n", replsec, txt)
    # Replace modifiers, such as :mod:`cape.pycart`
    txt = re.sub(":([\\w/ _-]+):`([^`\n]+)`", replfn, txt)
    # Simplify user names
    txt = re.sub(r"``(@\w+)``", repluid, txt)
    # Simplify bolds
    txt = re.sub(r"\*\*([\w ]+)\*\*", replemph, txt)
    # Mark string literals
    txt = re.sub("``([^`\n]+)``", repllit, txt)
    # Output
    return txt


# Set marked values
def setdocvals(doc, vals):
    r"""Replace tags such as ``"_tol_"`` with values from a dictionary

    :Call:
        >>> txt = setdocvals(doc, vals)
    :Inputs:
        *doc*: :class:`str`
            Docstring
        *vals*: :class:`dict`
            Dictionary of key names to replace and values to set them to
    :Outputs:
        *txt*: :class:`str`
            string with ``"_%s_"%k`` -> ``str(vals[k]))`` for *k* in *vals*
    :Versions:
        * 2017-02-17 ``@ddalle``: v1.0
    """
    # Loop through keys
    for k in vals:
        # Get value and convert to string
        v = str(vals[k])
        # Make replacement
        doc = doc.replace("_%s_" % k, v)
    # Output
    return doc


# Get number of incidences of character at beginning of string (incl. 0)
def get_nstart(line, c):
    r"""Count number of instances of character *c* at start of a line

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
        * 2017-02-17 ``@ddalle``: v1.0
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


def _o_of_thousands(x: Union[float, int]) -> int:
    r"""Get the floor of the order of thousands of a given number

    If ``0``, then *x* is between ``0`` and ``1000``. If the number is
    at least ``1000`` but less than one million, then the output is
    ``1``. This is to correspond roughly to English-language names for
    large numbers.
    """
    return int(math.log10(x) // 3)


def pprint_n(x: Union[float, int], nchar: int = 3) -> str:
    r"""Format a string as number thousands, millions, etc.

    :Call:
        >>> txt = pprint_n(x, nchar=3)
    :Inputs:
        *x*: :class:`int` | :class:`float`
            Number to format
        *nchar*: {``3``} | :class:`int`
            Number of digits to print
    :Outputs:
        *txt*: :class:`str`
            Formatted text
    """
    # Check for ``None``
    if x is None:
        return ''
    # Get number of tens
    nten = 0 if abs(x) < 1 else int(math.log10(x))
    # Get number of thousands
    nth = _o_of_thousands(x)
    # Extra orders of magnitude; 0 for 1.23k, 1 for 12.3k, 2 for 123k
    nround = nten - 3*nth
    # Number of digits after decimal we need to retain
    decimals = max(0, nchar - 1 - nround)
    # Divide *xround* by this to get to multiples of k, M, G, etc.
    exp2 = 3*nth
    div2 = 10 ** exp2
    # Final float b/w 0 and 1000, not including exponent
    y = x / div2
    # Get suffix
    suf = SI_SUFFIXES.get(nth)
    # Use appropriate flag
    if decimals and nth > 0:
        return "%.*f%s" % (decimals, y, suf)
    else:
        return "%i%s" % (int(y), suf)

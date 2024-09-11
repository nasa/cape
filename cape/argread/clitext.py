r"""
``argread.clitext``: Process help messages for console output
=============================================================

This module provides the function :func:`compile_rst` to turn a
docstring or other Python string into a format that is appropriate to
print to STDOUT. It uses ANSI escape characters to produce bold text if
available.
"""

# Standard library
import re


# Standard regular expressions
REGEX_DIRECTIVE = re.compile(r"\.\. +[a-z-]+::")

# Standard characters
BOLD = "\x1b[1m"
ITALIC = "\x1b[3m"
PLAIN = "\x1b[0m"
BOLDITALIC = f"{BOLD}{ITALIC}"


# Function to print bold text
def bold(txt: str) -> str:
    r"""Produce bold text string for console

    :Call:
        >>> out = bold(txt)
    :Inputs:
        *txt*: :class:`str`
            Text to mark as bold
    :Outputs:
        *out*: :class:`str`
            Compiled text
    """
    return BOLD + txt + PLAIN


# Function to print italic text
def italic(txt: str) -> str:
    r"""Produce italic text string for console

    :Call:
        >>> out = italic(txt)
    :Inputs:
        *txt*: :class:`str`
            Text to mark as bold
    :Outputs:
        *out*: :class:`str`
            Compiled text
    """
    return ITALIC + txt + PLAIN


# Function to print bold & italic
def bolditalic(txt: str) -> str:
    r"""Produce bold-italic text string for console

    :Call:
        >>> out = italic(txt)
    :Inputs:
        *txt*: :class:`str`
            Text to mark as bold
    :Outputs:
        *out*: :class:`str`
            Compiled text
    """
    return BOLDITALIC + txt + PLAIN


# Function to take out extensive markup for help messages
def compile_rst(doc: str) -> str:
    r"""Remove some extraneous markup for command-line help messages

    :Call:
        >>> txt = _compile_rst(doc)
    :Inputs:
        *doc*: :class:`str`
            A multiline string in Sphinx reST syntax
    :Outputs:
        *txt*: :class:`str`
            A string formatted for command-line output
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
            # No markup
            return "%s" % val

    # Remove literals around usernames
    def repluid(g):
        # Strip "``"
        return g.group(1)

    # Generic literals
    def repllit(g):
        # Strip "``", bold
        return bold(g.group(1))

    # Remove **emphasis**
    def replemph(g):
        # Strip "**", bold+italic
        return bolditalic(g.group(1))

    # Remove *italic*
    def replit(g):
        # Strip "*"
        return italic(g.group(1))

    # Split by lines
    lines_in = doc.split('\n')
    # Initialize previous raw line
    line_in_txt = ""
    # Initialize output lines
    lines_out = []
    # Loop through lines
    while len(lines_in):
        # Save previous (raw) line
        line_in_prev = line_in_txt
        # Get line
        line_in = lines_in.pop(0)
        # Stripped
        line_in_txt = line_in.strip()
        # Check for empty line
        if line_in_txt == "":
            # Remove blank line after directives, e.g. code-block::
            if REGEX_DIRECTIVE.match(line_in_prev) is None:
                # Blank line only included if not following directive
                lines_out.append("")
            # Done with empty line
            continue
        # Get first character
        c = line_in_txt[0]
        # Check for section header: repeats of first character
        if c in '=-*#^':
            # Get number of repeats
            nc = get_nstart(line_in_txt, c)
            # Check section marker
            if nc > 3:
                # Ignore hlines above/after section header
                continue
        # Check for code block
        if REGEX_DIRECTIVE.match(line_in_txt):
            # Search for leading spaces
            nw = get_nstart(line_in, " ")
            # Check lines after code block
            while len(lines_in):
                # Go to next line
                line_in = lines_in[0]
                line_in_prev = line_in_txt
                line_in_txt = line_in.strip()
                # Check for empty line
                if line_in_txt == "":
                    # Use it
                    lines_out.append("")
                    # Go to next line in block
                    lines_in.pop(0)
                    continue
                # For non-emoty line, get number of leading spaces
                ns = get_nstart(line_in, " ")
                # Check indent level
                if ns > nw:
                    # Shift left by whatever the additional indent is
                    lines_out.append(line_in[ns-nw:])
                    # Go to next line in block
                    lines_in.pop(0)
                    continue
                else:
                    # Detected a dedent; go back to main loop
                    line_in_txt = line_in_prev
                    break
            # Go to next line
            continue
        # Normal line
        lines_out.append(line_in.rstrip())
    # Reform doc string
    txt = '\n'.join(lines_out)
    # Replace section headers
    txt = re.sub(r":(\w[\w/ _-]*):\s*\n", replsec, txt)
    # Replace modifiers, such as :mod:`cape.pycart`
    txt = re.sub(r":(\w[\w/ _-]*):`([^`\n]+)`", replfn, txt)
    # Simplify user names
    txt = re.sub(r"``(@\w+)``", repluid, txt)
    # Simplify bolds
    txt = re.sub(r"\*\*(\w[\w ]*)\*\*", replemph, txt)
    # Simplify italic
    txt = re.sub(r"\*(\w[\w ]*)\*", replit, txt)
    # Mark string literals
    txt = re.sub(r"``([^`\n]*)``", repllit, txt)
    # Output
    return txt


# Get number of incidences of character at beginning of string (incl. 0)
def get_nstart(line: str, c: str) -> int:
    r"""Count number of instances of character *c* at start of a line

    :Call:
        >>> nc = get_nstart(line, c)
    :Inputs:
        *line*: :class:`str` | :class:`unicode`
            String
        *c*: :class:`str`
            Character
    :Outputs:
        *nc*: :class:`int` >= 0
            Number of times *c* occurs at beginning of string (can be 0)
    """
    # Initialize counter
    nc = 0
    # Check position *nc*
    while True:
        if len(line) > nc and line[nc] == c:
            nc += 1
        else:
            # Position *nc* is not *c* or line is over
            return nc

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
import shutil
from typing import Optional

# Third-party
try:
    import colorama
except ModuleNotFoundError:  # pragma no cover
    colorama = None


# Initialize colorama to support ANSI escape codes on Windows
if colorama is not None:
    colorama.init(autoreset=True)

# Standard regular expressions
REGEX_DIRECTIVE = re.compile(r"\.\. +[a-z-]+::")


# Console colors and attributes
CONSOLE = {
    # Foreground colors
    'black':     '\x1b[30m',
    'darkgray':  '\x1b[30;01m',
    'red':       '\x1b[31;01m',
    'darkred':   '\x1b[31m',
    'green':     '\x1b[32;01m',
    'darkgreen': '\x1b[32m',
    'yellow':    '\x1b[33;01m',
    'brown':     '\x1b[33m',
    'blue':      '\x1b[34;01m',
    'darkblue':  '\x1b[34m',
    'purple':    '\x1b[35m',
    'fuchsia':   '\x1b[35;01m',
    'turquoise': '\x1b[36m',
    'teal':      '\x1b[36;01m',
    'white':     '\x1b[37;01m',
    'lightgray': '\x1b[37m',
    # Bright foreground colors
    'bright-black':   '\x1b[90m',
    'bright-red':     '\x1b[91m',
    'bright-green':   '\x1b[92m',
    'bright-yellow':  '\x1b[93m',
    'bright-blue':    '\x1b[94m',
    'bright-magenta': '\x1b[95m',
    'bright-cyan':    '\x1b[96m',
    'bright-white':   '\x1b[97m',
    # Background colors
    'bg-black':     '\x1b[40m',
    'bg-red':       '\x1b[41m',
    'bg-green':     '\x1b[42m',
    'bg-yellow':    '\x1b[43m',
    'bg-blue':      '\x1b[44m',
    'bg-magenta':   '\x1b[45m',
    'bg-cyan':      '\x1b[46m',
    'bg-white':     '\x1b[47m',
    # Bright background colors
    'bg-bright-black':   '\x1b[100m',
    'bg-bright-red':     '\x1b[101m',
    'bg-bright-green':   '\x1b[102m',
    'bg-bright-yellow':  '\x1b[103m',
    'bg-bright-blue':    '\x1b[104m',
    'bg-bright-magenta': '\x1b[105m',
    'bg-bright-cyan':    '\x1b[106m',
    'bg-bright-white':   '\x1b[107m',
    # Text attributes
    'bold':          '\x1b[01m',
    'faint':         '\x1b[02m',
    'italic':        '\x1b[03m',
    'underline':     '\x1b[04m',
    'blink':         '\x1b[05m',
    'standout':      '\x1b[03m',
    'reverse':       '\x1b[07m',
    'conceal':       '\x1b[08m',
    'strikethrough': '\x1b[09m',
    # Special turn-off sequences
    'un-bold':          "\x1b[22m",
    'un-italic':        "\x1b[23m",
    'un-underline':     "\x1b[24m",
    'un-blink':         "\x1b[25m",
    'un-reverse':       "\x1b[27m",
    'un-strikethrough': "\x1b[29m",
    # Reset
    'plain':      '\x1b[0m',
    'reset':      '\x1b[39;49;00m',
}


# Standard characters
BOLD = CONSOLE["bold"]
ITALIC = CONSOLE["italic"]
PLAIN = CONSOLE["plain"]
BOLDITALIC = f"{BOLD}{ITALIC}"
UNDERLINE = CONSOLE["underline"]


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
    return BOLD + txt + CONSOLE["un-bold"]


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
    return ITALIC + txt + CONSOLE["un-italic"]


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
    return BOLDITALIC + txt + CONSOLE["un-bold"] + CONSOLE["un-italic"]


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

    # Markdown-style sections
    def replmdsec1(g):
        # Top-level section, add many levels
        return (
            BOLD + UNDERLINE +
            CONSOLE["bg-black"] + CONSOLE["yellow"] +
            g.group(1) + PLAIN + '\n')

    # Markdown-style sections
    def replmdsec2(g):
        # Top-level section, add many levels
        return (
            BOLD + UNDERLINE +
            g.group(1) + PLAIN + '\n')

    # Markdown-style sections
    def replmdsec3(g):
        # Top-level section, add many levels
        return (
            UNDERLINE +
            g.group(1) + PLAIN + '\n')

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

    # Replace with an arbitrary color
    def replcolor(g):
        # Get color name
        color = g.group(1)
        # Get text
        text = g.group(2)
        # CHeck if color was recognized
        if color in CONSOLE:
            # Put some text in a color
            return CONSOLE[color] + text + PLAIN
        else:
            # Return original string
            return BOLD + text + PLAIN

    # Insert general format character
    def replfmt(g):
        # Get char name
        color = g.group(1)
        # Use it if possible
        return CONSOLE.get(color, '')

    # Set an arbitrary RGB color
    def replrgb(g):
        # Get r,g,b values
        cr = g.group(2)
        cg = g.group(3)
        cb = g.group(4)
        # Insert special character
        return f"\x1b[38;2;{cr};{cg};{cb}m"

    # Set an arbitrary RGB background color
    def replrgbbg(g):
        # Get r,g,b values
        cr = g.group(1)
        cg = g.group(2)
        cb = g.group(3)
        # Insert special character
        return f"\x1b[48;2;{cr};{cg};{cb}m"

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
    # Shortcut for color spec
    c = "([012]?[0-9]?[0-9])"
    pat1 = rf"\&(fg)?\({c}[;,]{c}[;,]{c}\)"
    pat2 = rf"\&bg\({c}[;,]{c}[;,]{c}\)"
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
    # Take out markdown-style section headers
    txt = re.sub(r"^# +(.*)$", replmdsec1, txt)
    txt = re.sub(r"^## +(.*)$", replmdsec2, txt)
    txt = re.sub(r"^### +(.*)$", replmdsec3, txt)
    # Apply custom "roles"
    txt = re.sub(r":([a-zA-Z][\w_-]*):`([^`\n]+)`", replcolor, txt)
    # Replace field-list headers
    txt = re.sub(r":(\w[\w/ _.-]*):\s*\n", replsec, txt)
    # Replace modifiers, such as :mod:`cape.pycart`
    txt = re.sub(r":(\w[\w/ _.-]*):`([^`\n]+)`", replfn, txt)
    # Simplify user names
    txt = re.sub(r"``(@\w+)``", repluid, txt)
    # Simplify bolds
    txt = re.sub(r"\*\*([^*\n]*\**)\*\*", replemph, txt)
    # Simplify italic (more targeted)
    txt = re.sub(r"\*(\w[^*\n]*\**)\*", replit, txt)
    # Manual format characters
    txt = re.sub(r"\&\(([\w-]+)\)", replfmt, txt)
    # Arbitrary colors
    txt = re.sub(pat1, replrgb, txt)
    txt = re.sub(pat2, replrgbbg, txt)
    # Mark string literals
    txt = re.sub(r"``?([^`\n]*)``?", repllit, txt)
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


def wrapline(msg: str, w: Optional[int] = None) -> str:
    r"""Wrap a message at word boundaries

    :Call:
        >>> formatted = wrapline(msg, w=None)
    :Inputs:
        *msg*: :class:`str`
            Original message, possibly with long lines
        *w*: {``None``} | :class:`int`
            Max cols in one line, default is the smaller of 79 and the
            current terminal width
    :Outputs:
        *formatted*: :class:`str`
            Formatted line with line breaks inserted where necessary
    """
    # Default width
    if w is None:
        w = min(79, shutil.get_terminal_size().columns)
    # Start output
    lines = []
    # Loop through lines
    for line in msg.split('\n'):
        # Check if line is short enough already
        if len(line) <= w:
            lines.append(line)
            continue
        # Get leading whitespace for indentation
        nw = get_nstart(line, " ")
        indent = " " * nw
        # Strip leading whitespace for wrapping
        line_stripped = line[nw:]
        # Split into words
        words = line_stripped.split()
        # Start building wrapped lines
        current_line = indent
        for word in words:
            # Check if adding this word would exceed width
            if len(current_line) > len(indent):
                # Not the first word on this line
                test_line = current_line + " " + word
            else:
                # First word on this line
                test_line = current_line + word
            # Check length
            if len(test_line) <= w:
                # Word fits on current line
                current_line = test_line
            else:
                # Word doesn't fit; start new line
                if len(current_line) > len(indent):
                    # Save current line if it has content
                    lines.append(current_line)
                # Start new line with same indentation
                current_line = indent + word
        # Add final line if it has content
        if len(current_line) > len(indent):
            lines.append(current_line)
    # Join with newlines
    return '\n'.join(lines)


r"""
This module provides several functions for modifying Python docstrings.  One of
these, provided by the function :func:`markdown`, which removes much of the
reST syntax used to make this documentation and prints a more readable message
for command-line help messages.

There is another function :func:`setdocvals` which can be used to rapidly
substitute strings such as ``_atol_`` with a value for *atol* taken from a
:class:`dict`.

"""

# Regular expression tools
import re


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
    txt = re.sub(":([\w/ _-]+):\s*\n", replsec, txt)
    # Replace modifiers, such as :mod:`cape.pycart`
    txt = re.sub(":([\w/ _-]+):`([^`\n]+)`", replfn, txt)
    # Simplify user names
    txt = re.sub("``(@\w+)``", repluid, txt)
    # Simplify bolds
    txt = re.sub("\*\*([\w ]+)\*\*", replemph, txt)
    # Mark string literals
    txt = re.sub("``([^`\n]+)``", repllit, txt)
    # Output
    return txt
                
# Set marked values
def setdocvals(doc, vals):
    """Replace tags such as ``"_tol_"`` with values from a dictionary
    
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
        * 2017-02-17 ``@ddalle``: First version
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


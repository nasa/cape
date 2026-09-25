r"""
:mod:`cape.clidoc.cape`: CAPE command-line help
================================================

Auto-generated help message for the CAPE command-line interface,
including the help messages for each of its sub-commands.
"""

# Standard library modules
from typing import Optional

# Local modules
from ..cfdx import cli


def get_cmdparser_cls(
        parser: cli.CfdxFrontDesk,
        cmdname: str) -> Optional[cli.CfdxArgReader]:
    r"""Get the parser class for a ``cape`` sub-command

    :Call:
        >>> cls = get_cmdparser_cls(parser, cmdname)
    :Inputs:
        *parser*: :class:`cape.cfdx.cli.CfdxFrontDesk`
            Instance of main parser for the ``cape`` command
        *cmdname*: :class:`str`
            Name of the sub-command
    :Outputs:
        *cls*: ``None`` | :class:`cape.cfdx.cli.CfdxArgReader`
            Parser class for sub-command *cmdname*, if any
    """
    # Direct lookup in sub-command parser dict
    cls = parser._cmdparsers.get(cmdname)
    if cls is not None:
        return cls
    # Check aliases (e.g. "find-cases" resolves to "find" parser)
    for alias, target in parser._cmdmap.items():
        if target == cmdname:
            cls = parser._cmdparsers.get(alias)
            if cls is not None:
                return cls
    # No parser found
    return None


def genr8_cmd_help(
        parser: cli.CfdxFrontDesk,
        cmdname: str) -> str:
    r"""Generate a help section for one ``cape`` sub-command

    This creates the same message as ``cape CMD -h`` but with the
    title turned into an RST section header.

    :Call:
        >>> msg = genr8_cmd_help(parser, cmdname)
    :Inputs:
        *parser*: :class:`cape.cfdx.cli.CfdxFrontDesk`
            Instance of main parser for the ``cape`` command
        *cmdname*: :class:`str`
            Name of the sub-command
    :Outputs:
        *msg*: :class:`str`
            Help message for *cmdname*, marked up as a section
    """
    # Get the parser class for this sub-command
    cls = get_cmdparser_cls(parser, cmdname)
    # Special case for sub-commands with no dedicated parser
    if cls is None:
        # Get short description from front-desk list if available
        descr = parser._help_cmd.get(
            cmdname, "Show main help message and exit")
        # Create a title
        title = f"``cape {cmdname}``: {descr}"
        # Simple explanation (same as ``cape -h``)
        return (
            f"{title}\n{'~' * len(title)}\n\n"
            "Prints the main help message shown above.\n")
    # Generate full help message for the sub-command
    msg = cls().genr8_help()
    # Split off title (first line) and its adornment (second line)
    lines = msg.split("\n")
    body = "\n".join(lines[2:]).strip("\n")
    # Re-create title using actual sub-command name; this also
    # converts the title to a section header (``~~~~`` underline)
    title = f"``cape {cmdname}``: {cls._help_title}"
    # Combine title and body
    return f"{title}\n{'~' * len(title)}\n\n{body}\n"


def genr8_full_help() -> str:
    r"""Generate full help message, including all sub-commands

    :Call:
        >>> msg = genr8_full_help()
    :Outputs:
        *msg*: :class:`str`
            Help message for ``cape -h`` plus one section for each
            sub-command's ``-h``
    """
    # Instantiate parser
    parser = cli.CfdxFrontDesk()
    # Initialize document with main help message
    parts = [parser.genr8_help()]
    # Begin section listing each sub-command's help
    parts.append("\n\n")
    # Loop through sub-commands
    for cmdname in parser._cmdlist:
        parts.append("\n" + genr8_cmd_help(parser, cmdname))
    # Combine the sections
    return "".join(parts)


# Generate help
__doc__ = genr8_full_help()

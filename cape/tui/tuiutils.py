r"""
:mod:`cape.tui.tuiutils`: Render helpers for the CAPE TUI
=========================================================

This module provides the presentation helpers for :mod:`cape.tui`,
the Textual-based interactive terminal user interface to CAPE. These
functions build Rich *renderables* (tables, panels) which the app
writes to its scroll log or prints to the terminal on exit.

"""

from __future__ import annotations

# Standard library
import os
import socket
from datetime import timedelta
from typing import Optional

# Third-party imports
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# CAPE imports
from cape import __version__
from cape import capeconfig


# Exit commands (same as :mod:`cape.ui`)
EXIT_CMDS = (
    "exit",
    "quit",
    "exit()",
    "quit()",
)

# TUI meta-commands
META_CMDS = (
    ":cd",
    ":pwd",
    ":help",
    ":clear",
    ":history",
    ":status",
    ":exit",
    "TAB",
    "↑↓",
    "Ctrl-C",
)

# Descriptions of meta-commands
META_CMD_DESCS = {
    ":cd": "Change folder, e.g. ``:cd powerless/``",
    ":pwd": "Show current folder",
    ":clear": "Clear the scroll log",
    ":exit": "Exit the CAPE TUI",
    ":help": "Show CAPE command list or details of one command",
    ":history": "Show table of recently-run commands",
    ":status": "Show summary of current TUI session",
    "TAB": "complete",
    "↑↓": "search history",
    "Ctrl-C": "interrupt",
}

# Help-me topics for :help
META_HELP_TOPICS = ("tui", "meta")

# Maximum number of commands kept in the history file
CAPE_HISTORY_LENGTH = 1000

# tokyo-night colors for log content (CSS vars only style widgets)
TN_BLUE = "#7AA2F7"
TN_DIM = "#565F89"
TN_GREEN = "#9ECE6A"
TN_ORANGE = "#FF9E64"
TN_PURPLE = "#BB9AF7"
TN_RED = "#F7768E"
TN_SURFACE = "#24283B"


# Format a duration in seconds
def sprintf_duration(dt: float) -> str:
    r"""Format a duration in seconds as a compact string

    :Call:
        >>> txt = sprintf_duration(dt)
    :Inputs:
        *dt*: :class:`float`
            Duration in seconds
    :Outputs:
        *txt*: :class:`str`
            Compact string, e.g. ``"0.42 s"`` or ``"1 min 5 s"``
    """
    # Check size
    if dt < 1.0:
        return f"{1000*dt:4.0f} ms"
    elif dt < 100.0:
        return f"{dt:5.2f} s"
    else:
        return str(timedelta(seconds=round(dt)))


# Get last two folders of a path
def get_dirname(path: Optional[str] = None) -> str:
    r"""Get the last two folders of a path, as in :mod:`cape.ui`

    :Call:
        >>> dirname = get_dirname(path=None)
    :Inputs:
        *path*: {``None``} | :class:`str`
            Folder to summarize; default is current working folder
    :Outputs:
        *dirname*: :class:`str`
            Last two folders of *path*
    """
    # Use current folder if not given
    if path is None:
        path = os.getcwd()
    # Get last two parts
    _dir, basename = os.path.split(path)
    parname = os.path.basename(_dir)
    # Generate a short name
    return os.path.join(parname, basename)


# Get the name of the history file
def get_tui_histfile() -> str:
    r"""Get the name of the CAPE TUI history file

    The file is controlled by the *TUIHistoryFile* option from the
    user's ``~/.capeconfig.json`` (or the ``$CAPE_TUI_HISTORY_FILE``
    environment variable). Relative paths are joined with the CAPE
    *CacheDir*.

    :Call:
        >>> histfile = get_tui_histfile()
    :Outputs:
        *histfile*: :class:`str`
            Name of TUI command history file
    """
    # Get history file
    histfile = capeconfig.get_cape_opt("TUIHistoryFile")
    # If relative path, join with CacheDir
    if not os.path.isabs(histfile):
        cachedir = capeconfig.get_cape_opt("CacheDir")
        histfile = os.path.join(cachedir, histfile)
    # Output
    return os.path.expanduser(histfile)


# Render the full command help table
def cmd_table(cls: type) -> Table:
    r"""Build a Rich table of all CAPE sub-commands

    :Call:
        >>> table = cmd_table(cls)
    :Inputs:
        *cls*: :class:`type`
            :class:`cape.argread.ArgReader` subtype
    :Outputs:
        *table*: :class:`rich.table.Table`
            Table of CAPE commands, descriptions, and aliases
    """
    # Create table
    table = Table(
        title="CAPE commands",
        box=box.SIMPLE_HEAD,
        title_justify="left",
        pad_edge=False)
    # Columns
    table.add_column("Command", style="bold green", no_wrap=True)
    table.add_column("Description", overflow="fold")
    table.add_column("Aliases", style="purple", overflow="fold")
    # Invert the alias map: {alias -> cmd} -> {cmd -> [aliases]}
    cmd_aliases = _invert_cmdmap(cls)
    # Loop through commands
    for cmdname in cls._cmdlist:
        # Get subparser class
        subcls = cls._cmdparsers.get(cmdname)
        # Get description
        desc = "" if subcls is None else subcls._help_title
        # Get aliases
        aliases = ", ".join(cmd_aliases.get(cmdname, []))
        # Add row
        table.add_row(cmdname, desc, aliases)
    # Output
    return table


# Render help for one CAPE command
def cmd_help_panel(cmdname: str, subcls: type, cls: type) -> Panel:
    r"""Build a Rich help panel for a single CAPE sub-command

    :Call:
        >>> panel = cmd_help_panel(cmdname, subcls, cls)
    :Inputs:
        *cmdname*: :class:`str`
            Name of command
        *subcls*: :class:`type`
            :class:`cape.argread.ArgReader` for *cmdname*
        *cls*: :class:`type`
            :class:`cape.argread.ArgReader` for front desk
    :Outputs:
        *panel*: :class:`rich.panel.Panel`
            Panel with description, aliases, and options table
    """
    # Invert the alias map
    cmd_aliases = _invert_cmdmap(cls)
    # Build title line
    title = Text(f"cape {cmdname}", style="bold green")
    # Aliases
    alias_txt = ", ".join(cmd_aliases.get(cmdname, []))
    # Description line
    desc = Text(subcls._help_title, style="italic")
    # Assemble body
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(desc)
    if alias_txt:
        body.add_row(Text(f"(aliases: {alias_txt})", style="purple"))
    # Options table
    body.add_row("")
    # Collect options and types along sub-command's MRO
    optlist = _merged_optlist(subcls)
    opttypes = _merged_dict(subcls, "_opttypes")
    opthelps = _merged_dict(subcls, "_help_opt")
    # Table of options
    opttable = Table(box=box.SIMPLE, pad_edge=False)
    opttable.add_column("Option", style="bold green", no_wrap=True)
    opttable.add_column("Value", style="yellow")
    opttable.add_column("Description", overflow="fold")
    # Loop through options of this sub-command
    for opt in optlist:
        # Get option type
        vtype = opttypes.get(opt, bool)
        # Human-readable value type
        vlabel = _opt_type_label(vtype)
        # Description
        odesc = opthelps.get(opt, "")
        # Add row
        opttable.add_row(f"--{opt}" if len(opt) > 1 else f"-{opt}",
                         vlabel, odesc)
    # Options table as second row
    body.add_row(opttable)
    # Output
    return Panel(body, title=title, border_style="green", padding=(0, 1))


# TUI meta-command table
def meta_help_table() -> Table:
    r"""Build table of TUI meta-commands

    :Call:
        >>> table = meta_help_table()
    :Outputs:
        *table*: :class:`rich.table.Table`
            Table of TUI meta-commands and their descriptions
    """
    # Create table
    table = Table(
        title="CAPE TUI commands",
        box=box.SIMPLE_HEAD,
        title_justify="left",
        pad_edge=False)
    # Columns
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("Description")
    # Loop through meta-commands
    for metacmd in META_CMDS:
        table.add_row(metacmd, META_CMD_DESCS.get(metacmd, ""))
    # Add re-run
    table.add_row(":!N", "Rerun history command No. N")
    # Output
    return table


# Recent history as a table
def history_table(history: list, n: int = 25) -> Table:
    r"""Build a table of recently-run commands

    :Call:
        >>> table = history_table(history, n=25)
    :Inputs:
        *history*: :class:`list`\ [:class:`str`]
            Full session history (oldest first)
        *n*: {``25``} | :class:`int`
            Number of recent commands to show
    :Outputs:
        *table*: :class:`rich.table.Table`
            Table with 1-based history numbers and command text
    """
    # History length and first index to show
    nhist = len(history)
    j0 = max(0, nhist - n)
    # Create table
    table = Table(
        title="Command history",
        box=box.SIMPLE_HEAD,
        title_justify="left",
        pad_edge=False)
    # Columns
    table.add_column("No.", style="purple", justify="right")
    table.add_column("Command", overflow="fold")
    # Loop through recent history
    for j in range(j0, nhist):
        table.add_row(str(j + 1), history[j])
    # Output
    return table


# Startup banner panel
def banner_panel(histfile: Optional[str] = None) -> Panel:
    r"""Build the CAPE TUI startup banner

    :Call:
        >>> panel = banner_panel(histfile=None)
    :Inputs:
        *histfile*: {``None``} | :class:`str`
            Name of command history file
    :Outputs:
        *panel*: :class:`rich.panel.Panel`
            Startup banner with host info and key hints
    """
    # Get host information
    hostname = socket.gethostname().split('.')[0]
    # Build info table
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold green", justify="right")
    table.add_column()
    # Add rows
    table.add_row("Host:", hostname)
    table.add_row("Folder:", os.getcwd())
    # Optional history info
    if histfile:
        table.add_row("History:", histfile)
    # Hint line
    hint = Text()
    hint.append("TAB", style="bold cyan")
    hint.append(" completes · ")
    hint.append("up/down", style="bold cyan")
    hint.append(" recalls history · ")
    hint.append("Ctrl-C", style="bold cyan")
    hint.append(" interrupts · ")
    hint.append(":help", style="bold cyan")
    hint.append(" for TUI commands")
    # Combine into banner
    body = Table.grid(padding=(0, 0))
    body.add_column()
    body.add_row(table)
    body.add_row("")
    body.add_row(hint)
    # Output
    return Panel(
        body,
        title="[bold green]CAPE[/] [bold cyan]TUI[/]",
        subtitle=f"v{__version__}",
        border_style="green",
        padding=(0, 2))


# Summary of the TUI session
def session_stats_panel(
        stats: dict,
        histfile: Optional[str] = None,
        title: str = "CAPE TUI session") -> Panel:
    r"""Build a panel summarizing the TUI session

    :Call:
        >>> panel = session_stats_panel(stats, histfile=None)
        >>> panel = session_stats_panel(stats, histfile, title="Bye")
    :Inputs:
        *stats*: :class:`dict`
            Session statistics (``commands``, ``failures``, etc.);
            ``json_files`` and ``last_json_file`` add JSON context
        *histfile*: {``None``} | :class:`str`
            Name of command history file
        *title*: {``"CAPE TUI session"``} | :class:`str`
            Title for statistics panel
    :Outputs:
        *panel*: :class:`rich.panel.Panel`
            Session summary panel
    """
    # Get host information
    hostname = socket.gethostname().split('.')[0]
    # Build info table
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold green", justify="right")
    table.add_column()
    # Rows
    table.add_row("Host:", hostname)
    table.add_row("Folder:", os.getcwd())
    table.add_row("Commands:", str(stats.get("commands", 0)))
    table.add_row("Failures:", str(stats.get("failures", 0)))
    table.add_row("TUI cmds:", str(stats.get("tui_commands", 0)))
    table.add_row("Duration:", sprintf_duration(stats.get("duration", 0.0)))
    # Show the active JSON file and how many other files were loaded.
    last_json = stats.get("last_json_display_file") or \
        stats.get("last_json_file")
    if last_json:
        other_files = max(0, len(stats.get("json_files", ())) - 1)
        suffix = f" (+{other_files})" if other_files else ""
        table.add_row("JSON file:", f"{last_json}{suffix}")
    # Optional history
    if histfile:
        table.add_row("History:", histfile)
    # Output
    return Panel(table, title=title, border_style="green", padding=(0, 2))


# Collect option names from each class in an MRO, preserving order
def _merged_optlist(cls: type) -> list:
    r"""Collect ``_optlist`` from *cls* and its parents in MRO order

    :Call:
        >>> optlist = _merged_optlist(cls)
    :Inputs:
        *cls*: :class:`type`
            :class:`cape.argread.ArgReader` subtype
    :Outputs:
        *optlist*: :class:`list`\ [:class:`str`]
            Combined list of option names, no duplicates
    """
    # Initialize output
    optlist = []
    # Loop through classes in MRO, most-specific first
    for klass in cls.__mro__:
        # Get options defined by this class, if any
        for opt in klass.__dict__.get("_optlist", ()):
            # Append if new
            if opt not in optlist:
                optlist.append(opt)
    # Output
    return optlist


# Merge dict attributes from each class in an MRO
def _merged_dict(cls: type, name: str) -> dict:
    r"""Merge a :class:`dict` attribute along an inheritance chain

    :Call:
        >>> merged = _merged_dict(cls, name)
    :Inputs:
        *cls*: :class:`type`
            :class:`cape.argread.ArgReader` subtype
        *name*: :class:`str`
            Name of :class:`dict` attribute, e.g. ``"_opttypes"``
    :Outputs:
        *merged*: :class:`dict`
            Merged dict, child-class entries override parents'
    """
    # Initialize output
    merged = {}
    # Loop through classes in MRO, most-general first
    for klass in reversed(cls.__mro__):
        # Get dict defined by this class, if any
        merged.update(klass.__dict__.get(name, {}))
    # Output
    return merged


# Get human-readable label for an option's value type
def _opt_type_label(vtype) -> str:
    # Handle type tuples
    if isinstance(vtype, tuple):
        # Check for all-bool
        if all(v is bool for v in vtype):
            return "flag"
        # Take last member for combos such as (bool, str)
        vtype = vtype[-1]
    # Check single types
    if vtype is bool:
        return "flag"
    elif vtype is int:
        return "INT"
    elif vtype is float:
        return "FLOAT"
    elif vtype is str:
        return "STR"
    # Default
    return "VAL"


# Invert the {alias -> cmd} map to {cmd -> [aliases]}
def _invert_cmdmap(cls: type) -> dict:
    # Initialize map
    cmd_aliases = {}
    # Loop through alias map
    for alias, cmdname in cls._cmdmap.items():
        # Skip self-references
        if alias == cmdname:
            continue
        # Get current list of aliases
        aliaslist = cmd_aliases.setdefault(cmdname, [])
        # Add this alias
        aliaslist.append(alias)
    # Sort lists
    for cmdname, aliaslist in cmd_aliases.items():
        cmd_aliases[cmdname] = sorted(aliaslist)
    # Output
    return cmd_aliases

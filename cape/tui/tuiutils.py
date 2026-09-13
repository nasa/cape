r"""
:mod:`cape.tui.tuiutils`: Render helpers for the CAPE rich TUI
================================================================

This module provides the presentation layer for :mod:`cape.tui`, the
Rich-based interactive terminal user interface to CAPE. It includes
the startup banner, help and history tables, per-command status
rules, and the :class:`TuiCompleter` tab-completion class, which adds
TUI meta-command completion on top of
:class:`cape.ui.promptutils.CfdxCompleter`.

"""

from __future__ import annotations

# Standard library
import fnmatch
import os
import readline
import socket
from datetime import timedelta
from typing import Optional

# Third-party imports
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# CAPE imports
from cape import __version__
from cape.promptutils import clickable_prompt_ok

# Local imports
from ..ui.promptutils import CfdxCompleter, sprintf_color_rl


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
    ":clear",
    ":exit",
    ":help",
    ":history",
    ":pwd",
    ":quit",
    ":status",
)

# Descriptions of meta-commands
META_CMD_DESCS = {
    ":cd": "Change folder, e.g. ``:cd powerless/``",
    ":clear": "Clear the terminal screen",
    ":exit": "Exit the CAPE TUI",
    ":help": "Show CAPE command list or details of one command",
    ":history": "Show table of recently-run commands",
    ":pwd": "Print the current working folder",
    ":quit": "Exit the CAPE TUI",
    ":status": "Show summary of current TUI session",
}

# Global console
CONSOLE = Console(highlight=False)

# Prompt colors
PROMPT_COLOR = ("bold", "green")
PROMPT_OK_COLOR = ("bold", "green")
PROMPT_FAIL_COLOR = ("bold", "red")
META_PROMPT_STYLE = "bold cyan"
BORDER_STYLE_OK = "green"
BORDER_STYLE_FAIL = "red"
BORDER_STYLE_META = "purple"


# Readline-based autocompleter with TUI meta-commands
class TuiCompleter(CfdxCompleter):
    r"""CAPE autocompleter that adds TUI meta-command suggestions

    This extends :class:`cape.ui.promptutils.CfdxCompleter` to
    complete first words that start with ``:`` using the TUI
    meta-command list *META_CMDS*.

    :Versions:
        * 2026-08-28 ``@ddalle``: v1.0
    """

    def genr8_suggestions(self, text: str) -> list[str]:
        r"""Generate suggestions, including TUI meta-commands

        :Call:
            >>> suggestions = comp.genr8_suggestions(text)
        :Inputs:
            *comp*: :class:`TuiCompleter`
                CAPE TUI autocompleter
            *text*: :class:`str`
                Current text of current word
        :Outputs:
            *suggestions*: :class:`list`\ [:class:`str`]
                List of suggested completions for current word
        """
        # Get current line
        line = readline.get_line_buffer()
        # Check for meta-command completion on first word
        if text.startswith(":") and line.lstrip().startswith(text):
            # Complete TUI meta-command names
            self.role = "metacmd"
            return fnmatch.filter(META_CMDS, f"{text}*")
        # Defer to CAPE front-desk completions
        return CfdxCompleter.genr8_suggestions(self, text)


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


# Get a prompt string with readline-safe colors
def make_prompt(hostname: str, dirname: str, ok: bool = True) -> str:
    r"""Create the CAPE TUI prompt string

    The ``CAHP host:dir`` prefix is bold green, and the trailing
    ``"$"`` is green after a successful command and red after a
    failed one.

    :Call:
        >>> prompt = make_prompt(hostname, dirname, ok=True)
    :Inputs:
        *hostname*: :class:`str`
            Name of current host
        *dirname*: :class:`str`
            Last two folders of current working folder
        *ok*: {``True``} | :class:`bool`
            Whether the most recent command succeeded
    :Outputs:
        *prompt*: :class:`str`
            Prompt with readline-safe color instructions
    """
    # Get color of "$" based on last result
    dollar_color = PROMPT_OK_COLOR if ok else PROMPT_FAIL_COLOR
    # Form base prompt
    prompt = sprintf_color_rl(f"CAPE {hostname}:{dirname}", PROMPT_COLOR)
    # Append status-colored "$"
    return prompt + sprintf_color_rl("$ ", dollar_color)


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


# Render the startup banner
def render_banner(histfile: Optional[str] = None) -> None:
    r"""Render the CAPE TUI startup banner

    :Call:
        >>> render_banner(histfile=None)
    :Inputs:
        *histfile*: {``None``} | :class:`str`
            Name of command history file
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
    hint.append(" for history · ")
    hint.append(":help", style="bold cyan")
    hint.append(" for TUI commands")
    # Optional hint about clickable option menus
    if clickable_prompt_ok():
        hint.append(" · ")
        hint.append("click", style="bold cyan")
        hint.append(" answers option menus")
    # Combine into banner
    body = Table.grid(padding=(0, 0))
    body.add_column()
    body.add_row(table)
    body.add_row("")
    body.add_row(hint)
    # Render
    CONSOLE.print(
        Panel(
            body,
            title="[bold green]CAPE[/] [bold cyan]TUI[/]",
            subtitle=f"v{__version__}",
            border_style="green",
            padding=(0, 2)))


# Render one command's footer status rule
def render_status_rule(cmd: str, ierr: int, dt: float) -> None:
    r"""Render a rule after a command with exit code and duration

    :Call:
        >>> render_status_rule(cmd, ierr, dt)
    :Inputs:
        *cmd*: :class:`str`
            Command that was run
        *ierr*: :class:`int`
            Return code of the command
        *dt*: :class:`float`
            Wall-clock duration of command in seconds
    """
    # Check status
    ok = ierr == 0
    # Get icon and styles
    icon = "✓" if ok else "✗"
    style = BORDER_STYLE_OK if ok else BORDER_STYLE_FAIL
    # Truncate long command names in the rule
    cmdtxt = cmd if len(cmd) <= 60 else cmd[:57] + "..."
    # Build label
    label = Text.assemble(
        (f"{icon} ", f"bold {style}"),
        (cmdtxt, "italic"),
        (f"  exit {ierr}  ·  {sprintf_duration(dt)}", style))
    # Render
    CONSOLE.rule(label, style=style)


# Render the echoed command heading before it runs
def render_command_rule(cmd: str) -> None:
    r"""Render a left-aligned rule with the command about to run

    :Call:
        >>> render_command_rule(cmd)
    :Inputs:
        *cmd*: :class:`str`
            Command about to run
    """
    # Truncate long command names in the rule
    cmdtxt = cmd if len(cmd) <= 72 else cmd[:69] + "..."
    # Build label and render
    label = Text(f"$ {cmdtxt}", style="purple")
    CONSOLE.rule(label, style="purple", align="left")


# Render a simple error message
def render_error(msg: str) -> None:
    r"""Print a red-styled error message to the TUI console

    :Call:
        >>> render_error(msg)
    :Inputs:
        *msg*: :class:`str`
            Error message to display
    """
    CONSOLE.print(Text(msg, style="bold red"))


# Render the full command help table
def render_help(cls: type, query: Optional[str] = None) -> int:
    r"""Render help for CAPE commands as a Rich table

    If *query* is given, show details (aliases, description, and
    options) for one command; otherwise show the full command list.

    :Call:
        >>> ierr = render_help(cls, query=None)
    :Inputs:
        *cls*: :class:`type`
            :class:`cape.argread.ArgReader` subtype, usually
            :class:`cape.cfdx.cli.CfdxFrontDesk`
        *query*: {``None``} | :class:`str`
            Name of one CAPE command
    :Outputs:
        *ierr*: :class:`int`
            Return code; nonzero if *query* is an unknown command
    """
    # Check for query
    if query is None:
        # Render the full table
        return render_cmd_table(cls)
    # Check alternate names
    cmdname = cls._cmdmap.get(query, query)
    # Get subparser
    subcls = cls._cmdparsers.get(cmdname)
    # Check for unknown command
    if subcls is None:
        render_error(f"Unknown CAPE command: '{query}'")
        return 16
    # Render details for one command
    return render_cmd_help(cmdname, subcls, cls)


# Render table of all CAPE commands
def render_cmd_table(cls: type) -> int:
    r"""Render a Rich table of all CAPE sub-commands

    :Call:
        >>> ierr = render_cmd_table(cls)
    :Inputs:
        *cls*: :class:`type`
            :class:`cape.argread.ArgReader` subtype
    :Outputs:
        *ierr*: :class:`int`
            Return code, always ``0``
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
    # Render
    CONSOLE.print(table)
    # Hint
    CONSOLE.print(
        Text("Use :help <cmd> or cape <cmd> -h for details", style="cyan"))
    return 0


# Render help for one CAPE command
def render_cmd_help(cmdname: str, subcls: type, cls: type) -> int:
    r"""Render help for a single CAPE sub-command

    :Call:
        >>> ierr = render_cmd_help(cmdname, subcls, cls)
    :Inputs:
        *cmdname*: :class:`str`
            Name of command
        *subcls*: :class:`type`
            :class:`cape.argread.ArgReader` for *cmdname*
        *cls*: :class:`type`
            :class:`cape.argread.ArgReader` for front desk
    :Outputs:
        *ierr*: :class:`int`
            Return code, always ``0``
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
    # Render in a panel
    CONSOLE.print(
        Panel(body, title=title, border_style="green", padding=(0, 1)))
    return 0


# Render recent history as a table
def render_history(n: int = 25) -> int:
    r"""Render a table of recently-run commands

    :Call:
        >>> ierr = render_history(n=25)
    :Inputs:
        *n*: {``25``} | :class:`int`
            Number of recent commands to show
    :Outputs:
        *ierr*: :class:`int`
            Return code, always ``0``
    """
    # Get history length
    nhist = readline.get_current_history_length()
    # First index to show
    j0 = max(1, nhist - n + 1)
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
    for j in range(j0, nhist + 1):
        table.add_row(str(j), readline.get_history_item(j))
    # Render
    CONSOLE.print(table)
    # Hint
    CONSOLE.print(Text("Use :!N to rerun command No. N", style="cyan"))
    return 0


# Render TUI meta-command table
def render_meta_help() -> int:
    r"""Render table of TUI meta-commands

    :Call:
        >>> ierr = render_meta_help()
    :Outputs:
        *ierr*: :class:`int`
            Return code, always ``0``
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
    # Render
    CONSOLE.print(table)
    return 0


# Render a summary of the TUI session
def render_session_stats(stats: dict, histfile: Optional[str] = None,
                         title: str = "CAPE TUI session") -> None:
    r"""Render a panel summarizing the TUI session

    :Call:
        >>> render_session_stats(stats, histfile=None)
        >>> render_session_stats(stats, histfile=None, title="Bye")
    :Inputs:
        *stats*: :class:`dict`
            Session statistics (``commands``, ``failures``, etc.)
        *histfile*: {``None``} | :class:`str`
            Name of command history file
        *title*: {``"CAPE TUI session"``} | :class:`str`
            Title for statistics panel
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
    # Optional history
    if histfile:
        table.add_row("History:", histfile)
    # Render
    CONSOLE.print(
        Panel(table, title=title, border_style="green", padding=(0, 2)))


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

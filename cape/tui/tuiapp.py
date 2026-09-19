r"""
:mod:`cape.tui.tuiapp`: Textual app for the CAPE terminal UI
============================================================

This module contains the OpenCode-style terminal user interface to
CAPE, built on the optional ``textual`` package. It provides, in one
persistent app with a scroll log and a pinned command composer:

* **In-process CAPE commands**
    Commands starting with ``cape``, ``pycart``, ``pyfun``, etc. run
    in-process using :func:`cape.cfdx.cli.main` in a worker thread,
    with their output streamed into the scroll log.

* **External commands**
    Any other command runs as a subprocess in its own process group;
    its combined output is streamed into the scroll log.

* **Tab completion**
    Pressing TAB in the editor completes the current word using the
    same :class:`cape.ui.promptutils.CfdxCompleter` logic as
    :mod:`cape.ui`, plus TUI meta-commands. Multiple matches appear
    in a selectable box immediately above the editor.

* **Ctrl-C / ESC interrupts**
    Pressing Ctrl-C (or ESC) while a command is running interrupts
    it. External commands receive ``SIGINT`` in their process group;
    in-process CAPE commands get a :class:`KeyboardInterrupt`
    injected into their worker thread. With nothing running, Ctrl-C
    clears the editor.

* **Prompt bridge**
    Interactive prompts raised by commands (via
    :func:`cape.promptutils.prompt_color`) mount as clickable widgets
    in the stream. Answering with Ctrl-C aborts the command.

* **TUI meta-commands**
    Commands starting with ``:`` are handled by the TUI itself:
    ``:help`` [<cmd>], ``:history`` [<n>], ``:!N``, ``:status``,
    ``:cd`` <dir> (plain ``cd`` also works), ``:pwd``, ``:clear``,
    and ``:exit`` / ``:quit``.

* **OpenCode-style chrome**
    Dark command composer, context and command discovery below the
    prompt, a live spinner while a command runs, and backgrounded
    command bubbles in the scroll log. Ctrl-P opens the command picker;
    clicking a blue command chevron folds or expands its output.

* **Text selection**
    Dragging the mouse over log or editor text highlights it, and
    releasing copies the selection to the system clipboard (OSC 52),
    like OpenCode.

History (loaded from and saved to the CAPE TUI history file) is
recalled with the up/down arrow keys.
"""

from __future__ import annotations

# Standard library
import ctypes
import fnmatch
import io
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Optional, Tuple

# Third-party
from rich.segment import Segment
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

# CAPE imports
from ..promptutils import _new_prompt_widget
from ..ui.promptutils import CAPE_EXECS, CfdxCompleter

# Local imports
from .tuiutils import (
    CAPE_HISTORY_LENGTH,
    META_CMDS,
    META_CMD_DESCS,
    META_HELP_TOPICS,
    TN_BLUE,
    TN_DIM,
    TN_GREEN,
    TN_ORANGE,
    TN_RED,
    TN_SURFACE,
    cmd_help_panel,
    cmd_table,
    get_dirname,
    get_tui_histfile,
    history_table,
    meta_help_table,
    session_stats_panel,
)


# Match commands to run with CAPE's in-process CLI
REGEX_CAPE_CLI = re.compile(rf"\$?\s*({'|'.join(CAPE_EXECS)})( |$)")

# Exit commands
EXIT_CMDS = (":exit", ":quit", "exit", "quit", "exit()", "quit()")

# Placeholder for the idle editor
INPUT_PLACEHOLDER = "cape command"

# Busy spinner animation in the status bar
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL = 0.1

# Status bar hints
HINTS_IDLE = "ctrl+p commands"
HINTS_BUSY = "ctrl+c interrupt"
COMPOSER_HINT = "TAB complete · ↑↓ history · Ctrl-C interrupt · Ctrl-D exit"
SUGGESTION_HINT = "↑↓ select · Enter/Tab insert · Esc close"


class CommandPalette(ModalScreen[str]):
    r"""Small picker for the TUI's built-in commands."""

    BINDINGS = [
        Binding("escape", "dismiss_palette", "Close", show=False),
        Binding("ctrl+p", "dismiss_palette", "Close", show=False),
    ]

    CSS = """
    CommandPalette {
        align: center middle;
        background: #000000 65%;
    }
    #command-list {
        width: 60;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        background: #202020;
        border-left: solid #7aa2f7;
        padding: 1 2;
    }
    #command-list > .option-list--option-highlighted {
        background: #343a48;
    }
    """

    def compose(self) -> ComposeResult:
        yield OptionList(
            *(Option(f"{cmd:<15} {META_CMD_DESCS[cmd]}", id=cmd)
              for cmd in META_CMDS),
            id="command-list")

    def on_option_list_option_selected(
            self, event: OptionList.OptionSelected) -> None:
        cmd = event.option.id
        self.dismiss(cmd if cmd is not None and cmd.startswith(":") else None)

    def action_dismiss_palette(self) -> None:
        self.dismiss(None)


# Inject an exception into a running thread
def _raise_in_thread(thread: threading.Thread, exc) -> None:
    r"""Raise an exception asynchronously in another thread

    This uses ``PyThreadState_SetAsyncExc``; the exception is raised at
    the next Python bytecode in *thread*. It cannot interrupt blocking
    C-level calls (e.g. :func:`subprocess.Popen.wait`).

    :Call:
        >>> _raise_in_thread(thread, exc)
    :Inputs:
        *thread*: :class:`threading.Thread`
            Thread that should raise *exc*
        *exc*: ``type`` | :class:`BaseException`
            Exception (or class) to raise in *thread*
    """
    # Get thread ID
    tid = thread.ident
    if tid is None:
        return
    # Inject the exception
    n = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(exc))
    # More than one thread modified: undo (should not happen)
    if n > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


# Tab-completer reading the command line from the editor
class CapeTuiCompleter(CfdxCompleter):
    r"""CAPE tab-completer that reads the line from the TUI editor

    This overrides :func:`CfdxCompleter.get_line_buffer` to source the
    command line from the app's ``Input`` widget (text before the
    cursor) instead of :mod:`readline`, and adds suggestions for the
    TUI ``:`` meta-commands.

    :Attributes:
        *app*: :class:`CapeTuiApp`
            App whose editor is being completed
    """

    __slots__ = ("app",)

    def __init__(self, cls, app: "CapeTuiApp"):
        # Initialize hierarchy
        CfdxCompleter.__init__(self, cls)
        # Save the app
        self.app = app

    def get_line_buffer(self) -> str:
        r"""Get the editor text before the cursor

        :Call:
            >>> line = comp.get_line_buffer()
        :Inputs:
            *comp*: :class:`CapeTuiCompleter`
                TUI autocompleter
        :Outputs:
            *line*: :class:`str`
                Command input text, truncated at the cursor
        """
        # Get current text and cursor position
        value = self.app._input.value
        pos = self.app._input.cursor_position
        # Output text up to the cursor
        return value[:pos]

    def genr8_suggestions(self, text: str) -> list[str]:
        r"""Generate suggestions, including TUI meta-commands

        :Call:
            >>> suggestions = comp.genr8_suggestions(text)
        :Inputs:
            *comp*: :class:`CapeTuiCompleter`
                TUI autocompleter
            *text*: :class:`str`
                Current text of current word
        :Outputs:
            *suggestions*: :class:`list`\ [:class:`str`]
                List of suggested completions for current word
        """
        # Get current line
        line = self.get_line_buffer()
        # Check for meta-command completion on first word
        if text.startswith(":") and line.lstrip().startswith(text):
            # Complete TUI meta-command names
            self.role = "metacmd"
            return fnmatch.filter(META_CMDS, f"{text}*")
        # Defer to CAPE front-desk completions
        return CfdxCompleter.genr8_suggestions(self, text)


# File-like object that posts written lines to the scroll log
class LogWriter(io.TextIOBase):
    r"""Redirect writes (e.g. :data:`sys.stdout`) into a RichLog

    Lines are buffered; complete lines are posted to the app's scroll
    log from the writing thread via ``call_from_thread``. ANSI color
    codes are converted to Rich :class:`Text` styling.

    :Attributes:
        *app*: :class:`CapeTuiApp`
            App whose log receives written lines
        *_buf*: :class:`str`
            Buffer holding an incomplete line
    """

    def __init__(self, app: "CapeTuiApp"):
        # Initialize hierarchy
        super().__init__()
        # Save attributes
        self.app = app
        self._buf = ""

    def write(self, s: str) -> int:
        # Append to buffer
        self._buf += s
        # Post any complete lines
        while "\n" in self._buf:
            # Split off one line
            line, self._buf = self._buf.split("\n", 1)
            # Post it to the log, preserving ANSI colors
            self.app._post_to_log(Text.from_ansi(line))
        # Report number of characters accepted
        return len(s)

    def flush(self) -> None:
        # Post any incomplete trailing line
        if self._buf:
            buf, self._buf = self._buf, ""
            self.app._post_to_log(Text.from_ansi(buf))


class CommandLog(RichLog):
    r"""Scroll log whose command markers fold their associated output.

    Text selection is implemented here because :class:`RichLog` renders
    its lines to cached strips: by itself it neither highlights the
    selection nor can it extract the selected text for copying.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._groups = []
        self._entries = []
        self._active_group = None
        self._header_lines = {}
        self._replaying = False

    def render_line(self, y: int):
        strip = super().render_line(y)
        scroll_x, scroll_y = self.scroll_offset
        line = scroll_y + y
        # Tag segments with content offsets so the screen can resolve
        # the pointer position to text coordinates during a drag.
        strip = strip.apply_offsets(scroll_x, line)
        selection = self.text_selection
        if selection is not None:
            strip = self._apply_selection_style(strip, line, scroll_x)
        return strip

    def _apply_selection_style(self, strip, line: int, scroll_x: int):
        selection = self.text_selection
        span = selection.get_span(line)
        if span is None:
            return strip
        start, end = span
        start = max(0, start - scroll_x)
        if end < 0:
            end = strip.cell_length
        else:
            end = min(strip.cell_length, end - scroll_x)
        if end <= start or start >= strip.cell_length:
            return strip
        sel_style = self.screen.get_component_rich_style("screen--selection")
        parts = strip.divide([start, end, strip.cell_length])
        # Strip.apply_style() treats its argument as a base style, so the
        # log's existing foreground/background win. Apply selection as a
        # post-style instead so its colors visibly override log styling.
        selected = Strip(
            Segment.apply_style(parts[1], post_style=sel_style),
            parts[1].cell_length)
        out = parts[0] + selected
        if len(parts) > 2:
            out = out + parts[2]
        return out

    def get_selection(self, selection):
        # Match each rendered line against its source text; the strips
        # are space-padded, so trailing padding is not copied.
        lines = [strip.text.rstrip() for strip in self.lines]
        return selection.extract("\n".join(lines)), "\n"

    def write_command(self, cmd: str, header: Text, folded: Text) -> None:
        group = {"cmd": cmd, "header": header, "folded": folded,
                 "output": [], "collapsed": False}
        self._groups.append(group)
        self._entries.append(group)
        self._active_group = group
        self._write_header(len(self._groups) - 1)

    def start_section(
            self,
            name: str,
            header: Text,
            folded: Text,
            collapsed: bool = False) -> None:
        r"""Start a separately foldable output section."""
        group = {
            "cmd": name,
            "header": header,
            "folded": folded,
            "output": [],
            "collapsed": collapsed,
        }
        self._groups.append(group)
        self._entries.append(group)
        self._active_group = group
        self._write_header(len(self._groups) - 1)

    def end_group(self) -> None:
        r"""Route subsequent writes to the top-level transcript."""
        self._active_group = None

    def _write_header(self, index: int) -> None:
        group = self._groups[index]
        start = len(self.lines)
        super().write(group["folded"] if group["collapsed"]
                      else group["header"])
        for line in range(start, len(self.lines)):
            self._header_lines[line] = index

    def write(self, content, *args, **kwargs):
        if self._active_group is not None and not self._replaying:
            group = self._active_group
            group["output"].append((content, args, kwargs))
            if group["collapsed"]:
                return self
        elif not self._replaying:
            self._entries.append((content, args, kwargs))
        return super().write(content, *args, **kwargs)

    def clear(self):
        if not self._replaying:
            self._groups.clear()
            self._entries.clear()
            self._active_group = None
        self._header_lines.clear()
        return super().clear()

    def on_click(self, event) -> None:
        # Only the blue chevron is an affordance, not the whole log row.
        if event.x != self.styles.padding.left:
            return
        line = int(self.scroll_y) + event.y - self.styles.padding.top
        index = self._header_lines.get(line)
        if index is None:
            return
        self._groups[index]["collapsed"] ^= True
        old_y = self.scroll_y
        self._replaying = True
        try:
            super().clear()
            self._header_lines.clear()
            for entry in self._entries:
                if isinstance(entry, dict):
                    j = self._groups.index(entry)
                    self._write_header(j)
                    if not entry["collapsed"]:
                        for content, args, kwargs in entry["output"]:
                            super().write(content, *args, **kwargs)
                else:
                    content, args, kwargs = entry
                    super().write(content, *args, **kwargs)
        finally:
            self._replaying = False
        self.scroll_to(y=old_y, animate=False, immediate=True)


# Main CAPE TUI app
class CapeTuiApp(App):
    r"""OpenCode-style CAPE terminal user interface

    A scroll log plus a pinned command composer. Commands run in
    worker threads: CAPE CLI commands in-process with redirected
    :data:`sys.stdout`/:data:`sys.stderr`, other commands as
    subprocesses in their own process group. Interactive prompts mount
    clickable widgets above the editor while the worker waits.

    :Attributes:
        *_log*: :class:`textual.widgets.RichLog`
            Scroll log of commands and output
        *_input*: :class:`textual.widgets.Input`
            Command editor
        *_status*: :class:`textual.widgets.Static`
            One-line status bar below the composer
        *_prompt_widget*: ``None`` | :class:`textual.widget.Widget`
            Currently mounted interactive prompt, if any
        *_completer*: ``None`` | :class:`CapeTuiCompleter`
            Tab-completer for the editor
        *_proc*: ``None`` | :class:`subprocess.Popen`
            Currently running external command, if any
        *_worker*: ``None`` | :class:`threading.Thread`
            Currently running command's worker thread, if any
        *_running_cmd*: ``None`` | :class:`str`
            Text of the command currently running, if any
        *_histfile*: ``None`` | :class:`str`
            Name of the command history file
        *_json_files*: :class:`tuple`\ [:class:`str`]
            Successfully loaded JSON files, least to most recently used
        *_last_json_file*: ``None`` | :class:`str`
            Most recently used JSON file
        *_last_json_display_file*: ``None`` | :class:`str`
            Most recent JSON file relative to its controller's root
        *_history*: :class:`list`\ [:class:`str`]
            Session command history (oldest first)
        *_hist_ix*: ``None`` | :class:`int`
            Index being browsed during history recall, if any
        *_hist_draft*: :class:`str`
            Editor text saved when history browsing started
        *_stats*: :class:`dict`
            Session statistics (``commands``, ``failures``, etc.)
        *_t0_cmd*: :class:`float`
            Start time of the running command
        *_t0_session*: :class:`float`
            Start time of the session
        *_spin_ix*: :class:`int`
            Current frame of the busy spinner
        *_spin_timer*: ``None`` | :class:`textual.timer.Timer`
            Timer advancing the busy spinner
    """

    # Key bindings: priority so they win over defaults, gated by
    # check_action() so prompts and other widgets keep their own keys
    BINDINGS = [
        Binding(
            "ctrl+d", "quit", "Quit",
            show=False, priority=True),
        Binding(
            "ctrl+c", "interrupt", "Interrupt",
            show=False, priority=True),
        Binding(
            "escape", "interrupt", "Interrupt",
            show=False, priority=True),
        Binding(
            "tab", "tab_complete", "Complete",
            show=False, priority=True),
        Binding(
            "up", "history_prev", "Previous command",
            show=False, priority=True),
        Binding(
            "down", "history_next", "Next command",
            show=False, priority=True),
        Binding(
            "enter", "accept_suggestion", "Use suggestion",
            show=False, priority=True),
        Binding(
            "ctrl+p", "command_palette", "Commands",
            show=False, priority=True),
    ]

    # Style settings
    CSS = """
    Screen > .screen--selection {
        background: #264f78;
        color: #ffffff;
    }
    App, #body, RichLog {
        background: #0d0d0d;
        color: #d6d6d6;
    }
    #body {
        height: 100%;
    }
    RichLog {
        height: 1fr;
        padding: 1 2 0 2;
        scrollbar-color: #343434;
        scrollbar-color-hover: #565f89;
    }
    #suggestions {
        display: none;
        width: 64;
        max-width: 90%;
        height: auto;
        max-height: 9;
        margin: 0 2;
        padding: 0 1;
        background: #202020;
        border-left: solid #7aa2f7;
        color: #d6d6d6;
    }
    #suggestions > .option-list--option-highlighted {
        background: #343a48;
    }
    #composer {
        height: 4;
        margin: 0 2;
        padding: 1 1 0 1;
        background: #202020;
        border-left: solid #7aa2f7;
    }
    #composer.busy {
        border-left: solid #ff9e64;
    }
    #prompt-input {
        height: 1;
        border: none;
        background: transparent;
        color: #eeeeee;
        padding: 0;
    }
    #composer-hint {
        height: 1;
        color: #858585;
    }
    #status-bar {
        height: 2;
        margin: 0 2;
        padding: 0 0 1 0;
        background: #0d0d0d;
        color: #888888;
    }
    """

    # Presentation hooks for related CAPE Textual interfaces
    APP_TITLE = "CAPE TUI"
    INPUT_PLACEHOLDER = INPUT_PLACEHOLDER
    CONTEXT_LABEL = "CAPE"

    def __init__(
            self,
            cls: type,
            *a,
            histfile: Optional[str] = None,
            **kw):
        r"""Create the app with *cls* as the CAPE front-desk parser"""
        # Initialize hierarchy
        super().__init__(*a, **kw)
        # Save the front-desk class (for completions and :help)
        self._frontdesk_cls = cls
        # Optional history-file override for related interfaces
        self._histfile_override = histfile

    def compose(self) -> ComposeResult:
        with Vertical(id="body"):
            yield CommandLog(id="log", auto_scroll=True)
            yield OptionList(id="suggestions")
            with Vertical(id="composer"):
                yield Input(
                    placeholder=self.INPUT_PLACEHOLDER,
                    id="prompt-input")
                yield Static(COMPOSER_HINT, id="composer-hint")
            yield Static(id="status-bar")

    def on_mount(self) -> None:
        # Theme and window title
        self.title = self.APP_TITLE
        # Save widget references
        self._log = self.query_one("#log", CommandLog)
        self._suggestions = self.query_one("#suggestions", OptionList)
        self._suggestion_matches = []
        self._suggestion_span = (0, 0)
        self._input = self.query_one("#prompt-input", Input)
        self._composer = self.query_one("#composer", Vertical)
        self._composer_hint = self.query_one("#composer-hint", Static)
        self._status = self.query_one("#status-bar", Static)
        # No mounted prompt or running command so far
        self._prompt_widget = None
        self._proc = None
        self._worker = None
        self._running_cmd = None
        self._last_exit = None
        # Session clock and statistics
        self._t0_cmd = 0.0
        self._t0_session = time.perf_counter()
        self._stats = {
            "commands": 0,
            "failures": 0,
            "tui_commands": 0,
            "duration": 0.0,
            "json_files": (),
            "last_json_file": None,
            "last_json_display_file": None,
        }
        self._spin_ix = 0
        self._spin_timer = None
        # History file and browsing state
        self._histfile = self._histfile_override or get_tui_histfile()
        self._hist_ix = None
        self._hist_draft = ""
        self._load_history()
        # CAPE commands run in threads and share the CLI controller cache.
        from ..cfdx import cli
        self._sync_json_cache(tuple(cli.CNTL_CACHE))
        # Create tab-completer hooked to the editor
        self._completer = self._make_completer()
        # Track result data; visible context and hints are below the composer
        self._input.border_title = self._context_title()
        self._update_status()
        self.call_after_refresh(self._update_status)
        # Focus the editor
        self._input.focus()

    # Create the editor completer; subclasses may narrow its behavior
    def _make_completer(self):
        return CapeTuiCompleter(self._frontdesk_cls, self)

    # Finalize session statistics
    def finalize_stats(self) -> dict:
        r"""Compute the session duration and return the stats dict"""
        self._stats["duration"] = time.perf_counter() - self._t0_session
        return self._stats

    # Load the command history file
    def _load_history(self) -> None:
        # Default to empty history
        self._history = []
        # Read the history file if it exists
        try:
            with open(self._histfile, encoding="utf-8") as fp:
                lines = fp.read().splitlines()
        except OSError:
            return
        # Keep the most recent non-empty entries
        lines = [line for line in lines if line.strip()]
        self._history = lines[-CAPE_HISTORY_LENGTH:]

    # Record one submitted line in memory and in the history file
    def _record_history(self, cmd: str) -> None:
        # Append to the session history
        self._history.append(cmd)
        if len(self._history) > CAPE_HISTORY_LENGTH:
            self._history = self._history[-CAPE_HISTORY_LENGTH:]
        # Append to the history file (crash-safe)
        try:
            folder = os.path.dirname(self._histfile)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(self._histfile, "a", encoding="utf-8") as fp:
                fp.write(cmd + "\n")
        except OSError:
            pass

    # Rewrite the (trimmed) history file; called on exit
    def save_history(self) -> None:
        r"""Write the (trimmed) session history back to the file"""
        try:
            folder = os.path.dirname(self._histfile)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(self._histfile, "w", encoding="utf-8") as fp:
                for cmd in self._history:
                    fp.write(cmd + "\n")
        except OSError:
            pass

    # Post one line to the scroll log from another thread
    def _post_to_log(self, txt) -> None:
        self.call_from_thread(self._log.write, txt)

    def _sync_json_cache(self, paths: tuple[str, ...]) -> None:
        r"""Record the CLI cache's JSON files in recency order

        :Call:
            >>> app._sync_json_cache(paths)
        :Inputs:
            *paths*: :class:`tuple`\ [:class:`str`]
                Absolute JSON paths, least to most recently used
        :Outputs:
            ``None``
        """
        self._json_files = paths
        self._last_json_file = paths[-1] if paths else None
        self._last_json_display_file = None
        if self._last_json_file is not None:
            # Keep the canonical path for tracking, but show the path
            # relative to the controller's run-matrix root.
            from ..cfdx import cli
            entry = cli.CNTL_CACHE.get(self._last_json_file)
            root = getattr(entry[1], "RootDir", None) if entry else None
            if root:
                try:
                    self._last_json_display_file = os.path.relpath(
                        self._last_json_file, root)
                except ValueError:
                    pass
            if self._last_json_display_file is None:
                self._last_json_display_file = self._last_json_file
        self._stats["json_files"] = paths
        self._stats["last_json_file"] = self._last_json_file
        self._stats["last_json_display_file"] = \
            self._last_json_display_file
        self._update_composer_hint()

    def _update_composer_hint(self) -> None:
        r"""Show completion keys or the most recently used CAPE JSON

        :Call:
            >>> app._update_composer_hint()
        :Outputs:
            ``None``
        """
        if self._suggestions.display:
            hint = SUGGESTION_HINT
        elif self._last_json_display_file:
            hint = Text.assemble(
                ("CAPE file: ", "#858585"),
                (self._last_json_display_file, TN_BLUE))
        else:
            hint = COMPOSER_HINT
        self._composer_hint.update(hint)

    # Show a result in the editor frame (subtitle and class)
    def _show_result(self, ierr: int) -> None:
        # Keep the last result available on the input widget
        self._last_exit = ierr
        self._input.set_class(ierr == 0, "ok")
        self._input.set_class(ierr != 0, "fail")
        icon = "✓" if ierr == 0 else "✗"
        self._input.border_subtitle = f" {icon} exit {ierr} "
        self._input.border_title = self._context_title()
        self._update_status()

    # Re-enable the editor after a command finishes
    def _set_idle(self, ierr: int = 0) -> None:
        # Stop the busy spinner
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
        self._running_cmd = None
        self._input.remove_class("busy")
        self._composer.remove_class("busy")
        # Show the result in the editor frame
        self._show_result(ierr)
        # Re-enable the editor
        self._input.disabled = False
        self._input.placeholder = self.INPUT_PLACEHOLDER
        self._input.focus()

    # Short host and folder context for the input widget
    def _context_title(self) -> str:
        hostname = socket.gethostname().split('.')[0]
        return f" {self.CONTEXT_LABEL} {hostname}:{get_dirname()} "

    # Render the status bar for the current state
    def _update_status(self) -> None:
        # Available width (may be 0 before layout)
        width = max(10, self._status.size.width)
        # Check state
        if self._running_cmd is None:
            # Full path below the composer, with command discovery on right
            left = Text(os.getcwd(), style="#858585")
            right = Text(HINTS_IDLE, style="#d6d6d6")
            if self._last_exit is not None:
                color = TN_GREEN if self._last_exit == 0 else TN_RED
                icon = "✓" if self._last_exit == 0 else "✗"
                right = Text.assemble(
                    (f"{icon} exit {self._last_exit}  ", color), right)
        else:
            # Busy: spinning frame, command text, wall time
            frame = SPINNER_FRAMES[self._spin_ix % len(SPINNER_FRAMES)]
            dt = time.perf_counter() - self._t0_cmd
            left = Text.assemble(
                (f"{frame} ", f"bold {TN_ORANGE}"),
                (f"running '{self._running_cmd}' · {dt:.1f}s", ""))
            right = Text(HINTS_BUSY, style=TN_ORANGE)
        # Preserve the command hint on narrow terminals; keep the useful
        # tail of the path or running command when space is tight.
        available = max(1, width - len(right) - 1)
        if len(left) > available:
            tail = left.plain[-(available - 1):] if available > 1 else ""
            left = Text("…" + tail,
                        style=TN_ORANGE if self._running_cmd else "#858585")
        # Pad the gap between left and right
        pad = max(1, width - len(left) - len(right))
        self._status.update(Text.assemble(left, " " * pad, right))

    # Advance the busy spinner one frame
    def _tick_spinner(self) -> None:
        self._spin_ix = (self._spin_ix + 1) % len(SPINNER_FRAMES)
        self._update_status()

    # Keep the status bar padded on resize
    def on_resize(self, event) -> None:
        # Resize may fire before on_mount
        if hasattr(self, "_status"):
            self._update_status()

    # OpenCode-style copy: a mouse-drag selection goes to the clipboard
    def on_text_selected(self, event) -> None:
        text = self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)

    # Command echo rendered as an OpenCode-style message bubble
    def _bubble_text(self, cmd: str, folded: bool = False) -> Text:
        # Pad to the log's text width (2 padding + 1 scrollbar)
        head = "▸ " if folded else "❯ "
        width = max(10, self._log.size.width - 3)
        pad = max(1, width - len(head) - len(cmd))
        return Text.assemble(
            (head, f"bold {TN_BLUE} on {TN_SURFACE}"),
            (cmd + " " * pad, f"on {TN_SURFACE}"))

    def _write_command_header(self, cmd: str) -> None:
        r"""Write the submitted command and open its output group."""
        self._log.write_command(
            cmd, self._bubble_text(cmd), self._bubble_text(cmd, folded=True))

    # Exit-status line rendered as a thin, dim rule
    def _status_rule_text(self, ierr: int, dt: float) -> Text:
        ok = ierr == 0
        color = TN_GREEN if ok else TN_RED
        icon = "✓" if ok else "✗"
        label = f" {icon} exit {ierr} · {dt:.2f}s "
        width = max(10, self._log.size.width - 3)
        ndash = max(2, width - len(label) - 2)
        return Text.assemble(
            ("──", TN_DIM), (label, color), ("─" * ndash, TN_DIM))

    # Gate key bindings so prompts keep their own TAB and Ctrl-C
    def check_action(
            self,
            action: str,
            parameters: Tuple[object, ...]) -> Optional[bool]:
        # Let a modal screen handle its own keys (notably Escape/Ctrl-P).
        if isinstance(self.screen, CommandPalette):
            return False
        if action == "accept_suggestion":
            return self._suggestions.display and self._prompt_widget is None
        if action == "command_palette" and self._running_cmd is not None:
            return False
        # While a prompt is mounted, it owns TAB, Ctrl-C, and arrows
        if action in (
                "tab_complete", "interrupt", "history_prev",
                "history_next", "command_palette"):
            if self._prompt_widget is not None:
                return False
            # TAB and history recall only work in the active editor
            if action not in ("interrupt", "command_palette"):
                return self._input.has_focus and not self._input.disabled
        # All other actions enabled
        return True

    def action_command_palette(self) -> None:
        r"""Open a picker for built-in TUI commands (Ctrl-P)."""
        self.push_screen(CommandPalette(), self._palette_selected)

    def _palette_selected(self, cmd: Optional[str]) -> None:
        # Selecting a command executes it just like typing it in the editor.
        if cmd is not None:
            self._submit_command(cmd)
        if not self._input.disabled:
            self._input.focus()

    # Complete the current word of the editor
    def action_tab_complete(self) -> None:
        r"""Complete the word left of the cursor (TAB action)

        Completions come from :class:`CapeTuiCompleter`. A unique
        match is inserted directly; multiple matches open a picker
        directly above the command editor.
        """
        if self._suggestions.display:
            self.action_accept_suggestion()
            return
        self._show_suggestions()

    def _completion_span(self) -> tuple[int, int]:
        """Return the current word's start and the cursor position."""
        # Get the word left of the cursor and its start index
        value = self._input.value
        pos = self._input.cursor_position
        start = pos
        while start > 0 and value[start - 1] not in " \t\n":
            start -= 1
        return start, pos

    def _completion_matches(self) -> tuple[int, int, list[str]]:
        start, pos = self._completion_span()
        value = self._input.value
        text = value[start:pos]
        # Generate suggestions; tolerate partial/quoted input
        try:
            matches = self._completer.get_suggestions(text)
        except Exception:
            matches = []
        return start, pos, list(dict.fromkeys(matches))

    def _show_suggestions(self) -> None:
        start, pos, matches = self._completion_matches()
        # No completions
        if not matches:
            self._hide_suggestions()
            self.bell()
            return
        # Unique match: insert it (with trailing ' ' or os.sep)
        if len(matches) == 1:
            self._accept_match(matches[0], start, pos)
            return
        self._suggestion_matches = matches
        self._suggestion_span = (start, pos)
        self._suggestions.set_options(
            Option(match.rstrip(), id=str(j))
            for j, match in enumerate(matches))
        self._suggestions.highlighted = 0
        self._suggestions.display = True
        self._update_composer_hint()

    def _hide_suggestions(self) -> None:
        self._suggestions.display = False
        self._suggestion_matches = []
        self._update_composer_hint()

    def _accept_match(self, match: str, start: int, pos: int) -> None:
        # Unique matches from get_suggestions() already have a suffix.
        if not match.endswith((" ", os.sep)):
            role = self._completer.role
            if role == "filename" and os.path.isdir(match):
                match += os.sep
            else:
                match += " "
        self._hide_suggestions()
        self._replace_word(start, pos, match)
        self._hist_ix = None
        self._input.focus()

    def action_accept_suggestion(self) -> None:
        if not self._suggestions.display:
            return
        if self._completion_span() != self._suggestion_span:
            self._hide_suggestions()
            return
        index = self._suggestions.highlighted
        if index is None:
            index = 0
        match = self._suggestion_matches[index]
        start, pos = self._suggestion_span
        self._accept_match(match, start, pos)

    def on_option_list_option_selected(
            self, event: OptionList.OptionSelected) -> None:
        if event.option_list is self._suggestions:
            event.stop()
            self._suggestions.highlighted = event.option_index
            self.action_accept_suggestion()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is self._input and self._suggestions.display:
            # Refresh the candidates as the user edits the word.
            start, pos, matches = self._completion_matches()
            if not matches:
                self._hide_suggestions()
                return
            self._suggestion_matches = matches
            self._suggestion_span = (start, pos)
            self._suggestions.set_options(
                Option(match.rstrip(), id=str(j))
                for j, match in enumerate(matches))
            self._suggestions.highlighted = 0

    # Insert a completion, replacing the current word
    def _replace_word(self, start: int, pos: int, match: str) -> None:
        value = self._input.value
        self._input.value = value[:start] + match + value[pos:]
        self._input.cursor_position = start + len(match)

    # Recall the previous history entry (up-arrow action)
    def action_history_prev(self) -> None:
        if self._suggestions.display:
            n = self._suggestions.option_count
            if n:
                index = self._suggestions.highlighted or 0
                self._suggestions.highlighted = (index - 1) % n
            return
        # No history to browse
        if not self._history:
            self.bell()
            return
        # Start browsing, or step to an older entry
        if self._hist_ix is None:
            # Save the current editor text as the draft
            self._hist_draft = self._input.value
            self._hist_ix = len(self._history) - 1
        elif self._hist_ix > 0:
            self._hist_ix -= 1
        else:
            # Already at the oldest entry
            self.bell()
            return
        # Show the entry
        self._input.value = self._history[self._hist_ix]
        self._input.cursor_position = len(self._input.value)

    # Recall the next history entry (down-arrow action)
    def action_history_next(self) -> None:
        if self._suggestions.display:
            n = self._suggestions.option_count
            if n:
                index = self._suggestions.highlighted or 0
                self._suggestions.highlighted = (index + 1) % n
            return
        # Not browsing
        if self._hist_ix is None:
            self.bell()
            return
        # Step to a newer entry, or back to the draft
        if self._hist_ix < len(self._history) - 1:
            self._hist_ix += 1
            self._input.value = self._history[self._hist_ix]
        else:
            self._hist_ix = None
            self._input.value = self._hist_draft
        # Move cursor to the end
        self._input.cursor_position = len(self._input.value)

    # Interrupt the running command (Ctrl-C / ESC action)
    def action_interrupt(self) -> None:
        r"""Interrupt the running command, or clear the editor

        External commands run in their own process group and are sent
        ``SIGINT``; in-process CAPE commands get a
        :class:`KeyboardInterrupt` injected into the worker thread.
        With nothing running, the editor is cleared.
        """
        if self._suggestions.display:
            self._hide_suggestions()
            self._input.focus()
            return
        # Interrupt a running external command
        proc = self._proc
        if (proc is not None) and (proc.poll() is None):
            self._log.write(Text("^C", style=f"bold {TN_RED}"))
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGINT)
                else:
                    proc.send_signal(signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass
            return
        # Interrupt a running in-process CAPE command
        worker = self._worker
        if (worker is not None) and worker.is_alive():
            self._log.write(Text("^C", style=f"bold {TN_RED}"))
            _raise_in_thread(worker, KeyboardInterrupt)
            return
        # Idle: clear the editor
        self._input.value = ""

    # Run one submitted line
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._suggestions.display:
            self.action_accept_suggestion()
            return
        self._submit_command(event.value.strip())

    def _submit_command(self, cmd: str) -> None:
        # Get command text and clear the editor
        self._input.value = ""
        # Check for empty command
        if not cmd:
            return
        # Record history and reset browsing
        self._record_history(cmd)
        self._hist_ix = None
        # Echo the command in the log as a bubble
        self._write_command_header(cmd)
        # Check for exit commands
        if cmd in EXIT_CMDS:
            self.exit()
            return
        # TUI meta-commands (except worker-based ":prompt-demo")
        if cmd.startswith(":") and cmd != ":prompt-demo":
            self._run_meta(cmd)
            return
        # Folder-change commands
        if cmd == "cd" or cmd.startswith("cd "):
            self._stats["commands"] += 1
            ierr = self._run_cd_text(cmd)
            self._stats["failures"] += int(ierr != 0)
            self._show_result(ierr)
            return
        # Run in a worker (in-process CAPE CLI or subprocess)
        self._start_command(cmd)

    # Start a command's busy chrome and worker thread
    def _start_command(self, cmd: str) -> None:
        # Count it
        self._stats["commands"] += 1
        # Busy indicators: accent bar and spinner in the status bar
        self._input.disabled = True
        self._input.placeholder = "running..."
        self._input.remove_class("ok", "fail")
        self._input.add_class("busy")
        self._composer.add_class("busy")
        self._running_cmd = cmd
        self._t0_cmd = time.perf_counter()
        self._spin_ix = 0
        self._spin_timer = self.set_interval(
            SPINNER_INTERVAL, self._tick_spinner)
        self._update_status()
        # Run the command in a worker thread
        thread = threading.Thread(
            target=self._run_command, args=(cmd,), daemon=True)
        self._worker = thread
        thread.start()

    # Worker thread entry point for one command
    def _run_command(self, cmd: str) -> None:
        # Start timer
        t0 = time.perf_counter()
        # Run the command; never crash the thread
        try:
            ierr = self._execute_command(cmd)
        except KeyboardInterrupt:
            self._post_to_log(Text("KeyboardInterrupt",
                                   style=f"bold {TN_RED}"))
            ierr = 130
        except Exception:
            # Unexpected error: show the traceback in the log
            for line in traceback.format_exc().rstrip().split("\n"):
                self._post_to_log(Text(line, style=TN_RED))
            ierr = 128
        # Wall time
        dt = time.perf_counter() - t0
        # Command is no longer running
        self._worker = None
        self._stats["failures"] += int(ierr != 0)
        # Status line
        self._post_to_log(self._status_rule_text(ierr, dt))
        # Re-enable the editor
        self.call_from_thread(self._set_idle, ierr)

    # Execute one command; subclasses can provide another backend
    def _execute_command(self, cmd: str) -> int:
        if cmd == ":prompt-demo":
            return self._run_prompt_demo()
        elif REGEX_CAPE_CLI.match(cmd):
            return self._run_cape_cli(cmd)
        return self._run_subprocess(cmd)

    # Run an in-process CAPE CLI command with redirected output
    def _run_cape_cli(self, cmd: str) -> int:
        # Delayed import of CAPE CLI (to avoid circular imports)
        from ..cfdx import cli
        # Split the command
        parts = shlex.split(cmd)
        # Run with stdout/stderr redirected into the log
        writer = LogWriter(self)
        stdout_old, stderr_old = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            ierr = cli.main(argv=parts)
        finally:
            sys.stdout = stdout_old
            sys.stderr = stderr_old
            writer.flush()
            # Worker threads share this module cache with the TUI, but UI
            # state must be updated on Textual's app thread.
            self.call_from_thread(
                self._sync_json_cache, tuple(cli.CNTL_CACHE))
        # Output
        return int(ierr or 0)

    # Run an external command as a subprocess
    def _run_subprocess(self, cmd: str) -> int:
        # Use a new process group (POSIX) so Ctrl-C reaches children too
        popen_kw = {}
        if os.name == "posix":
            popen_kw["start_new_session"] = True
        # Start the command
        try:
            proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, errors="replace",
                **popen_kw)
        except FileNotFoundError:
            self._post_to_log(Text(
                f"Command not found: '{cmd.split()[0]}'",
                style=f"bold {TN_RED}"))
            return 127
        except PermissionError:
            self._post_to_log(Text(
                f"Permission denied: '{cmd.split()[0]}'",
                style=f"bold {TN_RED}"))
            return 13
        # Remember the running command (for Ctrl-C)
        self._proc = proc
        try:
            # Stream output lines into the log
            for line in proc.stdout or []:
                self._post_to_log(Text.from_ansi(line.rstrip("\n")))
            # Wait for the process
            ierr = proc.wait()
        finally:
            self._proc = None
        # Map signal terminations to 128 + signo (e.g. SIGINT -> 130)
        if ierr < 0:
            ierr = 128 - ierr
        # Output
        return ierr

    # Run a demo command that triggers an interactive prompt
    def _run_prompt_demo(self) -> int:
        # CAPE imports
        from .. import promptutils
        # Redirect output so the confirmation lands in the log
        writer = LogWriter(self)
        stdout_old = sys.stdout
        try:
            sys.stdout = writer
            v = promptutils.prompt_color(
                "Pick an action (click one)", "skip",
                ["next", "extend", "skip"], prompt="tui>")
            # Show what was resolved
            print(f"demo answer: {v}")
        finally:
            sys.stdout = stdout_old
            writer.flush()
        # Output
        return 0

    # Registered prompt handler; called from worker threads
    def _handle_prompt(
            self,
            txt: str,
            vdef,
            vopt,
            prompt: str = '>',
            oneline: bool = False) -> str:
        # Communicate with the app thread
        ev = threading.Event()
        result = {}

        # Callback receives the user's answer and releases the worker
        def on_answer(vraw) -> None:
            result["v"] = vraw
            ev.set()

        # Mount the prompt widget above the editor
        def mount() -> None:
            widget = _new_prompt_widget(
                txt, vdef, vopt, prompt, oneline, on_answer=on_answer)
            self._prompt_widget = widget
            body = self.query_one("#body", Vertical)
            body.mount(widget, before=self._composer)

        # Mount from the worker thread and wait for the answer
        self.call_from_thread(mount)
        ev.wait()

        # Remove the widget from the stream
        def unmount() -> None:
            widget, self._prompt_widget = self._prompt_widget, None
            if widget is not None:
                widget.remove()
            self._input.focus()

        self.call_from_thread(unmount)
        # Check for cancel (Ctrl-C in the prompt)
        vraw = result["v"]
        if vraw is None:
            raise KeyboardInterrupt
        # Output
        return vraw

    # Handle a TUI meta-command starting with ":"
    def _run_meta(self, cmd: str) -> None:
        # Count it
        self._stats["tui_commands"] += 1
        # Split meta-command
        try:
            parts = shlex.split(cmd)
        except ValueError as err:
            self._log.write(Text(str(err), style=f"bold {TN_RED}"))
            self._show_result(16)
            return
        # Name of meta-command and first argument
        metacmd = parts[0]
        arg = parts[1] if len(parts) > 1 else None
        # Rerun a command from history (runs like a normal command)
        if metacmd.startswith(":!"):
            self._meta_rerun(metacmd[2:])
            return
        # All other meta-commands produce immediate output
        ierr = self._meta_simple(metacmd, arg)
        self._show_result(ierr)

    # Handle the immediate (non-rerun) meta-commands
    def _meta_simple(self, metacmd: str, arg: Optional[str]) -> int:
        if metacmd in (":exit", ":quit"):
            self.exit()
            return 0
        elif metacmd == ":clear":
            self._log.clear()
            return 0
        elif metacmd == ":pwd":
            self._log.write(Text(os.getcwd(), style="cyan"))
            return 0
        elif metacmd == ":cd":
            # Change to folder *arg*, or home folder
            return self._run_cd_text(f"cd {arg}" if arg else "cd ~")
        elif metacmd == ":help":
            return self._meta_help(arg)
        elif metacmd == ":history":
            # Optional count of history entries
            try:
                n = int(arg) if arg else 25
            except ValueError:
                self._log.write(Text(
                    f"Bad history count: '{arg}'", style=f"bold {TN_RED}"))
                return 1
            self._log.write(history_table(self._history, n))
            self._log.write(Text(
                "Use :!N to rerun command No. N", style="cyan"))
            return 0
        elif metacmd == ":status":
            stats = dict(self._stats)
            stats["duration"] = time.perf_counter() - self._t0_session
            self._log.write(session_stats_panel(stats, self._histfile))
            return 0
        # Unknown meta-command
        metacmds = " ".join(META_CMDS)
        self._log.write(Text(
            f"Unrecognized TUI command: '{metacmd}'",
            style=f"bold {TN_RED}"))
        self._log.write(Text(f"Try one of: {metacmds}"))
        return 16

    # Render help for CAPE commands or TUI meta-commands
    def _meta_help(self, arg: Optional[str]) -> int:
        # Help about the TUI itself
        if arg in META_HELP_TOPICS:
            self._log.write(meta_help_table())
            self._log.write(Text(
                "Drag the mouse over log text to select; "
                "the selection is copied to the clipboard",
                style="cyan"))
            return 0
        # Full command table
        cls = self._frontdesk_cls
        if arg is None:
            self._log.write(cmd_table(cls))
            self._log.write(Text(
                "Use :help <cmd> or cape <cmd> -h for details",
                style="cyan"))
            return 0
        # Check alternate names
        cmdname = cls._cmdmap.get(arg, arg)
        # Get subparser
        subcls = cls._cmdparsers.get(cmdname)
        # Check for unknown command
        if subcls is None:
            self._log.write(Text(
                f"Unknown CAPE command: '{arg}'", style=f"bold {TN_RED}"))
            return 16
        # Render details for one command
        self._log.write(cmd_help_panel(cmdname, subcls, cls))
        return 0

    # Rerun command No. N from the history
    def _meta_rerun(self, txt: str) -> None:
        # Parse the index
        try:
            j = int(txt)
        except ValueError:
            self._log.write(Text(
                f"Bad history entry number: '{txt}'",
                style=f"bold {TN_RED}"))
            self._show_result(16)
            return
        # Check range (the ":!N" line itself is already recorded)
        nhist = len(self._history)
        if j < 1 or j > nhist:
            self._log.write(Text(
                f"History entry out of range 1:{nhist}: {j}",
                style=f"bold {TN_RED}"))
            self._show_result(16)
            return
        # Get the command
        cmd = self._history[j - 1]
        # Check for meta-command
        if cmd.startswith(":"):
            self._log.write(Text(
                f"Cannot rerun TUI command: {cmd}", style=f"bold {TN_RED}"))
            self._show_result(16)
            return
        # Echo the rerun command as a bubble
        self._log.write(Text(
            f"Rerunning history entry {j}:", style="cyan"))
        self._log.write(self._bubble_text(cmd))
        # Run it like a freshly submitted command
        self._start_command(cmd)

    # Handle folder-change commands like ``cd powerless/``
    def _run_cd_text(self, user_message: str) -> int:
        # Split off the folder name
        parts = user_message.split(' ', 1)
        # Get folder
        target = os.path.expanduser(parts[1]) if len(parts) > 1 else "~"
        # Change folder
        try:
            os.chdir(target)
            ierr = 0
        except FileNotFoundError:
            self._log.write(Text(
                f"Folder not found: '{target}'", style=f"bold {TN_RED}"))
            ierr = 2
        except PermissionError:
            self._log.write(Text(
                f"Permission denied: '{target}'", style=f"bold {TN_RED}"))
            ierr = 13
        # Output
        return ierr


# Run the app when executed as ``python3 -m cape.tui.tuiapp``
if __name__ == "__main__":
    # Local imports
    from . import main
    main()

r"""
:mod:`cape.tui.poc`: Proof-of-concept persistent CAPE TUI (experimental)
=========================================================================

This module is a **proof of concept** for an OpenCode-style terminal
user interface to CAPE, built on the optional ``textual`` package. It
is experimental and not part of the CAPE CLI. Run it with:

.. code-block:: console

    $ python3 -m cape.tui.poc

It demonstrates, in one persistent app with a scroll log and a pinned
input box:

* **In-process CAPE commands**
    Commands starting with ``cape``, ``pycart``, ``pyfun``, etc. run
    in-process using :func:`cape.cfdx.cli.main` in a worker thread,
    with their output streamed into the scroll log.

* **External commands**
    Any other command runs as a subprocess; its combined output is
    streamed into the scroll log.

* **Tab completion**
    Pressing TAB in the input box completes the current word using the
    same :class:`cape.tui.tuiutils.TuiCompleter` logic as
    :mod:`cape.ui`: CAPE executables, ``$PATH`` commands, sub-command
    names, options, option values, and file names. One match completes
    outright; many complete the longest common prefix first and are
    listed on a second TAB.

* **Ctrl-C interrupts**
    Pressing Ctrl-C while a command is running interrupts it. External
    commands run in their own process group and receive ``SIGINT``;
    in-process CAPE commands get a :class:`KeyboardInterrupt` injected
    into their worker thread (this interrupts Python-level blocking
    such as ``cape -c`` status loops, but not C-level waits, e.g. an
    in-flight solver subprocess keeps running). With nothing running,
    Ctrl-C clears the input line.

* **Prompt bridge**
    Interactive prompts raised by commands (the synchronous calls to
    :func:`cape.promptutils.prompt_color` in :mod:`cape.cfdx.cntl`)
    mount as clickable widgets in the stream. Try ``:prompt-demo`` to
    answer one. Answering with Ctrl-C aborts the command.

* **OpenCode-style chrome**
    Rounded, state-aware editor border (context on the left, last exit
    status on the right), a status bar with a live spinner and elapsed
    time while a command runs, backgrounded command "bubbles" in the
    scroll log, and the ``tokyo-night`` theme. ESC is a second
    interrupt key alongside Ctrl-C.

* ``:exit`` / ``:quit`` (or plain ``exit``) leave the app.

Not included (deferred to the full implementation): history files,
the ``:meta`` commands of :mod:`cape.tui`, and suspension for
interactive full-screen subprocesses.
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
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, RichLog, Static

# CAPE imports
from ..promptutils import _new_prompt_widget, register_prompt_handler
from ..ui.promptutils import CAPE_EXECS, CfdxCompleter

# Local imports
from .tuiutils import TuiCompleter, get_dirname


# Match commands to run with CAPE's in-process CLI (as in cape.tui)
REGEX_CAPE_CLI = re.compile(rf"\$?\s*({'|'.join(CAPE_EXECS)})( |$)")

# Exit commands
EXIT_CMDS = (":exit", ":quit", "exit", "quit")

# PoC meta-commands for completion
POC_META_CMDS = (":exit", ":prompt-demo", ":quit")

# Placeholder for the idle input box
INPUT_PLACEHOLDER = "cape command (TAB completes · Ctrl-C interrupts)"

# tokyo-night colors for log content (CSS vars only style widgets)
TN_BLUE = "#7AA2F7"
TN_DIM = "#565F89"
TN_GREEN = "#9ECE6A"
TN_ORANGE = "#FF9E64"
TN_PURPLE = "#BB9AF7"
TN_RED = "#F7768E"
TN_SURFACE = "#24283B"

# Busy spinner animation in the status bar
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL = 0.1

# Status bar hints
HINTS_IDLE = "TAB complete · ^C interrupt · :exit"
HINTS_BUSY = "^C interrupt"


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


# Tab-completer reading the command line from the PoC input box
class PocCompleter(TuiCompleter):
    r"""CAPE tab-completer that reads the line from the PoC input

    This overrides :func:`CfdxCompleter.get_line_buffer` to source the
    command line from the app's ``Input`` widget (text before the
    cursor) instead of :mod:`readline`. Suggestions for ``:``
    meta-commands are limited to the ones this app implements.

    :Attributes:
        *app*: :class:`CapePocApp`
            App whose input box is being completed
    """

    __slots__ = ("app",)

    def __init__(self, cls, app: "CapePocApp"):
        # Initialize hierarchy
        CfdxCompleter.__init__(self, cls)
        # Save the app
        self.app = app

    def get_line_buffer(self) -> str:
        r"""Get the input box text before the cursor

        :Call:
            >>> line = comp.get_line_buffer()
        :Inputs:
            *comp*: :class:`PocCompleter`
                PoC autocompleter
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
        r"""Generate suggestions, including PoC meta-commands

        :Call:
            >>> suggestions = comp.genr8_suggestions(text)
        :Inputs:
            *comp*: :class:`PocCompleter`
                PoC autocompleter
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
            # Complete PoC meta-command names
            self.role = "metacmd"
            return fnmatch.filter(POC_META_CMDS, f"{text}*")
        # Defer to CAPE front-desk completions
        return CfdxCompleter.genr8_suggestions(self, text)


# File-like object that posts written lines to the PoC scroll log
class LogWriter(io.TextIOBase):
    r"""Redirect writes (e.g. :data:`sys.stdout`) into a RichLog

    Lines are buffered; complete lines are posted to the app's scroll
    log from the writing thread via ``call_from_thread``. ANSI color
    codes are converted to Rich :class:`Text` styling.

    :Attributes:
        *app*: :class:`CapePocApp`
            App whose log receives written lines
        *_buf*: :class:`str`
            Buffer holding an incomplete line
    """

    def __init__(self, app: "CapePocApp"):
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


# Main proof-of-concept app
class CapePocApp(App):
    r"""Experimental OpenCode-style CAPE TUI app

    A scroll log plus a pinned input box. Commands run in worker
    threads: CAPE CLI commands in-process with redirected
    :data:`sys.stdout`/:data:`sys.stderr`, other commands as
    subprocesses. Interactive prompts (via
    :func:`cape.promptutils.prompt_color`) mount clickable widgets
    below the log while the worker thread waits for an answer.

    :Attributes:
        *_log*: :class:`textual.widgets.RichLog`
            Scroll log of commands and output
        *_input*: :class:`textual.widgets.Input`
            Command input box (the OpenCode-style editor frame)
        *_status*: :class:`textual.widgets.Static`
            One-line status bar below the editor
        *_prompt_widget*: ``None`` | :class:`textual.widget.Widget`
            Currently mounted interactive prompt, if any
        *_completer*: ``None`` | :class:`PocCompleter`
            Tab-completer for the command input box
        *_proc*: ``None`` | :class:`subprocess.Popen`
            Currently running external command, if any
        *_worker*: ``None`` | :class:`threading.Thread`
            Currently running command's worker thread, if any
        *_running_cmd*: ``None`` | :class:`str`
            Text of the command currently running, if any
        *_t0*: :class:`float`
            Start time of the running command
        *_spin_ix*: :class:`int`
            Current frame of the busy spinner
        *_spin_timer*: ``None`` | :class:`textual.timer.Timer`
            Timer advancing the busy spinner
    """

    # Key bindings: priority so they win over defaults, gated by
    # check_action() so prompts and other widgets keep their own keys
    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt", show=False,
                priority=True),
        Binding("escape", "interrupt", "Interrupt", show=False,
                priority=True),
        Binding("tab", "tab_complete", "Complete", show=False,
                priority=True),
    ]

    # Style settings
    CSS = (
        "#body {\n"
        "    height: 100%;\n"
        "}\n"
        "RichLog {\n"
        "    height: 1fr;\n"
        "    padding: 0 1;\n"
        "}\n"
        "Input {\n"
        "    border: round $primary;\n"
        "}\n"
        "Input.busy {\n"
        "    border: round $warning;\n"
        "}\n"
        "Input.ok {\n"
        "    border-subtitle-color: $success;\n"
        "}\n"
        "Input.fail {\n"
        "    border-subtitle-color: $error;\n"
        "}\n"
        "#status-bar {\n"
        "    height: 1;\n"
        "    padding: 0 1;\n"
        "    background: $surface;\n"
        "    color: $text-muted;\n"
        "}\n")

    def compose(self) -> ComposeResult:
        with Vertical(id="body"):
            yield RichLog(id="log", auto_scroll=True)
            yield Input(placeholder=INPUT_PLACEHOLDER, id="prompt-input")
            yield Static(id="status-bar")

    def on_mount(self) -> None:
        # Theme and window title
        self.theme = "tokyo-night"
        self.title = "CAPE TUI (proof of concept)"
        # Save widget references
        self._log = self.query_one("#log", RichLog)
        self._input = self.query_one("#prompt-input", Input)
        self._status = self.query_one("#status-bar", Static)
        # No mounted prompt or running command so far
        self._prompt_widget = None
        self._proc = None
        self._worker = None
        self._running_cmd = None
        self._t0 = 0.0
        self._spin_ix = 0
        self._spin_timer = None
        # Create tab-completer hooked to the command input
        from ..cfdx.cli import CfdxFrontDesk
        self._completer = PocCompleter(CfdxFrontDesk, self)
        # Editor frame: context on the left, hints in the status bar
        self._input.border_title = self._context_title()
        self._update_status()
        # Banner
        self._log.write(Text(
            "CAPE TUI proof of concept  ·  try 'cape help', "
            "'echo hello', ':prompt-demo', ':exit'",
            style=f"bold {TN_BLUE}"))
        # Focus the command input
        self._input.focus()

    # Post one line to the scroll log from another thread
    def _post_to_log(self, txt) -> None:
        self.call_from_thread(self._log.write, txt)

    # Re-enable the input box after a command finishes
    def _set_idle(self, ierr: int = 0) -> None:
        # Stop the busy spinner
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
        self._running_cmd = None
        # Editor frame: back to idle, subtitle = last result
        self._input.remove_class("busy")
        self._input.set_class(ierr == 0, "ok")
        self._input.set_class(ierr != 0, "fail")
        icon = "✓" if ierr == 0 else "✗"
        self._input.border_subtitle = f" {icon} exit {ierr} "
        self._input.border_title = self._context_title()
        # Re-enable the input
        self._input.disabled = False
        self._input.placeholder = INPUT_PLACEHOLDER
        self._input.focus()
        # Status bar back to idle hints
        self._update_status()

    # Title for the editor border: host and short folder
    def _context_title(self) -> str:
        hostname = socket.gethostname().split('.')[0]
        return f" CAPE {hostname}:{get_dirname()} "

    # Render the status bar for the current state
    def _update_status(self) -> None:
        # Available width (may be 0 before layout)
        width = max(10, self._status.size.width)
        # Check state
        if self._running_cmd is None:
            # Idle: short folder on the left, key hints on the right
            left = Text(f" {get_dirname()}")
            right = Text(HINTS_IDLE)
        else:
            # Busy: spinning frame, command text, wall time
            frame = SPINNER_FRAMES[self._spin_ix % len(SPINNER_FRAMES)]
            dt = time.perf_counter() - self._t0
            left = Text.assemble(
                (f" {frame} ", f"bold {TN_ORANGE}"),
                (f"running '{self._running_cmd}' · {dt:.1f}s", ""))
            right = Text(HINTS_BUSY, style=TN_ORANGE)
        # Pad the gap between left and right
        pad = max(1, width - len(left) - len(right) - 1)
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

    # Command echo rendered as an OpenCode-style message bubble
    def _bubble_text(self, cmd: str) -> Text:
        # Pad to the log's text width (2 padding + 1 scrollbar)
        head = "❯ "
        width = max(10, self._log.size.width - 3)
        pad = max(1, width - len(head) - len(cmd))
        return Text.assemble(
            (head, f"bold {TN_BLUE} on {TN_SURFACE}"),
            (cmd + " " * pad, f"on {TN_SURFACE}"))

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
        # While a prompt is mounted, it owns TAB and Ctrl-C
        if action in ("tab_complete", "interrupt"):
            if self._prompt_widget is not None:
                return False
            # TAB only completes in the active command input
            if action == "tab_complete":
                return self._input.has_focus and not self._input.disabled
        # All other actions enabled
        return True

    # Complete the current word of the command input
    def action_tab_complete(self) -> None:
        r"""Complete the word left of the cursor (TAB action)

        Completions come from :class:`PocCompleter`. A unique match is
        inserted directly; multiple matches extend to the longest
        common prefix, and if the word is already fully extended the
        candidates are listed in the log (like a second readline TAB).
        """
        # Get the word left of the cursor and its start index
        value = self._input.value
        pos = self._input.cursor_position
        start = pos
        while start > 0 and value[start - 1] not in " \t\n":
            start -= 1
        text = value[start:pos]
        # Generate suggestions; tolerate partial/quoted input
        try:
            matches = self._completer.get_suggestions(text)
        except Exception:
            matches = []
        # No completions
        if not matches:
            self.bell()
            return
        # Deduplicate (e.g. a CAPE exec that's also on $PATH)
        matches = list(dict.fromkeys(matches))
        # Unique match: insert it (with trailing ' ' or os.sep)
        if len(matches) == 1:
            match = matches[0]
            # Unique matches from get_suggestions() already have the
            # suffix; add it for matches left unique by deduplication
            if not match.endswith((" ", os.sep)):
                role = self._completer.role
                if (role == "filename") and os.path.isdir(match):
                    match += os.sep
                else:
                    match += " "
            self._replace_word(start, pos, match)
            return
        # Several matches: extend to the longest common prefix
        prefix = os.path.commonprefix(matches)
        if len(prefix) > len(text):
            self._replace_word(start, pos, prefix)
        else:
            # Nothing new to insert: list candidates like a 2nd TAB
            self._log.write(Text("  ".join(matches), style="cyan"))

    # Insert a completion, replacing the current word
    def _replace_word(self, start: int, pos: int, match: str) -> None:
        value = self._input.value
        self._input.value = value[:start] + match + value[pos:]
        self._input.cursor_position = start + len(match)

    # Interrupt the running command (Ctrl-C action)
    def action_interrupt(self) -> None:
        r"""Interrupt the running command, or clear the input line

        External commands run in their own process group and are sent
        ``SIGINT``; in-process CAPE commands get a
        :class:`KeyboardInterrupt` injected into the worker thread.
        With nothing running, the input line is cleared.
        """
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
        # Idle: clear the input line
        self._input.value = ""

    # Run one submitted command
    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Get command text and clear the input
        cmd = event.value.strip()
        self._input.value = ""
        # Check for empty command
        if not cmd:
            return
        # Echo the command in the log as a bubble
        self._log.write(self._bubble_text(cmd))
        # Check for exit commands
        if cmd in EXIT_CMDS:
            self.exit()
            return
        # Busy indicators: warning border, spinner in the status bar
        self._input.disabled = True
        self._input.placeholder = "running..."
        self._input.remove_class("ok", "fail")
        self._input.add_class("busy")
        self._running_cmd = cmd
        self._t0 = time.perf_counter()
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
            if cmd == ":prompt-demo":
                ierr = self._run_prompt_demo()
            elif REGEX_CAPE_CLI.match(cmd):
                ierr = self._run_cape_cli(cmd)
            else:
                ierr = self._run_subprocess(cmd)
        except KeyboardInterrupt:
            self._post_to_log(Text("KeyboardInterrupt", style="bold red"))
            ierr = 130
        except Exception:
            # Unexpected error: show the traceback in the log
            for line in traceback.format_exc().rstrip().split("\n"):
                self._post_to_log(Text(line, style="red"))
            ierr = 128
        # Wall time
        dt = time.perf_counter() - t0
        # Command is no longer running
        self._worker = None
        # Status line
        self._post_to_log(self._status_rule_text(ierr, dt))
        # Re-enable the input
        self.call_from_thread(self._set_idle, ierr)

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
                style="bold red"))
            return 127
        except PermissionError:
            self._post_to_log(Text(
                f"Permission denied: '{cmd.split()[0]}'",
                style="bold red"))
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
                ["next", "extend", "skip"], prompt="poc>")
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

        # Mount the prompt widget above the input box
        def mount() -> None:
            widget = _new_prompt_widget(
                txt, vdef, vopt, prompt, oneline, on_answer=on_answer)
            self._prompt_widget = widget
            body = self.query_one("#body", Vertical)
            body.mount(widget, before=self._input)

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


# Main entry point
def main() -> None:
    r"""Run the proof-of-concept CAPE TUI app

    Registers the app's prompt bridge with
    :func:`cape.promptutils.register_prompt_handler` and restores the
    previous handler on exit.
    """
    # Create the app
    app = CapePocApp()
    # Register its prompt handler, saving any previous one
    prev_handler = register_prompt_handler(app._handle_prompt)
    # Run the app
    try:
        app.run()
    finally:
        # Restore the previous prompt handler
        register_prompt_handler(prev_handler)


# Run the app when executed as ``python3 -m cape.tui.poc``
if __name__ == "__main__":
    main()

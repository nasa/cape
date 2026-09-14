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

* **Prompt bridge**
    Interactive prompts raised by commands (the synchronous calls to
    :func:`cape.promptutils.prompt_color` in :mod:`cape.cfdx.cntl`)
    mount as clickable widgets in the stream. Try ``:prompt-demo`` to
    answer one. Answering with Ctrl-C aborts the command.

* ``:exit`` / ``:quit`` (or plain ``exit``) leave the app.

Not included (deferred to the full implementation): history files,
tab-completion, the ``:meta`` commands of :mod:`cape.tui`, and
suspension for interactive full-screen subprocesses.
"""

from __future__ import annotations

# Standard library
import io
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback

# Third-party
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, RichLog

# CAPE imports
from ..promptutils import _new_prompt_widget, register_prompt_handler
from ..ui.promptutils import CAPE_EXECS


# Match commands to run with CAPE's in-process CLI (as in cape.tui)
REGEX_CAPE_CLI = re.compile(rf"\$?\s*({'|'.join(CAPE_EXECS)})( |$)")

# Exit commands
EXIT_CMDS = (":exit", ":quit", "exit", "quit")


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
            Command input box
        *_prompt_widget*: ``None`` | :class:`textual.widget.Widget`
            Currently mounted interactive prompt, if any
    """

    # Style settings
    CSS = (
        "#body {\n"
        "    height: 100%;\n"
        "}\n"
        "RichLog {\n"
        "    height: 1fr;\n"
        "    padding: 0 1;\n"
        "}\n")

    def compose(self) -> ComposeResult:
        with Vertical(id="body"):
            yield RichLog(id="log", auto_scroll=True)
            yield Input(
                placeholder="cape command (try :prompt-demo)",
                id="prompt-input")

    def on_mount(self) -> None:
        # Title
        self.title = "CAPE TUI (proof of concept)"
        # Save widget references
        self._log = self.query_one("#log", RichLog)
        self._input = self.query_one("#prompt-input", Input)
        # No mounted prompt so far
        self._prompt_widget = None
        # Banner
        self._log.write(Text(
            "CAPE TUI proof of concept  ·  try 'cape help', "
            "'echo hello', ':prompt-demo', ':exit'",
            style="bold green"))
        # Focus the command input
        self._input.focus()

    # Post one line to the scroll log from another thread
    def _post_to_log(self, txt) -> None:
        self.call_from_thread(self._log.write, txt)

    # Re-enable the input box after a command finishes
    def _set_idle(self) -> None:
        self._input.disabled = False
        self._input.placeholder = "cape command (try :prompt-demo)"
        self._input.focus()

    # Run one submitted command
    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Get command text and clear the input
        cmd = event.value.strip()
        self._input.value = ""
        # Check for empty command
        if not cmd:
            return
        # Echo the command in the log
        self._log.write(Text(f"$ {cmd}", style="purple"))
        # Check for exit commands
        if cmd in EXIT_CMDS:
            self.exit()
            return
        # Busy indicator
        self._input.disabled = True
        self._input.placeholder = "running..."
        # Run the command in a worker thread
        thread = threading.Thread(
            target=self._run_command, args=(cmd,), daemon=True)
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
        # Status line
        ok = ierr == 0
        style = "green" if ok else "red"
        icon = "✓" if ok else "✗"
        self._post_to_log(Text(f"{icon} exit {ierr} · {dt:.2f}s",
                               style=style))
        # Re-enable the input
        self.call_from_thread(self._set_idle)

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
        # Start the command
        try:
            proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, errors="replace")
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
        # Stream output lines into the log
        for line in proc.stdout or []:
            self._post_to_log(Text.from_ansi(line.rstrip("\n")))
        # Output
        return proc.wait()

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

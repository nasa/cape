r"""
:mod:`cape.agent.agenttui`: Textual interface for CAPE agent
============================================================

This module adapts :class:`cape.tui.tuiapp.CapeTuiApp` to use an
:class:`cape.agent.agentcntl.AgentCntl` conversation as its command backend.
The ordinary CAPE TUI supplies the composer, folding log, history, prompt
bridge, interruption behavior, and session chrome.
"""

from __future__ import annotations

# Standard library
import contextlib
import io
import pprint
import sys
from typing import Optional

# Third-party imports
from openai import InternalServerError
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Local imports
from . import agentutils
from .agentcntl import AgentCntl
from ..tui import run_app
from ..tui.tuiapp import CapeTuiApp, CapeTuiCompleter, LogWriter
from ..tui.tuiutils import TN_BLUE, TN_DIM, TN_GREEN, TN_PURPLE
from ..ui.promptutils import CAPE_EXECS


class AgentTuiCompleter(CapeTuiCompleter):
    r"""Complete meta-commands and explicit shell/CAPE commands only."""

    __slots__ = ()

    def genr8_suggestions(self, text: str) -> list[str]:
        # Natural-language prompts should not produce filename suggestions.
        line = self.get_line_buffer().lstrip()
        command = line.lstrip("$").lstrip()
        first = command.split(maxsplit=1)[0] if command else ""
        if not line.startswith(":") and \
                not line.startswith("$") and first not in CAPE_EXECS:
            self.role = None
            return []
        # Explicit commands retain the ordinary CAPE TUI completion behavior.
        return super().genr8_suggestions(text)


class AgentTuiApp(CapeTuiApp):
    r"""OpenCode-style Textual interface backed by a CAPE agent."""

    APP_TITLE = "CAPE Agent"
    INPUT_PLACEHOLDER = "Ask the CAPE agent"
    CONTEXT_LABEL = "CAPE Agent"

    def __init__(
            self,
            cls: type,
            cntl: AgentCntl,
            *a,
            startup_output: str = "",
            histfile: Optional[str] = None,
            **kw):
        # Save the agent before Textual mounts the app.
        self._agent = cntl
        self._startup_output = startup_output
        super().__init__(cls, *a, histfile=histfile, **kw)

    def on_ready(self) -> None:
        # Extend shared state after CapeTuiApp's mount handler has run.
        self._stats.update({
            "n_user_msgs": 0,
            "n_tool_calls": 0,
            "n_tool_fails": 0,
            "n_fails": 0,
        })
        # Identify the active endpoint/model inside the persistent log.
        self._log.write(_agent_banner(self._agent, self._histfile))
        for line in self._startup_output.rstrip().splitlines():
            self._log.write(Text.from_ansi(line))

    def _make_completer(self):
        return AgentTuiCompleter(self._frontdesk_cls, self)

    def _write_command_header(self, cmd: str) -> None:
        # Agent turns contain their own reasoning/tool folds.
        # Keep user's prompt at the transcript's top level
        self._log.end_group()
        self._log.write(self._bubble_text(cmd))

    def _section_text(self, title: str, folded: bool = False) -> Text:
        r"""Build a rule-like header for an agent output section."""
        # Variable prompt char for section start line
        head = "▶ " if folded else "▼ "
        # Overall width of window
        width = max(10, self._log.size.width - 3)
        # Rule starts after code on the fold headline
        tail = max(2, width - len(head) - len(title) - 1)
        # Assemble ">" + {text} + hline
        return Text.assemble(
            (head, f"bold {TN_BLUE}"),
            (title, f"italic {TN_PURPLE}"),
            (" " + "─" * tail, TN_DIM))

    def _handle_section(
            self,
            action: str,
            kind: str,
            title: str,
            expanded: bool) -> None:
        r"""Open or close a fold group from the agent worker thread."""
        if action == "start":
            self.call_from_thread(
                self._log.start_section,
                kind,
                self._section_text(title),
                self._section_text(title, folded=True),
                not expanded)
        else:
            self.call_from_thread(self._log.end_group)

    def _execute_command(self, cmd: str) -> int:
        # AgentCntl writes its progress, tools, reasoning, and answer to
        # stdout; route that to the shared RichLog.
        writer = LogWriter(self)
        stdout_old, stderr_old = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            # Match the readline loop by reporting completed background
            # work before processing the next user turn.
            self._agent.reap_tasks(section_handler=self._handle_section)
            self._stats["n_user_msgs"] += 1
            result = self._agent.run_agent(
                cmd,
                spinner=False,
                capture_subprocess=True,
                section_handler=self._handle_section)
            self._stats["n_tool_calls"] += result.get("n_tool_calls", 0)
            self._stats["n_tool_fails"] += result.get("n_tool_fails", 0)
            return 0
        except InternalServerError as err:
            self._stats["n_fails"] += 1
            parts = err.args[0].split(" - ", 1)
            if len(parts) > 1:
                pprint.pprint(parts[1])
            print(f"{type(err).__name__}: {parts[0]}")
            return 1
        finally:
            sys.stdout = stdout_old
            sys.stderr = stderr_old
            writer.flush()
            # Agent tools share the CLI controller cache with cape.tui.
            from ..cfdx import cli
            self.call_from_thread(
                self._sync_json_cache, tuple(cli.CNTL_CACHE))

    def finalize_stats(self) -> dict:
        stats = super().finalize_stats()
        stats["background_tasks"] = sum(
            not task.reaped and task.poll() is None
            for task in self._agent.tasks)
        return stats


def _agent_banner(cntl: AgentCntl, histfile: str) -> Panel:
    r"""Build the compact startup panel shown in the agent log."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style=f"bold {TN_GREEN}", justify="right")
    table.add_column()
    table.add_row("Endpoint:", Text(cntl.base_url, style=TN_BLUE))
    table.add_row("Model:", Text(cntl.model, style=TN_BLUE))
    table.add_row("History:", histfile)
    return Panel(
        table,
        title="[bold green]CAPE[/] [bold cyan]Agent[/]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 2))


def main(cls: Optional[type] = None):
    r"""Run the Textual CAPE-agent conversation interface."""
    if cls is None:
        from ..cfdx.cli import CfdxFrontDesk
        cls = CfdxFrontDesk
    # Preserve controller startup notices and replay them in the app log
    startup = io.StringIO()
    # Initialize agent instance
    with contextlib.redirect_stdout(startup), \
            contextlib.redirect_stderr(startup):
        cntl = AgentCntl()
    # Initialize TUI app
    app = AgentTuiApp(
        cls,
        cntl,
        startup_output=startup.getvalue(),
        histfile=agentutils.get_agent_histfile())
    # Execute TUI app
    return run_app(app, title="CAPE agent summary")

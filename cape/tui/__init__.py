r"""
OpenCode-style interactive terminal user interface to CAPE, built on
the optional ``textual`` package and launched with:

.. code-block:: console

    $ cape tui

It provides the same core features as :mod:`cape.ui` -- CAPE-aware
tab-completion and a dedicated history file -- inside a persistent app
with a scroll log, a pinned dark composer, and context below the prompt.
Ctrl-P opens a picker for the built-in TUI commands:

* **In-process CAPE commands**
    Commands starting with ``cape``, ``pycart``, ``pyfun``, etc. run
    in-process using :func:`cape.cfdx.cli.main`, with output streamed
    into the log; other commands run as subprocesses.

* **Interruptions and prompts**
    Ctrl-C (or ESC) interrupts the running command, and interactive
    CAPE prompts mount as clickable widgets above the editor.

* **TUI meta-commands**
    Commands starting with ``:`` are handled by the TUI itself:
    ``:help [<cmd>]``, ``:history [<n>]``, ``:!N``, ``:status``,
    ``:cd <dir>`` (plain ``cd`` also works), ``:pwd``, ``:clear``,
    and ``:exit`` / ``:quit``.

See :mod:`cape.tui.tuiapp` for the app itself and
:mod:`cape.tui.tuiutils` for the render helpers.
"""

from __future__ import annotations

# Standard library
from typing import Optional, Tuple

# CAPE imports
from ..promptutils import register_prompt_handler


# Main function
def main(cls: Optional[type] = None) -> Tuple[int, dict]:
    r"""Main Textual-based interactive TUI function

    :Call:
        >>> ierr, stats = main(cls)
    :Inputs:
        *cls*: {``None``} | :class:`type`
            :class:`cape.argread.ArgReader` subtype for completions
            and ``:help``; defaults to
            :class:`cape.cfdx.cli.CfdxFrontDesk`
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *stats*: :class:`dict`
            Information about results of commands run

            * ``commands``: :class:`int` number of commands run
            * ``failures``: :class:`int` number of failed commands
            * ``tui_commands``: :class:`int` number of TUI meta-cmds
            * ``duration``: :class:`float` session wall time (sec)
            * ``json_files``: :class:`tuple` of loaded JSON paths,
              least to most recently used
            * ``last_json_file``: :class:`str` | ``None`` most recent
              absolute JSON path
            * ``last_json_display_file``: :class:`str` | ``None`` path
              displayed relative to the controller's root
    """
    # Delayed imports of textual-dependent modules
    from .tuiapp import CapeTuiApp
    # Default completions class
    if cls is None:
        from ..cfdx.cli import CfdxFrontDesk
        cls = CfdxFrontDesk
    # Create the app
    app = CapeTuiApp(cls)
    # Run it using the shared Textual lifecycle
    return run_app(app, title="CAPE TUI summary")


def run_app(app, title: str = "CAPE TUI summary") -> Tuple[int, dict]:
    r"""Run a CAPE Textual app with shared prompt/history lifecycle"""
    # Delayed imports keep textual and rich optional
    from .tuiutils import session_stats_panel
    from rich.console import Console
    # Register its prompt handler, saving any previous one
    prev_handler = register_prompt_handler(app._handle_prompt)
    # Run the app
    try:
        app.run()
    finally:
        # Restore the previous prompt handler
        register_prompt_handler(prev_handler)
        # Write back the (trimmed) history file
        app.save_history()
    # Finalize statistics
    stats = app.finalize_stats()
    # Render exit summary on the restored terminal
    Console().print(
        session_stats_panel(stats, app._histfile, title=title))
    # Return code
    return 0, stats

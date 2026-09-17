# Standard library
import asyncio
import os

# Third-party
import pytest

# Only run these tests if the optional textual package is installed
textual = pytest.importorskip("textual")

# Local imports
from cape.cfdx.cli import CfdxFrontDesk  # noqa: E402
from cape.tui.tuiapp import (  # noqa: E402
    CapeTuiApp, CommandPalette, HINTS_IDLE)


# Run an async coroutine without needing a pytest async plugin
def run_async(coro):
    return asyncio.run(coro)


# Create an app with an isolated history file
def make_app(tmp_path, monkeypatch):
    # Point the TUI history file into the test temp dir
    histfile = os.path.join(str(tmp_path), "cape_tui_history")
    monkeypatch.setattr(
        "cape.tui.tuiapp.get_tui_histfile", lambda: histfile)
    # Create the app
    app = CapeTuiApp(CfdxFrontDesk)
    app._test_histfile = histfile
    return app


# Plain text of the scroll log
def log_text(app):
    return "\n".join(
        str(getattr(line, "text", line)) for line in app._log.lines)


# Plain text of the status bar
def status_text(app):
    return str(app._status.content)


# Wait for a condition inside the app, pumping the event loop
async def wait_for(pilot, cond, n=100, dt=0.1):
    for _ in range(n):
        await pilot.pause(dt)
        if cond():
            return True
    return False


# Table: TAB completes a CAPE executable name
def test_01_tab_exec(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = "pyca"
            app._input.cursor_position = 4
            await pilot.press("tab")
            await pilot.pause()
            assert app._input.value == "pycart "
    run_async(drive())


# Table: TAB completes a TUI meta-command
def test_02_tab_meta(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = ":he"
            app._input.cursor_position = 3
            await pilot.press("tab")
            await pilot.pause()
            assert app._input.value == ":help "
    run_async(drive())


# Ctrl-C with no command running clears the editor
def test_03_ctrlc_idle_clears(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = "some text"
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app._input.value == ""
    run_async(drive())


# Ctrl-C interrupts a running subprocess (and kills it)
@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_04_ctrlc_subprocess(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = "sleep 30"
            await pilot.press("enter")
            ok = await wait_for(pilot, lambda: app._proc is not None, n=50)
            assert ok
            pid = app._proc.pid
            # Busy chrome while running
            assert app._input.has_class("busy")
            assert "running 'sleep 30'" in status_text(app)
            # Interrupt
            await pilot.press("ctrl+c")
            await wait_for(
                pilot,
                lambda: app._proc is None and not app._input.disabled)
            out = log_text(app)
            # Killed by SIGINT, reported as exit 130
            assert "exit 130" in out
            try:
                os.kill(pid, 0)
                dead = False
            except ProcessLookupError:
                dead = True
            assert dead
            # Editor frame shows the failure and is re-enabled
            assert app._input.has_class("fail")
            assert "exit 130" in str(app._input.border_subtitle)
            assert HINTS_IDLE in status_text(app)
    run_async(drive())


# Ctrl-C interrupts a blocking in-process CAPE command
def test_05_ctrlc_inprocess(tmp_path, monkeypatch):
    import time

    # Local imports
    import cape.cfdx.cli as cli

    def slow_main(argv=None):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 30:
            time.sleep(0.02)
        return 0

    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = "cape -c --slow"
            await pilot.press("enter")
            t0 = time.perf_counter()
            await wait_for(
                pilot, lambda: app._worker is not None, n=50)
            await pilot.pause(0.5)
            await pilot.press("ctrl+c")
            await wait_for(
                pilot,
                lambda: app._worker is None and not app._input.disabled)
            dt = time.perf_counter() - t0
            out = log_text(app)
            assert "KeyboardInterrupt" in out
            assert "exit 130" in out
            assert dt < 10
    # Patch the CAPE CLI with a long Python-level loop
    monkeypatch.setattr(cli, "main", slow_main)
    run_async(drive())


# History recall with up/down arrows
def test_06_history_recall(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Seed history via the app's own loader
            with open(app._test_histfile, "w") as fp:
                fp.write("cape -c\necho hello\n")
            app._load_history()
            # Browse back and forth
            await pilot.press("up")
            await pilot.pause()
            assert app._input.value == "echo hello"
            await pilot.press("up")
            await pilot.pause()
            assert app._input.value == "cape -c"
            await pilot.press("down")
            await pilot.pause()
            assert app._input.value == "echo hello"
            await pilot.press("down")
            await pilot.pause()
            assert app._input.value == ""
    run_async(drive())


# History file is rewritten on save with submitted commands
def test_07_history_save(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = ":pwd"
            await pilot.press("enter")
            await pilot.pause()
            # Save and check the file
            app.save_history()
            with open(app._test_histfile) as fp:
                lines = fp.read().splitlines()
            assert lines == [":pwd"]
    run_async(drive())


# Folder-change commands update the editor frame
def test_08_cd(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        cwd0 = os.getcwd()
        try:
            async with app.run_test() as pilot:
                await pilot.pause()
                # Change to the temp folder
                app._input.value = f"cd {tmp_path}"
                await pilot.press("enter")
                await pilot.pause()
                assert os.getcwd() == str(tmp_path)
                assert os.path.basename(str(tmp_path)) in \
                    str(app._input.border_title)
                assert app._input.has_class("ok")
                # Bad folder gives a failure
                app._input.value = "cd /this/folder/does/not/exist"
                await pilot.press("enter")
                await pilot.pause()
                assert os.getcwd() == str(tmp_path)
                assert "exit 2" in str(app._input.border_subtitle)
                assert app._input.has_class("fail")
        finally:
            os.chdir(cwd0)
    run_async(drive())


# Meta-commands: help table, status panel, and :!N rerun
def test_09_meta(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            # :help writes the command table
            app._input.value = ":help"
            await pilot.press("enter")
            await pilot.pause()
            assert app._input.has_class("ok")
            assert app._stats["tui_commands"] == 1
            # :status writes the session panel
            app._input.value = ":status"
            await pilot.press("enter")
            await pilot.pause()
            assert app._stats["tui_commands"] == 2
            # Run a quick subprocess command
            app._input.value = "echo tui_meta_marker"
            await pilot.press("enter")
            await wait_for(pilot, lambda: not app._input.disabled, n=50)
            # Rerun it from history
            app._input.value = ":!3"
            await pilot.press("enter")
            await wait_for(pilot, lambda: not app._input.disabled, n=50)
            out = log_text(app)
            assert "Rerunning history entry 3" in out
            # Marker ran twice: once directly, once via :!3
            assert out.count("tui_meta_marker") >= 2
    run_async(drive())


# Unknown meta-command reports an error
def test_10_meta_unknown(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = ":bogus"
            await pilot.press("enter")
            await pilot.pause()
            assert "Unrecognized TUI command" in log_text(app)
            assert "exit 16" in str(app._input.border_subtitle)
            assert app._input.has_class("fail")
    run_async(drive())


# Ctrl-P lists commands, Escape cancels, Enter runs the selected command
def test_11_command_palette(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "TAB complete" in str(
                app.query_one("#composer-hint").content)
            assert os.getcwd() in status_text(app)
            app._input.value = "draft"
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert isinstance(app.screen, CommandPalette)
            assert app.screen.query_one("#command-list").option_count > 0
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, CommandPalette)
            assert app._input.value == "draft"
            await pilot.press("ctrl+p")
            await pilot.pause()
            app.screen.query_one("#command-list").highlighted = 3
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, CommandPalette)
            assert app._history[-1] == ":help"
    run_async(drive())

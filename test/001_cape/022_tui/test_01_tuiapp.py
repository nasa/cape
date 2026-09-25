# Standard library
import asyncio
from collections import OrderedDict
import os
from types import SimpleNamespace

# Third-party
import pytest

# Only run these tests if the optional textual package is installed
textual = pytest.importorskip("textual")

# Local imports
from cape.cfdx.cli import CfdxFrontDesk  # noqa: E402
from cape.tui.tuiapp import (  # noqa: E402
    CapeTuiApp, CommandPalette, HINTS_IDLE)
from cape.tui.tuiutils import session_stats_panel  # noqa: E402


# Run an async coroutine without needing a pytest async plugin
def run_async(coro):
    return asyncio.run(coro)


# Create an app with an isolated history file
def make_app(tmp_path, monkeypatch):
    # Point the TUI history file into the test temp dir
    histfile = os.path.join(str(tmp_path), "cape_tui_history")
    monkeypatch.setattr(
        "cape.tui.tuiapp.get_tui_histfile", lambda: histfile)
    # Isolate the shared CLI JSON-file cache from other tests
    monkeypatch.setattr("cape.cfdx.cli.CNTL_CACHE", OrderedDict())
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
            app.screen.query_one("#command-list").highlighted = 2
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, CommandPalette)
            assert app._history[-1] == ":help"
            app._input.value = "keep this draft"
            history_before = list(app._history)
            await pilot.press("ctrl+p")
            await pilot.pause()
            app.screen.query_one("#command-list").highlighted = 7
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, CommandPalette)
            assert app._input.value == "keep this draft"
            assert app._history == history_before
    run_async(drive())


# Clicking the blue chevron folds and restores only that command's output
def test_12_fold_command_output(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._composer.styles.padding.top == 1
            assert app._log.styles.padding.bottom == 0
            app._input.value = "echo fold_marker"
            await pilot.press("enter")
            await wait_for(pilot, lambda: not app._input.disabled)
            assert "fold_marker" in log_text(app)
            assert "exit 0" in log_text(app)
            await pilot.click("#log", offset=(2, 1))
            await pilot.pause()
            assert app._log._groups[0]["collapsed"]
            assert "exit 0" not in log_text(app)
            assert "▸ echo fold_marker" in log_text(app)
            app._log.write("late output")
            assert "late output" not in log_text(app)
            await pilot.click("#log", offset=(2, 1))
            await pilot.pause()
            assert not app._log._groups[0]["collapsed"]
            assert "exit 0" in log_text(app)
            assert "late output" in log_text(app)
            app._log.write_command(
                "second", app._bubble_text("second"),
                app._bubble_text("second", folded=True))
            app._log.write("second output")
            await pilot.click("#log", offset=(2, 1))
            await pilot.pause()
            assert "late output" not in log_text(app)
            assert "second output" in log_text(app)
            # Replaying a long transcript must not activate RichLog's
            # automatic scroll-to-end behavior.
            for j in range(40):
                app._log.write(f"extra output {j}")
            await pilot.pause()
            app._log.scroll_to(y=0, animate=False, immediate=True)
            await pilot.pause()
            assert int(app._log.scroll_y) == 0
            await pilot.click("#log", offset=(2, 1))
            await pilot.pause()
            assert int(app._log.scroll_y) == 0
            assert "late output" in log_text(app)
    run_async(drive())


# Multiple matches appear beside the editor, not in prior command output
def test_13_completion_box_keyboard(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = "echo previous_marker"
            await pilot.press("enter")
            await wait_for(pilot, lambda: not app._input.disabled)
            previous_output = list(app._log._groups[0]["output"])
            app._input.value = ":"
            app._input.cursor_position = 1
            await pilot.press("tab")
            await pilot.pause()
            assert app._suggestions.display
            assert app._suggestions.option_count > 1
            assert "Enter/Tab insert" in str(app._composer_hint.content)
            assert app._log._groups[0]["output"] == previous_output
            assert "previous_marker" in log_text(app)
            await pilot.press("down", "enter")
            await pilot.pause()
            assert app._input.value == ":pwd "
            assert not app._suggestions.display
            assert "TAB complete" in str(app._composer_hint.content)
            assert app._log._groups[0]["output"] == previous_output
    run_async(drive())


# Editing filters the list; clicking an option inserts it without executing
def test_14_completion_box_click(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._input.value = ":"
            app._input.cursor_position = 1
            await pilot.press("tab")
            await pilot.pause()
            app._input.value = ":h"
            app._input.cursor_position = 2
            await pilot.pause()
            assert app._suggestions.display
            assert app._suggestions.option_count == 2
            await pilot.press("escape")
            await pilot.pause()
            assert not app._suggestions.display
            assert app._input.value == ":h"
            await pilot.press("tab")
            await pilot.pause()
            await pilot.click("#suggestions", offset=(3, 1))
            await pilot.pause()
            assert app._input.value.startswith(":help")
            assert not app._suggestions.display
            assert app._stats["tui_commands"] == 0
    run_async(drive())


# Composer soft-wraps, grows to three rows, and then shrinks again
def test_18_composer_wrap_height(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(60, 24)) as pilot:
            await pilot.pause()
            assert app._input.size.height == 1
            app._input.value = "word " * 40
            app._input.cursor_position = len(app._input.value)
            await pilot.pause()
            await pilot.pause()
            assert app._input.wrapped_document.height > 3
            assert app._input.size.height == 3
            app._input.value = "short"
            await pilot.pause()
            await pilot.pause()
            assert app._input.size.height == 1
    run_async(drive())


# Multiline history records remain one physical line and round-trip intact
def test_19_multiline_history(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            cmd = "first prompt line\nsecond prompt line"
            app._record_history(cmd)
            app.save_history()
            with open(app._test_histfile, encoding="utf-8") as fp:
                records = fp.read().splitlines()
            assert len(records) == 1
            app._load_history()
            assert app._history == [cmd]
    run_async(drive())


# Ctrl-D is an additional quit shortcut alongside Textual's Ctrl-Q
def test_15_ctrl_d_quits(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            assert app.is_running
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert not app.is_running
    run_async(drive())


# In-process CLI reads update the TUI's loaded files and recency on cache hits
def test_16_json_cache_tracking(tmp_path, monkeypatch):
    from cape.cfdx import cli

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fname_a = run_dir / "a.json"
    fname_b = run_dir / "b.json"
    fname_a.write_text("{}")
    fname_b.write_text("{}")
    monkeypatch.setattr(cli, "CNTL_CACHE", OrderedDict())
    monkeypatch.setattr(
        cli, "importlib", SimpleNamespace(import_module=lambda name:
                                          SimpleNamespace(Cntl=lambda f:
                                                          SimpleNamespace(
                                                              RootDir=str(
                                                                  tmp_path)))))

    def fake_main(argv=None):
        cli.read_cntl_cache(argv[-1], solver="cfdx")
        return 0

    monkeypatch.setattr(cli, "main", fake_main)

    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            assert app._json_files == ()
            for fname, expected in (
                    (fname_a, (fname_a,)),
                    (fname_b, (fname_a, fname_b)),
                    (fname_a, (fname_b, fname_a))):
                app._input.value = f"cape -f {fname}"
                await pilot.press("enter")
                await wait_for(pilot, lambda: not app._input.disabled)
                assert app._json_files == tuple(map(str, expected))
                assert app._last_json_file == str(fname)
                assert app._last_json_display_file == os.path.join(
                    "run", fname.name)
                assert str(app._composer_hint.content) == \
                    f"CAPE file: run/{fname.name}"
            # Completion instructions temporarily take precedence, then
            # the file context returns when the suggestions are closed.
            app._input.value = ":"
            app._input.cursor_position = 1
            await pilot.press("tab")
            await pilot.pause()
            assert "Enter/Tab insert" in str(app._composer_hint.content)
            await pilot.press("escape")
            await pilot.pause()
            assert str(app._composer_hint.content) == "CAPE file: run/a.json"
            app._input.value = ""
            # A missing file does not alter the successful-use order.
            app._input.value = f"cape -f {tmp_path / 'missing.json'}"
            await pilot.press("enter")
            await wait_for(pilot, lambda: not app._input.disabled)
            assert app._json_files == (str(fname_b), str(fname_a))
            assert app._last_json_file == str(fname_a)
            stats = app.finalize_stats()
            assert stats["last_json_file"] == str(fname_a)
            assert stats["last_json_display_file"] == "run/a.json"
            assert stats["json_files"] == (str(fname_b), str(fname_a))
            panel = session_stats_panel(stats, title="CAPE TUI summary")
            labels = list(panel.renderable.columns[0].cells)
            values = list(panel.renderable.columns[1].cells)
            assert values[labels.index("JSON file:")] == "run/a.json (+1)"
    run_async(drive())


# Dragging over log text selects it and copies it to the clipboard
def test_17_drag_select_copies(tmp_path, monkeypatch):
    async def drive():
        app = make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._log.write("copyable one")
            app._log.write("copyable two")
            await pilot.pause()
            # Log padding (top=1, left=2) shifts content coordinates
            await pilot.mouse_down("#log", offset=(2, 1))
            await pilot.mouse_up("#log", offset=(14, 2))
            await pilot.pause()
            text = app.screen.get_selected_text()
            assert text == "copyable one\ncopyable two"
            assert app._clipboard == text
            # The selection highlight survives until the next click
            assert app._log.text_selection is not None
            selection_style = app.screen.get_component_rich_style(
                "screen--selection")
            assert selection_style.bgcolor.triplet == (38, 79, 120)
            assert selection_style.color.triplet == (255, 255, 255)
            selected_strip = app._log.render_line(0)
            selected_segment = selected_strip._segments[0]
            assert selected_segment.style.bgcolor.triplet == (38, 79, 120)
            assert selected_segment.style.color.triplet == (255, 255, 255)
            await pilot.click("#body", offset=(5, 20))
            await pilot.pause()
            assert app._log.text_selection is None
    run_async(drive())

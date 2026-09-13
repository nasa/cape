
# Standard library
import asyncio

# Third-party
import pytest

# Local imports
import cape.promptutils as pu

# Only run these tests if the optional textual package is installed
textual = pytest.importorskip("textual")


# Run an async coroutine without needing a pytest async plugin
def run_async(coro):
    return asyncio.run(coro)


# Drive a clickable prompt app and return its reply
async def drive(app, *presses, click=None):
    async with app.run_test() as pilot:
        await pilot.pause()
        if click is not None:
            selector, offset = click
            await pilot.click(selector, offset=offset or (0, 0))
        else:
            await pilot.press(*presses)
    return app.return_value


# Clicking an option answers "@N" (1-based)
def test_01_click_option():
    vopt = ["next", "extend", "skip"]
    for j in range(len(vopt)):
        app = pu._new_click_prompt("pick", "skip", vopt)
        # First option row is offset y=1 due to OptionList border
        vraw = run_async(drive(app, click=("#prompt-opts", (2, j + 1))))
        assert vraw == f"@{j + 1}"


# Typing free text and pressing Enter passes text through
def test_02_typed_reply():
    app = pu._new_click_prompt("pick", "skip", ["next", "extend", "skip"])
    vraw = run_async(drive(app, "h", "e", "l", "l", "o", "enter"))
    assert vraw == "hello"


# Typing "@N" in the text box answers "@N", like the readline prompt
def test_03_typed_at():
    app = pu._new_click_prompt("pick", "skip", ["next", "extend", "skip"])
    vraw = run_async(drive(app, "@", "2", "enter"))
    assert vraw == "@2"


# Empty input (just Enter) accepts the default, like the readline prompt
def test_04_empty_reply():
    app = pu._new_click_prompt("pick", "skip", ["next", "extend", "skip"])
    vraw = run_async(drive(app, "enter"))
    assert vraw == ""


# Escape accepts the default
def test_05_escape():
    app = pu._new_click_prompt("pick", "skip", ["next", "extend", "skip"])
    vraw = run_async(drive(app, "escape"))
    assert vraw == ""


# Arrow keys move focus to the option list; Enter selects it
def test_06_arrow_select():
    app = pu._new_click_prompt("pick", "next", ["next", "extend", "skip"])
    # First down moves focus to the option list (highlighting default,
    # "@1"); second down moves highlight to the next option
    vraw = run_async(drive(app, "down", "down", "enter"))
    assert vraw == "@2"


# The default option is highlighted on startup
def test_07_default_highlight():
    app = pu._new_click_prompt("pick", "extend", ["next", "extend", "skip"])

    async def check():
        async with app.run_test():
            opts = app.query_one("#prompt-opts")
            assert opts.highlighted == 1

    run_async(check())


# One-line mode (buttons) answers by clicking a button
def test_08_oneline_buttons():
    app = pu._new_click_prompt("delete?", "n", ["y", "n"], oneline=True)
    vraw = run_async(drive(app, click=("#prompt-opt-0", None)))
    assert vraw == "@1"


# Ctrl-C quits the app with no reply (maps to KeyboardInterrupt)
def test_09_ctrl_c():
    app = pu._new_click_prompt("pick", "skip", ["next", "extend", "skip"])
    vraw = run_async(drive(app, "ctrl+c"))
    assert vraw is None

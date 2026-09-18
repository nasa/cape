r"""
:mod:`cape.promptutils`: Simple tools for interactive CLI prompts
===================================================================

This module provides tools for auto-completion, colored formatting, and
more when prompting users for values interactively.

When the optional third-party ``textual`` package is installed and
standard input and output are both terminals with a reasonable
``$TERM``, prompts that present a list of options (via *vopt*) are
shown as a clickable menu instead: clicking an option (or pressing
``Enter`` on it) selects it, just like typing ``@N`` on the readline
prompt. Free text, ``@N`` answers, and empty input to accept the
default still work through a text box in the menu. This clickable
backend can be controlled with *clickable*, and the
``$CAPE_PROMPT_CLICK`` environment variable can be set to ``always``
or ``never`` to override auto-detection.
"""

# Standard library
import fnmatch
import glob
import importlib.util
import os
import re
import readline
import sys
from typing import Any, Callable, Optional

# Local imports
from .argread.clitext import CONSOLE

# Third-party OPTIONAL
try:
    from colorama import init

    # Initialize colorama to support ANSI escape codes on Windows
    init(autoreset=True)
except Exception:
    pass


# Regular expression to recognize "@{n}" entries
REGEX_AT = re.compile("@([0-9]+)")

# Name of environment variable to control clickable prompts
ENVVAR_PROMPT_CLICK = "CAPE_PROMPT_CLICK"

# Values of $CAPE_PROMPT_CLICK to force the clickable prompt backend
CLICKABLE_TRUE = ("1", "on", "true", "yes", "always")
CLICKABLE_FALSE = ("0", "off", "false", "no", "never")

# Cached result of clickable-prompt auto-detection
_CLICKABLE_OK: Optional[bool] = None

# Registered host handler for prompts (e.g. the CAPE TUI), if any
_PROMPT_HANDLER: Optional[Callable] = None

# Generic completer settings
readline.set_completer_delims(' \t\n')
readline.parse_and_bind("tab: complete")


class PromptCompleter:
    __slots__ = (
        "glob",
        "vopt",
        "func",
    )

    def __init__(self, glob: bool = False, vopt: Optional[list] = None):
        self.glob = glob
        self.vopt = vopt
        self.func = None

    def __call__(self, text: str, state: int) -> Optional[str]:
        # Get list of suggestions starting with *text*
        suggestions = self.genr8_suggestions(text)
        # Append ``None`` (for no-match case) and index it
        suggestions.append(None)
        return suggestions[state]

    def genr8_suggestions(self, text: str) -> list:
        # Initialize list of values
        suggestions = []
        # Get list of values
        if isinstance(self.vopt, (tuple, list)):
            suggestions.extend(fnmatch.filter(self.vopt, f"{text}*"))
        # Complete on file names if appropriate
        if self.glob:
            suggestions.extend(glob.glob(f"{text}*"))
        # Add any extra suggestions (custom)
        extra = self.genr8_extra_suggestions(text)
        suggestions.extend(extra)
        # Check for extra function
        if callable(self.func):
            custom = self.func(text)
            if isinstance(custom, (list, tuple)):
                suggestions.extend(custom)
        # Output
        return suggestions

    def genr8_extra_suggestions(self, text: str) -> list:
        return []


def complete_glob(text: str, state: int) -> Optional[str]:
    r"""Return a tab-completion suggestion based on file names

    :Call:
        >>> suggestion = complete_glob(text, state)
    :Inputs:
        *text*: :class:`str`
            User input so far
        *state*: :class:`int`
            Return the suggestion of index *state*, (usually ``0``)
    :Outputs:
        *suggestion*: :class:`str` | ``None``
            A file whose name starts with *text* if possible
    """
    return (glob.glob(text + '*') + [None])[state]


# Function to get user input using a colored prompt
def prompt_color(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        color: str = "green",
        prompt: str = '>',
        completer: Optional[Callable] = None,
        glob: bool = False,
        show: bool = True,
        oneline: bool = False,
        pcolor: Optional[str] = None,
        atcolor: Optional[str] = None,
        ocolor: Optional[str] = None,
        clickable: Optional[bool] = None,
        erase: bool = True) -> Any:
    r"""Get user input using a colorized prompt

    :Call:
        >>> v = prompt_color(txt, vdef=None)
    :Inputs:
        *txt*: :class:`str`
            Text of the question on the same line as prompt
        *vdef*: {``None``} | :class:`object`
            Default value (if any)
        *vopt*: {``None``} | :class:`list`
            List of possible or suggested values (optional)
        *color*: {``"green"``} | :class:`str`
            Color name for entire prompt
        *completer*: {``None``} | **callable**
            Function to return list of suggestions given current text
        *prompt*: {``">"``} | :class:`str`
            Character(s) to use as prompt
        *glob*: ``True`` | {``False``}
            Use glob to suggest file-name-based completions
        *show*: {``True``} | ``False``
            Print the final selection for confirmation
        *oneline*: ``True`` | {``False``}
            Option to display values as compact one-line list ``[y]/n``
        *clickable*: {``None``} | ``True`` | ``False``
            If ``True``, use a clickable menu for option lists (only
            when the ``textual`` package is installed); if ``False``,
            always use the colored readline prompt; if ``None``,
            auto-detect a TUI-capable terminal (see also
            :func:`clickable_prompt_ok`)
        *erase*: {``True``} | ``False``
            If ``True``, erase the option list and prompt line (the
            ``n+1`` lines below the question for ``n`` options) from
            the screen after the user answers, leaving only the
            question and the final selection; only applies to
            multi-line option lists on a terminal
    :Outputs:
        *v*: :class:`str` | *vdef* | ``vopt[j]``
            User input or default value
    """
    # Default vdef --> vopt
    vopt = vdef if (vopt is None) else vopt
    # Convert extra color options to color
    pcol = '' if pcolor is None else CONSOLE.get(pcolor, '')
    acol = '' if pcolor is None else CONSOLE.get(atcolor, '')
    ocol = '' if pcolor is None else CONSOLE.get(ocolor, '')
    # Args passed to option printers
    args = (txt, vdef, vopt, prompt, pcol, acol, ocol)
    # Three versions of option list; two will be empty
    if oneline:
        msg1 = _dumps_vopt_oneline(*args)
    else:
        msg1 = _dumps_vopt_list(*args)
    msg2 = _dumps_vdef(*args)
    msg3 = _dumps_plain(*args)
    # Combine all three
    msg = msg1 + msg2 + msg3
    # Create a completer
    comp = PromptCompleter(glob, vopt)
    # Check for custom function
    comp.func = completer
    # Turn custom completin class on
    readline.set_completer(comp)
    # Substantiate default
    vdef = vopt if vdef is None else vdef
    vdef = vdef if not isinstance(vdef, list) else vdef[0]
    # Check for a registered prompt handler (e.g. the CAPE TUI)
    handler = _PROMPT_HANDLER
    # Number of lines (options + prompt line) to erase after answering
    nerase = 0
    if erase and msg1 and not oneline:
        nerase = len(vopt) + 1
    if handler is not None:
        # Let the host render the prompt itself
        vraw = handler(txt, vdef, vopt, prompt, oneline)
        nerase = 0
    elif isinstance(vopt, (list, tuple)) and clickable_prompt_ok(clickable):
        try:
            # Use clickable menu (returns raw reply, like :func:`input`)
            vraw = prompt_click(txt, vdef, vopt, prompt, oneline)
        except KeyboardInterrupt:
            raise
        except Exception:
            # Fall back to colored readline prompt
            vraw = input_color(msg, color)
        else:
            # Clickable menu leaves its own rendering in the stream
            nerase = 0
    else:
        # Read input from command line (ignore lead/trail spaces)
        vraw = input_color(msg, color)
    # Check if it's an "@"
    if msg1 and REGEX_AT.fullmatch(vraw):
        # Get the number provided by user
        n = int(REGEX_AT.match(vraw).group(1))
        # Return that value (0-based)
        v = vopt[n - 1]
    elif vdef and (not vraw):
        # Use the default value instead
        v = vdef
    else:
        # Return the user's value, even if empty
        v = vraw
    # Erase the option list and prompt line from the screen
    if nerase:
        _erase_prompt_lines(nerase)
    # Inform user what value was used
    if show:
        print(f"--> using '{v}'")
    # Output
    return v


# Make a raw request
def input_color(prompt: str, color: str = "black") -> str:
    r"""Modify built-in :func:`input` to also set color

    :Call:
        >>> raw = input_color(prompt, color)
    :Inputs:
        *prompt*: :class:`str`
            Text to display prior to requesting input
        *color*: :class:`str`
            Common name of color to use for prompt
    :Outputs:
        *raw*: :class:`str`
            User's input
    """
    # Get color
    col = CONSOLE.get(color, '')
    reset = CONSOLE["reset"] if color else ''
    # Form a prompt with formatting
    prompt_txt = f"{col}{prompt}{reset}"
    # Request a response
    return input(prompt_txt).strip()


# Delete the most recent lines of terminal output
def _erase_prompt_lines(nlines: int) -> None:
    r"""Delete the most recent lines of terminal output

    Moves the cursor up *nlines* lines and clears from there to the end
    of the screen, effectively erasing lines that were just printed
    (e.g. an option-list prompt) once the user has answered. Does
    nothing if standard output is not a terminal.

    :Call:
        >>> _erase_prompt_lines(nlines)
    :Inputs:
        *nlines*: :class:`int`
            Number of preceding lines to erase
    """
    if (nlines <= 0) or not getattr(sys.stdout, "isatty", lambda: False)():
        return
    # Move cursor up to first line to erase; clear to end of screen
    sys.stdout.write(f"\x1b[{nlines}F\x1b[J")
    sys.stdout.flush()
    return


# Display simple list of options
def _dumps_vopt_oneline(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>',
        *args) -> str:
    # Check if options give are a list
    if not isinstance(vopt, (list, tuple)):
        return ''
    # Initial portion of prompt using pre-specified prompt
    msg = f"{txt} "
    # Default default value is first entry in *vopt_list*
    if vdef is None:
        vdef = vopt[0]
    # Check if given a list of default values
    if isinstance(vdef, (list, tuple)):
        # Use the first
        vdef = vdef[0]
    # Check if default is in option; if soe get the index
    jdef = None if vdef not in vopt else vopt.index(vdef)
    # Loop through options
    for j, opt in enumerate(vopt):
        # Add separator
        if j > 0:
            msg += '/'
        # Format message
        if j == jdef:
            # Highlight first option as the true default
            msgj = f"[{opt}]"
        else:
            # Use option number
            msgj = opt
        # Append to overall prompt
        msg += msgj
    # Append user input prompt
    return msg + prompt + ' '


# Display list of options
def _dumps_vopt_list(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>',
        pcol: str = '',
        acol: str = '',
        ocol: str = '') -> str:
    # Check if options give are a list
    if not isinstance(vopt, (list, tuple)):
        return ''
    # Resets
    rp = CONSOLE["reset"] if pcol else ''
    ra = CONSOLE["reset"] if acol else ''
    ro = CONSOLE["reset"] if ocol else ''
    # Initial portion of prompt using pre-specified prompt
    msg = f"{txt}:\n"
    # Default default value is first entry in *vopt_list*
    if vdef is None:
        vdef = vopt[0]
    # Check if given a list of default values
    if isinstance(vdef, (list, tuple)):
        # Use the first
        vdef = vdef[0]
    # Check if default is in option; if soe get the index
    jdef = None if vdef not in vopt else vopt.index(vdef)
    # Loop through options
    for j, opt in enumerate(vopt):
        # Format message
        if j == jdef:
            # Highlight first option as the true default
            msgj = f"    {acol}@{j+1}{ra}: [{ocol}{opt}{ro}]\n"
        else:
            # Use option number
            msgj = f"    {acol}@{j+1}{ra}: {ocol}{opt}{ro}\n"
        # Append to overall prompt
        msg += msgj
    # Append user input prompt
    return msg + pcol + prompt + rp + ' '


# Display list of options
def _dumps_vdef(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>', *args) -> str:
    # Check if options give are a list
    if isinstance(vopt, (list, tuple)):
        return ''
    # Check for any reasonable default
    if (vdef is None) and (vopt is None):
        return ''
    # Use *vdef* or *vopt*
    vdef = vopt if vdef is None else vdef
    # Form the prompt with single default value
    return f"{txt} [{vdef}]:\n{prompt} "


# Display with no list or default
def _dumps_plain(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>', *args) -> str:
    # Check for any default/option list; use prior function
    if (vdef is not None) or (vopt is not None):
        return ''
    # Form the prompt without default value
    return f"{txt}:\n{prompt} "


# Register a host handler for interactive prompts
def register_prompt_handler(handler: Optional[Callable]) -> Optional[Callable]:
    r"""Register (or clear) a host handler for interactive prompts

    If set, :func:`prompt_color` calls *handler* to obtain the user's
    raw reply instead of rendering a prompt itself. The handler
    signature mirrors :func:`prompt_click` and it must return a raw
    reply with the same format (i.e. ``"@{j+1}"`` for option *j*,
    typed text, or ``""`` to accept the default); it may raise
    :class:`KeyboardInterrupt` to abort. This allows a host
    application, such as the CAPE TUI, to render prompts in its own
    interface while its commands run in another thread.

    :Call:
        >>> prev_handler = register_prompt_handler(handler)
    :Inputs:
        *handler*: **callable** | {``None``}
            Prompt handler to register, or ``None`` to clear
    :Outputs:
        *prev_handler*: {``None``} | **callable**
            Previously registered handler (for restoration)
    """
    global _PROMPT_HANDLER
    # Save previous handler
    prev_handler = _PROMPT_HANDLER
    # Install new handler
    _PROMPT_HANDLER = handler
    # Output
    return prev_handler


# Check if the optional textual package is installed
def _textual_available() -> bool:
    try:
        return importlib.util.find_spec("textual") is not None
    except Exception:
        return False


# Check if clickable prompts are (or should be) available
def clickable_prompt_ok(clickable: Optional[bool] = None) -> bool:
    r"""Check if clickable option menus are currently available

    Clickable prompts require the optional third-party ``textual``
    package to be installed. Unless forced using *clickable* or the
    ``$CAPE_PROMPT_CLICK`` environment variable, clickable prompts are
    only activated if standard input and output are both terminals with
    a reasonable ``$TERM``.

    :Call:
        >>> ok = clickable_prompt_ok(clickable=None)
    :Inputs:
        *clickable*: {``None``} | ``True`` | ``False``
            If ``True`` or ``False``, force clickable prompts on or
            off; if ``None``, consult ``$CAPE_PROMPT_CLICK`` and then
            auto-detect the terminal
    :Outputs:
        *ok*: ``True`` | ``False``
            Whether clickable option menus should be used
    """
    global _CLICKABLE_OK
    # Check for explicit on/off argument
    if clickable is not None:
        return _textual_available() if clickable else False
    # Check for environment variable override
    env = os.environ.get(ENVVAR_PROMPT_CLICK, '').strip().lower()
    if env in CLICKABLE_FALSE:
        return False
    if env in CLICKABLE_TRUE:
        return _textual_available()
    # Use cached result of auto-detection
    if _CLICKABLE_OK is None:
        _CLICKABLE_OK = (
            _textual_available() and
            sys.stdin.isatty() and
            sys.stdout.isatty() and
            os.environ.get("TERM", "dumb") not in ("", "dumb"))
    return _CLICKABLE_OK


# Run a clickable menu; return a raw reply like :func:`input_color`
def prompt_click(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>',
        oneline: bool = False) -> str:
    r"""Run a clickable option prompt and return the user's reply

    The menu is rendered inline (when the terminal supports it), so
    that the question and options remain part of the terminal stream
    along with the user's answer; on terminals without inline-rendering
    support, it retries in full-screen mode. The reply has the same
    format as :func:`input_color`: clicking option *j* of *vopt*
    answers ``"@{j+1}"``, typing free text answers that text, and
    pressing ``Escape`` (like empty input) answers ``""``. Such replies
    can be parsed using the same logic as :func:`prompt_color`
    (i.e. ``@N`` selects ``vopt[N-1]``).

    :Call:
        >>> vraw = prompt_click(txt, vdef, vopt, prompt, oneline)
    :Inputs:
        *txt*: :class:`str`
            Text of the question
        *vdef*: {``None``} | :class:`object`
            Default value (if any)
        *vopt*: {``None``} | :class:`list`
            List of possible or suggested values
        *prompt*: {``">"``} | :class:`str`
            Character(s) to use as prompt
        *oneline*: ``True`` | {``False``}
            Use compact one-line buttons instead of a list
    :Outputs:
        *vraw*: :class:`str`
            User's raw reply
    :Raises:
        *KeyboardInterrupt*: if the user quits the prompt menu
    """
    # Create the app
    app = _new_click_prompt(txt, vdef, vopt, prompt, oneline)
    try:
        # Run app inline, leaving the menu in the terminal stream
        vraw = app.run(inline=True, inline_no_clear=True)
    except Exception:
        # If inline rendering fails, retry in full-screen mode
        app = _new_click_prompt(txt, vdef, vopt, prompt, oneline)
        vraw = app.run()
    # Check for quit (e.g. Ctrl-C); mimic readline Ctrl-C behavior
    if vraw is None:
        raise KeyboardInterrupt
    # Output
    return vraw


# Create a textual widget for a clickable prompt
def _new_prompt_widget(
        txt: str,
        vdef: Optional[Any],
        vopt: list,
        prompt: str = '>',
        oneline: bool = False,
        on_answer: Optional[Callable] = None):
    r"""Create a textual widget rendering a clickable option menu

    This function performs lazy imports of the optional ``textual``
    package, so calling it requires ``textual`` to be installed (see
    :func:`clickable_prompt_ok`). The widget can be run alone inside a
    small :class:`~textual.app.App` (as :func:`prompt_click` does) or
    mounted into a larger application, such as the CAPE TUI.

    :Call:
        >>> widget = _new_prompt_widget(txt, vdef, vopt, **kw)
    :Inputs:
        *txt*: :class:`str`
            Text of the question
        *vdef*: {``None``} | :class:`object`
            Default value (if any)
        *vopt*: :class:`list`
            List of options to display
        *prompt*: {``">"``} | :class:`str`
            Character(s) to use as prompt
        *oneline*: ``True`` | {``False``}
            Use compact one-line buttons instead of a list
        *on_answer*: {``None``} | **callable**
            Called with the user's raw reply (same format as
            :func:`prompt_click`); ``None`` on cancel (Ctrl-C)
    :Outputs:
        *widget*: :class:`textual.widget.Widget`
            Clickable option-menu widget
    """
    # Lazy imports of optional third-party textual package
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, Input, Label, OptionList

    # Index of default value in *vopt*, if any
    try:
        jdef = list(vopt).index(vdef)
    except ValueError:
        jdef = None
    # String version of default value for input placeholder
    vdef_txt = '' if vdef is None else str(vdef)
    # Compose option strings, highlighting the default
    opt_txts = [
        f"[{opt}]" if j == jdef else str(opt)
        for j, opt in enumerate(vopt)]

    # Define the widget class
    class PromptWidget(Vertical):
        # Style settings
        CSS = (
            "#prompt-root {\n"
            "    width: 100%;\n"
            "    height: auto;\n"
            "    padding: 1 2;\n"
            "}\n"
            "OptionList {\n"
            "    height: auto;\n"
            "    max-height: 16;\n"
            "}\n"
            "Horizontal {\n"
            "    height: auto;\n"
            "}\n"
            "Button {\n"
            "    margin-right: 1;\n"
            "}\n")
        # Key bindings
        BINDINGS = [
            Binding("ctrl+c", "cancel", show=False, priority=True),
            Binding("escape", "use_default", show=False),
            Binding("up", "focus_opts", show=False),
            Binding("down", "focus_opts", show=False),
        ]

        def __init__(self):
            # Initialize base container
            super().__init__(id="prompt-root")
            # Answer callback
            self.on_answer = on_answer

        def compose(self):
            # Question text
            yield Label(str(txt))
            # Render the option list
            if oneline:
                # Compact one-line buttons, e.g. "delete? [y] n"
                with Horizontal(id="prompt-opts"):
                    for j, otxt in enumerate(opt_txts):
                        yield Button(otxt, id=f"prompt-opt-{j}")
            else:
                # Scrollable list of options
                yield OptionList(*opt_txts, id="prompt-opts")
            # Free-text box, mirroring the readline prompt
            yield Input(placeholder=vdef_txt, id="prompt-inp")

        def on_mount(self) -> None:
            # Focus the free-text box by default
            self.query_one("#prompt-inp", Input).focus()
            # Highlight the default option
            if not oneline:
                opts = self.query_one("#prompt-opts", OptionList)
                opts.highlighted = 0 if jdef is None else jdef

        # Pass the user's reply to the answer callback
        def _answer(self, vraw) -> None:
            if self.on_answer is not None:
                self.on_answer(vraw)

        def on_option_list_option_selected(
                self, event: OptionList.OptionSelected) -> None:
            # Don't let host apps see this event
            event.stop()
            # Clicking (or Enter-ing) an option answers "@{j+1}"
            self._answer(f"@{event.option_index + 1}")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            # Parse button ID of the form "prompt-opt-{j}"
            btnid = event.button.id or ''
            if btnid.startswith("prompt-opt-"):
                event.stop()
                self._answer(f"@{int(btnid[11:]) + 1}")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            # Don't let host apps see this event
            event.stop()
            # Use the user's typed text
            self._answer(event.value.strip())

        def action_cancel(self) -> None:
            # Ctrl-C: abort the prompt (maps to :class:`KeyboardInterrupt`)
            self._answer(None)

        def action_use_default(self) -> None:
            # Escape key: accept default value (empty reply)
            self._answer('')

        def action_focus_opts(self) -> None:
            # Arrow key from free-text box: move to the option list
            self.query_one("#prompt-opts").focus()

    # Return an instance
    return PromptWidget()


# Create the textual App for a clickable prompt
def _new_click_prompt(
        txt: str,
        vdef: Optional[Any],
        vopt: list,
        prompt: str = '>',
        oneline: bool = False):
    r"""Create a textual :class:`App` for a clickable option menu

    This function performs lazy imports of the optional ``textual``
    package, so calling it requires ``textual`` to be installed (see
    :func:`clickable_prompt_ok`).

    :Call:
        >>> app = _new_click_prompt(txt, vdef, vopt, prompt, oneline)
    :Inputs:
        *txt*: :class:`str`
            Text of the question
        *vdef*: {``None``} | :class:`object`
            Default value (if any)
        *vopt*: :class:`list`
            List of options to display
        *prompt*: {``">"``} | :class:`str`
            Character(s) to use as prompt
        *oneline*: ``True`` | {``False``}
            Use compact one-line buttons instead of a list
    :Outputs:
        *app*: :class:`textual.app.App`
            App that exits with the user's raw reply; ``None`` on quit
    """
    # Lazy import of optional third-party textual package
    from textual.app import App

    # Define the app class, hosting one prompt widget
    class ClickPrompt(App):
        def __init__(self):
            super().__init__()
            self._pw = _new_prompt_widget(
                txt, vdef, vopt, prompt, oneline, on_answer=self.exit)

        def compose(self):
            yield self._pw

    # Return an instance
    return ClickPrompt()

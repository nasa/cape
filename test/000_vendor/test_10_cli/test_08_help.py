
# Local imports
from argread import ArgReader


# Custom class to delegate to subfunctions
class HelpDesk(ArgReader):
    __slots__ = ()

    _optlist = {
        "h",
        "v",
    }

    _optmap = {
        "help": "h",
    }

    _cmdlist = (
        "run",
        "stop",
    )

    _help_cmd = {
        "run": "Run one phase",
        "stop": "Stop current run",
    }


# Custom class with help messages
class HelpReader(ArgReader):
    __slots__ = ()

    _optlist = (
        "cmd",
        "f",
        "h",
        "subcmd",
    )

    _optmap = {
        "help": "h",
    }

    _arglist = (
        "cmd",
        "subcmd",
    )

    _nargmin = 1

    _rc = {
        "f": "argread.json",
        "subcmd": "read",
    }

    _name = "argread-test"

    _help_opt = {
        "cmd": "name of command to run",
        "f": "file to use",
        "h": "show this help message and exit",
        "subcmd": "name of subcommand to run",
    }

    _help_optarg = {
        "f": "JSON",
    }


# Test some features of docstrings
def test_help01():
    # Instantiate
    opts = HelpReader()
    # Generate help message for just "h"
    msg = opts.genr8_opthelp("h")
    assert msg.strip().startswith("-h, --help")
    # Generate help message for just "f"
    msgs = opts.genr8_opthelp("f").split('\n')
    assert "JSON" in msgs[0]
    assert HelpReader._help_opt["f"] in msgs[1]
    # Show help message
    msg = opts.genr8_help()
    assert msg.startswith("``argread-test``")


# Test help-desk message
def test_help02():
    # Instantiate
    opts = HelpDesk()
    # Generate help message
    usage = opts._genr8_help_usage()
    cmdhelp = opts._genr8_help_cmdlist()
    msg = opts.genr8_help()
    # Make sure it's reasonable
    assert "``stop``" in cmdhelp
    assert "CMD" in usage
    assert cmdhelp in msg
    assert ":Options:" in msg

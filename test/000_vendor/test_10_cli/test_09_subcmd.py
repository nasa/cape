

# Local
from argread import ArgReader


def test_subcmd01():
    # Create a parser
    parser = ArgReader()
    # Identify subcommand
    cmdname, subparser = parser.fullparse(
        ["p", "show", "more", "a=True", "-cj"])
    argv = subparser.reconstruct()
    # Check results
    assert cmdname == "show"
    assert argv == ["p-show", "more", "a=True", "-cj"]


def test_subcmd02():
    # Create a parser
    parser = ArgReader()
    # Sub-parse w/o any subcommand
    cmdname, subparser = parser.fullparse(["argread", "-f", "1", "--no-j"])
    assert cmdname is None
    assert subparser is parser
    # Inputs should already be parsed
    assert parser.reconstruct() == ["argread", "-f", "1", "--no-j"]

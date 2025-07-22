
# Local imports
from argread.clitext import (
    BOLD,
    BOLDITALIC,
    ITALIC,
    PLAIN,
    compile_rst)


# Test some features of docstrings
def test_clitext01():
    # Mark a code block
    txt1 = """.. code-block:: console

    $ ls

something"""
    # Dedent and remove directive and blank line
    out1 = compile_rst(txt1)
    assert out1 == "\n$ ls\n\nsomething"
    # Test bold
    txt2 = "**emph**"
    # Insert bold chars?
    out2 = compile_rst(txt2)
    assert out2 == f"{BOLDITALIC}emph{PLAIN}"
    # Test italic
    assert compile_rst("*it*") == f"{ITALIC}it{PLAIN}"
    # Test section
    txt3 = ":Usage:\n"
    out3 = compile_rst(txt3)
    assert out3 == "USAGE\n\n"
    # Test "roles"
    assert compile_rst(":func:`f`") == "f()"
    assert compile_rst(":class:`A`") == "A"
    # Test UID
    assert compile_rst("``@ddalle``") == "@ddalle"
    # Test general literal
    assert compile_rst("``fname``") == f"{BOLD}fname{PLAIN}"
    # Test section
    assert compile_rst("Title\n========") == "Title"



# Local
from cape.argread import ArgReader, readkeys, readkeys_full


def test_readkeys01():
    a, kw = readkeys(["p", "-cj"])
    assert a == []
    assert kw == {
        "cj": True,
    }


def test_readkeys02():
    a, kw = readkeys(["p", "a", "--extend", "2", "b"])
    assert a == ["a", "b"]
    assert kw == {
        "extend": "2",
    }


def test_readkeys03():
    a, kw = readkeys(["p", "a", "-v", "--qsub", "--no-start"])
    assert a == ["a"]
    assert kw == {
        "v": True,
        "qsub": True,
        "start": False,
    }


def test_readkeys04():
    a, kw = readkeys(["p", "a", "q=devel", "c=1"])
    assert a == ["a"]
    assert kw == {
        "q": "devel",
        "c": "1",
    }


def test_readkeys05():
    a, kw = readkeys_full(["p", "-x", "1.py", "-x", "2.py", "-x", "3.py"])
    assert a == []
    assert kw == {
        "x": "3.py",
        "__replaced__": [
            ("x", "1.py"),
            ("x", "2.py")
        ],
    }


def test_argtuple01():
    # Parse CLI w/ repeated option
    parser = ArgReader()
    parser.parse(["p", "a", "-x", "1.py", "-x", "2.py", "-x", "3.py"])
    # Parse as tuple
    arglist = parser.get_argtuple()
    # Check value
    assert arglist == (
        ("arg1", "a"),
        ("x", "1.py"),
        ("x", "2.py"),
        ("x", "3.py"),
    )
    # Parse a dict
    argdict = parser.get_argdict()
    # Check it
    assert argdict == {
        "arg1": "a",
        "x": "3.py",
    }

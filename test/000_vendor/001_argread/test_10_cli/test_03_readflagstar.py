

# Local
from cape.argread import readflagstar


def test_readflagstar01():
    a, kw = readflagstar(["p", "-cj"])
    assert a == []
    assert kw == {
        "c": True,
        "j": True,
    }


def test_readflagstar02():
    a, kw = readflagstar(["p", "a", "-cf", "fname", "-v", "--qsub"])
    assert a == ["a"]
    assert kw == {
        "c": True,
        "f": "fname",
        "v": True,
        "qsub": True,
    }



# Local imports
from cape.optdict import opterror


# Trivial tests
def test_isinstance01():
    opterror.assert_isinstance(1, None)


# Errors
def test_isinstance02():
    try:
        opterror.assert_isinstance("a", (int, list))
    except opterror.OptdictTypeError as err:
        assert str(err).startswith("G")
    else:
        assert False

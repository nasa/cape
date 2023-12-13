
# Standard library
import re

# Third-party
import pytest

# Local imports
from cape.nmlfile.nmlerr import (
    NmlTypeError,
    NmlValueError,
    assert_isinstance,
    assert_nextchar,
    assert_regex)


# Test type checker
def test_isinstance():
    # Null check
    assert_isinstance(3, None)
    # Valid check
    assert_isinstance(3, (int, float))
    # Failures
    with pytest.raises(NmlTypeError):
        assert_isinstance(3, float, "some float")
    with pytest.raises(NmlTypeError):
        assert_isinstance(3, (float, str))


# Character checker
def test_nextchar():
    # Pass
    assert_nextchar('c', 'cab')
    # Failures
    with pytest.raises(NmlValueError):
        assert_nextchar(',', ':=')
    with pytest.raises(NmlValueError):
        assert_nextchar('8', '01234567', 'octal digit')


# Regex checker
def test_regex():
    # Create a regex
    pat = re.compile(r"\([1-9][0-9]*:[1-9][0-9]*\)")
    # Pass
    assert_regex("(1:3)", pat)
    # Failures
    with pytest.raises(NmlValueError):
        assert_regex("(0:3)", pat)
    with pytest.raises(NmlValueError):
        assert_regex("(0:3)", pat, "namelist index range")

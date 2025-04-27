
# Third-party
import numpy as np
import pytest

# Local imports
from cape.nmlfile import (
    NmlFile,
    NmlIndexError,
    NmlTypeError,
    NmlValueError)


SEC1 = {
    "a": True,
    "b": 3,
    "name": "verygood",
    "c": np.array([1., -0.5, 0.0])
}

SEC2 = {
    "b": [True, True, True, False],
}

OPTS = {
    "sec1": SEC1,
    "sec2": SEC2,
}


# Test invalid defns
def test_nml01():
    # Too many inputs
    with pytest.raises(NmlTypeError):
        NmlFile(OPTS, 1)
    # Bad type
    with pytest.raises(NmlTypeError):
        NmlFile(1)


# Test some basic not-from-file instantiations
def test_nml02():
    # Use some kwargs
    nml = NmlFile(sec1=SEC1)
    # Get some options
    assert nml.get_opt("sec1", "a") is True
    assert nml.get_opt("sec1", "name") == SEC1["name"]
    assert nml.get_opt("sec1", "c", j=1) == SEC1["c"][1]
    # Read from arg
    nml = NmlFile(OPTS)
    # Check something
    assert nml.get_opt("sec1", "name") == SEC1["name"]


# Test set_opt() in detail
def test_setopt01():
    # Instantiate
    nml = NmlFile(sec1=SEC1)
    # Save a couple chars by save secname
    sec = "sec1"
    # Test bad *j*
    with pytest.raises(NmlTypeError):
        nml.set_opt(sec, "a", False, j=1.1)
    # Test with index
    nml.set_opt(sec, "c", 0.7, j=2)
    assert nml.get_opt(sec, "c", j=2) == 0.7
    # Extend
    nml.set_opt(sec, "c", 2.0, j=4)
    assert nml.get_opt(sec, "c", j=4) == 2.0
    # New field
    nml.set_opt(sec, "d", 2, j=2)
    assert nml.get_opt(sec, "d", j=2) == 2.0
    # Turn scalar into array
    nml.set_opt(sec, "b", 2, j=1)
    assert nml.get_opt(sec, "b", j=1) == 2
    # Set a slice
    nml.set_opt(sec, "b", np.array([3, 3]), j=slice(2, 4))
    assert nml.get_opt(sec, "b", 2) == 3
    # Implicit slice
    with pytest.raises(NmlValueError):
        nml.set_opt(sec, "b", np.array([1.0, 4.0, 2.0]), slice(1, None))
    # Slice with mismatching dimension
    with pytest.raises(NmlValueError):
        # Try to set value in 'b' as if it were 2D; but it's 1D
        nml.set_opt(sec, "b", 4, j=(1, 2))


# Test repeat sections
def test_setopt02():
    # Instantiate with two copies of "sec1"
    nml = NmlFile(sec1=[SEC1, SEC1])
    # Save a couple chars by save secname
    sec = "sec1"
    # Bad repeat index
    with pytest.raises(NmlValueError):
        nml.set_opt(sec, "b", 2, k=-1)
    # Turn off *a* in second copy
    nml.set_opt(sec, "a", False, k=1)
    assert nml[sec][0]["a"] is True
    assert nml[sec][1]["a"] is False
    # Use get_opt() on newly expanded thing
    with pytest.raises(NmlIndexError):
        nml.get_opt(sec, "a", k=3)
    # Use a valid get_opt()
    assert nml.get_opt(sec, "a", k=1) is False


# Test set_opt() with dict
def test_setopt03():
    # Instantiate
    nml = NmlFile(sec1=SEC1)
    # Save a couple chars by save secname
    sec = "sec1"
    # Set several entries at once
    val = {
        "1:5": -1,
        "2": 2,
        "3:4": 5
    }
    nml.set_opt(sec, "c", val)
    # Get value
    v = nml.get_opt(sec, "c")
    assert v[0] == -1
    assert list(v) == [-1, 2, 5, 5, -1]
    # Invalid indices
    with pytest.raises(NmlValueError):
        nml.set_opt(sec, "d", {"k": 1, "5:2:10": 2})


# Test string appending thing
def test_nml03():
    # Section and option name
    sec = "sec1"
    opt = "a"
    # Instantiate simple section
    nml = NmlFile({sec: {opt: ["two", "three"]}})
    # Check it
    assert nml.get_opt(sec, opt, j=1) == "three"

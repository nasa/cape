
# Third-party
import numpy as np
import pytest

# Local
from cape.optdict import optitem
from cape.optdict.opterror import (
    OptdictTypeError,
    OptdictKeyError,
    OptdictExprError
)


# Globals
MYLIST = [
    0,
    1,
    2,
    "a"
]

X = {
    "mach": np.linspace(0.5, 1.5, 5),
    "aoap": 0.0,
    "arch": [
        "sky",
        "sky",
        "cas",
        "cas",
        "rom"
    ],
}

MYEXPR = {
    "@expr": "$mach*$mach",
}

MYMAP = {
    "key": "arch",
    "@map": {
        "sky": 10,
        "rom": 4,
        "_default_": 20
    }
}

MYCONS = {
    "@cons": {
        "$mach <= 1.0": 1.0,
        "True": "1/$mach",
    }
}


MYCOMPOUND = {
    "key": "arch",
    "@map": {
        "sky": {
            "@expr": "np.ceil($mach * 10)"
        },
        "_default_": 10
    }
}


def test_getel01():
    v = optitem.getel(MYLIST, 2)
    assert v == MYLIST[2]


def test_getel02():
    v = optitem.getel(MYLIST)
    assert v == MYLIST


def test_getel03():
    v = optitem.getel(MYLIST, 5)
    assert v == MYLIST[-1]


def test_getel04():
    v = optitem.getel(MYLIST, 5, ring=True)
    assert v == 1


def test_getel05():
    v = optitem.getel(MYLIST, -5)
    assert v == MYLIST[0]


def test_getel06():
    v = optitem.getel(MYLIST, 1, listdepth=1)
    assert v == MYLIST


def test_getel07():
    v = optitem.getel([], 0)
    assert v is None


def test_getel_error01():
    with pytest.raises(TypeError):
        optitem.getel(MYLIST, 2.0)


def test_setel01():
    y = optitem.setel([0, 1], 2)
    assert y == 2


def test_setel02():
    y = optitem.setel([0, 1], 2, j=1)
    assert y == [0, 2]


def test_setel03():
    y = optitem.setel([0, 1], 2, j=3)
    assert y == [0, 1, 1, 2]


def test_setel04():
    y = optitem.setel([0, 1], 2, j=-4)
    assert y == [2, 0, 0, 1]


def test_setel05():
    y = optitem.setel((0, 1), 2, j=2)
    assert isinstance(y, list)
    assert y == [0, 1, 2]


def test_setel06():
    y = optitem.setel([0, 1], 2, j=1, listdepth=1)
    assert y == [[0, 1], 2]


def test_setel07():
    y = optitem.setel([0, 1], 2, j=1, listdepth=2)
    assert y == [[[0, 1]], 2]


def test_getel_expr01():
    # Working example
    assert abs(optitem.getel(MYEXPR, x=X, i=0) - 0.25) <= 1e-6
    # Test not-a-string
    try:
        optitem.getel({"@expr": 2})
    except OptdictTypeError:
        pass
    else:
        assert False
    # Test missing key
    try:
        optitem.getel({"@expr": "$m"}, x=X, i=0)
    except OptdictKeyError:
        pass
    else:
        assert False
    # Bad expression
    try:
        optitem.getel({"@expr": "$mach*sin(0.1)"}, x=X, i=0)
    except OptdictExprError:
        pass
    else:
        assert False


def test_getel_map01():
    # Test a valid map
    assert optitem.getel(MYMAP, x=X, i=0) == 10
    # This usese the _default_
    assert optitem.getel(MYMAP, x=X, i=3) == 20
    # Bad type for "key"
    try:
        assert optitem.getel({"@map": 1, "key": 1}, i=0)
    except OptdictTypeError:
        pass
    else:
        assert False
    # Missing key
    try:
        assert optitem.getel({"@map": 1, "key": "phip"}, x=X, i=0)
    except OptdictKeyError:
        pass
    else:
        assert False
    # Bad @map type
    try:
        assert optitem.getel({"@map": 1, "key": "mach"}, x=X, i=0)
    except OptdictTypeError:
        pass
    else:
        assert False


def test_getel_cons01():
    # Test a valid set of constraints
    assert optitem.getel(MYCONS, x=X, i=0) == 1
    # Bad type
    try:
        optitem.getel({"@cons": 1}, x=X, i=0)
    except OptdictTypeError:
        pass
    else:
        assert False


def test_getel_raw01():
    assert optitem.getel({"@raw": [0, 1]}, j=0) == [0, 1]


def test_getel_dict01():
    assert optitem.getel({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_getel_special01():
    # Use a @map with a bad key
    try:
        optitem.getel({"@expr": "$mach", "bad": True})
    except OptdictKeyError:
        pass
    else:
        assert False


def test_getel_compound01():
    assert optitem.getel(MYCOMPOUND, x=X, i=1) == 8


def test_getel_x01():
    # Bad type for *x* kwarg
    try:
        optitem.getel(MYEXPR, x=1, i=0)
    except OptdictTypeError:
        pass
    else:
        assert False


def test_getel_x02():
    # Test an expression whose run matrix value is a scalar
    assert optitem.getel({"@expr": "$aoap"}, x=X) == 0.0
    # Test list/tuple (not array)
    INDS = (0, 2)
    mach = optitem.getel({"@expr": "$mach"}, x=X, i=INDS)
    assert np.max(np.abs(mach - X["mach"][list(INDS)])) <= 1e-6
    # Test list/list
    xmach = [0.5, 0.75, 1.25]
    X1 = {"mach": xmach}
    mach = optitem.getel({"@expr": "$mach"}, x=X1, i=INDS)
    assert mach == [xmach[i] for i in INDS]


def test_getel_i01():
    # No index
    mach = optitem.getel({"@expr": "$mach"}, x=X)
    assert np.max(np.abs(mach - X["mach"])) <= 1e-6
    # Array index
    i = np.array([0, 2])
    mach = optitem.getel({"@expr": "$mach"}, x=X, i=i)
    assert np.max(np.abs(mach - X["mach"][i])) <= 1e-6
    # Bad entry
    i = [0, "2"]
    try:
        optitem.getel({"@expr": "$mach"}, x=X, i=i)
    except OptdictTypeError:
        pass
    else:
        assert False
    # Bad type
    try:
        optitem.getel({"@expr": "$mach"}, x=X, i="2")
    except OptdictTypeError:
        pass
    else:
        assert False

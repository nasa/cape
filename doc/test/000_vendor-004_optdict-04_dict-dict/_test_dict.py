
# Local
from cape.optdict import OptionsDict


def test_dict01_simple():
    # Input dictionary
    d = {
        "i": -3,
        "x": 24.3,
        "X": [
            24.3,
            14.7
        ],
        "d": {
            "name": "nasa",
            "on": True,
            "id": None
        }
    }
    # Read file
    opts = OptionsDict(d)
    # Test types
    assert isinstance(opts["i"], int)
    assert isinstance(opts["x"], float)
    assert isinstance(opts["X"], list)
    assert isinstance(opts["d"], dict)
    assert opts["d"]["on"] is True
    assert opts["d"]["id"] is None


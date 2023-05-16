
# Third-party
import numpy as np

# Local imports
from cape.optdict import OptionsDict


# Options
MYOPTS = {
    "v": 0.0,
    "a": [2, 4, 16],
    "b": {
        "r": "nice",
        "name": ["good", "place"],
    },
    "w": {
        "@expr": "$mach * np.sin($aoa * $DEG)",
    },
}

# Conditions
X = {
    "mach": np.linspace(0.5, 1.5, 5),
    "aoa": np.array([6.0, 4.0, 2.0, 2.0, -1.0]),
    "DEG": np.pi / 180.0
}
X0 = X["mach"][0] * np.sin(X["aoa"][0] * X["DEG"])

# Targets
MYOPTS00 = {
    "v": 0.0,
    "a": 2,
    "b": {
        "r": "nice",
        "name": "good",
    },
    "w": X0,
}
MYOPTS40 = {
    "v": 0.0,
    "a": 16,
    "b": {
        "r": "nice",
        "name": "place",
    },
    "w": X0,
}


# Access condition
def test_01_sampledict():
    # Initialize empty options
    opts = OptionsDict()
    # Set *x*
    opts.save_x(X)
    # Sample
    v0 = opts.sample_dict(MYOPTS, j=0, i=0)
    v4 = opts.sample_dict(MYOPTS, j=4, i=0)
    # Compare
    assert v0 == MYOPTS00
    assert v4 == MYOPTS40

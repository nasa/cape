# Local
from cape.pycart.options import functionalopts


# Raw options to test
OPTS = {
    "CA": {"compID": "STACK"},
    "CY": {"compID": "STACK", "force": 1},
    "CLM": {"compID": "CORE", "moment": 1, "type": "optMoment"},
}


def test_functionalopts():
    # Initialize options
    opts = functionalopts.FunctionalOpts(OPTS)
    # Filter forces
    fopts = opts.filter_FunctionalCoeffsByType("optForce")
    mopts = opts.filter_FunctionalCoeffsByType("optMoment")
    # Test them
    assert list(fopts.keys()) == ["CA", "CY"]
    assert list(mopts.keys()) == ["CLM"]
    # Test a single option and default
    assert opts.get_FunctionalCoeffOpt("CY", "force") == 1


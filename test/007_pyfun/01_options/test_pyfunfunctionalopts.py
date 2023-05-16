# Local
from cape.pyfun.options import functionalopts


# Raw options to test
OPTS = {
    "Coeffs": {
        "CA": {"compID": "STACK", "type": "adapt"},
        "CY": {"compID": "STACK", "force": 1},
        "CLM": {"compID": "CORE", "moment": 1, "type": "objective"},
    },
    "Functions": {
        "f1": {
            "type": "adapt",
            "coeffs": ["CA", "CY"],
        }
    }
}


def test_functionalopts():
    # Initialize options
    opts = functionalopts.FunctionalOpts(OPTS)
    # Filter forces
    ffuncs = opts.get_FunctionalFuncsByType("adapt")
    # Test them
    assert ffuncs == ["f1"]
    # Test list of all adaptation coeffs
    assert sorted(opts.get_FunctionalAdaptCoeffs()) == ["CA", "CY"]
    # Test individual option
    assert opts.get_FunctionalCoeffOpt("CLM", "compID") == "CORE"


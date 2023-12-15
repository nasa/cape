# Local
from cape.pyover.options import overnmlopts


def test_meshopts01():
    # Initialize options
    opts = overnmlopts.OverNmlOpts(
        {
            "GLOBAL": {
                "DTPHYS": [0.0, 0.0, 1.0],
                "NQT": 104,
            },
        }
    )
    # Sample entire namelist to phase 0
    opts0 = opts.select_namelist(j=0)
    # Test it
    assert opts0["GLOBAL"]["DTPHYS"] == 0.0
    # Get value
    v0 = opts.get_namelist_var(
        "GLOBAL", "DTPHYS", j=2)
    assert v0 == 1.0



# Third-party
import testutils

# Local
from cape.optdict import OptionsDict


# Globals
TEST_FILE = "simple01.json"


# Custom class
class MyOpts(OptionsDict):
    _ignore_case = True
    _lower_case = True


class MyUpperOpts(MyOpts):
    _lower_case = False

    _sec_cls = {
        "DATA": MyOpts,
    }


@testutils.run_sandbox(__file__, TEST_FILE)
def test_yaml01_simple():
    # Read file, converting all keys to lower-case
    opts0 = MyOpts(TEST_FILE)
    # Check case
    assert opts0["placestogo"] == "nowhere"
    assert "Data" not in opts0
    # Read file again, using upper-case
    opts1 = MyUpperOpts(TEST_FILE)
    assert opts1["PLACESTOGO"] == "nowhere"
    # Test that it applied to the subsection
    assert "DATA" in opts1
    assert opts1["DATA"]["name"] == "nasa"


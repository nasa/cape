
# Third-party
import testutils

# Local
from cape.optdict import OptionsDict


@testutils.run_testdir(__file__)
def test_json01_simple():
    fjson = "include01.json"
    OptionsDict(fjson)

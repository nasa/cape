
# Third-party
import testutils

# Local imports
import cape.attdb.ftypes.textdata as td


# File name
CSVFILE = "runmatrix.csv"


# Test boolean map
@testutils.run_testdir(__file__)
def test_01_boolmap():
    # Read as generic text with special first-column flag
    db = td.TextDataFile(
        CSVFILE,
        FirstColBoolMap={"PASS": "p", "ERROR": "e"})
    # Case number
    i = 6
    # Test the booleam map parameters
    assert db["_col1"][i] == "p"
    assert not db["ERROR"][i]
    assert db["PASS"][i]


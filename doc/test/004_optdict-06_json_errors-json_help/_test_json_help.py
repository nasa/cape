
# Third-party
import testutils

# Local
from cape.optdict import OptionsDict, OptdictJSONError


@testutils.run_testdir(__file__)
def test_error01():
    # Reliable path
    fjson = "error01.json"
    try:
        # Read broken file
        OptionsDict(fjson)
    except OptdictJSONError as e:
        # Get error message
        lines = e.args[0].split("\n")
        # Test lines
        assert lines[1].strip() == "Error occurred on line 10"
        assert lines[2].strip() == "Expecting ',' delimiter"
        assert lines[7].startswith("--> 10")
    else:
        # File should not have been read
        assert False


@testutils.run_testdir(__file__)
def test_error02():
    # Reliable path
    fjson = "error02.json"
    try:
        # Read broken file
        OptionsDict(fjson)
    except OptdictJSONError as e:
        # Get error message
        lines = e.args[0].split("\n")
        assert lines[1] == "Error occurred on line 13"
        assert lines[-1] == "--> 13 }"
    else:
        # File should not have been read
        assert False


@testutils.run_testdir(__file__)
def test_error06():
    # Reliable path
    fjson = "error06.json"
    try:
        # Read broken file
        OptionsDict(fjson)
    except OptdictJSONError as e:
        # Get error message
        lines = e.args[0].split("\n")
        # Test lines
        assert lines[1] == "Error occurred on line 2 of 'error06a.json'"
        assert lines[2].strip() == "Expecting ':' delimiter"
        assert lines[8].startswith("-->    1    2")
        assert lines[-1].strip() == "1: error06a.json"
    else:
        # File should not have been read
        assert False


# Standard library
import datetime

# Third-party
import testutils

# Local imports
from cape.cfdx import case


# Files to copy
TEST_FILES = (
    "case.json",
    "conditions.json"
)


# Read conditions
@testutils.run_sandbox(__file__, TEST_FILES)
def test_read_conditions():
    # Read conditions
    x = case.ReadConditions()
    # Check values
    assert abs(x["mach"] - 0.4) <= 1e-8
    assert abs(x["alpha"] - 4.0) <= 1e-8
    # Read specified key
    beta = case.ReadConditions("beta")
    # Check
    assert abs(beta + 0.2) <= 1e-8


# Read conditions
@testutils.run_sandbox(__file__, TEST_FILES)
def test_read_case_json():
    # Read options
    rc = case.ReadCaseJSON()
    # Test option values
    assert rc.get_PhaseSequence() == [0, 1, 2]
    assert rc.get_PhaseIters(1) == 750
    assert rc.get_PhaseIters(2) == 1000
    assert rc.get_PhaseIters(3) == 1000
    assert rc.get_qsub() is False


# Timing
@testutils.run_sandbox(__file__, TEST_FILES)
def test_case_timing():
    # Example file names
    fstrt = "cape_start.dat"
    ftime = "cape_time.dat"
    # Create initial time
    tic = datetime.datetime.now()
    # Read settings
    rc = case.ReadCaseJSON()
    # Write a flag for starting a program
    case.WriteStartTimeProg(tic, rc, 0, fstrt, "prog")
    # Read it
    nProc, t0 = case.ReadStartTimeProg(fstrt)
    # Calculate delta time
    dt = tic - t0
    # Test output
    assert nProc == 4
    assert dt.seconds == 0
    assert dt.microseconds < 1000000
    # Write output file
    case.WriteUserTimeProg(tic, rc, 0, ftime, "cape")


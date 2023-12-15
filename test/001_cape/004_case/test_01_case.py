
# Standard library
import os
import datetime

# Third-party
import pytest
import testutils

# Local imports
from cape.cfdx import case


# Files to copy
TEST_FILES = (
    "case.json",
    "conditions.json"
)


# Read runner
@testutils.run_sandbox(__file__, TEST_FILES)
def test_read_caserunner():
    # Instantiate runner
    runner = case.CaseRunner('.')
    # Check type
    assert isinstance(runner, case.CaseRunner)
    # Check attributes
    assert runner.j is None
    assert runner.root_dir == os.getcwd()
    # Run empty methods
    runner.prepare_files(0)
    runner.finalize_files(0)
    runner.run_phase(0)
    runner.getx_iter()
    runner.getx_restart_iter()
    runner.getx_phase(0)
    runner.check_error()
    # Now test the case start/stop functionality
    runner.mark_running()
    assert os.path.isfile(case.RUNNING_FILE)
    # Perform already-running test
    with pytest.raises(IOError):
        runner.check_running()
    # Stop a case
    runner.stop_case()
    assert not os.path.isfile(case.RUNNING_FILE)
    # Create an error
    runner.mark_failure()
    assert os.path.isfile(case.FAIL_FILE)


# Read conditions
@testutils.run_sandbox(__file__, TEST_FILES)
def test_read_conditions():
    # Get runner
    runner = case.CaseRunner()
    # Read conditions
    x = runner.read_conditions()
    # Check values
    assert abs(x["mach"] - 0.4) <= 1e-8
    assert abs(x["alpha"] - 4.0) <= 1e-8
    # Read specified key
    beta = runner.read_condition("beta")
    # Check
    assert abs(beta + 0.2) <= 1e-8


# Read conditions
@testutils.run_sandbox(__file__, TEST_FILES)
def test_read_case_json():
    # Read runner
    runner = case.CaseRunner()
    # Read options
    rc = runner.read_case_json()
    rc = runner.read_case_json()
    # Test option values
    assert rc.get_PhaseSequence() == [0, 1, 2]
    assert rc.get_PhaseIters(1) == 750
    assert rc.get_PhaseIters(2) == 1000
    assert rc.get_PhaseIters(3) == 1000
    assert rc.get_qsub() is False


# Timing
@testutils.run_sandbox(__file__, TEST_FILES)
def test_case_timing():
    # Read runner
    runner = case.CaseRunner()
    # Start timier
    runner.init_timer()
    # Create initial time
    tic = datetime.datetime.now()
    # Call timer reader when no file is present
    nproc, _ = runner.read_start_time()
    assert nproc is None
    # Read settings
    rc = runner.read_case_json()
    # Write a flag for starting a program
    runner.write_start_time(0)
    # Read it
    nProc, t0 = runner.read_start_time()
    # Calculate delta time
    dt = tic - t0
    # Test output
    assert nProc == rc.get_nProc(0)
    assert dt.seconds < 1
    # Write a end-of-phase timing file
    runner.write_user_time(0)


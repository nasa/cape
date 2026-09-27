
# Standard library
import glob
import os
import sys

# Third-party imports
import testutils


# List of file globs to copy into sandbox
TEST_FILES = (
    "case.json",
)


# Test 'cape run --dry-run'
@testutils.run_sandbox(__file__, TEST_FILES)
def test_dryrun():
    # Run the command
    cmdlist = [sys.executable, "-m", "cape", "run", "--dry-run"]
    # Run the command
    stdout, stderr, ierr = testutils.call_oe(cmdlist)
    # Check return code
    assert ierr == 0
    # Check that pre-command was printed, not run
    assert "echo pre-command" in stdout
    assert "echo pre-list" in stdout
    # Check that post-command was printed, not run
    assert "echo post-command" in stdout
    # Check that no STDOUT/STDERR log files were created
    assert len(glob.glob("precmd*")) == 0
    assert len(glob.glob("postcmd*")) == 0
    # Check that no case-state files or folders were created
    assert not os.path.isfile("RUNNING")
    assert not os.path.isfile("FAIL")
    assert not os.path.isdir("cape")


if __name__ == "__main__":
    test_dryrun()

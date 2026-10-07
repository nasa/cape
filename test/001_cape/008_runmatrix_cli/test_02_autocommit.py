# Standard library
import os
import subprocess

# Third-party
import testutils

# Local imports
from cape.cfdx import cli


# Files to copy to sandbox
TEST_FILES = (
    "cape.json",
    "matrix.csv",
)

# Identity for commits in the sandbox repo
GIT_ENV = {
    "GIT_AUTHOR_NAME": "CAPE Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "CAPE Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


# Run a git command in the sandbox
def _git(*args: str) -> str:
    env = dict(os.environ, **GIT_ENV)
    return subprocess.check_output(["git", *args], env=env).decode()


# Test ``cape auto-commit``
@testutils.run_sandbox(__file__, TEST_FILES)
def test_02_autocommit(capsys):
    os.environ.update(GIT_ENV)
    # Create a repo with the original matrix
    _git("init", "-q")
    _git("add", "matrix.csv", "cape.json")
    _git("commit", "-q", "-m", "Initial")
    nstart = len(_git("log", "--oneline").splitlines())
    # No changes: should still print the status table, but not commit
    ierr = cli.main(["cape", "auto-commit", "-f", "cape.json"])
    out = capsys.readouterr().out
    assert ierr == 0
    assert "No changes to 'matrix.csv'" in out
    assert "no changes" in out
    assert "Completed" not in out
    assert "mach values w/ no PASS cases:" in out
    assert len(_git("log", "--oneline").splitlines()) == nstart
    # Mark one case PASS and one ERROR
    cli.main(["cape", "-I", "5", "--PASS", "-f", "cape.json"])
    cli.main(["cape", "mark-error", "-I", "6", "-f", "cape.json"])
    capsys.readouterr()
    # Dry run: show message, don't commit
    ierr = cli.main(["cape", "auto-commit", "--dry-run", "-f", "cape.json"])
    out = capsys.readouterr().out
    assert ierr == 0
    assert "Auto-commit matrix.csv: PASS 1, FAIL 1" in out
    assert "PASS +1" in out
    assert "FAIL +1" in out
    assert len(_git("log", "--oneline").splitlines()) == nstart
    assert _git("status", "-s", "matrix.csv") != ""
    # Custom headline
    cli.main(["cape", "auto-commit", "-f", "cape.json", "-m", "My headline"])
    capsys.readouterr()
    assert len(_git("log", "--oneline").splitlines()) == nstart + 1
    msg = _git("log", "-1", "--format=%B")
    assert msg.startswith("My headline\n")
    assert "PASS +1" in msg
    assert _git("status", "-s", "matrix.csv") == ""


if __name__ == "__main__":
    test_02_autocommit()

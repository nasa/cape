
# Third party
import testutils

# Local imports
import cape.pyfun.cntl

# List of file globs to copy into sandbox
TEST_FILES = (
    "pyFun.json",
    "matrix.csv"
)


# Test cons in MeshFile
@testutils.run_sandbox(__file__, TEST_FILES)
def test_conditionals01():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Assert subssonic
    assert cntl.opts.get_MeshFile(i=0) == "subsonic.ugrid"
    # Assert supersonic
    assert cntl.opts.get_MeshFile(i=18) == "supersonic.ugrid"
    # Assert mpiprocs
    assert cntl.opts.get_PostPBS_mpiprocs(i=2) == 128



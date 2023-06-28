# Imports
import os

# Third party
import testutils

# Local imports
import cape.pyfun.cntl

# List of file globs to copy into sandbox
TEST_FILES = (
    "pyFun.json",
    "matrix.csv",
    "fun3d.nml",
    "bullet.xml",
    "subsonic.ugrid",
    "supersonic.ugrid",
    "bullet-inviscid.mapbc"
)

# Reference names
MESH_PREFIX = "arrow"


# Test conditionals with pyfun
@testutils.run_sandbox(__file__, TEST_FILES)
def test_conditionals01():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Assert subssonic - test cons
    assert cntl.opts.get_MeshFile(i=0) == "subsonic.ugrid"
    # Assert supersonic - test cons
    assert cntl.opts.get_MeshFile(i=18) == "supersonic.ugrid"
    # Assert mpiprocs - test map
    assert cntl.opts.get_PostPBS_mpiprocs(i=2) == 128


# Test conditionals propagating to pyFun input files
@testutils.run_sandbox(__file__, TEST_FILES)
def test_conditionals2():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Set up a case, but don't start
    cntl.SubmitJobs(I=0, start=False)
    # Get case folder
    case_folder = cntl.x.GetFullFolderNames(0)
    # Full path to ``aflr3.out`` and others
    mesh_file = os.path.join(case_folder, f"{MESH_PREFIX}.ugrid")
    # Name of intended mesh
    abs_mesh = "subsonic.ugrid"
    # Test that the mesh file is actually a file
    assert os.path.isfile(mesh_file)
    # Test that file is the correct mesh based on conditional
    assert testutils.compare_files(abs_mesh, mesh_file)




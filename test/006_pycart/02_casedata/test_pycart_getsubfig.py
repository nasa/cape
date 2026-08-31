
# Standard Library
import os
import shutil

# Third-party
import testutils

# CAPE
from cape.pycart.cntl import Cntl
from cape.dkit.rdb import DataKit


# Dir to case files
CASEDIR = os.path.join("poweroff", "m1.5a0.0b0.0")

# Test Files
TEST_FILES = (
    "pyCart.json",
    "c3dfunc.py",
    "matrix.csv",
    os.path.join(CASEDIR, "bullet_no_base.dat"),
    os.path.join(CASEDIR, "case.json"),
    os.path.join(CASEDIR, "history.dat"),
    os.path.join(CASEDIR, "run.00.200"),
    "test.[0-9][0-9].out"
)


# Test Files
TEST_FILES2 = (
    os.path.join(CASEDIR, "Components.i.triq"),
    os.path.join(CASEDIR, "bullet_no_base.dat"),
    os.path.join(CASEDIR, "case.json"),
    os.path.join(CASEDIR, "history.dat"),
    os.path.join(CASEDIR, "run.00.200"),
    "pyCart.json",
    "matrix.csv",
    "c3dfunc.py",
    "test.[0-9][0-9].out",
    os.path.join(CASEDIR, "lineload", "LineLoad_bullet_total_LL.dlds")
)

# Test Files
TEST_FILES3 = (
    "pyCart.json",
    "c3dfunc.py",
    "Config.xml",
    "cap-patch.uh3d",
    "matrix.csv",
    os.path.join(CASEDIR, "bullet_no_base.dat"),
    os.path.join(CASEDIR, "case.json"),
    os.path.join(CASEDIR, "history.dat"),
    os.path.join(CASEDIR, "run.00.200"),
    os.path.join(CASEDIR, "Components.i.triq"),
    "test.[0-9][0-9].out"
)


KW1 = {
    "I": [0],
    "fm": "bullet_no_base",
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW2 = {
    "I": [0],
    "fm": "bullet_no_base",
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW3 = {
    "I": [0],
    "ll": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW4 = {
    "I": [0],
    "ll": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW5 = {
    "I": [0],
    "dbpyfunc": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW6 = {
    "I": [0],
    "dbpyfunc": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW7 = {
    "I": [0],
    "triqfm": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW8 = {
    "I": [0],
    "triqfm": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}


@testutils.run_sandbox(__file__, TEST_FILES)
def test_updatedatabookfm():
    # Get cntl
    cntl = Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW1)
    # Location of output databook
    dbout = os.path.join("data/aero_bullet_no_base.csv")
    db = DataKit(dbout)
    # Compare output databook with reference result
    assert db["Mach"].size == 1
    assert db["CA"][0] > 0.7
    assert db["CA"][0] < 0.8


@testutils.run_sandbox(__file__, TEST_FILES)
def test_deletecasesfm():
    os.mkdir("data")
    # Use test.01.out as existing databook
    shutil.copy("test.01.out", os.path.join("data", "aero_bullet_no_base.csv"))
    # Get cntl
    cntl = Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW2)
    # Location of output databook
    dbout = os.path.join("data/aero_bullet_no_base.csv")
    # Read it
    db = DataKit(dbout)
    assert db["Mach"].size == 0


@testutils.run_sandbox(__file__, TEST_FILES3)
def test_get_subfig_bullet_ca():
    """Test that 'cape get-subfig bullet_CA -I 0' produces a figure.
    
    Uses CAPE_CACHE_DIR environment variable to set cache dir so that
    the figure file shows up in the work/ folder for verification.
    """
    # Set up cache directory in work folder
    cache_dir = "cache"
    os.environ["CAPE_CACHE_DIR"] = cache_dir
    
    try:
        # Create cache directory
        os.mkdir(cache_dir)
        
        # Get cntl
        cntl = Cntl()
        
        # Call get_subfigure for bullet_CA subfig
        result = cntl.get_subfigure("bullet_CA", I=[0])
        
        # Verify result structure
        assert "cachefiles" in result
        assert "case" in result
        assert "casename" in result
        assert "subfigname" in result
        
        # Verify subfigname matches
        assert result["subfigname"] == "bullet_CA"
        
        # Verify case index
        assert result["case"] == 0
        
        # Verify cache files were created
        cachefiles = result["cachefiles"]
        assert len(cachefiles) > 0, "Expected at least one cached figure file"
        
        # Verify cache files exist in the cache directory
        for f in cachefiles:
            assert os.path.exists(f), f"Cache file {f} does not exist"
            # Verify file is in the cache directory
            assert f.startswith(cache_dir), f"Cache file {f} not in cache directory"
            
        # Verify at least one image file exists (png, pdf, or svg)
        image_files = [f for f in cachefiles if any(f.endswith(ext) for ext in ['.png', '.pdf', '.svg'])]
        assert len(image_files) > 0, "Expected at least one image file (png, pdf, or svg)"
        
    finally:
        # Clean up environment variable
        if "CAPE_CACHE_DIR" in os.environ:
            del os.environ["CAPE_CACHE_DIR"]

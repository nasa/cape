
# Third-party
import numpy as np
import testutils

# Local imports
import cape.attdb.rdb as rdb


MAT_FILE = "CN-alpha-beta.mat"


@testutils.run_testdir(__file__)
def test_01_griddata():
    db = rdb.DataKit(MAT_FILE)
    # Number of break points
    n = 17
    # Reference points for regularization
    A0 = np.linspace(-2, 2, n)
    B0 = np.linspace(-2, 2, n)
    # Save break points
    db.bkpts = {"alpha": A0, "beta": B0}
    # Regularize
    db.regularize_by_griddata("CN", ["alpha", "beta"], prefix="reg")
    # Compare regularized alpha/beta to expected
    assert np.max(db["regalpha"][::n] - A0) <= 1e-8
    assert np.max(db["regbeta"][:n] - B0) <= 1e-8
    # Reguarlized *CN* should be monotonic on first alpha slice
    assert np.min(np.diff(db["regCN"][::n])) > 0


@testutils.run_testdir(__file__)
def test_02_rbf():
    db = rdb.DataKit(MAT_FILE)
    # Number of break points
    n = 17
    # Reference points for regularization
    A0 = np.linspace(-2, 2, n)
    B0 = np.linspace(-2, 2, n)
    # Save break points
    db.bkpts = {"alpha": A0, "beta": B0}
    # Regularize
    db.regularize_by_rbf("CN", ["alpha", "beta"], prefix="reg")
    # Compare regularized alpha/beta to expected
    assert np.max(db["regalpha"][::n] - A0) <= 1e-8
    assert np.max(db["regbeta"][:n] - B0) <= 1e-8
    # Reguarlized *CN* should be monotonic on first alpha slice
    assert np.min(np.diff(db["regCN"][::n])) > 0

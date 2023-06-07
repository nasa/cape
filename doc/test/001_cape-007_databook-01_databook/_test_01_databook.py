
# Third-party
import numpy as np
import testutils

# Local imports
import cape.cntl
import cape.cfdx.dataBook as databook


# Test DataBook Class
@testutils.run_testdir(__file__)
def test_01_databook():
    # Read settings
    cntl = cape.cntl.Cntl()
    # Read data book
    db = databook.DataBook(cntl)
    # Check the __repr__
    assert str(db) == "<DataBook nComp=10, nCase=30>"
    # Check the ocmponents
    assert db.Components == [
        "cap", "body", "fins",
        "arrow_no_base", "arrow_total",
        "fuselage", "fin1", "fin2", "fin3", "fin4"
    ]
    # Extract a component
    dbc = db["fin1"]
    # Test that we read a databook str with a comma in it
    assert ',' in db["fuselage"]["config"][0]
    # Display that
    assert str(dbc) == "<DBComp fin1, nCase=30>"
    # Match the trajectory to the actual data
    dbc.UpdateRunMatrix()
    # Filter cases at alpha=2
    I = dbc.x.Filter(["alpha==2"])
    # Check
    assert list(I) == [2, 7, 12, 17, 22, 27]
    # Test CaseFM Class
    # Create a force & moment history
    fm = databook.CaseFM("fin")
    # Some iterations
    n = 500
    # Create some iterations
    fm.i = np.arange(n)
    # Seed the random number generator
    np.random.seed(450)
    # Create some random numbers
    fm.CN = 1.4 + 0.3*np.random.randn(n)
    # Save properties
    fm.cols = ["i", "CN"]
    fm.coeffs = ["CN"]
    # Check __repr__
    assert str(fm) == "<dataBook.CaseFM('fin', i=500)>"
    # Calculate statistics
    stats = fm.GetStatsN(100)
    # Check values
    assert abs(stats["CN"] - 1.4149) <= 1e-4
    assert abs(stats["CN_min"] - 0.6555) <= 1e-4
    assert abs(stats["CN_max"] - 2.0462) <= 1e-4
    assert abs(stats["CN_std"] - 0.3095) <= 1e-4
    assert abs(stats["CN_err"] - 0.0190) <= 1e-4

 
if __name__ == "__main__":
    test_01_databook()


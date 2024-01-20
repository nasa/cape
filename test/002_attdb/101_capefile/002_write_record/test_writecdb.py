
# Third-party
import numpy as np
import testutils

# Local imports
from cape.attdb.ftypes.capefile import (
    RecordType,
    genr8_recordsize,
    write_record
)


# Test non-recursive sizes
@testutils.run_sandbox(__file__)
def test_write01():
    # Test size of ``None``
    _rs(None, size=0)
    # Test a string
    _rs("this string has len=22", size=26)
    # Test an integer
    _rs(14, "myint", 17)
    # Test a 1D array
    _rs(np.array([0.4, 12.1], dtype="f4"))
    # Test a 2D array
    _rs(np.ones((4, 2), dtype="int16"), size=24)
    # Peculiar case of 0D array
    _rs(np.array(3.4), size=12)


# Test recursive sizes
@testutils.run_sandbox(__file__)
def test_write02():
    # Test size of a list of None's
    _rs([None, None], size=20)
    # Test more complicated size
    _rs({"a": 1, "b": "name"}, size=46)


# Function to test a record size
def _rs(v, name=None, size=None):
    # Create a file
    with open("test.cdb", 'wb') as fp:
        # Write the value
        write_record(fp, v, name)
        # Calculate record type
        rtyp = RecordType.from_value(v, name)
        # Calculate size
        rs = genr8_recordsize(v, name)
        # Check position
        pos = fp.tell()
        # Test it
        assert pos == rs + 8 + 4*rtyp.xlong_record
    # Manual test of record size
    if size is not None:
        assert rs == size

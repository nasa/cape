
# Third-party
import numpy as np
import pytest
import testutils

# Local imports
from cape.attdb.ftypes.capefile import (
    CapeFile,
    CapeFileValueError,
    RecordType,
    check_header,
    read_nrecord,
    read_recordtype,
    write_record,
    HEADER_BYTES,
    _genr8_recordsize,
    _get_fp,
)


# Initial test
def test_io01():
    # Create empty instance
    db = CapeFile()
    # Save some data
    db.save_col(None, 14)
    # Make sure it worked
    assert db["_record0"] == 14
    # Test overwrite detection
    with pytest.raises(CapeFileValueError):
        db.save_col("_record0", 2, j=1)
    # Initialize from a dict
    db = CapeFile({"a": 1, "b": 'bob'})
    assert 'a' in db
    assert 'a' in db.cols
    assert 'b' in db
    assert db["a"] == 1
    assert db["b"] == "bob"


# Run write + reread test
@testutils.run_sandbox(__file__)
def test_io02():
    # Create empty instance
    db = CapeFile()
    # Reference quantities
    name1, v1 = None, 14
    name2, v2 = "myarray", np.arange(4, dtype="uint16")
    # Save them
    db.save_col(name1, v1)
    db.save_col(name2, v2)
    # Write to file
    db.write("io2.cdb")
    # Reread
    db1 = CapeFile("io2.cdb")
    # Check list of columns
    assert db1.cols[1] == name2
    # Check value
    assert db1[db1.cols[0]] == 14


# Test extra types
@testutils.run_sandbox(__file__, fresh=False)
def test_io03():
    # Create empty instance
    db = CapeFile()
    # Reference quantities
    name1, v1 = "col1", np.array(["á la", "Straße"])
    name2, v2 = "b1", np.array([b"a33", b"bf42ac"])
    name3, v3 = "a", None
    name4, v4 = "b", {"a": [1, 2, (3, 4)], "b": {"a", "b"}}
    name5, v5 = "c", np.array([[1, 2], [3, 4]])
    name6, v6 = "d", np.array(1.1)
    name7, v7 = "e", np.array([True, False])
    # Save them
    db.save_col(name1, v1)
    db.save_col(name2, v2)
    db.save_col(name3, v3)
    db.save_col(name4, v4)
    db.save_col(name5, v5)
    db.save_col(name6, v6)
    db.save_col(name7, v7)
    # Write to file
    db.write("io3.cdb")
    # Reread
    db1 = CapeFile("io3.cdb")
    # Test values
    assert db1[name1][0] == v1[0]
    assert db1[name2][1] == v2[1]
    assert db1[name3] is None
    assert isinstance(db1[name4], dict)
    assert isinstance(db1[name4]["a"], list)
    assert isinstance(db1[name5], np.ndarray)
    assert db1[name5].ndim == 2
    assert isinstance(db1[name6], np.ndarray)
    assert db1[name6].ndim == 0
    assert db1[name7].dtype.name == "bool"


# Test the *meta* option
@testutils.run_sandbox(__file__, fresh=False)
def test_io_meta():
    # Read previous file
    db = CapeFile("io3.cdb", meta=True)
    # Ensure that there's nothing in there
    assert db["e"] is None


# Test errors
@testutils.run_sandbox(__file__, fresh=False)
def test_io_e1():
    # Create an incomplete file
    with open("io4.cdb", 'wb') as fp:
        fp.write(HEADER_BYTES)
        # Write that there's going to be one record
        np.uint64(1).tofile(fp)
        # But don't write the record
        # Check the _get_fp() function
        fp1 = _get_fp(fp, mode='wb')
        assert fp1 is fp
    # Now try to read record type
    with open("io4.cdb", 'rb') as fp:
        # Read header
        check_header(fp)
        # Read number of records
        read_nrecord(fp)
        # Read number of headers
        with pytest.raises(CapeFileValueError):
            read_recordtype(fp)


# Other errors
@testutils.run_sandbox(__file__, fresh=False)
def test_io_e2():
    # Create a list with a named item
    with open("io5.cdb", 'wb') as fp:
        fp.write(HEADER_BYTES)
        np.uint64(1).tofile(fp)
        # Write a "list" header
        v = [1]
        rtyp = RecordType.from_value(v)
        rs = _genr8_recordsize(rtyp, v)
        # Write it
        rtyp.rt.tofile(fp)
        # Write size
        rs.tofile(fp)
        np.uint32(1).tofile(fp)
        # Write a named record
        write_record(fp, 1, "oops")
    # Try to read messed up file
    with pytest.raises(CapeFileValueError):
        CapeFile("io5.cdb")


# Other errors
@testutils.run_sandbox(__file__, fresh=False)
def test_io_e3():
    # Create a list with a named item
    with open("io6.cdb", 'wb') as fp:
        fp.write(HEADER_BYTES)
        np.uint64(1).tofile(fp)
        # Write a "list" header
        v = {"b": 1}
        rtyp = RecordType.from_value(v)
        rs = _genr8_recordsize(rtyp, v)
        # Write it
        rtyp.rt.tofile(fp)
        # Write size
        rs.tofile(fp)
        np.uint32(1).tofile(fp)
        # Write a named record
        write_record(fp, 1, None)
    # Try to read messed up file
    with pytest.raises(CapeFileValueError):
        CapeFile("io6.cdb")

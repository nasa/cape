
# Third party
import numpy as np

# Local imports
from cape.dkit.capefile import RecordType, genr8_recordtype


# Some basic types
def test_rt01():
    # Iterative types
    assert genr8_recordtype([1]) == 0x10 | 0x20000
    assert genr8_recordtype((1,)) == 0x20 | 0x20000
    assert genr8_recordtype({1}) == 0x30 | 0x20000
    assert genr8_recordtype({"a": 1}) == 0x40 | 0x20000


# More basic tpyes
def test_rt02():
    # Basic Python scalars
    assert genr8_recordtype(None) == 0
    assert genr8_recordtype(1) == 2 | 0x600
    assert genr8_recordtype(1.) == 3 | 0x600
    # Test class method of RecordType
    rtyp = RecordType.from_value(1.1 + 2j)
    assert rtyp.element_type == 4
    assert rtyp.element_bits == 7
    assert rtyp.record_array == 0
    assert rtyp.complex_flag == 0
    assert rtyp.record_title == 0
    assert rtyp.xlong_record == 0
    assert rtyp.el_byteorder == 0


# Arrays
def test_rt03():
    # Common array types
    v = np.array([1], dtype="uint32")
    rt = genr8_recordtype(v)
    assert rt == 1 | 0x500 | 0x1000
    assert isinstance(rt, np.uint32)



# Third-party
import testutils

# Local imports
from cape.cfdx import cmd as cmdgen
from cape.cfdx import case


# Test AFLR3 command
@testutils.run_testdir(__file__)
def test_01_aflr3():
    # Read settings
    rc = case.ReadCaseJSON()
    # Form command
    cmd1 = cmdgen.aflr3(rc)
    # Alternate form
    cmd2 = cmdgen.aflr3(
        i="pyfun.surf",
        o="pyfun.lb8.ugrid",
        blr=10,
        flags={"someflag": 2},
        keys={"somekey": 'c'})
    # Check
    assert cmd1[-1] == "somekey=c"
    assert cmd2 == [
        "aflr3",
        "-i", "pyfun.surf",
        "-o", "pyfun.lb8.ugrid",
        "-blr", "10",
        "-someflag", "2",
        "somekey=c"
    ]


# Test intersect command
@testutils.run_testdir(__file__)
def test_02_intersect():
    # Read settings
    rc = case.ReadCaseJSON()
    # Form commands
    cmd1 = cmdgen.intersect(rc)
    cmd2 = cmdgen.intersect(T=True)
    # Check
    assert cmd1 == [
        "intersect",
        "-i", "Components.tri",
        "-o", "Components.i.tri",
        "-ascii", "-T"
    ]
    assert cmd2 == [
        "intersect",
        "-i", "Components.tri",
        "-o", "Components.i.tri",
        "-ascii", "-T"
    ]

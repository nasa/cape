
# Third-party
import pytest
import testutils

# Local imports
from cape.cfdx import cmdgen
from cape.cfdx import casecntl
from cape.cfdx.options import Options


# Test AFLR3 command
@testutils.run_testdir(__file__)
def test_01_aflr3():
    # Get case runner
    runner = casecntl.CaseRunner()
    # Read settings
    rc = runner.read_case_json()
    # Form command
    cmd1 = cmdgen.aflr3(rc)
    # Alternate form
    cmd2 = cmdgen.aflr3(
        i="pyfun.surf",
        o="pyfun.lb8.ugrid",
        blr=1.2,
        flags={"someflag": 2, "boolflag": True},
        keys={"somekey": 'c'})
    # Check
    assert cmd1[-1] == "somekey=c"
    assert cmd2[:7] == [
        "aflr3",
        "-i", "pyfun.surf",
        "-o", "pyfun.lb8.ugrid",
        "-blr", "1.2"
    ]


# Test intersect command
@testutils.run_testdir(__file__)
def test_02_intersect():
    # Get case runner
    runner = casecntl.CaseRunner()
    # Read settings
    rc = runner.read_case_json()
    # Form commands
    cmd1 = cmdgen.intersect(rc)
    cmd2 = cmdgen.intersect(ascii=True)
    # Check
    assert cmd1 == [
        "intersect",
        "-i", "Components.tri",
        "-o", "Components.i.tri",
        "-T"
    ]
    assert cmd2 == [
        "intersect",
        "-i", "Components.tri",
        "-o", "Components.i.tri",
        "-ascii"
    ]


# Test options isolator branches
@testutils.run_testdir(__file__)
def test_03_isolate():
    # Read options
    opts = Options("cape.json")
    # Test bad calls
    with pytest.raises(TypeError):
        # Test w/o giving a class for *cls*
        cmdgen.isolate_subsection(opts, 3, ())
    with pytest.raises(TypeError):
        # Test w/ non-OptionsDict *cls*
        cmdgen.isolate_subsection(opts, dict, ())
    with pytest.raises(TypeError):
        # Test with non-tuple subsections
        cmdgen.isolate_subsection(opts, Options, ["RunControl"])
    with pytest.raises(TypeError):
        # Test with non-string subsections
        cmdgen.isolate_subsection(opts, Options, (1, 2))
    with pytest.raises(TypeError):
        # Test with invalid options instance
        cmdgen.isolate_subsection([], Options, ("RunControl",))
    with pytest.raises(ValueError):
        # Test with unrecognized section
        cmdgen.isolate_subsection(opts, Options, ("RunSpaceNope",))
    # Test default output, no subsections
    opts1 = cmdgen.isolate_subsection(None, Options, ())
    assert isinstance(opts1, Options)
    # Test trivial output
    opts2 = cmdgen.isolate_subsection(opts, Options, ())
    assert opts2 is opts
    # Test conversion of :class:`dict`
    opts3 = cmdgen.isolate_subsection(
        {"RunControl": {"qsub": True}}, Options, ())
    assert opts3["RunControl"]["qsub"]
    assert isinstance(opts3, Options)
    # Subsection
    opts4 = cmdgen.isolate_subsection(opts, Options, ("RunControl", "aflr3"))
    assert opts4.__class__.__name__ == "AFLR3Opts"


# Test command appenders
def test_04_append():
    # Test normal appending
    cmdi = ["verify"]
    cmdgen.append_cmd_if(cmdi, True, ['-v'])
    assert cmdi == ["verify", "-v"]
    # Test "None" version
    cmdgen.append_cmd_if_not_none(cmdi, 0, ['-a'])
    assert cmdi == ["verify", "-v", "-a"]
    # Test exceptions
    with pytest.raises(ValueError):
        cmdgen.append_cmd_if(cmdi, False, [], ValueError)
    with pytest.raises(ValueError):
        cmdgen.append_cmd_if_not_none(cmdi, None, [], ValueError)


# Test verify() command
@testutils.run_testdir(__file__)
def test_05_verify():
    # Read options
    opts = Options("cape.json")
    # Make a command for verify
    cmdi = cmdgen.verify(opts, i="in.tri")
    assert cmdi == ["verify", "in.tri", "-ascii"]

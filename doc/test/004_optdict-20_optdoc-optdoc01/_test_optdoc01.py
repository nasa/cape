
# Standard library
import os

# Third-party
import testutils

# Local imports
from cape.optdict import OptionsDict
from cape.optdict import optdoc


# Section 1 subsections
class Sec1Subsec1Opts(OptionsDict):
    _name = "Sub1"
    _label = "section-1-1"
    _optlist = (
        "a",
        "b"
    )


# Components of section 1
class Sec1CompOpts(OptionsDict):
    _optlist = (
        "A",
        "B",
        "Type",
    )


class Sec1CompAOpts(Sec1CompOpts):
    _rc = {
        "A": True,
    }


class Sec1CompBOpts(Sec1CompOpts):
    _rc = {
        "B": True,
    }


# Section 1 class
class Section1Opts(OptionsDict):
    _name = "Options for Section 1"
    _subsec_name = "component"
    _description = "This section has options for '1'?"
    _optlist = (
        "sec1opt1",
        "sec1opt2",
        "subsec2",
    )
    _opttypes = {
        "subsec2": Sec1CompAOpts,
    }
    _sec_cls = {
        "subsec1": Sec1Subsec1Opts,
    }
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "_default_": Sec1CompOpts,
        "A": Sec1CompAOpts,
        "B": Sec1CompBOpts,
    }


# Component options for section 2
class Sec2CompOpts(OptionsDict):
    _optlist = (
        "Type",
        "r",
        "s",
    )


class Sec2CompAOpts(OptionsDict):
    _optlist = (
        "A",
    )


class Sec2CompBOpts(OptionsDict):
    _optlist = (
        "B",
    )


# Section 2 class
class Section2Opts(OptionsDict):
    _optlist = (
        "sec2opt1",
        "sec2opt2",
    )
    _optmap = {
        "opt1": "sec2opt1",
        "opt2": "sec2opt2",
    }
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "A": Sec2CompAOpts,
        "B": Sec2CompBOpts,
    }


# Section 3 subclasses
class Sec3A1Opts(OptionsDict):
    pass


class Sec3AOpts(OptionsDict):
    _sec_cls_optmap = {
        "_default_": Sec3A1Opts,
    }


# Section 3 class
class Section3Opts(OptionsDict):
    _sec_cls = {
        "A": Sec3AOpts,
    }


# Overall class
class MyOpts(OptionsDict):
    _optlist = (
        "Section1",
        "opt1"
    )
    _sec_cls = {
        "Section1": Section1Opts,
        "Section2": Section2Opts,
        "Section3": Section3Opts,
    }


# Special options for improper _opttypes
class BrokenOptsA(OptionsDict):
    _optlist = (
        "section1",
    )
    _opttypes = {
        "section1": (3, int),
    }


class Base1AOpts(OptionsDict):
    pass


class Base1Opts(Base1AOpts, dict):
    pass


# Create class with complex bases
class BaseOpts(Base1Opts):
    pass


# Create class with diverse documentation strings
class TypeDocOpts(OptionsDict):
    # List of options
    _optlist = (
        "a",
        "b",
        "c",
    )
    # Types
    _opttypes = {
        "a": bool,
        "b": bool,
        "c": bool,
    }
    # Defaults
    _rc = {
        "a": True,
        "b": [False, True],
    }


# Options for documentation
DOC_OPTS = {
    "myopts": {
        "file": "index",
        "module": __name__,
        "class": "MyOpts",
        "recurse": False,
        "recurse_sec_clsmap": False,
        "verbose": False,
    },
    "Section1": {
        "recurse": True,
        "narrow": True,
    },
}

# Alternate documentation options
DOC_OPTS1 = {
    "sec3a": {
        "module": __name__,
        "folder": "sec3",
        "class": "Sec3AOpts",
    },
}


# Run first test
@testutils.run_sandbox(__file__)
def test_optdoc01():
    # Build the documentation
    optdoc.make_rst(DOC_OPTS, "myopts")
    # Assert expected files
    assert os.path.isfile("index.rst")
    assert os.path.isfile("Section2.rst")
    assert os.path.isfile("Section2-A.rst")
    assert not os.path.isfile("Section1-A.rst")
    # Read the initial table of contents
    txt = open("index.rst").read()
    # Test
    assert txt.rstrip().endswith("Section3")
    # Build it again to test caching
    optdoc.make_rst(DOC_OPTS, "myopts")
    # Test smaller section to test out the "fo
    optdoc.make_rst(DOC_OPTS1, "sec3a")
    # That should have created a folder
    assert os.path.isdir(DOC_OPTS1["sec3a"]["folder"])


# Cover a line of _filter_children() w/o title in *children*
def test_optdoc_filter():
    # Test children w/o title given
    children = Sec3AOpts._filter_children({"s1": Sec3A1Opts})
    # Should be empty
    assert children == {}


# Cover lines of _getx_cls_optmap() w/ not-OptionsDict types
def test_optdoc_opttypes():
    # Get children from broken class
    children = BrokenOptsA._getx_sec_cls_optmap()
    # Should be no children
    assert children == {}


# Test bases
def test_optdoc_bases():
    # Test empty basis class method
    bases = BaseOpts._getx_bases(narrow=False)
    assert bases == []
    # Get bases of ``BaseOpts``
    bases = BaseOpts._getx_bases(narrow=True)
    assert bases == [Base1Opts, Base1AOpts]


# Test additional type strings
def test_optdoc_optinfo():
    # Test arms of genr8_rst_opttypes
    optinfo = TypeDocOpts._genr8_rst_opt("a", indent=0)
    # Test first line
    line = optinfo.split("\n")[0]
    assert line.strip() == "*a*: {``True``} | ``False``"
    # Assert default of ``False``
    optinfo = TypeDocOpts._genr8_rst_opt("b", indent=0)
    line = optinfo.split("\n")[0]
    assert line.strip() == "*b*: {``False``} | ``True``"
    # Test boolean w/ no default
    optinfo = TypeDocOpts._genr8_rst_opt("c", indent=0)
    line = optinfo.split("\n")[0]
    assert line.strip() == "*c*: {``None``} | ``True`` | ``False``"


# Test class name features
def test_optdoc_name():
    # Get _name from a class
    name = Section1Opts.getcls_name()
    assert name == Section1Opts._name

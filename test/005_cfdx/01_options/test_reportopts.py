
# Third-party
import pytest

# Local
from cape.cfdx.options import reportopts


# Some templates
OPTS1 = {
    "Reports": [
        "report1",
        "report2"
    ],
    "report1": {
        "Title": "Report #1",
        "Restriction": "UU",
    },
    "report2": {
        "Parent": "report1",
        "Figures": [
            "fig1",
            "fig2",
        ],
    },
}

# Sweep options
OPTS2 = {
    "Sweeps": {
        "sweep1": {
            "Figures": [
                "mach-FM",
                "m050-alpha-FM"
            ],
            "MinCases": 5,
        },
    },
}

FIGOPTS1 = {
    "fig1": {
        "align": "left",
        "Header": "force histories",
        "Subfigures": [
            "CA",
            "CY",
            "CN",
        ],
    },
    "fig2": {
        "Parent": "fig1",
        "Subfigures": [
            "CA1",
            "CA2",
            "CA3",
        ],
    },
}

SUBFIGOPTS1 = {
    "STACK": {
        "Type": "PlotCoeff",
        "Component": "STACK",
        "CaptionComponent": "IV",
    },
    "STACK_CA": {
        "Type": "STACK",
        "Coefficient": "CA",
        "Delta": 0.02,
    },
}


def test_reportopts1():
    # Initialize options
    opts = reportopts.ReportOpts(OPTS1)
    # Test cascading
    assert opts.get_ReportOpt("report2", "Title") == OPTS1["report1"]["Title"]
    # List of reports
    assert opts.get_ReportList() == ["report1", "report2"]
    assert opts.get_ReportErrorFigures("report2") == ["fig1", "fig2"]


def test_sweepopts1():
    # Parse options
    opts = reportopts.ReportOpts(OPTS2)
    # Check it
    assert opts.get_SweepOpt("sweep1", "MinCases") == 5
    # Test list of sweeps
    assert opts.get_SweepList() == ["sweep1"]


def test_reportfigopts1():
    # Initialize figure options
    opts = reportopts.FigureCollectionOpts(FIGOPTS1)
    # Test types
    assert isinstance(opts["fig2"], reportopts.FigureOpts)
    # Test cascase
    assert opts.get_FigOpt("fig2", "Header") == opts["fig1"]["Header"]


def test_reportfigopts2():
    # Initialize report with figures
    opts = reportopts.ReportOpts({"Figures": FIGOPTS1})
    # Test list of figures
    assert opts.get_FigList() == ["fig1", "fig2"]


def test_subfigopts1():
    # Initialize subfigure options
    opts = reportopts.ReportOpts({"Subfigures": SUBFIGOPTS1})
    # Test list of subfigures
    assert opts.get_SubfigList() == ["STACK", "STACK_CA"]
    # Test cascading options
    assert opts.get_SubfigOpt("STACK_CA", "Component") == "STACK"
    # Test "base" type
    assert opts.get_SubfigBaseType("STACK_CA") == "PlotCoeff"
    # Construct fully expanded opts for "STACK_CA"
    sfigopts = dict(SUBFIGOPTS1["STACK_CA"], **SUBFIGOPTS1["STACK"])
    # Test the result
    assert opts.get_SubfigCascade("STACK_CA") == sfigopts
    # Test a nonsense subfig
    with pytest.raises(KeyError):
        opts.get_SubfigCascade("STACK_CY")


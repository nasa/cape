
# Third-party

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
        "Parent": "report1"
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


def test_reportopts1():
    # Initialize options
    opts = reportopts.ReportOpts(OPTS1)
    # Test cascading
    assert opts.get_ReportTitle("report2") == OPTS1["report1"]["Title"]


def test_reportfigopts1():
    # Initialize figure options
    opts = reportopts.FigureCollectionOptions(FIGOPTS1)
    # Test types
    assert isinstance(opts["fig2"], reportopts.FigureOptions)
    # Test cascase
    assert opts.get_FigHeader("fig2") == opts["fig1"]["Header"]

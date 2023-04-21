
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


def test_reportopts1():
    # Initialize options
    opts = reportopts.ReportOpts(OPTS1)
    # Test cascading
    assert opts.get_ReportTitle("report2") == OPTS1["report1"]["Title"]


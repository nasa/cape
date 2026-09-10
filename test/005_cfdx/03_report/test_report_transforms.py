# Standard library
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

# Third-party
import numpy as np
import pytest

# CAPE
from cape.cfdx.casecntl import CaseRunner
from cape.cfdx.casedata import CaseFM
from cape.cfdx.report import Report


COEFFS = ("CA", "CY", "CN", "CLL", "CLM", "CLN")
TRANSFORMS = [
    [],
    [{"Type": "Euler321", "phi": 90}],
    [{"Type": "ScaleCoeffs", "CLL": -1.0, "CLN": -1.0}],
    [{"Type": "ShiftMRP", "FromMRP": [0, 0, 0],
      "ToMRP": [1, 0, 0], "RefLength": 1.0}],
    [{"Type": "Euler321", "phi": 90},
     {"Type": "ScaleCoeffs", "CLL": -1.0, "CLN": -1.0}],
]
EXPECTED = [
    [1, 2, 3, 4, 5, 6],
    [1, -3, 2, 4, -6, 5],
    [1, 2, 3, -4, 5, -6],
    [1, 2, 3, 4, 8, 4],
    [1, -3, 2, -4, -6, -5],
]


class MemoryRunner(CaseRunner):
    """Use real data extraction and transforms with an in-memory history."""

    def __init__(self, cntl, transforms):
        self.cntl = cntl
        self.transforms = transforms
        self.histories = []

    def get_dex_type(self, comp):
        return "fm"

    def get_dex_opt(self, comp, opt, vdef=None):
        return {"CompID": comp, "Transformations": self.transforms}[opt]

    def read_cntl(self):
        return self.cntl

    def get_case_index(self):
        return 0

    def prep_dex(self, comp):
        pass

    def read_dex_element(self, comp, compid):
        fm = CaseFM(comp)
        fm.save_col("i", np.arange(1., 5.))
        for value, coeff in enumerate(COEFFS, start=1):
            fm.save_col(coeff, np.full(4, float(value)))
        fm.GetStats = Mock(wraps=fm.GetStats)
        fm.PlotCoeff = Mock(return_value=None)
        self.histories.append(fm)
        return fm


@pytest.mark.parametrize(
    "transforms, expected", list(zip(TRANSFORMS, EXPECTED)))
@pytest.mark.parametrize("view", ["summary", "plot"])
def test_report_applies_transformations_once(
        monkeypatch, tmp_path, transforms, expected, view):
    """Summary and plot consume the same once-transformed data extraction."""
    monkeypatch.chdir(tmp_path)
    transforms = deepcopy(transforms)
    original = deepcopy(transforms)
    subfig_opts = {
        "NStats": 1, "NMinStats": 0, "Iteration": 4,
        "Components": ["entire"], "Component": "entire",
        "Coefficients": list(COEFFS), "Coefficient": "CA",
        "Position": "b", "Width": 1., "Header": "Coefficients",
        "MuFormat": "%.3f", "SigmaFormat": "%.3f",
    }
    opts = Mock()
    opts.get_SubfigOpt.side_effect = lambda name, key, *a: subfig_opts.get(key)
    opts.get_DataBookNStats.return_value = 1
    opts.get_DataBookNMin.return_value = 0
    opts.get_DataBookNMaxStats.return_value = None
    opts.get_DataBookDNStats.return_value = 1
    opts.get_DataBookTransformations.return_value = transforms
    opts.get_RefPoint.return_value = [0., 0., 0.]
    opts.get_RefLength.return_value = 1.
    opts.expand_Point.side_effect = lambda point: point
    cntl = SimpleNamespace(
        opts=opts, RootDir=str(tmp_path), CheckCase=lambda i: 4,
        x=SimpleNamespace(GetFullFolderNames=lambda i: "."),
        PreparePoints=Mock())
    runner = MemoryRunner(cntl, transforms)
    report = Report.__new__(Report)
    report.cntl = cntl
    report.read_runner = lambda i: runner
    report.get_case_name = lambda i: "."
    report.SubfigCaption = lambda sfig: "Coefficients"
    report.SubfigInit = lambda sfig: []

    # Exercise the shared reader directly, just as the databook does.
    baseline = report.ReadCaseFM(0, "entire")
    for coeff, value in zip(COEFFS, expected):
        np.testing.assert_allclose(baseline[coeff], value, atol=1e-12)

    # Render twice to check repeatability and option immutability.
    for _ in range(2):
        if view == "summary":
            lines = report.SubfigSummary("summary", 0)
        else:
            lines = report.SubfigPlotCoeff("plot", 0, True)
        history = runner.histories[-1]
        for coeff, value in zip(COEFFS, expected):
            np.testing.assert_allclose(history[coeff], value, atol=1e-12)
            if view == "summary":
                assert f"& ${value:.3f}$ " in "".join(lines)
        if view == "plot":
            history.PlotCoeff.assert_called_once()
        else:
            history.GetStats.assert_called_once()
        assert transforms == original


def test_pycart_summary_matches_transformed_history(monkeypatch, tmp_path):
    """Read the committed Cart3D fixture through the report and case reader."""
    from cape.pycart.cntl import Cntl

    source = Path(__file__).resolve().parents[2]
    source = source / "006_pycart/02_casedata"
    for name in ("pyCart.json", "c3dfunc.py", "matrix.csv", "Config.xml"):
        shutil.copy(source / name, tmp_path / name)
    shutil.copytree(source / "poweroff", tmp_path / "poweroff")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    cntl = Cntl()
    settings = json.loads(json.dumps(cntl.opts))
    comp = "bullet_no_base"
    settings["DataBook"][comp] = {
        "Type": "FM", "Transformations": [{"Type": "ScaleCoeffs", "CA": 2.0}]}
    settings["Report"]["Subfigures"]["summary"] = {
        "Type": "Summary", "Components": [comp], "Coefficients": ["CA"],
        "NStats": 1, "NMinStats": 0, "MuFormat": "%.8f"}
    (tmp_path / "pyCart.json").write_text(json.dumps(settings))
    cntl = Cntl()
    report = Report.__new__(Report)
    report.cntl = cntl
    history = report.ReadCaseFM(0, comp)
    expected = history.GetStats(nStats=1, nLast=200)["CA"]
    assert expected > 1.0
    lines = report.SubfigSummary("summary", 0)
    assert f"& ${expected:.8f}$ " in "".join(lines)

import numpy as np

from cape.cfdx.casedata import (
    CaseData,
    CaseFM,
    WindowRank,
    _recommend_action,
    rank_windows,
)


def test_getstats_window_method(monkeypatch):
    fm = CaseFM("test")
    fm.save_col("i", np.arange(200))
    fm.save_col("CA", np.linspace(0.0, 1.0, 200))
    calls = []

    def get_autocorr_state(self, col, **kw):
        calls.append(("autocorrelation", col))
        return {
            "mean": 1.0,
            "std": 0.1,
            "n_stats": 80,
            "min": 0.5,
            "max": 1.5,
            "error": 0.01,
        }

    def get_welch_state(self, col, **kw):
        calls.append(("welch", col))
        return {
            "mean": 2.0,
            "std": 0.2,
            "n_stats": 100,
            "min": 1.0,
            "max": 3.0,
            "error": 0.02,
        }

    monkeypatch.setattr(CaseFM, "get_col_state", get_autocorr_state)
    monkeypatch.setattr(CaseFM, "get_col_state_welch", get_welch_state)

    stats = fm.GetStats(50, 150)
    assert stats["CA"] == 1.0
    assert stats["nStats"] == 80
    assert calls == [("autocorrelation", "CA")]

    calls.clear()
    stats = fm.GetStats(50, 150, WindowMethod="welch")
    assert stats["CA"] == 2.0
    assert stats["nStats"] == 100
    assert calls == [("welch", "CA")]


def test_rank_windows_sinusoid_amplitude():
    n = 240
    period = 20
    x = np.arange(n, dtype=float)
    y = (
        2.0 +
        3.0*np.cos(2*np.pi*x/period) +
        4.0*np.sin(2*np.pi*x/period))
    ranks = rank_windows(x, y, period, np.array([20, 40, 80]))
    assert isinstance(ranks, WindowRank)
    assert np.allclose(ranks.sinusoid_amplitude, 5.0)


def test_get_col_state_period_sequence():
    n = 240
    period = 20
    x = np.arange(n, dtype=float)
    y = np.sin(2*np.pi*x/period)
    db = CaseData()
    db.save_col("i", x)
    db.save_coeff("signal", y)
    state = db.get_col_state("signal", nstats=1)

    periods = np.asarray(state["n_period"])
    pmax = int(periods[-1])
    periods2 = 2 ** np.arange(int(np.log2(pmax)) + 1)
    expected = np.unique(np.hstack((periods2, 3*periods2, [pmax])))
    expected = expected[expected <= pmax]

    assert np.array_equal(periods, expected)
    assert "sinusoid_amplitude" in state
    for window in state["windows"]:
        assert "sinusoid_amplitude" in state[str(window)]


def test_recommend_increasing_oscillatory_amplitude():
    state = {
        "n": 200,
        "n_min": 0,
        "n_stats": 40,
        "class": "oscillatory",
        "full_range": 10.0,
        "frequency": 20,
        "linear_fit_a1": 0.0,
        "autocorrelation": 1.0,
        "windows": [80, 20, 60, 40],
        "20": {"mean": 1.0, "sinusoid_amplitude": 1.23},
        "40": {
            "mean": 1.0,
            "sign_change_rate": 0.0,
            "sinusoid_amplitude": 1.11,
        },
        "60": {"mean": 1.0, "sinusoid_amplitude": 1.0},
        "80": {"mean": 1.0, "sinusoid_amplitude": 20.0},
    }
    _recommend_action(state)
    assert state["recommendation"] == "extend"
    assert state["reason"] == "increasing oscillatory amplitude"

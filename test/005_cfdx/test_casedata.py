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


def test_get_col_state_nlast_iteration_cutoff():
    nsample = 240
    period = 20
    iters = 2*np.arange(nsample)
    signal = np.sin(2*np.pi*np.arange(nsample)/period)
    signal[160:] += 100.0
    db = CaseData()
    db.save_col("i", iters)
    db.save_coeff("signal", signal)

    # The cutoff is an iteration value, not a number of samples
    state = db.get_col_state("signal", nstats=20, nlast=319)
    assert state["n"] == 160
    assert np.isclose(state["full_range_ptp"], 2.0)
    assert np.isclose(state["full_range_std"], 6.0/np.sqrt(2.0))
    assert state["full_range"] == state["full_range_std"]

    # Negative values are offsets from the final iteration value
    state = db.get_col_state("signal", nstats=20, nlast=-40)
    assert state["n"] == 220


def test_get_col_state_startup_range():
    nsample = 240
    period = 20
    iters = np.arange(nsample)
    signal = np.sin(2*np.pi*iters/period)
    signal[:40] += 100.0
    db = CaseData()
    db.save_col("i", iters)
    db.save_coeff("signal", signal)

    state0 = db.get_col_state("signal", nstats=20)
    state1 = db.get_col_state("signal", nstats=20, nstartup=40)
    assert state0["full_range"] > 100.0
    assert np.isclose(state1["full_range_ptp"], 2.0)
    assert np.isclose(state1["full_range_std"], 6.0/np.sqrt(2.0))
    assert state1["full_range"] == state1["full_range_std"]
    assert state1["full_mean"] == state0["full_mean"]

    state2 = db.get_col_state("signal", nstats=20, nstartup=300)
    assert state2["recommendation"] == "continue"
    assert state2["reason"] == "startup iterations not complete"


def test_get_col_state_full_range_candidates():
    nsample = 240
    period = 20
    iters = np.arange(nsample)

    # A nearly constant coefficient is scaled by its standard deviation
    signal = 10.0 + 0.01*np.sin(2*np.pi*iters/period)
    db = CaseData()
    db.save_col("i", iters)
    db.save_coeff("signal", signal)
    state = db.get_col_state("signal", nstats=20)
    assert state["full_range"] == state["full_range_std"]

    # A sufficiently broad history retains its observed peak-to-peak range
    signal = np.sin(2*np.pi*iters/period)
    signal[100] = 20.0
    db.save_coeff("signal", signal)
    state = db.get_col_state("signal", nstats=20)
    assert state["full_range"] == state["full_range_ptp"]
    assert np.isclose(state["full_range_ptp"], 21.0)


def test_get_col_state_configurable_targets():
    nsample = 240
    period = 20
    iters = np.arange(nsample)
    signal = np.sin(2*np.pi*iters/period)
    db = CaseData()
    db.save_col("i", iters)
    db.save_coeff("signal", signal)
    state = db.get_col_state(
        "signal",
        nstats=20,
        FullRangeStdFactor=4.0,
        MaxOscillatoryAmplitudeRatio=1.2,
        MaxSignChangeRate=0.2,
        MinOscillatoryAmplitudeFraction=0.02,
        TargetAutocorrelation=0.8,
        TargetDriftFraction=0.01,
        TargetDriftFractionMap={"signal": 0.02},
        TargetMeanRangeFraction=0.03,
    )
    assert state["full_range_std_factor"] == 4.0
    assert np.isclose(state["full_range_std"], 4.0/np.sqrt(2.0))
    assert state["target_drift_fraction"] == 0.02
    assert np.isclose(state["target_drift"], 0.02*state["full_range"])
    assert state["target_mean_range_fraction"] == 0.03
    assert np.isclose(
        state["target_mean_range"], 0.03*state["full_range"])
    assert state["target_autocorrelation_base"] == 0.8
    assert np.isclose(
        state["target_autocorrelation"],
        0.8*np.sqrt(state["frequency"]/state["n_stats"]))
    assert state["max_sign_change_rate"] == 0.2
    assert state["min_oscillatory_amplitude_fraction"] == 0.02
    assert state["max_oscillatory_amplitude_ratio"] == 1.2


def test_get_col_state_iteration_cutoff(monkeypatch):
    nsample = 240
    period = 20
    iters = 2*np.arange(nsample)
    signal = np.sin(2*np.pi*np.arange(nsample)/period)
    db = CaseData()
    db.save_col("i", iters)
    db.save_coeff("signal", signal)

    def fail_recommendation(state):
        raise AssertionError("_recommend_action() should not be called")

    monkeypatch.setattr(
        "cape.cfdx.casedata._recommend_action", fail_recommendation)
    # Compare against the actual iteration value, not number of samples
    state = db.get_col_state("signal", nstats=20, ncutoff=300)
    assert state["recommendation"] == "approve"
    assert state["reason"] == "maximum iteration reached"

    # The absolute cutoff also overrides early convergence exits
    state = db.get_col_state(
        "signal", nstats=20, nstartup=500, ncutoff=300)
    assert state["recommendation"] == "approve"
    assert state["reason"] == "maximum iteration reached"


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
        "windows": [20, 40, 60, 80],
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

    # A configurable amplitude significance floor can permit approval
    state["min_oscillatory_amplitude_fraction"] = 0.2
    _recommend_action(state)
    assert state["recommendation"] == "approve"
    assert state["reason"] == "stationary mean"

    # A configurable sign-change limit controls the retry gate
    state["40"]["sign_change_rate"] = 0.15
    state["max_sign_change_rate"] = 0.2
    _recommend_action(state)
    assert state["recommendation"] == "approve"
    state["max_sign_change_rate"] = 0.1
    _recommend_action(state)
    assert state["recommendation"] == "retry"

import numpy as np

from cape.cfdx.casedata import CaseFM


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

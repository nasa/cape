#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import json

# Third party modules
import numpy as np
import testutils

# Local imorts
import cape.runmatrix


# Universal test tolerance
TOL = 1e-4


# Settings for run matrix
TEST_OPTS = {
    "Keys":   ["mach", "alpha", "beta", "config"],
    "mach":   1.4,
    "alpha":  np.array([0.0, 4.0, 0.0, 4.0]),
    "beta":   np.array([0.0, 0.0, 4.0, 4.0]),
    "config": "poweroff",
    "Prefix": None,
}


# Test from options
def test_01_opts():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(**TEST_OPTS)
    # Check folder name
    assert x.GetFolderNames(0) == "m1.4a0.0b0.0"
    assert x.GetFullFolderNames(1) == "poweroff/m1.4a4.0b0.0"


# Test 02: definitions
def test_02_defns():
    # Create run matrix with extra option
    x = cape.runmatrix.RunMatrix(
        Definitions={
            "mach": {"Format": "%.2f"},
            "alpha": {"Abbreviation": "_a"},
        },
        **TEST_OPTS)
    # Check modified format
    assert x.GetFolderNames(2) == "m1.40_a0.0b4.0"


# Test 03: write conditions file
@testutils.run_sandbox(__file__)
def test03_conditions_json():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(**TEST_OPTS)
    # Write conditions file for case 1
    x.WriteConditionsJSON(1)
    # Read the file
    with open("conditions.json") as fp:
        conds = json.load(fp)
    # Test conditions
    assert conds["config"] == "poweroff"
    assert abs(conds["mach"] - 1.4) <= TOL


# Test 04: convert *aoap*, *phip* to *a*, *b*
@testutils.run_testdir(__file__)
def test_04_ab():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(
        Keys=["aoap", "phip"],
        aoap=np.array([0.0, 4.0, 4.0, 4.0, 4.0]),
        phip=np.array([0.0, 0.0, 45.0, 90.0, 235.0]))
    # Check some lookups
    assert abs(x.GetAlphaTotal(1) - 4.0) <= TOL
    assert abs(x.GetPhi(3) - 90.0) <= TOL
    assert abs(x.GetAlpha(2) - 2.8307) <= TOL
    assert abs(x.GetBeta(4) + 3.2757) <= TOL


# Test 05: convert *a*, *b* to *aoap*, *phip*
@testutils.run_testdir(__file__)
def test_05_ab():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(
        Keys=["alpha", "beta"],
        alpha=np.array([0.0, 4.0, 4.0, -4.0]),
        beta=np.array([0.0, 0.0, 4.0, -2.0]))
    # Test some values
    assert abs(x.GetAlphaTotal(2) - 5.6546) <= TOL
    assert abs(x.GetAlphaManeuver(3) + 4.4714) <= TOL
    assert abs(x.GetPhiManeuver(3) - 26.5930) <= TOL


# Test 06: convert *mach*, *q* to *p*
@testutils.run_testdir(__file__)
def test_06_mq():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(
        Keys=["mach", "alpha", "beta", "q"],
        mach=2.0,
        alpha=0.0,
        beta=0.0,
        q=np.array([100.0, 250.0, 300.0]))
    # Check some values
    assert abs(x.GetMach(0) - 2.) <= TOL
    assert abs(x.GetDynamicPressure(0) - 100.0) <= TOL
    assert abs(x.GetPressure(1) - 89.2857) <= TOL
    assert abs(x.GetTotalPressure(2) - 838.3338) <= TOL


# Test 07: convert *mach*, *q*, *T* to *Re*
@testutils.run_testdir(__file__)
def test_07_reynolds():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(
        Keys=["mach", "alpha", "beta", "q", "T"],
        mach=2.0,
        alpha=0.0,
        beta=0.0,
        q=np.array([100.0, 250.0, 300.0]),
        T=450.0)
    # Check values
    assert abs(x.GetTemperature(1) - 450.0) <= TOL
    assert abs(x.GetReynoldsNumber(0) - 23996.4884) <= TOL



#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Third party modules
import numpy as np

# Import cape module
import cape.trajectory


# Test 01: convert *aoap*, *phip* to *a*, *b*
def test01_ab():
    # Create run matrix
    x = cape.trajectory.Trajectory(
        Keys=["aoap", "phip"],
        aoap=np.array([0.0, 4.0, 4.0. 4.0, 4.0]),
        phip=np.array([0.0, 0.0, 45.0, 90.0, 235.0]))
    # Status update
    print("01: aoap, phip --> a, b")
    # Display folder names
    for i in range(x.nCase):
        # Get conditions
        a = x.GetAlpha(i)
        b = x.GetBeta(i)
        # Display
        print("%i: a=%5.2f, b=%5.2f" % (i, a, b))


# Test 02: full folder names
def test02_full_names():
    # Create run matrix
    x = cape.trajectory.Trajectory(**opts)
    # Status update
    print("02: Full folder names")
    # Display folder names
    for fdir in x.GetFullFolderNames():
        print(fdir)


# Test 03: total angle of attack
def test03_alpha_t():
    # Create run matrix
    x = cape.trajectory.Trajectory(**opts)
    # Status update
    print("03: Total angle of attack")
    # Display total angle of attack and roll angle
    for i in range(x.nCase):
        # Get conditions
        aoap = x.GetAlphaTotal(i)
        phip = x.GetPhi(i)
        # Display
        print("aoap = %.2f, phip = %5.2f" % (aoap, phip))

# Test 04: fixed format
def test04_format():
    # Create run matrix with extra option
    x = cape.trajectory.Trajectory(
        Definitions={
            "mach": {"Format": "%.2f"},
            "alpha": {"Abbreviation": "_a"},
        },
        **opts)
    # Status update
    print("04: Modified format")
    # Display folder names of case 2
    frun = x.GetFolderNames(2)
    # Display it
    print(frun)

# Test 05: conditions file
def test05_conditions_json():
    # Create run matrix
    x = cape.trajectory.Trajectory(**opts)
    # Status update
    print("05: Conditions JSON file")
    # Write conditions file for case 1
    x.WriteConditionsJSON(1)
    # Read the file
    lines = open("conditions.json").readlines()
    # Display stripped lines
    for line in lines:
        # Strip trailing white space
        line = line.rstrip()
        # Skip empty lines
        if line == "": continue
        # Display it
        print(line)
    


if __name__ == "__main__":
    test01_ab()


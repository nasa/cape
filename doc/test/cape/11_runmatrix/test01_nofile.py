#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Third party modules
import numpy as np

# Import cape module
import cape.trajectory

# Settings for run matrix
opts = {
    "Keys":   ["mach", "alpha", "beta", "config"],
    "mach":   1.4,
    "alpha":  np.array([0.0, 4.0, 0.0, 4.0]),
    "beta":   np.array([0.0, 0.0, 4.0, 4.0]),
    "config": "poweroff",
    "Prefix": None,
}


# Test 01: display folder names
def test01_folder_names():
    # Create run matrix
    x = cape.trajectory.Trajectory(**opts)
    # Status update
    print("01: Folder names")
    # Display folder names
    for fdir in x.GetFolderNames():
        print(fdir)


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
    test01_folder_names()
    test02_full_names()
    test03_alpha_t()
    test04_format()
    test05_conditions_json()


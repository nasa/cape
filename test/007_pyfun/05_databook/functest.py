
# Standard library
import os

# Third-party
import cape.pyfun.databook as dbook
import cape.pyfun.cntl as pfcntl
import numpy as np

XCOLS = (
    "mach",
    "alpha",
    "beta",
    "q",
    "T",
    "config",
    "Label"
)

COEFFS = (
    "CA",
    "CY",
    "CN"
)

def f3dfunc(cntl, i):
    cntl = pfcntl.Cntl()
    db = dbook.DataBook(cntl)
    # Write and read databooks
    fname1 = db["cap"].fname
    fname2 = db["bullet_total"].fname
    db["cap"].Write()
    db["bullet_total"].Write()
    db["cap"].Read(fname1)
    db["bullet_total"].Read(fname2)
    # Take XCOLS from bullet_total
    for col in XCOLS:
        db["functest"].save_col(col, db["bullet_total"])
    # Random operations on COEFFS
    for cof in COEFFS:
        val = (db["bullet_total"] - 1.2*db["cap"])
        db["functest"].save_col(f"{cof}_delta", val)
    return db["functest"]

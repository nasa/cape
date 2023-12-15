# -*- coding: utf-8 -*-

# Third-party
import numpy as np
import testutils

# Local imports
import cape.tri as trifile
import cape.plt as pltfile


# Config file
CONFIGJSONFILE = "arrow.json"
# Prefix
OUTPUT_PREFIX = "arrow"
# XML file
XMLFILE = OUTPUT_PREFIX + ".xml"
# Source uh3d
SOURCE = "arrow.uh3d"
# Output tri file
TRIFILE = "arrow-tri10k.lr4.tri"
PLTFILE = "arrow-tri10k.plt"
TRIFILEOUT = "arrow.tri"

# Test files
TEST_FILES = (
    CONFIGJSONFILE,
    SOURCE
)


# Execute nominal UH3D-to-TRI process
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_convertuh3d():
    # Read source
    tri = trifile.Tri(uh3d=SOURCE, c=CONFIGJSONFILE)
    # Write XML file
    tri.WriteConfigXML(XMLFILE)
    # Map the AFLR3 boudnary conditions
    tri.MapBCs_ConfigAFLR3()
    # Write the AFLR3 boundary condition summary
    tri.config.WriteAFLR3BC(OUTPUT_PREFIX + ".bc")
    # Map the FUN3D boundary conditions
    tri.config.WriteFun3DMapBC(OUTPUT_PREFIX + ".mapbc")
    # Write surface
    print("Writing surface TRI file")
    # Number of triangles
    ntrik = (tri.nTri - 10) // 1000 + 1
    assert ntrik == 10
    # File name
    fname = OUTPUT_PREFIX + ("-tri%ik.lr4.tri" % ntrik)
    # Write it
    tri.WriteTri_lr4(fname)
    # Reread tri file
    tri0 = trifile.Tri(fname, c=XMLFILE)
    # Create PLTFile interface
    plt = pltfile.Plt(triq=tri0, c=XMLFILE)
    # File name
    fname = OUTPUT_PREFIX + ("-tri%ik.plt" % ntrik)
    # Write it
    plt.Write(fname)


@testutils.run_sandbox(__file__, fresh=False)
def test_02_compids():
    # Read TRI file
    tri = trifile.Tri(fname=TRIFILE)
    # Print the unique IDs
    compids = list(np.unique(tri.CompID))
    # Check
    assert compids == [1, 2, 3, 11, 12, 13, 14]


@testutils.run_sandbox(__file__, fresh=False)
def test_03_readplt():
    # Read PLT file
    plt = pltfile.Plt(PLTFILE)
    # Check it
    assert plt.Vars == ["x", "y", "z"]
    assert plt.Zones == [
        "boundary 1 cap",
        "boundary 2 body",
        "boundary 3 base",
        "boundary 11 fin1",
        "boundary 12 fin2",
        "boundary 13 fin3",
        "boundary 14 fin4",
    ]

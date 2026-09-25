# Standard library
import os

# Local imports
from cape.pyvul.inpfile import VulcanInpFile


# Test file
TEST_FILE = os.path.join(os.path.dirname(__file__), "vulcan.inp")


# Test string representations
def test_repr():
    assert repr(VulcanInpFile()) == "<VulcanInpFile>"
    inp = VulcanInpFile(TEST_FILE)
    assert repr(inp) == "<VulcanInpFile('vulcan.inp')>"
    assert str(inp) == repr(inp)


# Test reading of basic options
def test_read_basic():
    inp = VulcanInpFile(TEST_FILE)
    assert inp.get_mach() == 4.0
    assert inp.get_alpha() == 1.5
    assert inp.get_beta() == 0.0
    assert inp.get_processors() == 1.0
    assert inp.get_pressure() == 1600.96
    assert inp.get_temperature() == 64.722222
    assert inp.get("GAMMA") == 1.4
    assert inp.get("MAX. STATIC TEMP") == 450.0
    assert inp.get("UNIT REYNOLDS NO.") == -1.0
    assert inp.get("THREED") is None
    assert inp.get("PROCESSORS") == 1.0


# Test continuation lines
def test_continuations():
    inp = VulcanInpFile(TEST_FILE)
    assert inp.get_gridfile() == "./my_example.b8.ugrid"
    assert inp.get_cont("RESTART IN") == ["Restart_files/restart"]
    assert inp.get_cont("TURBULENCE MODEL") == ["MENTER-SST-2003"]
    names = inp.get_cont("PLOT FUNCTION")
    assert len(names) == 10
    assert names[0] == "DENSITY"
    assert names[2] == "MACH NO."


# Test read/write round trip is byte-identical
def test_roundtrip():
    inp = VulcanInpFile(TEST_FILE)
    orig = open(TEST_FILE, 'r').read()
    assert '\n'.join(inp.to_lines()) + '\n' == orig


# Test value changes write back correctly
def test_set_values(tmp_path):
    inp = VulcanInpFile(TEST_FILE)
    inp.set_mach(3.5)
    inp.set_alpha(-2.0)
    inp.set_processors(24)
    inp.set_gridfile("./new_grid.lb8.ugrid")
    fname = str(tmp_path / "vulcan_mod.inp")
    inp.write(fname)
    # Reread
    inp2 = VulcanInpFile(fname)
    assert inp2.get_mach() == 3.5
    assert inp2.get_alpha() == -2.0
    assert inp2.get_processors() == 24.0
    assert inp2.get_gridfile() == "./new_grid.lb8.ugrid"
    # Comments and untouched lines survive
    lines = open(fname, 'r').read().splitlines()
    assert lines[0].startswith('$**')
    assert any('SUTHERLANDS LAW MU0' in ll for ll in lines)
    assert any('(no. of processors to invoke)' in ll for ll in lines)


# Test BC groups interface
def test_bcgroups():
    inp = VulcanInpFile(TEST_FILE)
    bcg = inp.bcgroups
    assert len(bcg) == 9
    assert list(bcg.find_type('AWALL')) == ['fbody', 'oml', 'side']
    assert list(bcg.find_type('FIX_IN')) == ['inflow', 'farfield']
    grp = bcg['fbody']
    assert grp['TYPE'] == 'AWALL'
    assert grp['OPTIONS'] == ['PHYSICAL_IBL']
    assert grp['BL_delta'] == 0.005
    # FIX_IN groups keep auxiliary lines
    aux = bcg['inflow']._rawlines
    assert len(aux) == 4
    assert 'Density' in aux[1]


# Test writing modified BC groups
def test_bcgroups_write(tmp_path):
    inp = VulcanInpFile(TEST_FILE)
    inp.bcgroups.set_bl_delta('fbody', 0.006)
    inp.bcgroups['oml']['TYPE'] = 'IWALL'
    fname = str(tmp_path / "vulcan_bc.inp")
    inp.write(fname)
    inp2 = VulcanInpFile(fname)
    assert inp2.bcgroups['fbody']['BL_delta'] == 0.006
    assert inp2.bcgroups['oml']['TYPE'] == 'IWALL'
    # Untouched group keeps its state data verbatim
    lines = open(fname, 'r').read().splitlines()
    assert any('0.10000  500.2  0.0      25.0    85.0' in ll
               for ll in lines)


# Test BC objects interface
def test_bcobjects():
    inp = VulcanInpFile(TEST_FILE)
    bco = inp.bcobjects
    assert set(bco.keys()) == {'prop_walls', 'inl_walls'}
    assert bco['prop_walls'] == ['fbody', 'side']


# Test BC objects write-back
def test_bcobjects_write(tmp_path):
    inp = VulcanInpFile(TEST_FILE)
    inp.bcobjects.set_members('prop_walls', ['fbody', 'side', 'diverter'])
    fname = str(tmp_path / "vulcan_bco.inp")
    inp.write(fname)
    inp2 = VulcanInpFile(fname)
    assert inp2.bcobjects['prop_walls'] == ['fbody', 'side', 'diverter']
    assert inp2.bcobjects['inl_walls'] == ['fbody', 'side']


# Test block configuration table
def test_blockconfig():
    inp = VulcanInpFile(TEST_FILE)
    bc = inp.blockconfig
    assert set(bc.keys()) == {0}
    assert bc[0]['VISC'] == 'F'
    assert bc[0]['TURB'] == 'Y'
    assert bc[0]['REAC'] is None
    assert bc[0]['REGION'] == '1'


# Test block config write-back
def test_blockconfig_write(tmp_path):
    inp = VulcanInpFile(TEST_FILE)
    inp.blockconfig[0]['REGION'] = '2'
    fname = str(tmp_path / "vulcan_blk.inp")
    inp.write(fname)
    inp2 = VulcanInpFile(fname)
    assert inp2.blockconfig[0]['REGION'] == '2'
    assert inp2.blockconfig[0]['VISC'] == 'F'


# Test region control specification
def test_regions():
    inp = VulcanInpFile(TEST_FILE)
    assert set(inp.regions.keys()) == {1}
    reg = inp.regions[1]
    assert reg['SOLVER/STATUS']['SOLVER/STATUS'] == 'E/A'
    fmg = reg['FMG-LVLS']
    assert fmg['FMG-LVLS'] == 1.0
    assert fmg['REL RES'] == -99.0
    assert fmg['ABS RES'] == -99.0
    scheme = reg['SCHEME']
    assert scheme['SCHEME'] == 'SGS'
    assert scheme['CFL-VALS'] == 7.0
    # Implicit scheme line has fewer values than header tokens
    sgs = reg['SGS']
    assert sgs['JAC-UPDATE'] == 0.0
    assert sgs['NUM-SLV'] == 5.0
    # Unnamed CFL schedule lines stored as lists
    assert len(sgs.extras) == 2
    assert sgs.extras[0] == [1.0, 500.0, 1000.0, 1001.0,
                             1500.0, 10000.0, 15000.0]
    assert sgs.extras[1] == [0.1, 1.0, 50.0, 0.5, 50.0, 50.0, 25.0]


# Test region control write-back
def test_regions_write(tmp_path):
    inp = VulcanInpFile(TEST_FILE)
    inp.regions[1]['FMG-LVLS']['ABS RES'] = -8.0
    fname = str(tmp_path / "vulcan_reg.inp")
    inp.write(fname)
    inp2 = VulcanInpFile(fname)
    reg2 = inp2.regions[1]
    assert reg2['FMG-LVLS']['ABS RES'] == -8.0
    # Untouched rows stay verbatim
    assert reg2['KAPPA']['FLUX SCHEME'] == 'LDFSS'
    # CFL schedule lines preserved
    assert len(reg2['SGS'].extras) == 2

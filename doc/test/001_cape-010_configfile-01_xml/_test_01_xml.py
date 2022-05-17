
# Third-party
import testutils

# Local imports
import cape.config as configfile


CONFIG_FILE = "arrow.xml"


# Basic test
@testutils.run_testdir(__file__)
def test_01_xml():
    # Read config file
    cfg = configfile.ConfigXML(CONFIG_FILE)
    # Test some component IDs
    assert cfg.GetCompID("base") == [3]
    assert cfg.GetCompID("fins") == [11, 12, 13, 14]


# Write and reread
@testutils.run_sandbox(__file__, "arrow.xml")
def test_02_xmlwrite():
    # Read config file
    cfg = configfile.ConfigXML(CONFIG_FILE)
    # Write
    cfg.WriteXML("arrow2.xml")
    # Rearead it
    cfg2 = configfile.ConfigXML("arrow2.xml")
    # Test
    assert cfg.faces == cfg2.faces

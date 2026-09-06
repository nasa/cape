from cape.cfdx.options import Options


JUMP_HOSTS = {
    "athfe0[1-4]": {
        "@map": {"tur": None, "rom": "afe02"},
        "key": "arch",
    },
    "x1[0-9]+c[0-9]s[0-9]b[0-9]n[0-9]": {
        "@map": {
            "tur": None,
            "rom": ["athfe04", "afe02"],
        },
        "key": "arch",
    },
}


def test_JumpHosts():
    opts = Options(
        JumpHosts=JUMP_HOSTS,
        PBS={"PBSPrefix": "/PBS/bin"})
    opts.save_x({"arch": ["tur", "rom"]})

    assert opts.get_JumpHost(i=0, hostname="athfe02") is None
    assert opts.get_JumpHost(i=1, hostname="athfe02") == "afe02"
    assert opts.get_JumpHost(i=1, hostname="x12c3s4b5n6") == [
        "athfe04", "afe02"]
    assert opts.get_JumpHost(i=1, hostname="other") is None
    assert opts.get_PBS_PBSPrefix(i=1) == "/PBS/bin"

from cape.cfdx import queue


def test_pbs_cmd_local():
    assert queue._pbs_cmd("qstat", ["-u", "user"]) == [
        "qstat", "-u", "user"]
    assert queue._pbs_cmd("qdel", ["123"], prefix="/PBS/bin") == [
        "/PBS/bin/qdel", "123"]


def test_pbs_cmd_remote():
    assert queue._pbs_cmd(
        "qsub", ["/work/case/run.pbs"], "afe02", "/PBS/bin") == [
            "ssh", "afe02", "/PBS/bin/qsub /work/case/run.pbs"]
    assert queue._pbs_cmd(
        "qstat", ["-u", "user"], ["athfe04", "afe02"]) == [
            "ssh", "-J", "athfe04", "afe02", "qstat -u user"]

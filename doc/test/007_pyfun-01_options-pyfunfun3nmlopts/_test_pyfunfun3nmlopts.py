# Local
from cape.pyfun.options import fun3dnmlopts


def test_meshopts01():
    # Initialize options
    opts = fun3dnmlopts.Fun3DNmlOpts(
        {
            "nonlinear_solver_parameters": {
                "cfl_schedule": [
                    [0.1, 1.0],
                    [1.0, 10.0],
                    [10.0, 50.0]],
            },
        }
    )
    # Sample entire namelist to phase 0
    opts0 = opts.select_namelist(j=0)
    # Test it
    assert opts0["nonlinear_solver_parameters"]["cfl_schedule"] == [0.1, 1.0]
    # Get value
    v0 = opts.get_namelist_var(
        "nonlinear_solver_parameters", "cfl_schedule", j=1)
    assert v0 == [1.0, 10.0]
    # Set a value
    opts.set_namelist_var("code_run_control", "restart_read", "off")
    # Test it
    assert opts.get_namelist_var("code_run_control", "restart_read") == "off"
    # Get specific namelists
    assert opts.get_project() == {}
    assert opts.get_raw_grid() == {}
    # Set project name
    opts.set_namelist_var("project", "project_rootname", "pyfun00", 0)
    opts.set_namelist_var("project", "project_rootname", "pyfun01", 1)
    # Set grid type
    opts.set_namelist_var("raw_grid", "grid_format", "aflr3")
    # Test
    assert opts.get_project_rootname(j=1) == "pyfun01"
    assert opts.get_grid_format() == "aflr3"


# CAPE Agent Instructions

## Project Overview

CAPE (Computational Aerosciences Productivity & Execution) is a NASA CFD
run-matrix management tool that:
- Executes different CFD solvers in `cape/py[a-z0-9]+` based on `cape/cfdx/`
- Processes modified databases in `cape/dkit/`

## Architecture

Generic CFD run-matrix functionality lives in `cape/cfdx/`; solver-specific
packages generally extend it. The main run-matrix controller is
`cape.cfdx.cntl.Cntl`.

CLI parsing and dispatch are primarily implemented in `cape/cfdx/cli.py`.

## Build & Test Commands

### Build

Create Python-only wheel:

```bash
python3 setup.py build
python3 setup.py bdist_wheel
```

Create wheel with extensions:

```bash
python3 setup_with_extensions.py build
python3 setup_with_extensions.py bdist_wheel
```

Build extensions for local use:

```bash
python3 build.py
```

The build instructions, which would normally be in `setup.py`, are defined in
`cape/setup_py/__init__.py`. This work-around allows the extension build
process to work for different Python versions using version- and host-specific
`.cfg` files in the same folder.

### Test

Run nightly test suite:

```bash
bash test/qsub_pytest.sh
```

Run pytest with coverage using current Python version(s):

```bash
python3 drive_pytest.py
```

Tests are added in the ``test/`` folder and roughly organized by module name.
Each module usually has its own folder, e.g. ``test/001_cape/001_runmatrix``
for ``cape.cfdx.runmatrix``. Test folders numbered 900 or higher (e.g.
``test/901_pycart``) require actually running CFD and can only be run on HPC
systems.

To execute command-line calls in any tests, use `testutils.call_o`. Standard
`capsys` calls will usually break on at least one of the testing HPC systems.

### Lint
```bash
flake8 cape/              # Check with flake8 (see .flake8 for config)
```

## Module Structure

The folders `cape/py[a-z0-9]+/` each control a single CFD solver, e.g.

- `pycart/` - Cart3D solver
- `pyfun/` - FUN3D solver

The common CFD solver code is in `cape/cfdx/`.

The `cape/dkit/` module is the base for post-processing and delivering data.


## Key Conventions

1. **CLI Commands**: All commands defined in `cape/cfdx/cli.py` via `CMD_DICT`
   mapping command names to `cape_*` functions
2. **Option Handling**: Options defined in `_optlist`, types in `_opttypes`,
   aliases in `_optmap`
3. **Docstrings**: RST format with `:Call:`, `:Inputs:`, `:Outputs:` sections
   for new functions, don't add the `:Versions:` section
4. **Slots**: Classes use `__slots__` for memory efficiency
   (`cape.cfdx.cntl.Cntl` and `cape.dkit.rdb.DataKit` are exceptions to this
   directive.)


## Documentation

- API docs: `doc/api/cape/index.rst` and subfolders
- Build docs: `doc/` folder with Sphinx configuration
- New modules should add RST files to `doc/api/cape/` and update index
- If the new module is an `__init__.py` module, put the title in `index.rst`
  instead of the module's docstring; otherwise Sphinx can show grandchildren at
  the wrong TOC depth.

## Common Tasks

### Adding a CLI command
When adding or modifying CLI commands, follow the patterns in
`cape/cfdx/cli.py`; command dispatch is controlled by `CMD_DICT`.

### Adding a CLI option
1. Add to `CfdxFrontDesk._optlist`
2. Add to other `CfdxFrontDesk` attributes `._opt*` and `_help*` as appropriate
3. Add to sub-command parser's `_optlist`, e.g.

Do not add ordinary subcommand-specific options to `CfdxArgReader._optlist`;
options there are inherited by every subcommand.

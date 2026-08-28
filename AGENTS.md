# CAPE Agent Instructions

## Project Overview

CAPE (Computational Aerosciences Productivity & Execution) is a NASA CFD
run-matrix management tool that:
- Executes and post-processes multiple CFD solvers: Cart3D, FUN3D, OVERFLOW,
  Kestrel, LAVA
- Creates "datakits" - databases/toolkits for aerosciences data

## Key Entry Points

### CLI Commands
- `cape/cfdx/cli.py` - Main CLI with `CfdxFrontDesk` parser and `CMD_DICT` routing
- `cape/cli.py` - Auxiliary commands (e.g., `cape-expandjson`)
- `cape/agent/__init__.py` - Agentic LLM interface (`cape --agentic`)
- `cape/ui/__init__.py` - Readline-based interactive UI (`cape --ui`)

### Core Modules
- `cape/cfdx/cntl.py` - Main `Cntl` class for run matrix control
- `cape/cfdx/databook.py` - `DataBook` class for CFD data database
- `cape/cfdx/casecntl.py` - Case control class
- `cape/argread/` - Argument parsing base classes

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
python3 build.py          # Build extensions for local use
```

### Test

Run nightly test suite:

```bash
bash test/qsub_pytest.sh
```

Run pytest with coverage using current Python version:

```bash
python3 drive_pytest.py
```

Run limited tests for debugging
```bash
bash run_capetest.sh        # Run pytest with coverage
```

### Lint
```bash
flake8 cape/              # Check with flake8 (see .flake8 for config)
```

## Module Structure

```
cape/
├── agent/       # LLM agentic interface (new)
├── argread/     # Argument parsing base
├── cfdx/        # Generic CFD run matrix (core)
│   ├── cli.py       # CLI commands (CMD_DICT, CfdxFrontDesk)
│   ├── cntl.py      # Cntl class
│   ├── databook.py  # DataBook class
│   └── options/     # JSON option definitions
├── dkit/        # Data kit modules
├── filecntl/    # File control classes
├── gruvoc/      # Grid/visualization formats
├── nmlfile/     # Fortran namelist handling
├── optdict/     # Option dictionary
└── ui/          # Readline-based UI
```

## Solver-Specific Modules
- `pycart/` - Cart3D solver
- `pyfun/` - FUN3D solver
- `pyover/` - OVERFLOW solver
- `pykes/` - Kestrel solver
- `pylava/` - LAVA solver
- `pylch/` - CREATE-AV LCH

## Test Organization
- `test/000_vendor/` - Vendor/third-party tests
- `test/001_cape/` - Base cape module tests
- `test/005_cfdx/` - CFD run matrix tests
- `test/006_pycart/` - Cart3D-specific tests
- `test/007_pyfun/` - FUN3D-specific tests
- `test/008_pyover/` - OVERFLOW-specific tests

## Key Conventions

1. **CLI Commands**: All commands defined in `cape/cfdx/cli.py` via `CMD_DICT`
   mapping command names to `cape_*` functions
2. **Option Handling**: Options defined in `_optlist`, types in `_opttypes`,
   aliases in `_optmap`
3. **Docstrings**: RST format with `:Call:`, `:Inputs:`, `:Outputs:` sections
4. **Slots**: Classes use `__slots__` for memory efficiency
5. **Agentic Mode**: New `--agentic` flag uses `cape/agent/__init__.py` with
   LLM tool calling

## Documentation

- API docs: `doc/api/cape/index.rst` and subfolders
- Build docs: `doc/` folder with Sphinx configuration
- New modules should add RST files to `doc/api/cape/` and update index

## Common Tasks

### Adding a CLI command
1. Add `cape_<name>()` function in `cape/cfdx/cli.py`
2. Add `"<cmd>": cape_<name>` to `CMD_DICT`
3. Create `Cfdx<Name>Args` class with options
4. Add to `CMD_NAMES` if needed

### Adding an option
1. Add to `CfdxArgReader._optlist`
2. Add type to `_opttypes` if non-boolean
3. Add to `CfdxFrontDesk._optlist`

### Adding documentation
1. Create `doc/api/cape/<module>.rst` with `.. automodule:: cape.<module>`
2. Add to `doc/api/cape/index.rst` toctree

r"""
:mod:`cape.cfdx.quickstart`: Create templates for a new CAPE project
=====================================================================

This module provides the backend for the ``cape quickstart`` command,
which creates a new JSON (or YAML) settings file for one of CAPE's CFD
solvers. It also creates small template input files such as
``inputs/matrix.csv`` that the settings file references.
"""

# Standard library
import importlib
import os
import shutil
from typing import Optional, Tuple

# CAPE modules
from ..errors import CapeValueError
from ..optdict import OptionsDict


# Name of folder for template input files
INPUT_DIR = "inputs"
# Name of run matrix file
MATRIX_NAME = "matrix.csv"

# Run matrix keys to use when settings have not keys
DEFAULT_KEYS_PYCART = ("mach", "aoap", "phip", "config")
DEFAULT_KEYS = ("mach", "aoap", "phip", "q", "T", "config")

# Values for the single case of a generated run matrix file
MATRIX_KEY_VALUES = {
    "mach": "0.80",
    "aoap": "2.00",
    "phip": "0.00",
    "q": "150.0",
    "T": "518.67",
}

# Fallback for unrecognized run matrix keys
DEFAULT_MATRIX_VALUE = "0.0"

# Packaged template files to copy into *INPUT_DIR* for each solver; each
# template file is mapped to the option that references it
TEMPLATE_FILES = {
    "pycart": {
        "aero.csh": "AeroCsh",
        "input.cntl": "InputCntl",
    },
    "pyfun": {
        "fun3d.nml": "Fun3DNamelist",
        "rubber.data": "RubberDataFile",
    },
    "pyover": {
        "overflow.inp": "OverNamelist",
    },
}


# Main function
def quickstart(
        solver: str,
        fjson: Optional[str] = None,
        force: bool = False,
        fdir: str = INPUT_DIR) -> list:
    r"""Create a new CAPE settings file and template input files

    Files that already exist are not overwritten unless *force* is
    ``True``.

    :Call:
        >>> files = quickstart(solver, fjson=None, force=False, **kw)
    :Inputs:
        *solver*: :class:`str`
            Name of CAPE solver module, e.g. ``"pycart"``
        *fjson*: {``None``} | :class:`str`
            Name of settings file to create; default from *solver*
        *force*: {``False``} | :class:`bool`
            Option to overwrite existing settings file
        *fdir*: {``"inputs"``} | :class:`str`
            Name of folder in which to create template input files
    :Outputs:
        *files*: :class:`list`\ [:class:`str`]
            List of files created
    """
    # Get name of settings file to write
    if fjson is None:
        fjson = get_default_json_name(solver)
    # Write the JSON file unless it already exists
    files = []
    if write_json_settings(solver, fjson, force=force, fdir=fdir):
        files.append(fjson)
    # Write template input files
    files += write_input_templates(solver, fjson, fdir=fdir)
    return files


# Write a settings file
def write_json_settings(
        solver: str,
        fjson: str,
        force: bool = False,
        fdir: str = INPUT_DIR) -> bool:
    r"""Create a new CAPE settings file unless it already exists

    :Call:
        >>> created = write_json_settings(solver, fjson, force=False, **kw)
    :Inputs:
        *solver*: :class:`str`
            Name of CAPE solver module, e.g. ``"pycart"``
        *fjson*: :class:`str`
            Name of settings file to create
        *force*: {``False``} | :class:`bool`
            Option to overwrite existing settings file
        *fdir*: {``"inputs"``} | :class:`str`
            Name of folder for template input files
    :Outputs:
        *created*: :class:`bool`
            Whether the file was written
    """
    # Check for existing file
    if os.path.isfile(fjson) and not force:
        print(f"Unchanged {fjson}")
        return False
    # Instantiate bare options interface
    opts = read_solver_options(solver)
    # Point file options at the template files
    for fname, opt in TEMPLATE_FILES.get(solver, {}).items():
        opts.set_opt(opt, f"{fdir}/{fname}")
    # Get "RunMatrix" section
    rmo = opts["RunMatrix"]
    # Check for any trajectory keys in the dict itself
    if not rmo.get("Keys"):
        # Use defaults
        rmo.set_opt("Keys", list(get_default_keys(solver)))
    # Point to run matrix file in input folder
    rmo.set_opt("File", f"{fdir}/{MATRIX_NAME}")
    # Remove any unregistered top-level options
    prune_unknown_opts(opts)
    # Pick writer based on file extension
    if fjson.endswith((".yaml", ".yml")):
        writer = opts.write_yamlfile
    else:
        writer = opts.write_jsonfile
    # Status update
    print(f"Created {fjson}")
    # Write the file
    writer(fjson)
    return True


# Write input templates
def write_input_templates(
        solver: str,
        fjson: str,
        fdir: str = INPUT_DIR) -> list:
    r"""Create template input files for a new CAPE project

    Files that already exist are not overwritten.

    :Call:
        >>> files = write_input_templates(solver, fjson, **kw)
    :Inputs:
        *solver*: :class:`str`
            Name of CAPE solver module, e.g. ``"pycart"``
        *fjson*: :class:`str`
            Name of settings file; only the base name is used
        *fdir*: {``"inputs"``} | :class:`str`
            Name of folder in which to create template input files
    :Outputs:
        *files*: :class:`list`\ [:class:`str`]
            List of files created
    """
    # Initialize output
    files = []
    # Make input folder if necessary
    os.makedirs(fdir, exist_ok=True)
    # Write run matrix file
    fmat = os.path.join(fdir, MATRIX_NAME)
    if os.path.isfile(fmat):
        print(f"Unchanged {fmat}")
    else:
        with open(fmat, "w") as fp:
            fp.write(genr8_matrix_file(solver, get_config_name(fjson)))
        print(f"Created {fmat}")
        files.append(fmat)
    # Copy solver-specific template files
    for fname in TEMPLATE_FILES.get(solver, {}):
        # Path to target file
        ftarg = os.path.join(fdir, fname)
        # Skip if present
        if os.path.isfile(ftarg):
            print(f"Unchanged {ftarg}")
            continue
        # Path to packaged template
        fsrc = get_template_file(solver, fname)
        # Copy it
        shutil.copyfile(fsrc, ftarg)
        print(f"Created {ftarg}")
        files.append(ftarg)
    return files


# Remove top-level options not registered for their class
def prune_unknown_opts(opts: OptionsDict):
    r"""Remove top-level options not registered for their class

    Some solvers have legacy options that are not registered for the
    current solver class; these usually generate warnings when read.

    :Call:
        >>> prune_unknown_opts(opts)
    :Inputs:
        *opts*: :class:`cape.optdict.OptionsDict`
            Options interface with default settings
    """
    # Get full class attribute lists
    known = set(opts.getx_cls_set("_optlist"))
    known |= set(opts.getx_cls_set("_sec_cls"))
    # Add option aliases to list of known options
    optmap = opts.getx_cls_set("_optmap")
    if isinstance(optmap, dict):
        known |= set(optmap.keys()) | set(optmap.values())
    # Remove unregistered top-level options
    for opt in list(opts.keys()):
        if opt not in known:
            del opts[opt]


# Generate a run matrix file
def genr8_matrix_file(solver: str, config: str) -> str:
    r"""Generate the content of a template run matrix file

    The file has a header line naming the run matrix keys and a single
    case using nominal values.

    :Call:
        >>> txt = genr8_matrix_file(solver, config)
    :Inputs:
        *solver*: :class:`str`
            Name of CAPE solver module, e.g. ``"pycart"``
        *config*: :class:`str`
            Value for the ``"config"`` run matrix key
    :Outputs:
        *txt*: :class:`str`
            Content of a template run matrix file
    """
    # Get list of keys
    keys = get_default_keys(solver)
    # Initialize values
    vals = []
    # Loop through keys
    for key in keys:
        if key == "config":
            # Use settings file name
            vals.append(config)
        else:
            # Nominal value
            vals.append(MATRIX_KEY_VALUES.get(key, DEFAULT_MATRIX_VALUE))
    # Assemble text
    return "# %s\n  %s\n" % (", ".join(keys), ", ".join(vals))


# Default trajectory keys for a solver
def get_default_keys(solver: str) -> Tuple[str, ...]:
    r"""Get default run matrix keys for a solver

    :Call:
        >>> keys = get_default_keys(solver)
    :Inputs:
        *solver*: :class:`str`
            Name of CAPE solver module, e.g. ``"pycart"``
    :Outputs:
        *keys*: :class:`tuple`\ [:class:`str`]
            List of run matrix key names
    """
    if solver == "pycart":
        return DEFAULT_KEYS_PYCART
    else:
        return DEFAULT_KEYS


# Name for "config" key
def get_config_name(fjson: str) -> str:
    r"""Get default value of the ``"config"`` key from a file name

    :Call:
        >>> config = get_config_name(fjson)
    :Inputs:
        *fjson*: :class:`str`
            Name of settings file; only the base name is used
    :Outputs:
        *config*: :class:`str`
            File base name, lower case, with extension removed
    """
    return os.path.basename(fjson).split('.')[0].lower()


# Path to a packaged template file
def get_template_file(solver: str, fname: str) -> str:
    r"""Get full path to a packaged template file

    :Call:
        >>> fabs = get_template_file(solver, fname)
    :Inputs:
        *solver*: :class:`str`
            Name of CAPE solver module, e.g. ``"pycart"``
        *fname*: :class:`str`
            Name of packaged template file
    :Outputs:
        *fabs*: :class:`str`
            Full path to template file
    """
    # Get path to main ``cape`` package
    import cape
    cape_dir = os.path.dirname(cape.__file__)
    # Full path to template
    return os.path.join(cape_dir, solver, "templates", fname)


# Default settings file name for a solver
def get_default_json_name(solver: str) -> str:
    r"""Get default settings file name for a solver

    :Call:
        >>> fjson = get_default_json_name(solver)
    :Inputs:
        *solver*: :class:`str`
            Name of CAPE solver module, e.g. ``"pycart"``
    :Outputs:
        *fjson*: :class:`str`
            Default settings file name, e.g. ``"pyCart.json"``
    :Raises:
        :class:`CapeValueError`
            If *solver* is not a known CAPE solver module
    """
    # Import the solver's :mod:`cntl` module
    try:
        cntlmod = importlib.import_module(f"cape.{solver}.cntl")
    except ModuleNotFoundError:
        raise CapeValueError(f"No CAPE solver '{solver}'")
    # Use the class attribute
    return cntlmod.Cntl._fjson_default


# Instantiante bare options for a solver
def read_solver_options(solver: str) -> OptionsDict:
    r"""Instantiate bare options for a solver

    :Call:
        >>> opts = read_solver_options(solver)
    :Inputs:
        *solver*: :class:`str`
            Name of CAPE solver module, e.g. ``"pycart"``
    :Outputs:
        *opts*: :class:`cape.optdict.OptionsDict`
            Options interface with default settings
    :Raises:
        :class:`CapeValueError`
            If *solver* is not a known CAPE solver module
    """
    # Import the solver's :mod:`options` module
    try:
        optmod = importlib.import_module(f"cape.{solver}.options")
    except ModuleNotFoundError:
        raise CapeValueError(f"No CAPE solver '{solver}'")
    # Instantiate bare options
    return optmod.Options()

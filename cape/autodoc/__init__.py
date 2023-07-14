
# Standard library
import importlib
import os

# Local imports
from .optdict import optdoc


# Global settings
THIS_DIR = os.path.dirname(__file__)
CAPE_DIR = os.path.dirname(THIS_DIR)
REPO_DIR = os.path.dirname(CAPE_DIR)
DOC_DIR = os.path.join(REPO_DIR, "doc")

# Destination folders for each module
OPT_DIRS = {
    "cfdx": os.path.join(DOC_DIR, "common", "json"),
    "pycart": os.path.join(DOC_DIR, "pycart", "json"),
    "pyfun": os.path.join(DOC_DIR, "pyover", "json"),
    "pyover": os.path.join(DOC_DIR, "pyover", "json"),
}


# Which modules to import
DOC_OPTS = {
    "cfdx": {
        "folder": OPT_DIRS["cfdx"],
        "file": "index",
        "module": "cape.cfdx.options",
        "class": "Options",
    },
}


# Main function
def main():
    # Loop through sections
    for sec, opts in DOC_OPTS.items():
        # Parse options
        modname = opts.get("module", sec)
        fdir = opts.get("folder", OPT_DIRS.get(sec, sec))
        fname = opts.get("file", sec)
        clsname = opts["class"]
        # Import the module
        mod = importlib.import_module(modname)
        # Get the class
        cls = getattr(mod, clsname)
        # Form absolute file name
        frst = os.path.join(fdir, fname + ".rst")
        # Write files
        optdoc.write_rst(cls, frst, recurse=True)


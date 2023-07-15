
# Standard library
import os

# Local imports
from ..argread import readkeys
from ..optdict import optdoc


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
        "recurse": False,
        "recurse_sec_clsmap": True,
    },
    "Report-Subfigures": {
        "narrow": True,
    },
}


# Main function
def main():
    # Parse options
    _, kw = readkeys()
    # Loop through sections
    for sec in ["cfdx"]:
        # Write files
        optdoc.make_rst(DOC_OPTS, sec, **kw)


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import json
import os
import sys

# Third-party
import setuptools

# Local modules
from .. import argread
from .. import text as textutils
from ..tnakit import promptutils


# Docstring for CLI
HELP_QUICKSTART = r"""
--------------------------------------------------------------
``dkit-quickstart``: Create template for a new datakit package
--------------------------------------------------------------

Create several files and folders to create a basic template for a
DataKit 3.0 package.

:Usage:
    .. code-block:: console

        $ dkit-quickstart [PKG [WHERE]] [OPTIONS]

:Arguments:
    * *PKG*: name of Python package relative to *WHERE*
    * *WHERE*: location to create package

:Options:

    -h, --help
        Display this help message and quit

    -t, --target TARGET
        Use *TARGET* as a prefix to the package name *PKG*

    --where WHERE
        Create package in folder *WHERE* {.}

    --title TITLE
        Use *TITLE* as the one-line description for the package

:Versions:
    * 2021-08-24 ``@ddalle``: Version 1.0
"""


# Defaults
DEFAULT_TITLE = "DataKit quickstart package"

# Template docstring for a module
DEFAULT_DOCSTRING = r"""
%(hline_after-)s
%(mod)s: %(meta_title)s
%(hline-)s

Module metadata:

    %(meta)s

The following code block shows how to import the module and access the
primary database object(s).

    .. code-block:: python
        
        # Import module
        import %(pymod)s as dat

        # Get interface to processed trajectory data
        db = dat.read_db()

"""

# Template setup.py file
SETUP_PY = r"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard library
import os

# Third-part modules
import setuptools

# Cape modules
from cape.tnakit.metautils import ModuleMetadata


# Create the build
def main():
    # Find packages
    pkgs = setuptools.find_packages()
    # Main package
    pkg = pkgs[0]
    # Read metadata
    try:
        meta = ModuleMetadata(pkg)
    except ValueError:
        meta = {}
    # Get title
    title = meta.get("title", pkg)
    # Create the egg/wheel
    setuptools.setup(
        name=pkg,
        packages=pkgs,
        package_data={
            pkg: [
                "meta.json",
                "db/mat/ATT-VM-CLVTOPS-003-1002.mat",
                "db/csv/ATT-VM-CLVTOPS-003-1002.csv"
             ],
        },
        description=title,
        version="1.0")


# Compile
if __name__ == "__main__":
    main()

"""


# Main function
def main():
    r"""Main command-line interface function

    :Call:
        >>> main()
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Process command-line arguments
    a, kw = argread.readkeys(sys.argv)
    # Real main function
    quickstart(*a, **kw)


# Primary API function
def quickstart(*a, **kw):
    r"""Create templates for a new datakit module

    :Call:
        >>> quickstart(*a, **kw)
        >>> quickstart(pkg[, where], **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of Python package relative to *where*
        *where*: {``"."``} | :class:`str`
            Path from which to begin
        *t*, *target*: {``None``} | :class:`str`
            Optional subdir of *where* to put package in
        *title*: {``None``} | :class:`str`
            Title to use for this package (not module name)
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Check for help flag
    if kw.get('h') or kw.get('help'):
        # Display help message and quit
        print(textutils.markdown(HELP_QUICKSTART))
        return
    # Process positional args
    if len(a) > 2:
        print("Too many inputs!")
        print(textutils.markdown(HELP_QUICKSTART))
        return
    # Process "where"
    if len(a) > 1:
        # Get the "where" from *kw*, but default to second arg
        where = kw.pop("where", a[1])
    else:
        # Get "where", and default to current $PWD
        where = kw.pop("where", ".")
    # Process package name
    if len(a) > 0:
        # Get the package
        pkg = a[0]
    else:
        # Look for an existing package folder
        pkgs = setuptools.find_packages(where)
        # Get a default
        if len(pkgs) > 0:
            # Default to the first package found
            pkg = pkgs[0]
        else:
            # No default
            pkg = None
        # Prompt for a package
        pkg = promptutils.prompt("Python package name", vdef=pkg)
    # Path to default settings JSON file
    fjson = os.path.join(os.path.realpath(where), "datakit.json")
    # Load default settings
    if os.path.isfile(fjson):
        # Read options from it
        with open(fjson) as f:
            opts = json.load(f)
    else:
        # No defaults
        opts = {}
    # Set default target
    kw.setdefault("target", opts.get("target"))
    # Create folder
    create_pkgdir(pkg, where, **kw)
    # Prompt for a title
    kw["title"] = _prompt_title(**kw)
    # Write metadata
    create_metadata(pkg, opts, where, **kw)
    # Create the package files
    create_pkg(pkg, opts, where, **kw)
    # Write the vendorize.json file
    create_vendorize_json(pkg, opts, where, **kw)


# Ensure folder
def create_pkgdir(pkg, where=".", **kw):
    r"""Create folder(s) for a package

    :Call:
        >>> create_pkgdir(pkg, where=".", **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of Python module/package to create
        *where*: {``"."``} | :class:`str`
            Path from which to begin
        *t*, *target*: {``None``} | :class:`str`
            Optional subdir of *where* to put package in
    :Examples:
        This would create the folder ``c008/f3d/db001/`` if needed:
        
            >>> create_pkgdir("c008.f3d.db001")

        This would create the folder ``att_vm_clvtops3/db001":

            >>> create_pkgdir("db001", t="att_vm_clvtops3")

    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get absolute path to target
    basepath = expand_target(where)
    # Get relative folder
    path = get_pkgdir(pkg, **kw)
    # Create folders as needed
    mkdirs(basepath, path)


# Ensure folder
def create_pkg(pkg, opts, where=".", **kw):
    r"""Create ``__init__.py`` files in package folders if needed

    :Call:
        >>> create_pkg(pkg, where=".", **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of Python module/package to create
        *opts*: :class:`dict`
            Settings from ``datakit.json`` if available
        *where*: {``"."``} | :class:`str`
            Path from which to begin
        *t*, *target*: {``None``} | :class:`str`
            Optional subdir of *where* to put package in
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get absolute path to target
    basepath = expand_target(where)
    # Get relative folder
    path = get_pkgdir(pkg, **kw)
    # Full path to package
    pkgdir = os.path.join(basepath, path)
    # Loop through parts
    for fdir in path.split(os.sep)[:-1]:
        # Append to basepath
        basepath = os.path.join(basepath, fdir)
        # Name of Python file
        fpy = os.path.join(basepath, "__init__.py")
        # Check if file exists
        if os.path.isfile(fpy):
            continue
        # Otherwise create empty file
        with open(fpy, "w") as f:
            f.write("#!%s\n" % sys.executable)
            f.write("# -*- coding: utf-8 -*-\n\n")
    # Write template for the main Python file
    write_init_py(pkgdir, opts)
    # Write template for setup.py
    write_setup_py(os.path.dirname(pkgdir), opts)


# Create the metadata
def create_metadata(pkg, opts, where=".", **kw):
    r"""Write ``meta.json`` template in package folder

    :Call:
        >>> create_metadata(pkg, opts, where=".", **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of Python module/package to create
        *opts*: :class:`dict`
            Use metadata from ``opts["meta"]`` if able
        *title*: {``None``} | :class:`str`
            Title to use for this package (not module name)
        *where*: {``"."``} | :class:`str`
            Path from which to begin
        *t*, *target*: {``None``} | :class:`str`
            Optional subdir of *where* to put package in
        *meta.key*: :class:`str`
            Save this value as *key* in ``meta.json``
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get the path to the package
    basepath = expand_target(where)
    pkgpath = get_pkgdir(pkg, **kw)
    # Absolute path
    pkgdir = os.path.join(basepath, pkgpath)
    # Path to metadata file
    fjson = os.path.join(pkgdir, "meta.json")
    # Check if file exists
    if os.path.isfile(fjson):
        return
    # Get title
    title = kw.get("title", DEFAULT_TITLE)
    # Initialize metadata
    metadata = {
        "title": title,
    }
    # Get "meta" section from *opts*
    if isinstance(opts, dict):
        # Get "meta" section
        opts_meta = opts.get("meta", {})
        # Merge with "metadata" if able
        if isinstance(opts_meta, dict):
            metadata.update(opts_meta)
    # Add any other metadata
    for k in kw:
        # Check if it's a metadata key
        if k.startswith("meta."):
            # Normalized metadata key name
            col = k[5:].replace("_", "-")
            # Save value as shortened key
            metadata[col] = kw[k]
    # Write the JSON file
    with open(fjson, "w") as f:
        json.dump(metadata, f, indent=4)


# Write vendorization template
def create_vendorize_json(pkg, opts, where=".", **kw):
    r"""Write ``vendorize.json`` template in package folder

    :Call:
        >>> create_vendorize_json(pkg, opts, where=".", **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of Python module/package to create
        *opts*: :class:`dict`
            Use settings from ``opts["vendor"]`` if able
        *where*: {``"."``} | :class:`str`
            Path from which to begin
        *t*, *target*: {``None``} | :class:`str`
            Optional subdir of *where* to put package in
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get "vendor" section from *opts*
    if not isinstance(opts, dict):
        return
    # Get "vendor" section
    opts_vendor = opts.get("vendor")
    # Merge with "metadata" if able
    if not isinstance(opts_vendor, dict):
        return
    # Get the path to the package
    basepath = expand_target(where)
    pkgpath = get_pkgdir(pkg, **kw)
    # Absolute path
    pkgdir = os.path.join(basepath, pkgpath)
    # Vendorize in parent folder
    setupdir = os.path.dirname(pkgdir)
    # Path to vendorize file
    fjson = os.path.join(setupdir, "vendorize.json")
    # Check if file exists
    if os.path.isfile(fjson):
        return
    # Set default "target" in vendorize.json
    vendor = dict(target="%s/_vendor" % pkg.split(".")[-1])
    # Merge settings from *opts_vendor*
    vendor.update(opts_vendor)
    # Write the JSON file
    with open(fjson, "w") as f:
        json.dump(vendor, f, indent=4)


# Write the starter for a module
def write_init_py(pkgdir, opts):
    r"""Create ``__init__.py`` template for DataKit package

    :Call:
        >>> write_init_py(pkgdir, opts)
    :Inputs:
        *pkgdir*: :class:`str`
            Folder in which to create ``__init__.py`` file
        *opts*: :class:`dict`
            Settings from ``datakit.json`` if available
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Path to Python file
    fpy = os.path.join(pkgdir, "__init__.py")
    # Check if it exists
    if os.path.isfile(fpy):
        return
    # Check for a datakitloader module
    if isinstance(opts, dict):
        # Try to get module with database name settings
        dklmod = opts.get("datakitloader-module")
        # Check for vendorize settings
        vendor = opts.get("vendor", {})
        # Check for vendorized packages
        vendor_pkgs = vendor.get("packages", [])
    else:
        # No special settings
        dklmod = None
        vendor_pkgs = None
    # Create the file
    with open(fpy, "w") as f:
        # Write header info
        f.write("#!%s\n" % sys.executable)
        f.write("# -*- coding: utf-8 -*-\n")
        # Write template docstring
        f.write('r"""\n')
        f.write(DEFAULT_DOCSTRING)
        f.write('"""\n\n')
        # Write import
        f.write("# Standard library modules\n\n\n")
        f.write("# Third-party modules\n\n\n")
        f.write("# CAPE modules\n")
        f.write("import cape.attdb.datakitloader as dkloader\n")
        f.write("import cape.tnakit.modutils as modutils\n\n")
        f.write("# Local modules\n")
        # Check for vendorized packages
        for pkg in vendor_pkgs:
            f.write("from ._vendor imort %s\n" % pkg)
        f.write("\n")
        # Automatic docstring update
        f.write("# Update docstring\n")
        f.write("__doc__ = modutils.rst_docstring(")
        f.write("__name__, __file__, __doc__)\n\n")
        # Requirements template
        f.write("# Supporting modules\n")
        f.write("REQUIREMENTS = [\n]\n\n")
        # Template DataKitLoader
        f.write("# Get datakit loader settings\n")
        f.write("DATAKIT_LOADER = dkloader.DataKitLoader(__name__, __file__")
        # Check for existing settings
        if dklmod:
            # Use settings from specified modules
            f.write(", **%s.DB_NAMES)\n\n" % dklmod)
        else:
            # No expanded settings
            f.write(")\n\n")


# Write the setup.py script
def write_setup_py(pkgdir, opts):
    r"""Create ``__init__.py`` template for DataKit package

    :Call:
        >>> write_setup_py(pkgdir, opts)
    :Inputs:
        *pkgdir*: :class:`str`
            Folder in which to create ``__init__.py`` file
        *opts*: :class:`dict`
            Settings from ``datakit.json`` if available
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Path to Python file
    fpy = os.path.join(pkgdir, "setup.py")
    # Check if it exists
    if os.path.isfile(fpy):
        return
    # Create the file
    with open(fpy, "w") as f:
        # Write the template
        f.write(SETUP_PY)


# Get relative path to package folder
def get_pkgdir(pkg, **kw):
    r"""Get relative folder to a package

    :Call:
        >>> pkgdir = get_pkgdir(pkg, **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of Python module/package to create
        *t*, *target*: {``None``} | :class:`str`
            Optional subdir of *where* to put package in
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get target if any
    target = kw.get("t", kw.get("target"))
    # Convert package name to folder (mypkg.mymod -> mypkg/mymod/)
    pkgdir = pkg.replace(".", os.sep)
    # Total path
    if target:
        # Combine two-part package path
        return os.path.join(target.replace("/", os.sep), pkgdir)
    else:
        # No target; just the package
        return pkgdir



# Create folders
def mkdirs(basepath, path):
    r"""Create one or more folders within fixed *basepath*

    :Call:
        >>> mkdirs(basepath, path)
    :Inputs:
        *basepath*: :class:`str`
            Absolute folder in which to start creating folders
        *path*: :class:`str`
            Path to folder to create relative to *basepath*
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Ensure *basepath* exists
    if not os.path.isdir(basepath):
        raise SystemError("basepath '%s' is not a folder" % basepath)
    # Loop through remaining folders
    for pathj in path.split(os.sep):
        # Append to path
        basepath = os.path.join(basepath, pathj)
        # Check if it exists
        if not os.path.isdir(basepath):
            # Create it
            os.mkdir(basepath)


# Expand "target"
def expand_target(target="."):
    r"""Expand a relative path

    :Call:
        >>> cwd = expand_target(target=".")
    :Inputs:
        *target*: {``"."``} | ``None`` | :class:`str`
            Relative or absolute path
    :Outputs:
        *cwd*: :class:`str`
            Absolute path
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get absolute path
    if target is None:
        # Use current path
        return os.getcwd()
    else:
        # Expand
        return os.path.realpath(target)


# Ensure a title is present
def _prompt_title(**kw):
    r"""Prompt for a title if none provided

    :Call:
        >>> title = _prompt_title(**kw)
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get existing title from kwargs
    title = kw.get("title", kw.get("Title"))
    # If non-empty, return it
    if title:
        return title
    # Otherwise prompt one
    title = promptutils.prompt("One-line title for package")
    # Check for an answer again
    if title is None:
        return DEFAULT_TILE
    else:
        return title


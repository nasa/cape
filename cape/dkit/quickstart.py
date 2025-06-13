#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import importlib
import json
import os
import sys

# Third-party

# Local modules
from .. import argread
from .. import textutils
from .datakitast import DataKitAssistant
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
    * 2021-08-24 ``@ddalle``: v1.0
    * 2021-09-15 ``@ddalle``: v1.1; more STDOUT
"""


# Defaults
DEFAULT_TITLE = "DataKit quickstart package"

# Template docstring for a module
DEFAULT_DOCSTRING = r"""
%(hline_after-)s
%(modname)s: %(meta_title)s
%(hline-)s

Package metadata:

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

# CAPE modules
from cape.dkit import pkgutils


# Create the build
def main():
    # Create the egg/wheel
    pkgutils.setup(
        name=None,
        packages=None,
        package_data=None,
        description=None,
        db=True,
        meta=True,
        rawdata=False,
        version="1.0")


# Compile
if __name__ == "__main__":
    main()

"""


# Main function
def main():
    # Process command-line arguments
    a, kw = argread.readkeys(sys.argv)
    # Real main function
    quickstart(*a, **kw)


main.__doc__ = HELP_QUICKSTART


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
        * 2021-08-24 ``@ddalle``: v1.0
        * 2021-10-21 ``@ddalle``: v1.1; improve ``setup.py``
        * 2021-10-22 ``@ddalle``: v1.2; auto suffix
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
    # Expand the package name
    pkg = get_full_pkgname(pkg, opts, where, **kw)
    # Remove *target* options
    kw.pop("target", None)
    kw.pop("t", None)
    # Set title if applicable
    if "title" in opts.get("meta", {}):
        kw["title"] = opts["meta"]["title"]
    # Prompt for a title
    kw["title"] = _prompt_title(**kw)
    # Create folder
    create_pkgdir(pkg, where, **kw)
    # Write metadata
    create_metadata(pkg, opts, where, **kw)
    # Create the package files
    create_pkg(pkg, opts, where, **kw)
    # Write the vendorize.json file
    create_vendorize_json(pkg, opts, where, **kw)


# Ensure folder
def create_pkgdir(pkg: str, where: str = ".", **kw):
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

        This would create the folder ``att_vm_clvtops3/db001``:

            >>> create_pkgdir("db001", t="att_vm_clvtops3")

    :Versions:
        * 2021-08-24 ``@ddalle``: v1.0
    """
    # Get absolute path to target
    basepath = expand_target(where)
    # Get relative folder
    path = get_pkgdir(pkg, **kw)
    # Create folders as needed
    mkdirs(basepath, path)


# Ensure folder
def create_pkg(pkg: str, opts: dict, where: str = ".", **kw):
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
        * 2021-08-24 ``@ddalle``: v1.0
    """
    # Get absolute path to target
    basepath = expand_target(where)
    basepath0 = basepath
    # Get relative folder
    path = get_pkgdir(pkg, **kw)
    # Full path to package
    pkgdir = os.path.join(basepath, path)
    # Split package name into parts
    pkgparts = path.split(os.sep)
    # Last part is the name of the stand-alone package
    pkgname = pkgparts.pop()
    # Number of parts
    npart = len(pkgparts)
    # Loop through parts
    for j, fdir in enumerate(pkgparts):
        # Append to basepath
        basepath = os.path.join(basepath, fdir)
        # Name of Python file
        fpy = os.path.join(basepath, "__init__.py")
        # Check if file exists
        if os.path.isfile(fpy):
            continue
        # Status update
        print("Writing file '%s'" % os.path.relpath(fpy, basepath0))
        # Otherwise create empty file
        with open(fpy, "w") as f:
            # Write import statement for final level
            if j + 1 == npart:
                f.write("\nfrom .%s import read_db\n" % pkgname)
                f.write("from .%s import __doc__\n\n" % pkgname)
    # Write template for the main Python file
    write_init_py(pkgdir, opts, where=where)
    # Write template for setup.py
    write_setup_py(os.path.dirname(pkgdir), opts, where=where)


# Create the metadata
def create_metadata(pkg: str, opts: dict, where: str = ".", **kw):
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
        * 2021-08-24 ``@ddalle``: v1.0
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
    # Status update
    print("Writing file '%s'" % os.path.relpath(fjson, basepath))
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
def create_vendorize_json(pkg: str, opts: dict, where: str = ".", **kw):
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
        * 2021-08-24 ``@ddalle``: v1.0
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
    # Status update
    print("Writing file '%s'" % os.path.relpath(fjson, basepath))
    # Set default "target" in vendorize.json
    vendor = dict(target="%s/_vendor" % pkg.split(".")[-1])
    # Merge settings from *opts_vendor*
    vendor.update(opts_vendor)
    # Write the JSON file
    with open(fjson, "w") as f:
        json.dump(vendor, f, indent=4)


# Write the starter for a module
def write_init_py(pkgdir, opts, where="."):
    r"""Create ``__init__.py`` template for DataKit package

    :Call:
        >>> write_init_py(pkgdir, opts, basepath)
    :Inputs:
        *pkgdir*: :class:`str`
            Folder in which to create ``__init__.py`` file
        *opts*: :class:`dict`
            Settings from ``datakit.json`` if available
    :Versions:
        * 2021-08-24 ``@ddalle``: v1.0
    """
    # Path to Python file
    fpy = os.path.join(pkgdir, "__init__.py")
    # Check if it exists
    if os.path.isfile(fpy):
        return
    # Get root for status message
    basepath = expand_target(where)
    # Status update
    print("Writing file '%s'" % os.path.relpath(fpy, basepath))
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
        f.write('r"""')
        f.write(DEFAULT_DOCSTRING)
        f.write('"""\n\n')
        # Write import
        f.write("# Standard library modules\n")
        f.write("from typing import Optional\n\n")
        f.write("# Third-party modules\n\n\n")
        f.write("# CAPE modules\n")
        f.write("from cape.dkit.rdb import DataKit\n")
        f.write("from cape.dkit.datakitast import DataKitAssistant\n")
        f.write("from cape.dkit import modutils\n\n")
        f.write("# Local modules\n")
        # Check for vendorized packages
        for pkg in vendor_pkgs:
            f.write("from ._vendor import %s\n" % pkg)
        f.write("\n\n")
        # Automatic docstring update
        f.write("# Update docstring\n")
        f.write("__doc__ = modutils.rst_docstring(")
        f.write("__name__, __file__, __doc__)\n\n")
        # Requirements template
        f.write("# Supporting modules\n")
        f.write("REQUIREMENTS = [\n]\n\n")
        # Template DataKitLoader
        f.write("# Get datakit loader settings\n")
        f.write("AST = DataKitAssistant(\n")
        f.write("    __name__, __file__")
        # Check for existing settings
        if dklmod:
            # Use settings from specified modules
            f.write(", **%s.DB_NAMES)\n\n\n" % dklmod)
        else:
            # No expanded settings
            f.write(")\n\n\n")
        # Write reader stubs
        f.write("# Read datakit from MAT file\n")
        f.write("def read_db_mat() -> DataKit:\n")
        f.write("    return AST.read_db_mat()\n\n\n")
        f.write("# Read best datakit\n")
        f.write("def read_db() -> Optional[DataKit]:\n")
        f.write("    try:\n        return read_db_mat()\n")
        f.write("    except Exception:\n        pass\n\n\n")
        # Last reader stub
        f.write("# Read source data\n")
        f.write("def read_db_source() -> DataKit:\n")
        f.write("    pass\n\n\n")
        # Writer stub
        f.write("# Write datakit\n")
        f.write("def write_db(f: bool = True, **kw):\n")
        f.write("    AST.write_db_mat(f=f)")
        f.write("\n\n")


# Write the setup.py script
def write_setup_py(pkgdir, opts, where="."):
    r"""Create ``__init__.py`` template for DataKit package

    :Call:
        >>> write_setup_py(pkgdir, opts)
    :Inputs:
        *pkgdir*: :class:`str`
            Folder in which to create ``__init__.py`` file
        *opts*: :class:`dict`
            Settings from ``datakit.json`` if available
    :Versions:
        * 2021-08-24 ``@ddalle``: v1.0
    """
    # Path to Python file
    fpy = os.path.join(pkgdir, "setup.py")
    # Check if it exists
    if os.path.isfile(fpy):
        return
    # Get root for status message
    basepath = expand_target(where)
    # Status update
    print("Writing file '%s'" % os.path.relpath(fpy, basepath))
    # Create the file
    with open(fpy, "w") as f:
        # Write the template
        f.write(SETUP_PY)


# Create maximum-length package name
def get_full_pkgname(pkg: str, opts: dict, where: str = ".", **kw) -> str:
    r"""Expand shortened package name

    For example, this may expand

        * ``"c007.f3d.db201"`` to
        * ``"sls28sfb.c007.f3d.db201.sls28sfb_c007_f3d_201"``

    if the *target* is ``"sls28sfb"``. This prevents the user from
    needing to type the base module name, which doesn't change, and the
    final portion, which can be determined from the preceding portion.

    :Call:
        >>> fullpkg = get_full_pkgname(pkg, opts, where, **kw)
    :Inputs:
        *pkg*: :class:`str`
            Package name, possibly abbreviated
        *opts*: ``None`` | :class:`dict`
            Options read from ``datakit.json``
        *where*: {``"."``} | ``None`` | :class:`str`
            Base folder in which *fullpkg* resides
        *t*, *target*: {``None``} | :class:`str`
            Prefix from command line
    :Outputs:
        *fullpkg*: :class:`str`
            Full package name with prefix and suffix if needed
    :Versions:
        * 2021-10-22 ``@ddalle``: v1.0
    """
    # Prepend *target* if needed
    pkg = expand_pkg1(pkg, opts, **kw)
    # Read a *DataKitLoader* to get full names
    dkl = read_datakitloader(pkg, opts, where)
    breakpoint()
    # See if we can get full list of candidate module names
    if dkl is None:
        # No candidates
        return pkg
    else:
        # Generate list of candidates
        modnames = dkl.genr8_modnames()
        # Initialize final candidate
        modname = pkg
        # Loop through them
        for name in modnames:
            # Check if it starts with *pkg*
            if name.startswith(pkg):
                # Check if it's longer than the current one
                if len(name) > len(modname):
                    modname = name
        # Use longest correct module name
        return modname


# Read datakitloader
def read_datakitloader(pkg, opts, where="."):
    r"""Read :class:`DataKitLoader` to assist with module/db names

    :Call:
        >>> dkl = read_datakitloader(pkg, opts, where=".")
    :Inputs:
        *pkg*: :class:`str`
            Module name to quickstart, possibly shortened
        *opts*: ``None`` | :class:`dict`
            Options read from ``datakit.json``
        *where*: ``"."`` | :class:`str`
            Base folder from which *pkg* is imported
    :Outputs:
        *dkl*: ``None`` | :class:`DataKitLoader`
            Datakit reader and db/module name interchanger
    :Versions:
        * 2021-10-22 ``@ddalle``: v1.0
    """
    # Check for settings
    if not isinstance(opts, dict):
        return
    # Get module name
    modname = opts.get("datakitloader-module")
    # Check if one was found
    if modname is None:
        return
    # Attempt to load it
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return
    # Check for absolute path
    if where is None:
        # Let DataKitLoader figure it out from current path
        fdir = None
    else:
        # Absolutize from "where"
        fdir = os.path.abspath(os.path.join(where, pkg.replace(".", os.sep)))
    # Create DataKitLoader
    try:
        dkl = DataKitAssistant(pkg, fdir, **mod.DB_NAMES)
        return dkl
    except Exception:
        raise


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
        * 2021-08-24 ``@ddalle``: v1.0
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
def mkdirs(basepath: str, path: str):
    r"""Create one or more folders within fixed *basepath*

    :Call:
        >>> mkdirs(basepath, path)
    :Inputs:
        *basepath*: :class:`str`
            Absolute folder in which to start creating folders
        *path*: :class:`str`
            Path to folder to create relative to *basepath*
    :Versions:
        * 2021-08-24 ``@ddalle``: v1.0
        * 2025-06-12 ``@ddalle``: v1.1; make rawdata/ folder
    """
    # Ensure *basepath* exists
    if not os.path.isdir(basepath):
        raise SystemError("basepath '%s' is not a folder" % basepath)
    # Save original base for status updates
    basepath0 = basepath
    # Add "rawdata"
    rpath = os.path.join(path, "rawdata")
    # Loop through remaining folders
    for pathj in rpath.split(os.sep):
        # Append to path
        basepath = os.path.join(basepath, pathj)
        # Check if it exists
        if not os.path.isdir(basepath):
            # Status update
            print(
                "Creating folder '%s%s'"
                % (os.path.relpath(basepath, basepath0), os.sep))
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
        * 2021-08-24 ``@ddalle``: v1.0
    """
    # Get absolute path
    if target is None:
        # Use current path
        return os.getcwd()
    else:
        # Expand
        return os.path.realpath(target)


# Apply the "target" to a package name
def expand_pkg1(pkg, opts=None, **kw):
    r"""Apply *target* prefix if appropriate

    :Call:
        >>> pkg1 = expand_pkg1(pkg, opts=None, **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of package, possibly w/o prefix
        *opts*: {``None``} | :class:`dict`
            Options read from ``datakit.json``
        *t*, *target*: {``None``} | :class:`str`
            Target read from command line or kwargs
    :Outputs:
        *pkg1*: :class:`str`
            *pkg* prepended with *target* if necessary
    :Versions:
        * 2021-10-22 ``@ddalle``: v1.0
    """
    # Get target from *opts*
    if isinstance(opts, dict):
        # Check for it from valid *opts*
        opts_target = opts.get("target")
    else:
        # No JSON *target*
        opts_target = None
    # Check for *kw*
    kw_target = kw.get("target", kw.get("t"))
    # Pick between *kw* and *opts*
    if kw_target is None:
        # Use JSON value
        target = opts_target
    else:
        # Use *kw*
        target = kw_target
    # Check if *pkg* starts with *target*
    if target is None or pkg.startswith(target):
        # Already good
        return pkg
    else:
        # Prepend *pkg1*
        return f"{target}.{pkg}"


# Ensure a title is present
def _prompt_title(**kw):
    r"""Prompt for a title if none provided

    :Call:
        >>> title = _prompt_title(**kw)
    :Versions:
        * 2021-08-24 ``@ddalle``: v1.0
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
        return DEFAULT_TITLE
    else:
        return title


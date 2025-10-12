#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import importlib
import json
import os
import shutil
import sys
from typing import Optional

# Third-party

# Local modules
from .. import argread
from .. import textutils
from .datakitloader import DataKitLoader, DataKitLoaderOptions
from .metautils import ModulePropDB, merge_dict
from ..optdict import OptionsDict
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
        * 2025-06-13 ``@ddalle``: v2.0; use class
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
    # Create instance
    starter = DataKitQuickStarter(*a, **kw)
    # Call main method
    return starter.quickstart()


# Options for vendorize section
class VendorOptions(OptionsDict):
    r"""Options for ``dkit-vendorize`` definitions"""
    # No attributes
    __slots__ = ()

    # Allowed options
    _optlist = (
        "hub",
        "packages",
        "target",
    )

    # Aliases
    _optmap = {
        "pkgs": "packages",
    }

    # Types
    _opttypes = {
        "hub": str,
        "packages": str,
        "target": str,
    }

    # List options
    _optlistdepth = {
        "hub": 1,
        "packages": 1,
    }


# Options for "datakit.json"
class QuickStartOptions(OptionsDict):
    r"""Options class for ``dkit-quickstart``

    :Call:
        >>> opts = QuickStartOptions(fname=None, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of JSON/YAML file to read
        *kw*: :class:`dict`
            Additional options to parse
    :Outputs:
        *opts*: :class:`QuickStartOptions`
            Options for ``dkit-quickstart``
    """
    # No attributes
    __slots__ = ()

    # Allowed options
    _optlist = (
        "datakitloader",
        "datakitloader-module",
        "meta",
        "requirements",
        "target",
        "template",
        "title",
        "vendor",
    )

    # Aliases
    _optmap = {
        "r": "requirements",
        "reqs": "requirements",
        "t": "target",
    }

    # Types
    _opttypes = {
        "datakitloader-module": str,
        "meta": dict,
        "requirements": str,
        "target": str,
        "title": str,
        "template": str,
    }

    # Lists
    _optlistdepth = {
        "requirements": 1,
    }

    # Subsections
    _sec_cls = {
        "datakitloader": DataKitLoaderOptions,
        "vendor": VendorOptions,
    }


# Class to run dkit-quickstart
class DataKitQuickStarter:
    r"""Class to enable ``dkit-quickstart``

    :Call:
        >>> starter = DataKitQuickStarter(*a, **kw)
    :Outputs:
        *starter*: :class:`DataKitQuickStarter`
            Utility to create new DataKit packages
    """
    __slots__ = (
        "cwd",
        "opts",
        "pkg_input",
        "pkgdir",
        "pkgname",
    )

    def __init__(self, *a, **kw):
        # Process "where"
        if len(a) > 1:
            # Get the "where" from *kw*, but default to second arg
            where = kw.pop("where", a[1])
        else:
            # Get "where", and default to current $PWD
            where = kw.pop("where", ".")
        # Save location
        self.cwd = os.path.realpath(where)
        # Process package name
        if len(a) > 0:
            # Get the package
            pkg = a[0]
        else:
            # Prompt for a package
            pkg = promptutils.prompt("Python package name", vdef=pkg)
        # Save package name
        self.pkg_input = pkg
        # Clean kwargs
        kw.pop("__replaced__", None)
        # Initialize output
        kw_meta = {}
        # Loop through all kwargs
        for k, v in kw.items():
            # Check for "meta" prefix
            if not k.startswith("meta."):
                continue
            # Get option name
            opt = k.split('.', 1)[1]
            # Save it
            kw_meta[opt] = v
            # Delete it from kwargs
            kw.pop(k)
        # Path to default settings JSON file
        fjson = os.path.join(os.path.realpath(where), "datakit.json")
        # Load default settings
        self.opts = QuickStartOptions(fjson, **kw)
        # Add in kwarg metadata
        self.opts["meta"].update(kw_meta)
        # Prompt for a title
        self.prompt_title()
        # Expand the package name
        self.pkgname = self.get_full_pkgname()
        self.pkgdir = self.get_pkgdir()

    def quickstart(self,) -> int:
        r"""Create new datakits

        :Call:
            >>> ierr = starter.quickstart(**kw)
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions*:
            * 2021-08-24 ``@ddalle``: v1.0
            * 2021-10-21 ``@ddalle``: v1.1; improve ``setup.py``
            * 2021-10-22 ``@ddalle``: v1.2; auto suffix
            * 2025-06-13 ``@ddalle``: v2.0; instance method
        """
        # Create folder
        self.create_pkgdir()
        # Write metadata
        self.create_metadata()
        # Create the package files
        self.create_pkg()
        # Write the vendorize.json file
        self.create_vendorize_json()
        # Write requirements
        self.write_requirements()
        # Prepare rawdata/ folder
        self.prepare_rawdata()
        # Return code
        return 0

    # Ensure folder
    def create_pkgdir(self):
        r"""Create folder(s) for a package

        :Call:
            >>> starter.create_pkgdir(**kw)
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
            *t*, *target*: {``None``} | :class:`str`
                Optional subdir of *where* to put package in
        :Examples:
            This would create the folder ``c008/f3d/db001/`` if needed:

                >>> create_pkgdir("c008.f3d.db001")

            This would create the folder ``att_vm_clvtops3/db001``:

                >>> create_pkgdir("db001", t="att_vm_clvtops3")

        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
            * 2025-06-13 ``@ddalle``: v2.0; instance method
        """
        # Create folders as needed
        mkdirs(self.cwd, self.pkgdir)

    # Ensure folder
    def create_pkg(self, **kw):
        r"""Create ``__init__.py`` files in package folders if needed

        :Call:
            >>> starter.create_pkg(**kw)
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
            *t*, *target*: {``None``} | :class:`str`
                Optional subdir of *where* to put package in
        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
            * 2025-06-13 ``@ddalle``: v2.0; instance method
        """
        # Get absolute path to target
        basepath = self.cwd
        basepath0 = basepath
        # Get relative folder
        path = self.pkgdir
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
        self.write_init_py()
        # Write template for setup.py
        self.write_setup_py()

    # Create the metadata
    def create_metadata(self, **kw):
        r"""Write ``meta.json`` template in package folder

        :Call:
            >>> starter.create_metadata(**kw)
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
            *t*, *target*: {``None``} | :class:`str`
                Optional subdir of *where* to put package in
            *meta.{key}*: :class:`str`
                Save this value as *key* in ``meta.json``
        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
            * 2025-06-13 ``@ddalle``: v2.0; instance method
            * 2025-06-18 ``@ddalle``: v2.1; use template
        """
        # Get the path to the package
        basepath = self.cwd
        pkgpath = self.get_pkgdir(**kw)
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
        title = self.opts.get("title", DEFAULT_TITLE)
        # Initialize metadata
        metadata = kw.get("meta", {})
        metadata.setdefault("title", title)
        # Get template
        template = self.opts.get_opt("template")
        # Get "meta" section from *opts*
        if isinstance(self.opts, dict):
            # Get "meta" section
            opts_meta = self.opts.get("meta", {})
            # Merge with "metadata" if able
            if isinstance(opts_meta, dict):
                merge_dict(metadata, opts_meta)
        # Exit if no template
        if template is not None:
            # Get full name
            pkg = self.get_full_pkgname(template)
            # Get path to that folder
            pkgdir = pkg.replace('.', os.sep)
            # Absolute path to template file
            absdir = os.path.join(self.cwd, pkgdir)
            srcfile = os.path.join(absdir, "meta.json")
            # Read that
            if os.path.isfile(srcfile):
                # Read the template's options
                template_opts = ModulePropDB(srcfile)
                # Combine
                merge_dict(metadata, template_opts)
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
    def create_vendorize_json(self):
        r"""Write ``vendorize.json`` template in package folder

        :Call:
            >>> starter.create_vendorize_json(**kw)
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
            *t*, *target*: {``None``} | :class:`str`
                Optional subdir of *where* to put package in
        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
            * 2025-06-18 ``@ddalle``: v2.0; use template
        """
        # Get "vendor" section
        opts_vendor = self.opts.get("vendor")
        # Merge with "metadata" if able
        if not isinstance(opts_vendor, dict):
            return
        # Get the path to the package
        basepath = self.cwd
        pkgpath = self.pkgdir
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
        vendor = dict(target="%s/_vendor" % self.pkgname.split(".")[-1])
        # Merge settings from *opts_vendor*
        vendor.update(opts_vendor)
        pkglist = vendor.setdefault("packages", [])
        # Get template
        template = self.opts.get_opt("template")
        # Exit if no template
        if template is not None:
            # Get full name
            pkg = self.get_full_pkgname(template)
            # Get path to that folder
            pkgdir = pkg.replace('.', os.sep)
            # Absolute path to template file
            absdir = os.path.join(self.cwd, pkgdir)
            srcfile = os.path.join(os.path.dirname(absdir), "vendorize.json")
            # Read that
            if os.path.isfile(srcfile):
                # Read the template's options
                template_opts = VendorOptions(srcfile)
                # Blend the list of packages
                for vendorpkg in template_opts.get_opt("packages", vdef=[]):
                    if vendorpkg not in pkglist:
                        pkglist.append(vendorpkg)
        # Write the JSON file
        with open(fjson, "w") as f:
            json.dump(vendor, f, indent=4)

    # Prepare rawdata/ folder
    def prepare_rawdata(self):
        r"""Make ``rawdata/`` folder and copy ``datakit-sources.json``

        :Call:
            >>> starter.prepare_rawdata()
        :Versions:
            * 2025-06-14 ``@ddalle``: v1.0
        """
        # Make rawdata/ folder
        mkdirs(os.path.join(self.cwd, self.pkgdir), "rawdata")
        # Get template
        template = self.opts.get_opt("template")
        # Exit if no template
        if template is None:
            return
        # Get full name
        pkg = self.get_full_pkgname(template)
        # Get path to that folder
        pkgdir = pkg.replace('.', os.sep)
        pkgdir = os.path.join(pkgdir, "rawdata")
        # Path to this rawdata/ folder
        rawdir = os.path.join(self.pkgdir, "rawdata")
        # Absolute path to template file
        srcfile = os.path.join(self.cwd, pkgdir, "datakit-sources.json")
        dstfile = os.path.join(rawdir, "datakit-sources.json")
        rawfile = os.path.join(self.cwd, rawdir, "datakit-sources.json")
        # Check if it exists
        if os.path.isfile(srcfile) and not os.path.isfile(rawfile):
            print(f"Copying '{dstfile}' from {template}")
            shutil.copy(srcfile, rawfile)

    # Write the starter for a module
    def write_init_py(self):
        r"""Create ``__init__.py`` template for DataKit package

        :Call:
            >>> starter.write_init_py()
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
            * 2025-06-13 ``@ddalle``: v2.0; allow templates
        """
        # Packeg folder
        pkgdir = os.path.join(self.cwd, self.pkgdir)
        # Path to Python file
        fpy = os.path.join(pkgdir, "__init__.py")
        # Check if it exists
        if os.path.isfile(fpy):
            return
        # Get root for status message
        basepath = self.cwd
        # Status update
        print("Writing file '%s'" % os.path.relpath(fpy, basepath))
        # Get template option
        template_mod = self.opts.get_opt("template")
        # Use template if appropriate
        if template_mod is None:
            # Write default
            self.write_init_py_default(fpy)
        else:
            # Write from template
            self.write_init_py_template(fpy, template_mod)

    # Write the standard new-datakit Python file
    def write_init_py_default(self, fpy: str):
        r"""Create ``__init__.py`` template for DataKit package

        :Call:
            >>> starter.write_init_py()
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
        """
        # Check for a datakitloader module
        # Try to get module with database name settings
        dklmod = self.opts.get("datakitloader-module")
        # Check for vendorize settings
        vendor = self.opts.get("vendor", {})
        # Check for vendorized packages
        vendor_pkgs = vendor.get("packages", [])
        # Create the file
        with open(fpy, "w") as fp:
            # Write template docstring
            fp.write('r"""')
            fp.write(DEFAULT_DOCSTRING)
            fp.write('"""\n\n')
            # Write import
            fp.write("# Standard library modules\n")
            fp.write("from typing import Optional\n\n")
            fp.write("# Third-party modules\n\n\n")
            fp.write("# CAPE modules\n")
            fp.write("from cape.dkit.rdb import DataKit\n")
            fp.write("from cape.dkit.datakitast import DataKitAssistant\n")
            fp.write("from cape.dkit import modutils\n\n")
            fp.write("# Local modules\n")
            # Check for vendorized packages
            for pkg in vendor_pkgs:
                fp.write("from ._vendor import %s\n" % pkg)
            fp.write("\n\n")
            # Automatic docstring update
            fp.write("# Update docstring\n")
            fp.write("__doc__ = modutils.rst_docstring(")
            fp.write("__name__, __file__, __doc__)\n\n")
            # Requirements template
            fp.write("# Supporting modules\n")
            fp.write("REQUIREMENTS = [\n]\n\n")
            # Template DataKitLoader
            fp.write("# Get datakit loader settings\n")
            fp.write("AST = DataKitAssistant(\n")
            fp.write("    __name__, __file__")
            # Check for existing settings
            if dklmod:
                # Use settings from specified modules
                fp.write(", **%s.DB_NAMES)\n\n\n" % dklmod)
            else:
                # No expanded settings
                fp.write(")\n\n\n")
            # Write reader stubs
            fp.write("# Read datakit from MAT file\n")
            fp.write("def read_db_mat() -> DataKit:\n")
            fp.write("    return AST.read_db_mat()\n\n\n")
            fp.write("# Read best datakit\n")
            fp.write("def read_db() -> Optional[DataKit]:\n")
            fp.write("    try:\n        return read_db_mat()\n")
            fp.write("    except Exception:\n        pass\n\n\n")
            # Last reader stub
            fp.write("# Read source data\n")
            fp.write("def read_db_source() -> DataKit:\n")
            fp.write("    pass\n\n\n")
            # Writer stub
            fp.write("# Write datakit\n")
            fp.write("def write_db(f: bool = True, **kw):\n")
            fp.write("    AST.write_db_mat(f=f)")
            fp.write("\n\n")

    # Write __init__.py from template
    def write_init_py_template(self, fpy: str, modname: str):
        r"""Create ``__init__.py`` template for DataKit package

        :Call:
            >>> starter.write_init_py()
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
            *fpy*: :class:`str`
                Full path to ``__init__.py`` file to write
            *modname*: :class:`str`
                Partial (or full) module name to use as template
        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
        """
        # Get full name
        pkg = self.get_full_pkgname(modname)
        # Get path to that folder
        pkgdir = pkg.replace('.', os.sep)
        # Absolute path to template file
        pyfile = os.path.join(self.cwd, pkgdir, "__init__.py")
        # Check for such a file
        if not os.path.isfile(pyfile):
            raise ValueError(
                f"Cannot use template {pkg}; __init__.py not found")
        # Status update
        print(f"  Using template '{pkgdir}/__init__.py'")
        # Read the file
        with open(fpy, "w") as fp:
            fp.write(open(pyfile).read())

    # Write __init__.py using default template
    def write_requirements(self):
        r"""Write ``requirements.json`` if *requirements* are given

        :Call:
            >>> starter.write_requirements()
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
        :Versions:
            * 2025-06-13 ``@ddalle``: v1.0
        """
        # Get requirements moption
        reqs = self.opts.get_opt("requirements")
        # Exit if none
        if reqs is None or len(reqs) == 0:
            return
        # Ensure list
        if isinstance(reqs, str):
            reqs = [req.strip() for req in reqs.split(',')]
        # Package folder
        pkgdir = os.path.join(self.cwd, self.pkgdir)
        # Write to file
        with open(os.path.join(pkgdir, "requirements.json"), 'w') as fp:
            json.dump(reqs, fp, indent=4)

    # Write the setup.py script
    def write_setup_py(self):
        r"""Create ``__init__.py`` template for DataKit package

        :Call:
            >>> starter.write_setup_py()
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
        """
        # Absolute path to package
        pkgdir = os.path.join(self.cwd, self.pkgdir)
        # Path to Python file
        fpy = os.path.join(pkgdir, "setup.py")
        # Check if it exists
        if os.path.isfile(fpy):
            return
        # Get root for status message
        basepath = self.cwd
        # Status update
        print("Writing file '%s'" % os.path.relpath(fpy, basepath))
        # Create the file
        with open(fpy, "w") as f:
            # Write the template
            f.write(SETUP_PY)

    # Create maximum-length package name
    def get_full_pkgname(self, rawpkg: Optional[str] = None) -> str:
        r"""Expand shortened package name

        For example, this may expand

            * ``"c007.f3d.db201"`` to
            * ``"sls28sfb.c007.f3d.db201.sls28sfb_c007_f3d_201"``

        if the *target* is ``"sls28sfb"``. This prevents the user from
        needing to type the base module name, which doesn't change, and the
        final portion, which can be determined from the preceding portion.

        :Call:
            >>> fullpkg = starter.get_full_pkgname(**kw)
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
        :Outputs:
            *fullpkg*: :class:`str`
                Full package name with prefix and suffix if needed
        :Versions:
            * 2021-10-22 ``@ddalle``: v1.0
        """
        # Prepend *target* if needed
        pkg = self.expand_pkg1(rawpkg)
        # Read a *DataKitLoader* to get full names
        dkl = self.read_datakitloader(pkg)
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
    def read_datakitloader(self, pkg: str):
        r"""Read :class:`DataKitLoader` to assist with module/db names

        :Call:
            >>> ast = read_datakitloader(pkg)
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
            *pkg*: :class:`str`
                Module name to quickstart, possibly shortened
        :Outputs:
            *dkl*: ``None`` | :class:`DataKitAssistant`
                Datakit reader and db/module name interchanger
        :Versions:
            * 2021-10-22 ``@ddalle``: v1.0
        """
        # Get module name
        modname = self.opts.get("datakitloader-module")
        # Check if one was found
        if modname is None:
            return
        # Attempt to load it
        try:
            mod = importlib.import_module(modname)
        except Exception:
            return
        # Check for absolute path
        fdir = os.path.join(self.cwd, pkg.replace(".", os.sep))
        # Create DataKitLoader
        return DataKitLoader(pkg, fdir, **mod.DB_NAMES)

    # Get relative path to package folder
    def get_pkgdir(self):
        r"""Get relative folder to a package

        :Call:
            >>> pkgdir = get_pkgdir(pkg, **kw)
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
            *pkg*: :class:`str`
                Name of Python module/package to create
            *t*, *target*: {``None``} | :class:`str`
                Optional subdir of *where* to put package in
        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
        """
        # Get target if any
        target = self.opts.get_opt("target")
        # Convert package name to folder (mypkg.mymod -> mypkg/mymod/)
        pkgdir = self.pkgname.replace(".", os.sep)
        # Total path
        if target and not pkgdir.startswith(target):
            # Combine two-part package path
            return os.path.join(target.replace("/", os.sep), pkgdir)
        else:
            # No target; just the package
            return pkgdir

    # Apply the "target" to a package name
    def expand_pkg1(self, rawpkg: Optional[str] = None) -> str:
        r"""Apply *target* prefix if appropriate

        :Call:
            >>> pkg1 = expand_pkg1()
        :Inputs:
            *starter*: :class:`DataKitQuickStarter`
                Utility to create new DataKit packages
        :Outputs:
            *pkg1*: :class:`str`
                *pkg* prepended with *target* if necessary
        :Versions:
            * 2021-10-22 ``@ddalle``: v1.0
        """
        # Pick between *kw* and *opts*
        target = self.opts.get_opt("target")
        # Get raw package name from user
        rawpkg = self.pkg_input if rawpkg is None else rawpkg
        # Check if *pkg* starts with *target*
        if target is None or rawpkg.startswith(target):
            # Already good
            return rawpkg
        else:
            # Prepend *pkg1*
            return f"{target}.{rawpkg}"

    # Ensure a title is present
    def prompt_title(self):
        r"""Prompt for a title if none provided

        :Call:
            >>> title = starter.prompt_title()
        :Versions:
            * 2021-08-24 ``@ddalle``: v1.0
        """
        # Get existing title from kwargs
        title = self.opts.get_opt("title")
        # If non-empty, return it
        if title:
            return title
        # Otherwise prompt one
        title = promptutils.prompt("One-line title for package")
        # Check for an answer again
        title = DEFAULT_TITLE if title is None else title
        # Save it
        self.opts.set_opt("title", title)
        # Return it
        return title


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
    # Loop through remaining folders
    for pathj in path.split(os.sep):
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


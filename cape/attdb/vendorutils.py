#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================
:mod:`cape.attdb.vendorutils`: Package Vendorization Utilities
================================================================

This module provides some CAPE-specific alterations to the
:mod:`vendorize` package available from PyPI

"""

# Standard library modules
import json
import os
import re
import shutil
import subprocess as sp
import sys

# Third-party modules
import pip
import setuptools
import vendorize

# Local modules
from .. import argread
from .. import text as textutils


# Default values for various options
DEFAULT_FIND_EXCLUDE = ()
DEFAULT_FIND_INCLUDE = ("*",)
DEFAULT_FIND_REGEX = None
DEFAULT_FIND_WHERE = "."

# Valid vendorize.toml options
OPTLIST_TOML = {
    "target",
    "packages",
}
# Valid vendorize.json options
OPTLIST_JSON = {
    "target",
    "hub",
    "packages",
}

# Regular expression for plain package name
REGEX_PKG_PLAIN = re.compile("([A-z]\w+)(\[[A-z]\w+\])?")
REGEX_PKG_EGG = re.compile("#egg=([A-z]\w+)$")


# Docstring for CLI
HELP_VENDORIZE = r"""
---------------------------------------------------------------
``dkit-vendorize``: Vendorize one or more packages
---------------------------------------------------------------

This command provides an extension to the :mod:`vendorize` package
available from PyPI. It allows you to "vendorize," that is, install a
local copy of, one or more packages. It can be called in the same folder
as one of the two files:

    * ``vendorize.json``
    * ``vendorize.toml``

:Usage:
    .. code-block:: bash

        $ dkit-vendorize [PKGS] [OPTIONS]

:Arguments:
    * *PKG1*: vendorize each package matching this regular expression
    * *PKGN*: vendorize each package matching this regular expression

:Options:

    -h, --help
        Display this help message and quit

    -t, --target REGEX
        Only vendoroze in parent packages including regular expression
        *REGEX* (default is all packages with any vendorize file)

    --cwd WHERE
        Location from which to search for packages (``"."``)

    --no-install
        List packages to vendorize but don't install

:Versions:

    * 2021-08-23 ``@ddalle``: Version 1.0
"""


# Main function
def main():
    r"""Main command-line interface for vendorize command

    :Call:
        >>> main()
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Process command-line arguments
    a, kw = argread.readkeys(sys.argv)
    # Real main function
    vendorize_repo(*a, **kw)


# Top-level function
def vendorize_repo(*a, **kw):
    r"""Vendorize in all repos

    :Call:
        >>> vendorize_repo(*a, **kw)
    :Inputs:
        *a*: :class:`tuple`\ [:class:`str`]
            Regular expressions for vendorized package(s)
        *cwd*, *where*: {``"."``} | :class:`str`
            Location from which to search for target packages
        *t*, *target*: {``None``} | :class:`str`
            Regular expression for packages in which to vendor
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Check for help flag
    if kw.get('h') or kw.get('help'):
        # Display help message and quit
        print(textutils.markdown(HELP_VENDORIZE))
        return
    # Use args as package regexes
    pkg_regexes = a
    # Location
    where = kw.get("cwd", kw.get("where", DEFAULT_FIND_WHERE))
    # Absolutize
    fabs = os.path.realpath(where)
    # Get target
    target_regex = kw.pop("target", kw.pop("t", None))
    # Find all parents
    targets = find_vendors(where, regex=target_regex)
    # Remember current location
    fpwd = os.getcwd()
    # Install/check options
    install = kw.get("install", not kw.get("check", False))
    # Vendorixe requested packages in each target
    for target in targets:
        # Get folder version of package
        pkgdir = target.replace(".", os.sep)
        # Absolutize
        absdir = os.path.join(fabs, pkgdir)
        # Two file candidates
        fjson = os.path.join(absdir, "vendorize.json")
        ftoml = os.path.join(absdir, "vendorize.toml")
        # Check for JSON file
        if os.path.isfile(fjson):
            # Read JSON file
            opts = VendorizeJSON(fjson)
        elif os.path.isfile(ftoml):
            # Read TOML file
            opts = VendorizeTOML(ftoml)
        else:
            # Unreachable unless file got deleted very recently
            continue
        # Status update
        print("In '%s':" % target)
        # Vendorize requested packages
        opts.vendorize(regex=pkg_regexes, install=install)


# Find vendors
def find_vendors(where=".", **kw):
    r"""Find packages that have vendorization inputs

    This looks for all packages that have either

    * ``vendorize.json``
    * ``vendorize.toml``

    files in them.

    :Call:
        >>> pkgs = find_vendors(where=".", **kw)
    :Inputs:
        *where*: {``"."``} | :class:`str`
            Location from which to search for packages
        *exclude*: {``()``} | :class:`tuple`\ [:class:`str`]
            List of globs to exclude from package search
        *include*: {``("*",)``} | :class:`tuple`\ [:class:`str`]
            List of globs to include during package search
        *re*, *regex*, {``None``} | :class:`str`
            Only include packages including regular expression *regex*
    :Outputs:
        *pkgs*: :class:`list`\ [:class:`str`]
            List of packages with vendorization inputs (the package
            ``''`` means the current folder has vendor inputs)
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Options for find_packages()
    o_exclude = kw.pop("exclude", DEFAULT_FIND_EXCLUDE)
    o_include = kw.pop("include", DEFAULT_FIND_INCLUDE)
    # Other options
    o_regex = kw.pop("re", kw.pop("regex", DEFAULT_FIND_REGEX))
    # Replace any Nones
    if where is None:
        where = DEFAULT_FIND_WHERE
    if o_exclude is None:
        o_exclude = DEFAULT_FIND_EXCLUDE
    if o_include is None:
        o_include = DEFAULT_FIND_INCLUDE
    # Generate base path (absolute)
    absdir = os.path.realpath(where)
    # Find packages
    pkg_list = setuptools.find_packages(
        where, exclude=o_exclude, include=o_include)
    # Also look in the current folder
    pkg_list.insert(0, "")
    # Initialize list of packages with vendorize inputs
    pkgs = []
    # Loop through found packages
    for pkg in pkg_list:
        # Check for regex
        if o_regex:
            # Check for regular expression match
            if re.search(o_regex, pkg) is None:
                continue
        # Get folder name for package
        pkgdir = pkg.replace(".", os.sep)
        # Absolutize
        pkgdir = os.path.join(absdir, pkgdir)
        # Full path to vendorize input files
        fjson = os.path.join(pkgdir, "vendorize.json")
        ftoml = os.path.join(pkgdir, "vendorize.toml")
        # Check for either file
        if os.path.isfile(fjson):
            # Found JSON file
            pkgs.append(pkg)
        elif os.path.isfile(ftoml):
            # Found TOML file
            pkgs.append(pkg)
    # Sort the package list
    pkgs.sort()
    # Output
    return pkgs


# Vendorize from a TOML file
def vendorize_toml(ftoml, **kw):
    r"""Vendorize packages using ``vendorize.toml`` file

    :Call:
        >>> vendorize_toml(ftoml)
    :Inputs:
        *ftmol*: :class:`str`
            Absolute path to a ``vendorize.toml`` path
        *re*, *regex*: {``None``} | :class:`str`
            Include packages matching optional regular expression
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Read file
    opts = VendorizeTOML(ftoml)
    # Vendorize packages therein
    opts.vendorize(**kw)


# Vendorize from a JSON file
def vendorize_json(fjson, **kw):
    r"""Vendorize packages using ``vendorize.json`` file

    :Call:
        >>> vendorize_json(fjson)
    :Inputs:
        *fjson*: :class:`str`
            Absolute path to a ``vendorize.json`` path
        *re*, *regex*: {``None``} | :class:`str`
            Include packages matching optional regular expression
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Read file
    opts = VendorizeJSON(ftoml)
    # Vendorize packages therein
    opts.vendorize(**kw)


# Base class for vendorize file formats
class VendorizeConfig(dict):
    r"""Common methods for two vendorize options file formats

    :Call:
        >>> opts = VendorizeConfig(fname=None, **kw)
        >>> opts = VendorizeConfig(d, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of file from which to read
        *d*: :class:`dict`
            If first input is a :class:`dict`, copy options from it
        *kw*: :class:`dict`
            Additional inputs to override first two
    :Outputs:
        *opts*: :class:`VendorizeConfig`
            TOML vendorization options interface
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Initialization method
    def __init__(self, fname=None, **kw):
        r"""Initialization method

        :Versions:
            * 2021-08-23 ``@ddalle``: Version 1.0
        """
        # Check input type
        if fname is None:
            # No file to read
            self.root_dir = os.getcwd()
        elif isinstance(fname, dict):
            # Preexisting dictionary
            self.update(fname)
            # Use current folder as base path
            self.root_dir = os.getcwd()
        else:
            # Absolutize
            fabs = os.path.abspath(fname)
            # Read it as a file
            self.read(fabs)
            # Use base folder
            self.root_dir = os.path.dirname(fabs)
        # Append any kwargs
        self.update(**kw)

    # Read a file
    def read(self, fname):
        pass

    # Get list of packages
    def get_package_list(self, **kw):
        r"""Get list of package names from full requirements

        :Call:
            >>> pkgs = opts.get_package_list(**kw)
        :Inputs:
            *opts*: :class:`VendorizeConfig`
                Vendorization options interface
            *re*, *regex*: {``None``} | :class:`str`
                Include packages matching optional regular expression
        :Outputs:
            *pkgs*: :class:`list`\ [:class:`str`]
                List of package base names
        :Versions:
            * 2021-08-23 ``@ddalle``: Version 1.0
        """
        # Get regex option
        regex = kw.get("regex", kw.get("re", None))
        # Initialize list
        pkgs = []
        # Loop through packages
        for req in self.get("packages", []):
            # Get package name
            try:
                pkg = get_package_name(req)
            except ValueError:
                continue
            # Check for regular expression
            if regex:
                # Check for match
                if re.search(regex, pkg) is None:
                    continue
            # Save package name
            pkgs.append(pkg)
        # Output
        return pkgs
        
    # Vendorize
    def vendorize(self, **kw):
        r"""Get list of package names from full requirements

        :Call:
            >>> opts.vendorize(**kw)
        :Inputs:
            *opts*: :class:`VendorizeConfig`
                Vendorization options interface
            *re*, *regex*: {``None``} | :class:`str` | :class:`list`
                Include packages matching optional regular expression
            *check*: ``True`` | {``False``}
                Option to only list packages and not install
            *install*: {``True``} | ``False``
                Opposite of *check*
        :Versions:
            * 2021-08-23 ``@ddalle``: Version 1.0
        """
        # Get regex option
        regex = kw.get("regex", kw.get("re", None))
        # Get option for just checking
        install = kw.get("install", not kw.get("check", False))
        # Listify
        regexs = _listify(regex)
        # Loop through packages
        for req in self.get("packages", []):
            # Get package name
            try:
                pkg = get_package_name(req)
            except ValueError:
                continue
            # Check for regular expression
            if regex:
                # Check for match
                for regexj in regexs:
                    if re.search(regexj, pkg):
                        match = True
                        break
                else:
                    # Matched no filters
                    match = False
                # Go to next requirement if no matches
                if not match:
                    continue
            # Status update
            print("Vendorizing package '%s'" % pkg)
            # Check for --no-install option
            if not install:
                continue
            # Vendorize
            self.vendorize_requirement(req)

    # Vendorize one package
    def vendorize_requirement(self, req):
        r"""Vendorize one package, updating if necessary
    
        This slightly modifies the function of the same name from
        :mod:`vendorize`.
    
        :Call:
            >>> ierr = opts.vendorize_requirement(req)
        :Inputs:
            *opts*: :class:`VendorizeConfig`
                Vendorization options interface
            *req*: :class:`str`
                Package to install with ``pip``
        :Outputs:
            *ierr*: :class:`int`
                Exit code from :func:`pip.main`
        :Versions:
            * 2021-08-23 ``@ddalle``: Version 1.0
        """
        # Get target folder
        target = self["target"]
        # Remember current folder
        fpwd = os.getcwd()
        # Go to vendorize folder
        os.chdir(self.root_dir)
        # Status update
        sys.stdout.write("  requirement='%s' ... " % req)
        sys.stdout.flush()
        # Vendorize
        ierr = vendorize_requirement(req, target)
        # Return to original location
        os.chdir(fpwd)
        # Status update
        if ierr:
            sys.stdout.write("failed\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("succeeded\n")
            sys.stdout.flush()
        # Return output
        return ierr


# Class for TOML packages
class VendorizeTOML(VendorizeConfig):
    r"""Common methods from ``vendorize.toml`` file

    :Call:
        >>> opts = VendorizeTOML(ftoml=None, **kw)
        >>> opts = VendorizeTOML(d, **kw)
    :Inputs:
        *ftoml*: {``None``} | :class:`str`
            Name of file from which to read
        *d*: :class:`dict`
            If first input is a :class:`dict`, copy options from it
        *kw*: :class:`dict`
            Additional inputs to override first two
    :Outputs:
        *opts*: :class:`VendorizeConfig`
            TOML vendorization options interface
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Read a file
    def read(self, ftoml):
        r"""Read a ``vendorize.toml`` file

        :Call:
            >>> opts.read(ftoml)
        :Inputs:
            *opts*: :class:`VendorizeTOML`
                TOML vendorization options interface
            *ftoml*: :class:`str`
                Name of file to read
        :Versions:
            * 2021-08-23 ``@ddalle``: Version 1.0
        """
        # Open the file
        with open(ftoml) as f:
            # Read the file
            opts = vendorize.toml.load(f)
        # Loop through options
        for k, v in opts.items():
            # Check validity
            if k not in OPTLIST_TOML:
                print("Unrecognized option '%s'" % k)
                continue
            # Save it
            self[k] = v


# Class for TOML packages
class VendorizeJSON(VendorizeConfig):
    r"""Common methods from ``vendorize.json`` file

    :Call:
        >>> opts = VendorizeTOML(ftoml=None, **kw)
        >>> opts = VendorizeTOML(d, **kw)
    :Inputs:
        *ftoml*: {``None``} | :class:`str`
            Name of file from which to read
        *d*: :class:`dict`
            If first input is a :class:`dict`, copy options from it
        *kw*: :class:`dict`
            Additional inputs to override first two
    :Outputs:
        *opts*: :class:`VendorizeConfig`
            TOML vendorization options interface
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Read a file
    def read(self, fjson):
        r"""Read a ``vendorize.json`` file

        :Call:
            >>> opts.read(ftoml)
        :Inputs:
            *opts*: :class:`VendorizeJSON`
                JSON vendorization options interface
            *fjson*: :class:`str`
                Name of file to read
        :Versions:
            * 2021-08-23 ``@ddalle``: Version 1.0
        """
        # Open the file
        with open(fjson) as f:
            # Read the file
            opts = json.load(f)
        # Loop through options
        for k, v in opts.items():
            # Check validity
            if k not in OPTLIST_JSON:
                print("Unrecognized option '%s'" % k)
                continue
            # Save it
            self[k] = v

    # Vendorize one package
    def vendorize_requirement(self, req):
        r"""Vendorize one package, updating if necessary
    
        This slightly modifies the function of the same name from
        :mod:`vendorize`.
    
        :Call:
            >>> ierr = opts.vendorize_requirement(req)
        :Inputs:
            *opts*: :class:`VendorizeJSON`
                Vendorization options interface
            *req*: :class:`str`
                Package to install with ``pip``
        :Outputs:
            *ierr*: :class:`int`
                Exit code from :func:`pip.main`
        :Versions:
            * 2021-08-23 ``@ddalle``: Version 1.0
        """
        # Get target folder
        target = self["target"]
        # Get list of hubs
        hubs = _listify(self.get("hub", []))
        # Remember current folder
        fpwd = os.getcwd()
        # Go to vendorize folder
        os.chdir(self.root_dir)
        # Try each "hub" until one works
        for hub in hubs:
            # Combine paths
            if hub:
                # Check if it's a full package
                if REGEX_PKG_PLAIN.fullmatch(req):
                    # Append extra markers
                    # * "tnas53" -> "tnas53.git#egg=tnas53"
                    relativereq = "%s.git#egg=%s" % (req, req)
                else:
                    # Full descriptor
                    # * "tnas53.git#egg=tnas53"
                    # * "tnas53.git@v1.0#egg=tnas53"
                    relativereq = req
                # Prepend non-empty hub
                fullreq = os.path.join(hub.rstrip("/"), relativereq)
            else:
                # For no hub or empty hub, install *req* as is
                fullreq = req
            # Status update
            sys.stdout.write("  requirement='%s' ... " % fullreq)
            sys.stdout.flush()
            # Vendorize
            ierr = vendorize_requirement(fullreq, target)
            # Status update
            if ierr:
                sys.stdout.write("failed\n")
                sys.stdout.flush()
            else:
                sys.stdout.write("succeeded\n")
                sys.stdout.flush()
            # Exit if successful
            if ierr == 0:
                break
        # Return to original location
        os.chdir(fpwd)
        # Return output
        return ierr


# Vendorize a folder
def vendorize_requirement(req, target):
    r"""Vendorize one package, updating if necessary

    This slightly modifies the function of the same name from
    :mod:`vendorize`.

    :Call:
        >>> ierr = vendorize_requirement(req, target)
    :Inputs:
        *req*: :class:`str`
            Package to install with ``pip``
        *target*: :class:`str`
            Folder in which to install vendorized packages
    :Outputs:
        *ierr*: :class:`int`
            Exit code from :func:`pip.main`
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Ensure folder exists
    vendorize.mkdir_p(target)
    # Install command
    cmd = [
        sys.executable, "-m", "pip",
        "install", "--no-dependencies", "--upgrade",
        "--target", target, req
    ]
    # Install it
    ierr = sp.call(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    # Check for errors
    if ierr:
        return ierr
    # Get top-level package folders
    top_level_names = find_top_level_packages(target)
    # Otherwise, rewrite imports
    vendorize._rewrite_imports(target, top_level_names)
    # Python file to ensure folder is a package
    fpy = os.path.join(target, "__init__.py")
    # Make sure the target folder is a package
    if not os.path.isfile(fpy):
        with open(fpy, "w") as f:
            f.write("")
    # Output
    return ierr


# Find top-level packages in a folder
def find_top_level_packages(target):
    r"""Find top-level packages in a folder

    :Call:
        >>> pkgs = find_top_level_packages(target)
    :Inputs:
        *target*: :class:`str`
            Name of folder to look in
    :Outputs:
        *pkgs*: :class:`list`\ [:class:`str`]
            List of top-level packages in *target*
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Use find_packages
    return setuptools.find_packages(where=target, exclude=("*.*",))


# Find base package name from full direction
def get_package_name(req):
    r"""Get package name from full ``pip`` requirement

    :Call:
        >>> pkg = get_package_name(req)
    :Inputs:
        *req*: :class:`str`
            Any full requirement for ``pip`` install
    :Outputs:
        *pkg*: :class:`str`
            Name of package provided by install
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Try to interpret *req* as a plain package name
    match = REGEX_PKG_PLAIN.fullmatch(req)
    # Check for a match
    if match:
        # Get package name:
        #   * "dvc" --> "dvc"
        #   * "dvc[ssh]" --> "dvc"
        return match.group(1)
    # Try to interpret *req* as a PIP+VCS package
    match = REGEX_PKG_EGG.search(req)
    # Check for a match
    if match:
        # Get package name:
        #   * "git+file:///mypackage.git#egg=mypackage" -> "mypackage"
        return match.group(1)
    # Failure
    raise ValueError("Could not determine package from '%s'" % req)


# Convert to list
def _listify(v):
    r"""Convert scalar or ``None`` to :class:`list`

    :Call:
        >>> V = _listify(V)
        >>> V = _listify(v)
        >>> V = _listify(vnone)
    :Inputs:
        *V*: :class:`list` | :class:`tuple`
            Preexisting :class:`list` or :class:`tuple`
        *vnone*: ``None``
            Empty input of ``None``
        *v*: **scalar**
            Anything else
    :Outputs:
        *V*: :class:`list`
            Enforced list, either

            * ``V`` --> ``V``
            * ``V`` --> ``list(V)``
            * ``None`` --> ``[]``
            * ``v`` --> ``[v]``
    :Versions:
        * 2021-08-18 ``@ddalle``: Version 1.0
    """
    # Check type
    if isinstance(v, list):
        # Return it
        return v
    elif isinstance(v, tuple):
        # Just convert it
        return list(v)
    elif v is None:
        # Empty list
        return []
    else:
        # Create singleton
        return [v]


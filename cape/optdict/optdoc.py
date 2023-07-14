r"""
``optdict.optdoc``: Documentation writers for ``OptionsDict`` subclasses
========================================================================

This module provides functions to write reStructuredText (``.rst``)
files listing all available information for a given subclass of
:class:`OptionsDict`.
"""

# Standard library
import importlib
import os


# Sequence of title markers for reST, (character, overline option)
RST_SECTION_CHARS = (
    ("-", True),
    ("=", False),
    ("-", False),
    ("^", False),
    ("*", False),
)


# Write documentation for a class
def write_rst(cls: type, fname: str, **kw):
    r"""Write all options for an :class:`OptionsDict` subclass to reST

    :Call:
        >>> write_rst(cls, fname, **kw)
    :Inputs:
        *cls*: :class:`type`
            A :class:`OptionsDict` subclass
        *fname*: :class:`str`
            Name of ``.rst`` file to write (unless up-to-date)
        *force_update*, *f*: ``True`` | {``False``}
            Write *fname* even if it appears to be up-to-date
        *recurse*: {``True``} | ``False``
            Option to recurse into subsections of *cls*
        *narrow*: ``True`` | {``False``}
            Option to only include ``_optlist`` specific to *cls*
        *depth*: {``0``} | :class:`int`
            Depth to use for headers
        *verbose*, *v*: ``True`` | {``False``}
            Option to use verbose text for options
    :Versions:
        * 2023-07-13 ``@ddalle``: v1.0
    """
    # Modification of *fname* if already existing
    if os.path.isfile(fname):
        # Get last modification time of existing file
        t_rst = os.path.getmtime(fname)
    else:
        # Use ``0`` modification time to guarantee update
        t_rst = 0.0
    # Initialize modification time of source code
    t_mod = 0.0
    # Parse options
    force_update = kw.pop("force_update", kw.pop("f", False))
    recurse = kw.pop("recurse", True)
    narrow = kw.pop("narrow", False)
    depth = kw.pop("depth", 0)
    verbose = kw.pop("verbose", kw.pop("v", False))
    # Options for print_rst()
    kw_rst = {
        "recurse": recurse,
        "narrow": narrow,
        "depth": depth,
        "v": verbose,
    }
    # Get unique files whose source code is part of this class
    modlist = _find_cls_modfile(cls, narrow, recurse)
    # Get latest mod time
    for modfile in modlist:
        t_mod = max(t_mod, os.path.getmtime(modfile))
    # Check if documentation is out of date
    if (not force_update) and (t_mod <= t_rst):
        return
    # Open file
    with open(fname, 'w') as fp:
        # Generate text
        txt = cls.print_rst(**kw_rst)
        # Write to file
        fp.write(txt)


def _find_cls_modfile(cls: type, narrow: bool, recurse: bool) -> set:
    # Initialize set of classes whose source code affects *cls*
    modlist = {_get_cls_modfile(cls)}
    # If there's no *recurse* option, just exit
    if not recurse:
        return modlist
    # Get children
    if narrow:
        # Just get class attributes of *cls*
        cls_optmap = cls.__dict__.get("_sec_cls_optmap", {})
        cls_secmap = cls.__dict__.get("_sec_cls", {})
    else:
        # Include attributes of basis class(es)
        cls_optmap = cls.getx_cls_dict("_sec_cls_optmap")
        cls_secmap = cls.getx_cls_dict("_sec_cls")
    # Get unique classes from these two attributes
    cls_subcls = set(cls_optmap.values()).union(cls_secmap.values())
    # Get module for each
    cls_submod = {_get_cls_modfile(subcls) for subcls in cls_subcls}
    # Include those
    modlist.update(cls_submod)
    # Recurse
    for subcls in cls_subcls:
        # Get subsection class list
        submodlist = _find_cls_modfile(subcls, narrow, recurse)
        # Combine
        modlist.update(submodlist)
    # Output
    return modlist


def _get_cls_modfile(cls: type):
    # Import the module
    mod = importlib.import_module(cls.__module__)
    # Get the file
    return mod.__file__

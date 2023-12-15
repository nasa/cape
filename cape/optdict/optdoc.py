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
import sys


# Sequence of title markers for reST, (character, overline option)
RST_SECTION_CHARS = (
    ("-", True),
    ("=", False),
    ("-", False),
    ("^", False),
    ("*", False),
)


# Drive overall documentaion for a class
def make_rst(opts: dict, name: str, **kw):
    r"""Write full list of options to a collection of ``.rst`` files

    :Call:
        >>> make_rst(opts, name, **kw)
    :Inputs:
        *opts*: :class:`dict`\ [:class:`dict`]
            Options for each *name*
        *name*: :class:`str`
            Name of top-level remaining section ``opts[name]``
    :Versions:
        * 2023-07-14 ``@ddalle``: v1.0
    """
    # Get options for this section
    sec_opts = opts.get(name, {})
    # Parse options
    fdir = sec_opts.get("folder", kw.get("folder", os.getcwd()))
    fname = sec_opts.get("file", name)
    prefix = sec_opts.get("prefix", kw.get("prefix", ""))
    modname = sec_opts.get("module", kw.get("module", name))
    clsname = sec_opts.get("class", kw.get("class", name))
    # Formatting options
    narrow = sec_opts.get("narrow", kw.get("narrow", False))
    verbose = sec_opts.get("verbose", kw.get("verbose", False))
    recurse = sec_opts.get("recurse", kw.get("recurse", True))
    # Proper cascase for recursion options
    recurse_sec_cls = sec_opts.get(
        "recurse_sec_cls",
        sec_opts.get("recurse", kw.get("recurse_sec_cls", recurse)))
    # Proper cascase for recursion options
    recurse_sec_clsmap = sec_opts.get(
        "recurse_sec_clsmap",
        sec_opts.get("recurse", kw.get("recurse_sec_clsmap", recurse)))
    # Add prefix if not top-level
    if kw.get("parent"):
        # Strip out any prefixes already in *name*
        prefixname = name.split('-')[-1]
        # Append to prefix
        prefix = f"{prefix}{prefixname}-"
    # Process options
    kw_sec = {
        "narrow": narrow,
        "prefix": prefix,
        "verbose": verbose,
        "recurse": recurse,
        "recurse_sec_cls": recurse_sec_cls,
        "recurse_sec_clsmap": recurse_sec_clsmap,
        "force_update": kw.get("force_update", kw.get('f', False)),
    }
    # Process defaults for child sections
    kw_children = dict(kw_sec)
    kw_children.update({
        "folder": fdir,
        "prefix": sec_opts.get("child_prefix", prefix),
        "parent": name,
    })
    # Include title if specified
    if "title" in kw:
        kw_sec["title"] = kw["title"]
    # Import module
    mod = importlib.import_module(modname)
    # Create folder if necessary (only one level, though)
    if not os.path.isdir(fdir):
        os.mkdir(fdir)
    # Get the class
    cls = getattr(mod, clsname)
    # Absolute path
    frst = os.path.join(fdir, fname + ".rst")
    # Process parent class
    children = write_rst(cls, frst, **kw_sec)
    # Loop through children
    for sec, cls_or_tuple in children.items():
        # Copy options
        kw_child = dict(kw_children)
        # Unpack if necessary
        if isinstance(cls_or_tuple, tuple):
            # Unpack
            sectitle, seccls = cls_or_tuple
            # Save title
            kw_child["title"] = sectitle
        else:
            # Class only
            seccls = cls_or_tuple
        # Get default module name and class name
        modname = seccls.__module__
        clsname = seccls.__name__
        # Set dfault module name and class name
        kw_child.setdefault("module", modname)
        kw_child.setdefault("class", clsname)
        # Include prefix in name
        secname = f"{prefix}{sec}"
        # Recurse
        make_rst(opts, secname, **kw_child)
    # Status update
    tw = 72
    msg = f"{cls.__module__}.{cls.__name__} -> {os.path.basename(fname)}"
    msg = msg[:tw - 2]
    sys.stdout.write((" "*tw) + "\r")
    sys.stdout.flush()


# Write documentation for a (single) class
def write_rst(cls: type, fname: str, **kw):
    r"""Write all options for an :class:`OptionsDict` subclass to reST

    :Call:
        >>> children = write_rst(cls, fname, **kw)
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
    :Outputs:
        *children*: :class:`dict`\ [:class:`type`]
            Dictionary of child classes not written to *fname*
    :Versions:
        * 2023-07-13 ``@ddalle``: v1.0
        * 2023-07-14 ``@ddalle``: v1.1; process "toctree"
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
    prefix = kw.pop("prefix", "")
    narrow = kw.pop("narrow", False)
    depth = kw.pop("depth", 0)
    verbose = kw.pop("verbose", kw.pop("v", False))
    recurse = kw.pop("recurse", True)
    recurse_sec_cls = kw.pop("recurse_sec_cls", recurse)
    recurse_sec_clsmap = kw.pop("recurse_sec_clsmap", recurse)
    # Options for print_rst()
    kw_rst = {
        "narrow": narrow,
        "depth": depth,
        "v": verbose,
        "recurse": recurse,
        "recurse_sec_cls": recurse_sec_cls,
        "recurse_sec_clsmap": recurse_sec_clsmap,
    }
    # Set title if appropriate
    if "title" in kw:
        kw_rst["title"] = kw.pop("title")
    # Get unique files whose source code is part of this class
    modlist = _find_cls_modfile(cls, narrow, recurse)
    # Get latest mod time
    for modfile in modlist:
        t_mod = max(t_mod, os.path.getmtime(modfile))
    # Check if documentation is out of date
    if (not force_update) and (t_mod <= t_rst):
        # Status update
        tw = 72
        msg = f"{cls.__module__}.{cls.__name__} up-to-date"
        msg = msg[:tw - 2]
        sys.stdout.write((" "*tw) + "\r")
        sys.stdout.write(msg + "\r")
        sys.stdout.flush()
        # Initialize children
        children = {}
        # Add sections to children as appropriate
        if not recurse_sec_cls:
            children.update(cls._getx_sec_cls(narrow))
        if not recurse_sec_clsmap:
            children.update(cls._getx_sec_cls_optmap(narrow))
        # Return children
        return children
    # Status update
    tw = 72
    msg = f"{cls.__module__}.{cls.__name__} -> {os.path.basename(fname)}"
    msg = msg[:tw - 2]
    sys.stdout.write((" "*tw) + "\r")
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()
    # Open file
    with open(fname, 'w') as fp:
        # Generate text
        txt, children = cls.print_rst(**kw_rst)
        # Write to file
        fp.write(txt)
        # Check for child subsections not described in *txt*
        if len(children) == 0:
            return {}
        # Write table of contents for children
        fp.write("**Subsections:**\n\n")
        fp.write(".. toctree::\n")
        fp.write("    :maxdepth: 1\n\n")
        # Loop through children
        for child in children:
            fp.write(f"    {prefix}{child}\n")
    # Outputs
    return children


def _find_cls_modfile(cls: type, narrow: bool, recurse: bool) -> set:
    # Initialize set of classes whose source code affects *cls*
    modlist = {_get_cls_modfile(cls)}
    # If there's no *recurse* option, just exit
    if not recurse:
        return modlist
    # Get children
    cls_optmap = cls._getx_sec_cls_optmap(narrow)
    cls_secmap = cls._getx_sec_cls(narrow)
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


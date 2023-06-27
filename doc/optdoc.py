import os
import importlib
import inspect

# Solvers to make options rsts for
PYX = [
    "pycart",
    "pyfun",
    "pyover",
]


def write_info_rsts(pyx, optsecs, odir="json"):
    r"""Write option info rsts for all option modules in *pyx*
    """
    # Get cwd
    cwd = os.getcwd()
    # Build root output dir
    orootdir = os.path.join(cwd, odir)
    # Go to output dir
    os.chdir(orootdir)
    # Loop through opt classes
    for optsec, optcls in optsecs.items():
        # RST file name
        fname = "%s.rst" % optsec
        # Open rst
        f = open(fname, "w")
        sphinx_ref = odir.replace("/", "-")
        # Start writing the file with sphinx ref
        f.write("\n.. _%s-%s:\n\n" % (sphinx_ref, optsec.lower()))
        # Write Section for this class
        class_substr = "Options for ``%s`` Section" % optsec
        # Proper number of under dashes
        title = "*" * len(class_substr)
        # Write section title
        f.write(title + "\n")
        f.write(class_substr + "\n")
        f.write(title + "\n")
        # Write descpription for user
        uhelp = ("The options below are the available options in the " +
                 "``%s`` Section of the " % optsec + "``%s.json`` " % pyx +
                 "control file")
        f.write(uhelp + "\n\n")
        # Write help rst
        _write_info_rst(f, optsec, optcls)
        # Close file
        f.close()
    # Go back to cwd
    os.chdir(cwd)


def _write_info_rst(f, optsec, optcls, optclsp=False):
    r"""Recursively write option info rsts for *optcls*
    """
    # Get instance of opt class
    ocls = optcls()
    # Get optlist
    optlist = ocls.getx_cls_set("_optlist")
    # If there's an optlist
    if optlist:
        # Write section for each opt
        for opt in optlist:
            # Write comment to indicate start of opt help
            f.write("\n")
            # Get opt info output
            o_info = ocls.getx_optinfo(opt)
            # Check for ending line break
            if not o_info[-2:] == "\n":
                o_info += "\n"
            # Write body of opt help
            f.write(o_info)
            # Write comment to indicate end of opt help
            f.write("\n")
    # See if any nested classes to write
    _sec_cls = ocls.getx_cls_dict("_sec_cls")
    # See if there are any nested optmap
    _sec_cls_optmap = ocls.getx_cls_dict("_sec_cls_optmap")
    # If there are
    if _sec_cls:
        # Go through them recursively
        for optsec, optcls in _sec_cls.items():
            sub_ocls = optcls()
            # See if there are any nested optmap
            sub_sec_cls_optmap = sub_ocls.getx_cls_dict("_sec_cls_optmap")
            # If _default_ in optmap, write a subsection header instead
            if "_default_" in sub_sec_cls_optmap.keys():
                # Write sub sub section for this class
                class_substr = "Options for all ``%s``" % optsec
                # Proper number of under dashes
                subtitle = "=" * len(class_substr)
            else:
                subchar = "="
                # Write sub sub section for this class
                class_substr = "%s Options" % optsec
                # Proper number of under dashes
                subtitle = "-" * len(class_substr)
            # Write section title
            f.write(class_substr + "\n")
            f.write(subtitle + "\n")
            _write_info_rst(f, optsec, optcls, optclsp=optclsp)
    if _sec_cls_optmap:
        # Go through them recursively
        for optsec, optcls in _sec_cls_optmap.items():
            # If optmap default
            if optsec == "_default_":
                optclsp = _sec_cls_optmap[optsec]
                # Set the optsec to the prefix to Opts
                optsec = optcls.__name__.split("Opts")[0]
                subchar = "-"
            else:
                # Get all bases of this class
                bases = inspect.getmro(optcls)
                # If it shares a bases wit the default
                if optclsp in bases:
                    # It's a subsection
                    subchar = "-"
                else:
                    # Otherwise it's its own section
                    subchar = "="
                # Write sub sub section for this class
                class_substr = "%s Options" % optsec
                # Proper number of under dashes
                subtitle = subchar * len(class_substr)
                # Write section title
                f.write(class_substr + "\n")
                f.write(subtitle + "\n")
            _write_info_rst(f, optsec, optcls, optclsp=optclsp)
    # Reset parent class just in case
    optclsp = None


def get_optsecs(pyx):
    r"""Get list of option modules in *pyx*
    """
    # Get pyx options module
    options_mod = importlib.import_module("cape.%s.options" % pyx)
    # Get cls dict from instance of Options
    sec_cls = options_mod.Options().getx_cls_dict("_sec_cls")
    return sec_cls


def make_rsts():
    r"""Make all option rsts for each *pyx* solver
    """
    # Write options for each solver
    for pyx in PYX:
        odir = os.path.join(f"{pyx}", "json")
        # Get list of options modules
        optsecs = get_optsecs(pyx)
        # Write info rsts for these modules
        write_info_rsts(pyx, optsecs, odir=odir)

make_rsts()
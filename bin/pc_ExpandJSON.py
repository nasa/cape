#!/usr/bin/env python
"""
Expand one or more JSON files: ``pc_ExpandJSON.py``
===================================================

This function performs two tasks:
    
    * Removes comments starting with either ``'//'`` or ``'#'``
    * Replaces ``JSONFile(fname)`` with the contents of *fname*

The comments are removed recursively, so comments within children files are
allowed.  If no output file is specified, the suffix ``.old`` is added to
the original file, and the expanded contents are written to a new file with
the same name as the input file.

:Call:
    
    .. code-block:: bash
    
        pc_ExpandJSON.py IFILE1 IFILE2 [OPTIONS]
        pc_ExpandJSON.py -i IFILE1 -o OFILE1 [OPTIONS]
        
:Inputs:
    *IFILE1*: name of first input file
    *IFILE2*: name of second input file
    *OFILE1*: output file for *IFILE1*
    
:OPTIONS:
    -h, --help
        Display this help message and exit
        
    -i IFILE
        Use *IFILE* as last input file
        
    -o OFILE
        Use *OFILE* as output file for ``-i IFILE``

:Versions:
    * 2015-10-27 ``@ddalle``: First version
"""

# Standard library modules
import sys
import shutil

# CAPE modules
import cape.text
import cape.argread
import cape.cfdx.options.util


# Interpreter function
def ExpandJSON(*a, **kw):
    """Expand one or more JSON files
    
    This function performs two tasks:
    
        * Removes comments starting with either ``'//'`` or ``'#'``
        * Replaces ``JSONFile(fname)`` with the contents of *fname*
    
    The comments are removed recursively, so comments within children files are
    allowed.  If no output file is specified, the suffix ``.old`` is added to
    the original file, and the expanded contents are written to a new file with
    the same name as the input file.
    
    :Call:
        >>> main(f1, f2, ...)
        >>> main(i=None, o=None)
    :Inputs:
        *f1*: :class:`str`
            Name of first input file
        *f2*: :class:`str`
            Name of second input file
        *i*: :class:`str`
            Name of input file, alternate input format
        *o*: :class:`str`
            Name of output file for input *i* (defaluts to *i*) 
    :Versions:
        * 2015-10-27 ``@ddalle``: First version
    """
    # Initialize list of input files
    ifile = a
    # Initialize output files
    ofile = a
    # Check for directly specified input and output files.
    if 'i' in kw:
        # Append to input list.
        ifile.append(kw['i'])
        # Check for output file name
        ofile.append(kw.get('o', kw['i']))
        
    # Loop through files.
    for i in range(len(ifile)):
        # Extract names
        fi = ifile[i]
        fo = ofile[i]
        # Check for matches
        if fi == fo:
            # Copy original file
            shutil.copy(fi, '%s.old'%fi)
        # Read the file
        lines = open(fi).readlines()
        # Strip the comments and expand subfiles
        lines = cape.cfdx.options.util.expandJSONFile(lines)
        # Write to the output file
        f = open(fo, 'w')
        f.write(lines)
        f.close()

# Check for script
if __name__ == "__main__":
    # Read inputs
    a, kw = cape.argread.readflags(sys.argv)
    # Check for help flag.
    if 'h' in kw or 'help' in kw:
        # Print help message and exit.
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Apply to inputs
    ExpandJSON(*a, **kw)


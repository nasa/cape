#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`pyUS.inputInp`: US3D primary input file interface
========================================================

This is a module built off of the :mod:`cape.fileCntl.FileCntl` module
customized for manipulating US3D input files.  Such files are split into
"blocks" with a syntax such as the following:

    .. code-block:: none
    
        [CFD_SOLVER]
        !-----------------------------------------------------------------------
        !   nstop      ires     nplot     iconr      impl      kmax    kmaxo
            30000         0       300       0         21        10       4
        !
        !   ivisc      ivib     ichem      itrb     ibase   idiss_g
              11          2         1         0         0       1
        !
        !   ivmod     ikmod     idmod       ikv      icfl     dtfix
                3      -999         3        11         1     0.0d0
        !
        !  iorder      iuem      ikve       kbl      iman
                2         3        11        80       100
        !
        !   npfac     npvol
               0         0
        !
        !     cfl      epsj      wdis
             1.1d0      0.3     0.001d+0
        !-----------------------------------------------------------------------
        [/CFD_SOLVER]

This module is designed to recognize such sections.  The main feature of the
module is methods to set specific properties of an input file according to
certain named blocks.  Most of the parameters occur in ordered sections without
labels, and so therefore the contents are quite solver-specific.

"""

# Standard library
import re

# Standard third-party modules
import numpy as np

# Base file control class
import cape.fileCntl
import cape.namelist

# Base this class off of the main file control class
class InputInp(cape.namelist.Namelist):
    """
    Input file class for US3D primary input files
    
    This class is derived from the :class:`cape.fileCntl.FileCntl` class, so
    all methods applicable to that class can also be used for instances of this
    class.
    
    :Call:
        >>> inpt = InputInp()
        >>> inpt = InputInp(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of namelist file to read, defaults to ``'input.inp'``
    :Outputs:
        *inp*: :class:`pyUS.inputInp.InputInp`
            Namelist file control instance
        *inp.Sections*: :class:`dict` (:class:`list` (:class:`str`))
            Dictionary of sections containing contents of each namelist
        *inp.SectionNames*: :class:`list` (:class:`str`)
            List of section names
    :Version:
        * 2019-06-06 ``@ddalle``: First version
    """
   # --- Hard-Coded Orders ---
    # Parameter names in [CFD_SOLVER] block
    CFD_SOLVER_keys = [
        ["nstop",  "ires",  "nplot",  "iconr", "impl",   "kmax",   "kmaxo"],
        ["ivisc",  "ivib",  "ichem",  "itrb",  "ibase",  "idiss_g"],
        ["ivmod",  "ikmod", "idmod",  "ikv",   "icfl",   "dtfix"],
        ["iorder", "iuem",  "ikve",   "kbl",   "iman"],
        ["npfac",  "npvol"],
        ["cfl",    "epsj",  "wdis"]
    ]
    # Current values of BCs table
    BCNames = []
    BCTable = {}
    # Mass fractions
    BC_Y = {}
    # Direction cosines
    BC_cos = {}
    # Number of lines in the BC Table
    BCTable_rows = 0
    
   # --- Config ---
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="input.inp"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.SplitToBlocks(reg="\[/?([\w_]+)\]", endreg="\[/([\w_]+)\]")
        
    
  # --- Conversions ---
    # Conversion to text
    def ConvertToText(self, v, exp="d", fmt="%s"):
        """Convert a value to text to write in the namelist file
        
        :Call:
            >>> val = inp.ConvertToText(v)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *v*: :class:`str` | :class:`int` | :class:`float` | :class:`list`
                Evaluated value of the text
            *exp*: {``"d"``} | ``"e"`` | ``"E"`` | ``"D"``
                Character to use for exponential notation
            *fmt*: {``"%s"``} | :class:`str`
                C-style format string
        :Outputs:
            *val*: :class:`str` | :class:`unicode`
                Text of the value from file
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # Get the type
        t = type(v).__name__
        # Form the output line.
        if t in ['str', 'unicode']:
            # Force quotes
            return '"%s"' % v
        elif t in ['bool'] and v:
            # Boolean
            return ".true."
        elif t in ['bool']:
            # Boolean
            return ".false."
        elif t in ['list', 'ndarray', "tuple"]:
            # List (convert to string first)
            V = [str(vi) for vi in v]
            return " ".join(V)
        elif isinstance(v, float):
            # Use the built-in string converter
            txt = fmt % v
            # Check for integer format
            if ("e" in txt) and ("." not in txt):
                # Redo with forced decimal
                txt = "%.1e" % v
            # Replace "e" with "d", or whatever character
            txt = txt.replace("e", exp)
            # Output
            return txt
        else:
            # Use the built-in string converter
            return str(v)
            
  # --- Sections ---
    # Add a section
    def AddSection(self, sec):
        """Add a section to the ``input.inp`` interface
        
        :Call:
            >>> inp.AddSection(sec)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2019-06-06 ``@ddalle``: Lightly modified from Namelist
        """
        # Escape if already present
        if sec in self.SectionNames:
            return
        # Append the section
        self.SectionNames.append(sec)
        # Add the lines
        self.Section[sec] = [
            '[%s]\n' % sec,
            '[/%s]\n' % sec,
        ]
            
  # --- Data ---
    # Convert a single line of text to values
    def ConvertLineToList(self, line, **kw):
        """Convert a line of space-separated values into parts
        
        :Call:
            >>> header, vals, LV, LS = inp.ConvertLineToList(line, **kw)
        :Inputs:
            *line*: :class:`str`
                Line of text with space-separated values
            *indent*: {``0``} | :class:`int` >= 0
                Number of characters to ignore at beginning of line
        :Outputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *header*: :class:`str`
                First *indent* characters of *line*
            *vals*: :class:`list`\ [:class:`str`]
                List of non-whitespace groups
            *LV*: :class:`list`\ [:class:`int`]
                Lengths of strings in *vals*
            *LS*: :class:`list`\ [:class:`int`]
                Lengths of whitespace groups between values
        :Versions:
            * 2019-06-05 ``@ddalle``: First version
        """
        # Number of spaces in tab
        indent = kw.get("indent", 0)
        # Save tab
        header = line[:indent]
        # Split the remaining portions by white space
        groups = re.findall("\s+[^\s]+", line[indent:])
        # Get non-whitespace groups and lengths of whitespace sections
        tags = [(t.lstrip(), len(t) - len(t.lstrip())) for t in groups]
        # Split columns into values and whitespace lengths
        vals = [t[0] for t in tags]
        lens_sep = [t[1] for t in tags]
        # Get lengths of values
        lens_val = [len(t) for t in vals]
        # Output
        return header, vals, lens_val, lens_sep
        
    # Set one value within a line of text
    def SetLineValueSequential(self, line, i, val, **kw):
        """Set a value in a line that assumes space-separated values
        
        :Call:
            >>> txt = inp.SetLineValueSequential(line, i, val, **kw)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *line*: :class:`str`
                Line of text in space-separated format
            *i*: :class:`int` >= 0
                Index of entry to change/set
            *val*: :class:`any`
                Value to print to entry *i* of modified *line*
            *align*: ``"left"`` | ``"center"`` | {``"right"``}
                Alignment option; determines output spacing
            *delimiter*, *delim*, *sep*: {``"    "`` } | :class:`str`
                Separator when *line* has less than *i* entries
            *delim_len*: {``len(delim)``} | :class:`int` > 0
                Number of spaces to use in default delimiter
            *vdef*: {``"_"``} | :class:`str`
                Default value if line needs additional entries
        :Outputs:
            *txt*: :class:`str`
                Modified *line* with entry *i* set to printed version of *val*
        :See also:
            * :func:`pyUS.inputInp.InputInp.ConvertToText`
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
       # --- Options ---
        # Alignment option
        align = kw.pop("align", "right")
        # Number of spaces in header
        indent = kw.pop("indent", 0)
        # Default separator
        delim = kw.pop("delimiter", kw.pop("delim", kw.pop("sep", "    ")))
        # Number of spaces in delimiter
        LD = kw.pop("delim_len", len(delim))
        # Default value
        vdef = kw.pop("vdef", kw.pop("val_def", "_"))
       # --- Inputs ---
        # Convert to string
        txt = self.ConvertToText(val, **kw)
        # Length of value
        L = len(txt)
       # --- Pad Lists ---
        # Split up line
        header, vals, LV, LS = self.ConvertLineToList(line, indent=indent)
        # Number of values found
        nval = len(vals)
        # Check length
        if i >= nval:
            # Number of additional values needed
            jval = i - nval + 1
            # Append empty values
            vals = vals + ([vdef] * jval)
            # Expand length lists
            LV = LV + ([len(vdef)] * jval)
            LS = LS + ([LD] * jval)
       # --- Text Updates ---
        # Updated length
        nval = len(vals)
        # Save new value
        vals[i] = txt
        # Update spaces, checking alignment option
        if align == "right":
            # Available string lengths
            L_avail = LS[i] + LV[i]
            # Check if we will overfill the slot
            LS[i] = max(1, L_avail - L)
        elif align == "center":
            # Check for end entry
            if i == nval:
                # Only spaces to the left (so far)
                L1 = LS[i]
                L2 = L1
            else:
                # Spaces on both sides
                L1 = LS[i]
                L2 = LS[i+1]
            # Length of current value
            L0 = LV[i]
            # Total available length
            L_avail = L1 + L0 + L2
            # Spaces left over
            L_ws = max(1, L_avail - L)
            # Calculate space on the left (attempt to keep L/R ratios)
            LS[i] = (L_ws * L1) // (L1 + L2)
            # Space on the right is whatever is left
            if i < nval:
                LS[i+1] = max(1, L_avail - L - LS[i])
        elif align == "right":
            # Check for end entry
            if i < nval:
                # Available string length
                L_avail = LS[i+1] + LV[i] - 1
                # Update space on the right, checking for overfills
                LS[i] = max(1, L_avail - L)
        else:
            raise ValueError(
                ("Alignment option '%s' unknown; " % align) +
                ("options are 'left', 'center', 'right'"))
       # --- Update ---
        # Convert gaps to spaces
        txt_ws = [" "*t for t in LS] 
        # Make a list of spaces followed by values
        txts = [txt_ws[i] + vals[i] for i in range(nval)]
        # Recreate the entire line
        line = header + "".join(txts) + "\n"
        # Output
        return line
        
    # Set a value in a space-separated table section
    def SetSectionTableValue(self, sec, row, col, val, **kw):
        """Set one value in a table-like space-separated section
        
        :Call:
            >>> inp.SetSectionTableValue(sec, row, col, val, **kw)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *row*: :class:`int` >= 0
                Index of non-comment line in section
            *col*: :class:`int` >= 0
                Index of entry in line of text to change/set
            *val*: :class:`any`
                Value to print to entry *i* of modified line
            *comment*: {``"!"``} | :class:`str`
                Character denoting start of comment line
            *skiprows*: {``0``} | :class:`int` >= 0
                Number of non-comment rows to not count towards *row*
            *align*: ``"left"`` | ``"center"`` | {``"right"``}
                Alignment option; determines output spacing
            *delimiter*, *delim*, *sep*: {``"    "`` } | :class:`str`
                Separator when *line* has less than *i* entries
            *delim_len*: {``len(delim)``} | :class:`int` > 0
                Number of spaces to use in default delimiter
            *vdef*: {``"_"``} | :class:`str`
                Default value if line needs additional entries
        :See also:
            * :func:`pyUS.inputInp.InputInp.SetLineValueSequential`
            * :func:`pyUS.inputInp.InputInp.ConvertToText`
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        # Get comment character
        comment = kw.pop("comment", kw.pop("comment_char", "!"))
        # Number of non-comment rows to skip
        skiprows = kw.pop("skiprows", 0)
        # Overall row number
        jrow = row + skiprows
        # Counter for number of lines found
        n = 0
        # Check if section is present
        if sec not in self.Section:
            # Add the section
            self.AddSection(sec)
        # Start with line *i0* and loop
        for (i, line) in enumerate(self.Section[sec][1:-1]):
            # Skip if comment
            if line.lstrip().startswith(comment):
                continue
            # Otherwise, increment counter
            n += 1
            # Otherwise, check count
            if n > jrow:
                break
        # Check if we actually found enough non-comment rows
        for j in range(n, jrow + 1):
            # Use blank line
            line = "\n"
            # Append to section (before EOS marker)
            self.Section[sec].insert(-1, line)
            # Update section line counter
            i += 1
        # We skipped section title row
        i += 1
        # Process line
        txt = self.SetLineValueSequential(line, col, val, **kw)
        # Save updated row
        self.Section[sec][i] = txt
        
    # Get a value from a space-separated table section
    def GetSectionTableValue(self, sec, row, col, vdef=None, **kw):
        """Get one value in a table-like space-separated section
        
        :Call:
            >>> inp.GetSectionTableValue(sec, row, col, val, **kw)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *row*: :class:`int` >= 0
                Index of non-comment line in section
            *col*: :class:`int` >= 0
                Index of entry in line of text to change/set
            *vdef*: {``None``} | :class:`any`
                Default value if line needs additional entries
            *comment*: {``"!"``} | :class:`str`
                Character denoting start of comment line
            *skiprows*: {``0``} | :class:`int` >= 0
                Number of non-comment rows to not count towards *row*
        :Outputs:
            *val*: {*vdef*} | :class:`any`
                Converted value if found; else *vdef*
        :See also:
            * :func:`cape.namelist.Namelist.ConvertToVal`
            * :func:`pyUS.inputInp.InputInp.SetSectionTableValue`
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        # Get comment character
        comment = kw.pop("comment", kw.pop("comment_char", "!"))
        # Number of non-comment rows to skip
        skiprows = kw.pop("skiprows", 0)
        # Overall row number
        jrow = row + skiprows
        # Counter for number of lines found
        n = 0
        # Check if section is present
        if sec not in self.Section:
            return vdef
        # Start with line *i0* and loop
        for (i, line) in enumerate(self.Section[sec][1:-1]):
            # Skip if comment
            if line.lstrip().startswith(comment):
                continue
            # Otherwise, increment counter
            n += 1
            # Otherwise, check count
            if n > jrow:
                break
        # Check if we actually found enough non-comment rows
        if n < jrow:
            return vdef
        # Number of spaces in header
        indent = kw.pop("indent", 0)
        # Split row into values
        vals = line[indent:].split()
        # Check if line has enough values
        if col >= len(vals):
            return vdef
        # Otherwise, convert value
        return self.ConvertToVal(vals[col])
        
  # --- Specific Settings ---
   # [CFD_SOLVER]
    # Generic parameter (get)
    def get_CFDSOLVER_key(self, key):
        """Get value of parameter *key* from ``CFD_SOLVER`` section
        
        :Call:
            >>> val = inp.get_CFDSOLVER_key(key)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *key*: :class:`str`
                Name of parameter
        :Outputs:
            *val*: :class:`int` | :class:`float` | ``None``
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        # Initialize column index
        col = None
        # Loop through rows
        for (i, row) in enumerate(self.CFD_SOLVER_keys):
            # Check if *k* is in row
            if key in row:
                # Get index
                col = row.index(key)
                # Stop searching
                break
        # If not found, raise exception
        if col is None:
            raise KeyError("CFD_SOLVER parameter '%s' not known" % k)
        # Otherwise, return the value
        return self.GetSectionTableValue("CFD_SOLVER", i, col)
    
    # Generic parameter (set)
    def set_CFDSOLVER_key(self, key, val):
        """Get value of parameter *key* from ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_key(key, val)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *key*: :class:`str`
                Name of parameter
            *val*: :class:`int` | :class:`float` | ``None``
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        # Initialize column index
        col = None
        # Loop through rows
        for (i, row) in enumerate(self.CFD_SOLVER_keys):
            # Check if *k* is in row
            if key in row:
                # Get index
                col = row.index(key)
                # Stop searching
                break
        # If not found, raise exception
        if col is None:
            raise KeyError("CFD_SOLVER parameter '%s' not known" % key)
        # Otherwise, return the value
        return self.SetSectionTableValue("CFD_SOLVER", i, col, val)
        
    # Hard-coded methods
    def get_CFDSOLVER_nstop(self):
        """Get *nstop* from ``CFD_SOLVER`` section
        
        :Call:
            >>> nstop = inp.get_CFDSOLVER_nstop()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *nstop*: :class:`int`
                Maximum iteration number
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 0, 0)
        
    def set_CFDSOLVER_nstop(self, nstop):
        """Set *nstop* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_nstop(nstop)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *nstop*: :class:`int`
                Maximum iteration number
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 0, 0, nstop)
    
    def get_CFDSOLVER_ires(self):
        """Get *ires* from ``CFD_SOLVER`` section
        
        :Call:
            >>> ires = inp.get_CFDSOLVER_ires()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *ires*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 0, 1)
        
    def set_CFDSOLVER_ires(self, ires):
        """Set *ires* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_nstop(nstop)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *ires*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 0, 1, ires)
    
    def get_CFDSOLVER_nplot(self):
        """Get *nplot* from ``CFD_SOLVER`` section
        
        :Call:
            >>> nplot = inp.get_CFDSOLVER_nplot()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *nplot*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 0, 2)
        
    def set_CFDSOLVER_nplot(self, nplot):
        """Set *nplot* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_nstop(nstop)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *nplot*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 0, 2, nplot)
    
    def get_CFDSOLVER_iconr(self):
        """Get *iconr* from ``CFD_SOLVER`` section
        
        :Call:
            >>> iconr = inp.get_CFDSOLVER_iconr()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *iconr*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 0, 3)
        
    def set_CFDSOLVER_iconr(self, iconr):
        """Set *nplot* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_iconr(iconr)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *iconr*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 0, 3, iconr)
    
    def get_CFDSOLVER_impl(self):
        """Get *impl* from ``CFD_SOLVER`` section
        
        :Call:
            >>> impl = inp.get_CFDSOLVER_impl()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *impl*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 0, 4)
        
    def set_CFDSOLVER_impl(self, impl):
        """Set *impl* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_impl(impl)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *impl*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 0, 4, impl)
    
    def get_CFDSOLVER_kmax(self):
        """Get *kmax* from ``CFD_SOLVER`` section
        
        :Call:
            >>> kmax = inp.get_CFDSOLVER_kmax()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *kmax*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 0, 5)
        
    def set_CFDSOLVER_kmax(self, nplot):
        """Set *kmax* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_kmax(kmax)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *kmax*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 0, 5, kmax)
    
    def get_CFDSOLVER_kmaxo(self):
        """Get *kmaxo* from ``CFD_SOLVER`` section
        
        :Call:
            >>> kmaxo = inp.get_CFDSOLVER_kmaxo()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *kmaxo*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 0, 6)
        
    def set_CFDSOLVER_kmaxo(self, nplot):
        """Set *kmaxo* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_kmaxo(kmaxo)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *kmaxo*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 0, 6, ires)
    
    def get_CFDSOLVER_ivisc(self):
        """Get *ivisc* from ``CFD_SOLVER`` section
        
        :Call:
            >>> ivisc = inp.get_CFDSOLVER_ivisc()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *ivisc*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 1, 0)
        
    def set_CFDSOLVER_ivisc(self, ivisc):
        """Set *ivisc* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_(ivisc)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *ivisc*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 1, 0, ivisc)
    
    def get_CFDSOLVER_ivib(self):
        """Get *ivib* from ``CFD_SOLVER`` section
        
        :Call:
            >>> ivib = inp.get_CFDSOLVER_ivib()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *ivib*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 1, 1)
        
    def set_CFDSOLVER_ivib(self, nplot):
        """Set *ivib* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_ivib(ivib)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *ivib*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 1, 1, ivib)
    
    def get_CFDSOLVER_ichem(self):
        """Get *ichem* from ``CFD_SOLVER`` section
        
        :Call:
            >>> ichem = inp.get_CFDSOLVER_ichem()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *ichem*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 1, 2)
        
    def set_CFDSOLVER_ichem(self, nplot):
        """Set *ichem* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_ichem(ichem)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *ichem*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 1, 2, ichem)
    
    def get_CFDSOLVER_itrb(self):
        """Get *itrb* from ``CFD_SOLVER`` section
        
        :Call:
            >>> itrb = inp.get_CFDSOLVER_itrb()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *itrb*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 1, 3)
        
    def set_CFDSOLVER_itrb(self, itrb):
        """Set *itrb* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_itrb(itrb)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *itrb*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 1, 3, itrb)
    
    def get_CFDSOLVER_ibase(self):
        """Get *ibase* from ``CFD_SOLVER`` section
        
        :Call:
            >>> ibase = inp.get_CFDSOLVER_ibase()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *ibase*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 1, 4)
        
    def set_CFDSOLVER_ibase(self, ibase):
        """Set *ibase* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_ibase(ibase)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *ibase*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 1, 4, ibase)
    
    def get_CFDSOLVER_idiss_g(self):
        """Get *idiss_g* from ``CFD_SOLVER`` section
        
        :Call:
            >>> idiss_g = inp.get_CFDSOLVER_idiss_g()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *idiss_g*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 1, 5)
        
    def set_CFDSOLVER_idiss_g(self, nplot):
        """Set *idiss_g* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_idiss_g(idiss_g)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *idiss_g*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 1, 5, idiss_g)
    
    def get_CFDSOLVER_ivmod(self):
        """Get *ivmod* from ``CFD_SOLVER`` section
        
        :Call:
            >>> ivmod = inp.get_CFDSOLVER_ivmod()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *ivmod*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 2, 0)
        
    def set_CFDSOLVER_ivmod(self, ivmod):
        """Set *ivmod* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_ivmod(ivmod)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *ivmod*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 2, 0, ivmod)
    
    def get_CFDSOLVER_ikmod(self):
        """Get *ikmod* from ``CFD_SOLVER`` section
        
        :Call:
            >>> ikmod = inp.get_CFDSOLVER_ikmod()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *ikmod*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 2, 1)
        
    def set_CFDSOLVER_ikmod(self, ikmod):
        """Set *ikmod* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_(ikmod)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *ikmod*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 2, 1, ikmod)
    
    def get_CFDSOLVER_idmod(self):
        """Get *idmod* from ``CFD_SOLVER`` section
        
        :Call:
            >>> idmod = inp.get_CFDSOLVER_idmod()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *idmod*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 2, 2)
        
    def set_CFDSOLVER_idmod(self, idmod):
        """Set *idmod* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_idmod(idmod)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *idmod*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 2, 2, idmod)
    
    def get_CFDSOLVER_ikv(self):
        """Get *ikv* from ``CFD_SOLVER`` section
        
        :Call:
            >>> ikv = inp.get_CFDSOLVER_ikv()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *ikv*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 2, 3)
        
    def set_CFDSOLVER_ikv(self, ikv):
        """Set *ikv* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_ikv(ikv)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *ikv*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 2, 3, ikv)
    
    def get_CFDSOLVER_icfl(self):
        """Get *icfl* from ``CFD_SOLVER`` section
        
        :Call:
            >>> icfl = inp.get_CFDSOLVER_icfl()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *icfl*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 2, 4)
        
    def set_CFDSOLVER_icfl(self, icfl):
        """Set *icfl* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_icfl(icfl)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *icfl*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 2, 4, icfl)
    
    def get_CFDSOLVER_dtfix(self):
        """Get *dtfix* from ``CFD_SOLVER`` section
        
        :Call:
            >>> dtfix = inp.get_CFDSOLVER_dtfix()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *dtfix*: :class:`float`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 2, 5)
        
    def set_CFDSOLVER_dtfix(self, dtfix):
        """Set *dtfix* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_dtfix(dtfix)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *dtfix*: :class:`float`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 2, 5, dtfix)
    
    def get_CFDSOLVER_iorder(self):
        """Get *iorder* from ``CFD_SOLVER`` section
        
        :Call:
            >>> iorder = inp.get_CFDSOLVER_iorder()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *iorder*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 3, 0)
        
    def set_CFDSOLVER_iorder(self, iorder):
        """Set *iorder* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_iorder(iorder)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *iorder*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 3, 0, iorder)
    
    def get_CFDSOLVER_iuem(self):
        """Get *iuem* from ``CFD_SOLVER`` section
        
        :Call:
            >>> iuem = inp.get_CFDSOLVER_iuem()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *iuem*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 3, 1)
        
    def set_CFDSOLVER_iuem(self, iuem):
        """Set *iuem* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_iuem(iuem)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *iuem*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 3, 1, iuem)
    
    def get_CFDSOLVER_ikve(self):
        """Get *nplot* from ``CFD_SOLVER`` section
        
        :Call:
            >>> ikve = inp.get_CFDSOLVER_ikve()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *ikve*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 3, 2)
        
    def set_CFDSOLVER_ikve(self, ikve):
        """Set *ikve* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_ikve(ikve)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *ikve*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 3, 2, ikve)
    
    def get_CFDSOLVER_kbl(self):
        """Get *kbl* from ``CFD_SOLVER`` section
        
        :Call:
            >>> kbl = inp.get_CFDSOLVER_kbl()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *kbl*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 3, 3)
        
    def set_CFDSOLVER_kbl(self, kbl):
        """Set *kbl* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_kbl(kbl)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *kbl*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 3, 3, kbl)
    
    def get_CFDSOLVER_iman(self):
        """Get *iman* from ``CFD_SOLVER`` section
        
        :Call:
            >>> iman = inp.get_CFDSOLVER_iman()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *iman*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 3, 4)
        
    def set_CFDSOLVER_iman(self, iman):
        """Set *iman* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_iman(iman)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *iman*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 3, 4, iman)
    
    def get_CFDSOLVER_npfac(self):
        """Get *npfac* from ``CFD_SOLVER`` section
        
        :Call:
            >>> npfac = inp.get_CFDSOLVER_npfac()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *npfac*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 4, 0)
        
    def set_CFDSOLVER_npfac(self, npfac):
        """Set *npfac* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_npfac(npfac)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *npfac*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 4, 0, npfac)
    
    def get_CFDSOLVER_npvol(self):
        """Get *npvol* from ``CFD_SOLVER`` section
        
        :Call:
            >>> npvol = inp.get_CFDSOLVER_npvol()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *npvol*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 4, 1)
        
    def set_CFDSOLVER_npvol(self, npvol):
        """Set *npvol* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_npvol(npvol)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *npvol*: :class:`int`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 4, 1, npvol)
    
    def get_CFDSOLVER_cfl(self):
        """Get *cfl* from ``CFD_SOLVER`` section
        
        :Call:
            >>> cfl = inp.get_CFDSOLVER_cfl()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *cfl*: :class:`float`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 5, 0)
        
    def set_CFDSOLVER_cfl(self, cfl):
        """Set *cfl* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_cfl(cfl)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *cfl*: :class:`float`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 5, 0, cfl)
    
    def get_CFDSOLVER_epsj(self):
        """Get *epsj* from ``CFD_SOLVER`` section
        
        :Call:
            >>> epsj = inp.get_CFDSOLVER_epsj()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *epsj*: :class:`float`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 5, 1)
        
    def set_CFDSOLVER_epsj(self, epsj):
        """Set *epsj* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_epsj(epsj)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *epsj*: :class:`float`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 5, 1, epsj)
    
    def get_CFDSOLVER_wdis(self):
        """Get *wdis* from ``CFD_SOLVER`` section
        
        :Call:
            >>> wdis = inp.get_CFDSOLVER_wdis()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *wdis*: :class:`float`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.GetSectionTableValue("CFD_SOLVER", 5, 2)
        
    def set_CFDSOLVER_wdis(self, wdis):
        """Set *wdis* in ``CFD_SOLVER`` section
        
        :Call:
            >>> inp.set_CFDSOLVER_wdis(wdis)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *wdis*: :class:`float`
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        return self.SetSectionTableValue("CFD_SOLVER", 5, 2, wdis)
   # [/CFD_SOLVER]
   
   # [CFD_SOLVER_OPTS]
    # Generic parameter (get)
    def get_CFDSOLVEROPTS_key(self, key):
        """Get parameter *key* from ``CFD_SOLVER_OPTS`` section
        
        :Call:
            >>> val = inp.get_CFDSOLVEROPTS_key(key)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *key*: :class:`str`
                Name of parameter
        :Outputs:
            *val*: :class:`int` | :class:`float` | ``None``
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        # Output
        return self.GetVar("CFD_SOLVER_OPTS", key)

    # Generic parameter (set)
    def set_CFDSOLVEROPTS_key(self, key, val):
        """Get parameter *key* from ``CFD_SOLVER_OPTS`` section
        
        :Call:
            >>> val = inp.get_CFDSOLVEROPTS_key(key)
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
            *key*: :class:`str`
                Name of parameter
            *val*: :class:`int` | :class:`float` | ``None``
                Value in ``input.inp`` file
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        # Output
        return self.SetVar("CFD_SOLVER_OPTS", key, val, tab="")
   # [/CFD_SOLVER_OPTS]
   
   # [CFD_BCS]
    # Read BCs table
    def ReadBCs(self):
        """Read boundary condition table
        
        :Call:
            >>> BCs = inp.ReadBCs()
        :Inputs:
            *inp*: :class:`pyUS.inputInp.InputInp`
                Namelist file control instance
        :Outputs:
            *BCs*: :class:`dict`
                Dictionary of *zone*, *bcn*, *igrow*, *name*, and
                *params* for each boundary condition
        :Attributes:
            *inp.BCNames*: :class:`list`\ [:class:`str`]
                List of boundary condition names
            *inp.BCTable*: *BCs*
                Boundary condition properties
            *inp.BCTable_rows*: :class:`int`
                Number of rows in the BC table section
        :Versions:
            * 2019-06-06 ``@ddalle``: First version
        """
        # Name of section
        sec = "CFD_BCS"
        # Check if section is present
        if sec not in self.Section:
            # Don't update tables
            return self.BCTable
        # Initialize properties
        BCTable = {}
        BCNames = []
        BCRows = {}
        BC_Y = {}
        BC_cos = {}
        # Number of zones found
        nzone = 0
        # Loop through rows
        for (i, line) in enumerate(self.Section[sec][1:-1]):
            # Check if line is a comment
            if line.lstrip().startswith("!"):
                # Comment
                continue
            elif line.strip() == "":
                # Empty line
                continue
            # Check if table is over
            if line.strip() == "done":
                # End of table
                self.BCTable_rows = i + 1
                break
            # Otherwise, process the line
            V = line.strip().split()
            # Check line length
            if len(V) < 4:
                raise ValueError(
                    "Boundary condition %i has only %i/%i required columns"
                    % (nzone+1, len(V), 4))
            # Get name
            name = V[3].strip('"').strip("'")
            # Append to list
            BCNames.append(name)
            # Save required parameters
            BCTable[name] = {
                "row":    i + 1,
                "zone":   self.ConvertToVal(V[0]),
                "bcn":    self.ConvertToVal(V[1]),
                "igrow":  self.ConvertToVal(V[2]),
                "params": " ".join(V[4:]),
            }
        # Loop through remaining rows
        # Save values
        self.BCTable = BCTable
        self.BCNames = BCNames
        # Output
        return BCTable

    # Read mass fractoins
    def GetBCMassFraction(self, name, i=None):
        
        # Name of section
        sec = "CFD_BCS"
        # Check if section is present
        if sec not in self.Section:
            # Nothing to search
            return
        # Regular expression to search for
        regex = "^\s*['\"]%s['\"]" % name
        # Use section searcher for lines starting with whitespace plus name
        lines = self.GetLineInSectionSearch(sec, regex, 1)
        # Check for a match
        if len(lines) < 1:
            return
        # Separate line into space-separated values
        txts = lines[0].split()[1:]
        # Evaluate each entry
        vals = [self.ConvertToVal(txt) for txt in txts]
        # Check for index
        if i is None:
            # Return entire
            return vals
        elif i < len(vals):
            # Return indexed value
            return vals[i]
            
   # [/CFD_BCS]
# class InputInp

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
        *nml*: :class:`cape.namelist.Namelist`
            Namelist file control instance
        *nml.Sections*: :class:`dict` (:class:`list` (:class:`str`))
            Dictionary of sections containing contents of each namelist
        *nml.SectionNames*: :class:`list` (:class:`str`)
            List of section names
    :Version:
        * 2015-10-15 ``@ddalle``: Started
    """
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
            >>> val = nml.ConvertToText(v)
        :Inputs:
            *nml*: :class:`cape.namelist.Namelist`
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
            
  # --- Data ---
    # Convert a single line of text to values
    def ConvertLineToList(self, line, **kw):
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
        
        
    # Convert a section to a list of literals
    def GetSectionAsTable(self, sec, i, i0=0, comment="!"):
        """Convert a line of space-separated data into converted values
        
        """
        # Counter for number of lines found
        n = 0
        # Start with line *i0* and loop
        for line in self.Section[sec][i0:]:
            # Skip if comment
            if line.lstrip().startswith(comment):
                continue
    
# class InputInp

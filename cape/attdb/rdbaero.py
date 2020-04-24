#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.attdb.rdbaero`: Database Template for Aerospace Variables
======================================================================

This module contains extensions and extra methods that are common to
many (atmospheric) aerospace databases.  This includes definitions for
variables like "angle of attack," "total angle of attack," etc.


"""



# CAPE modules
import cape.tnakit.kwutils as kwutils

# ATTDB modules
from . import rdb
from . import convert


# Class definition
class AeroDataKit(rdb.DataKit):
    r""":class:`DataKit` extension for aerospace applications

    :Call:
        >>> db = AeroDataKit(fname=None, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            File name; extension is used to guess data format
        *csv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVFile` read
        *textdata*: {``None``} | :class:`str`
            Explicit file name for :class:`TextDataFile`
        *simplecsv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVSimple`
        *xls*: {``None``} | :class:`str`
            File name for :class:`XLSFile`
        *mat*: {``None``} | :class:`str`
            File name for :class:`MATFile`
    :Outputs:
        *db*: :class:`cape.attdb.rdbaero.AeroDataKit`
            Generic database
    :Versions:
        * 2020-03-19 ``@ddalle``: First version
    """
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Tags ---
    # Built-in list of default tags based on column name
    _tagmap = {
        "ALPH":      "alpha",
        "ALPHA":     "alpha",
        "ALPHA":     "alpha",
        "ALPHA_T":   "aoap",
        "AOA":       "alpha",
        "AOAP":      "aoap",
        "AOAV":      "aoav",
        "Alpha":     "alpha",
        "Alpha_T":   "aoap",
        "Alpha_t":   "aoap",
        "BETA":      "beta",
        "Beta":      "beta",
        "MACH":      "mach",
        "Mach":      "mach",
        "PHI":       "phip",
        "PHIP":      "phip",
        "PHIV":      "phiv",
        "Phi":       "phip",
        "RE":        "Re",
        "REY":       "Re",
        "REYNODLDS": "Re",
        "Rey":       "Re",
        "Re":        "Re",
        "T":         "T",
        "Tinf":      "T",
        "alph":      "alpha",
        "alpha":     "alpha",
        "alpha_t":   "aoap",
        "aoa":       "alpha",
        "aoap":      "aoap",
        "aos":       "beta",
        "beta":      "beta",
        "mach":      "mach",
        "phi":       "phip",
        "phip":      "phip",
        "phiv":      "phiv",
        "q":         "q",
        "qbar":      "q",
        "qinf":      "q",
    }

    # Tags that could be computed using other tags
    _tagsubmap = {
        "alpha": {"aoap", "aoav", "phip", "phiv"},
        "aoap": {"alpha", "aoav", "beta", "phiv"},
        "aoav": {"alpha", "aoap", "beta", "phip"},
        "beta": {"aoap", "aoav", "phip", "phiv"},
        "phip": {"alpha", "aoav", "beta", "phiv"},
        "phiv": {"alpha", "aoap", "beta", "phip"},
    }
  # >

  # ================
  # Conversions
  # ================
  # <
   # --- Angle of Attack ---
    # Create angles of attack and sideslip
    def make_alpha_beta(self, col1="alpha", col2="beta"):
        r"""Build *alpha* and *beta* cols if necessary and possible

        :Call:
            >>> db.make_alpha_beta(col1="alpha", col2="beta")
        :Inputs:
            *db*: :class:`cape.attdb.rdbaero.AeroDataKit`
                Data container with aerospace tags
            *col1*: {``"alpha"``} | :class:`str`
                Name of new column for angle of attack
            *col2*: {``"beta"``} | :class:`str`
                Name of new column for angle of sideslip
        :Effects:
            *db[col1]*: :class:`np.ndarray`
                Created if no other key with ``"alpha"`` tag present
            *db[col2]*: :class:`np.ndarray`
                Created if no other key with ``"beta"`` tag present
        :Versions:
            * 2020-03-18 ``@ddalle``: First version
        """
        # Get *alpha* and *beta* keys, if any
        ka = self.get_col_by_tag("alpha")
        kb = self.get_col_by_tag("beta")
        # Check if present
        if not (ka is None or kb is None):
            # Nothing to do; already computed
            return
        # Get *aoap* and *phip* keys, if any
        kaoap = self.get_col_by_tag("aoap")
        kphip = self.get_col_by_tag("phip")
        # Try *aoav* and *phiv*
        kaoav = self.get_col_by_tag("aoav")
        kphiv = self.get_col_by_tag("phiv")
        # Check if usable
        if kaoap and kphip:
            # Get values from total angle of attack
            aoap = self.get_all_values(kaoap)
            phip = self.get_all_values(kphip)
            # Convert
            a, b = convert.AlphaTPhi2AlphaBeta(aoap, phip)
        elif kaoav and kphiv:
            # Get values from maneuver angle of attack
            aoav = self.get_all_values(kaoav)
            phiv = self.get_all_values(kphiv)
            # Convert
            a, b = convert.AlphaTPhi2AlphaBeta(aoav, phiv)
        else:
            # No conversion possible
            return
        # Save them (using default key names)
        self.save_col(col1, a)
        self.save_col(col2, b)
        # Create definitions
        self.create_defn(col1, a, Tag="alpha")
        self.create_defn(col2, b, Tag="beta")

    # Create total angle of attack and roll
    def make_aoap_phip(self, col1="aoap", col2="phip"):
        r"""Build *aoap* and *phip* if necessary and possible

        :Call:
            >>> db.make_aoap_phip(col1="aoap", col2="phip")
        :Inputs:
            *db*: :class:`cape.attdb.rdbaero.AeroDataKit`
                Data container with aerospace tags
            *col1*: {``"aoap"``} | :class:`str`
                Name of new column for total angle of attack
            *col2*: {``"phip"``} | :class:`str`
                Name of new column for missile-axis roll
        :Effects:
            *db[col1]*: :class:`np.ndarray`
                Created if no other key with ``"aoap"`` tag present
            *db[col2]*: :class:`np.ndarray`
                Created if no other key with ``"phip"`` tag present
        :Versions:
            * 2020-03-18 ``@ddalle``: First version
        """
        # Get *aoap* and *phip* keys, if any
        kaoap = self.get_col_by_tag("aoap")
        kphip = self.get_col_by_tag("phip")
        # Check if present
        if not (kaoap is None or kphip is None):
            # Nothing to do; already computed
            return
        # Get *alpha* and *beta* keys, if any
        ka = self.get_col_by_tag("alpha")
        kb = self.get_col_by_tag("beta")
        # Try *aoav* and *phiv*
        kaoav = self.get_col_by_tag("aoav")
        kphiv = self.get_col_by_tag("phiv")
        # Check if usable
        if ka and kb:
            # Get values from angle of attack and sideslip
            a = self.get_all_values(ka)
            b = self.get_all_values(kb)
            # Convert
            aoap, phip = convert.AlphaBeta2AlphaTPhi(a, b)
        elif kaoav and kphiv:
            # Get values from maneuver angle of attack
            aoav = self.get_all_values(kaoav)
            phiv = self.get_all_values(kphiv)
            # Convert
            aoap, phip = convert.AlphaMPhi2AlphaTPhi(aoav, phiv)
        else:
            # No conversion possible
            return
        # Save them (using default key names)
        self.save_col(col1, aoap)
        self.save_col(col2, phip)
        # Create definitions
        self.create_defn(col1, aoap, Tag="aoap")
        self.create_defn(col2, phip, Tag="phip")

    # Create missile-axis angle of attack and roll
    def make_aoav_phiv(self, col1="aoav", col2="phiv"):
        r"""Build *aoav* and *phiv* if necessary and possible

        :Call:
            >>> db.make_aoav_phiv(col1="aoav", col2="phiv")
        :Inputs:
            *db*: :class:`cape.attdb.rdbaero.AeroDataKit`
                Data container with aerospace tags
            *col1*: {``"aoav"``} | :class:`str`
                Name of new column for missile-axis angle of attack
            *col2*: {``"phiv"``} | :class:`str`
                Name of new column for missile-axis roll angle
        :Effects:
            *db[col1]*: :class:`np.ndarray`
                Created if no other key with ``"aoav"`` tag present
            *db[col2]*: :class:`np.ndarray`
                Created if no other key with ``"phiv"`` tag present
        :Versions:
            * 2020-03-18 ``@ddalle``: First version
        """
        # Get *aoav* and *phiv* keys, if any
        kaoav = self.get_col_by_tag("aoav")
        kphiv = self.get_col_by_tag("phiv")
        # Check if present
        if not (kaoav is None or kphiv is None):
            # Nothing to do; already computed
            return
        # Get *alpha* and *beta* keys, if any
        ka = self.get_col_by_tag("alpha")
        kb = self.get_col_by_tag("beta")
        # Try *aoap* and *phip*
        kaoap = self.get_col_by_tag("aoap")
        kphip = self.get_col_by_tag("phip")
        # Check if usable
        if ka and kb:
            # Get values from angle of attack and sideslip
            a = self.get_all_values(ka)
            b = self.get_all_values(kb)
            # Convert
            aoav, phiv = convert.AlphaBeta2AlphaMPhi(a, b)
        elif kaoap and kphip:
            # Get values from maneuver angle of attack
            aoap = self.get_all_values(kaoap)
            phip = self.get_all_values(kphip)
            # Convert
            aoav, phiv = convert.AlphaTPhi2AlphaMPhi(aoap, phip)
        else:
            # No conversion possible
            return
        # Save them (using default key names)
        self.save_col(col1, aoav)
        self.save_col(col2, phiv)
        # Create definitions
        self.create_defn(col1, aoav, Tag="aoav")
        self.create_defn(col2, phiv, Tag="phiv")
  # >


# Combine options
kwutils._combine_val(AeroDataKit._tagmap, rdb.DataKit._tagmap)
kwutils._combine_val(AeroDataKit._tagsubmap, rdb.DataKit._tagsubmap)

# Invert the _tagmap
AeroDataKit.create_tagcols()

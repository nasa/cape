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
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Tags ---
    _tagmap = {
        "ALPHA":   "alpha",
        "ALPHA":   "alpha",
        "ALPHA_T": "aoap",
        "AOAP":    "aoap",
        "Alpha":   "alpha",
        "Alpha_T": "aoap",
        "Alpha_t": "aoap",
        "BETA":    "beta",
        "Beta":    "beta",
        "MACH":    "mach",
        "Mach":    "mach",
        "PHI":     "phip",
        "PHIP":    "phip",
        "PHIV":    "phiv",
        "Phi":     "phip",
        "alpha":   "alpha",
        "aoa":     "alpha",
        "aoap":    "aoap",
        "aos":     "beta",
        "beta":    "beta",
        "mach":    "mach",
        "phi":     "phip",
        "phip":    "phip",
        "phiv":    "phiv",
    }
  # >

  # ================
  # Conversions
  # ================
  # <
   # --- Angle of Attack ---
    def make_alpha_beta(self):
        r"""Build *alpha* and *beta* cols exist, if possible

        :Call:
            >>> db.make_alpha_beta()
        :Inputs:
            *db*: :class:`cape.attdb.rdbaero.AeroDataKit`
                Data container with aerospace tags
        :Effects:
            *db["alpha"]*: :class:`np.ndarray`
                Created if no other key with ``"alpha"`` tag present
            *db["beta"]*: :class:`np.ndarray`
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
        # Check if usable
        if kaoap and kphip:
            # Get values
            aoap = self.get_all_values(kaoap)
            phip = self.get_all_values(kphip)
            # Convert
            a, b = convert.AlphaTPhi2AlphaBeta(aoap, phip)
            # Save them (using default key names)
            self.save_col("alpha", a)
            self.save_col("beta", b)
            # Done
            return
        # Try *aoav* and *phiv*
        kaoav = self.get_col_by_tag("aoav")
        kphiv = self.get_col_by_tag("phiv")
        # Check if usable
        if kaoav and kphiv:
            # Get values
            aoav = self.get_all_values(kaoav)
            phiv = self.get_all_values(kphiv)
            # Convert
            a, b = convert.AlphaTPhi2AlphaBeta(aoav, phiv)
            # Save them (using default key names)
            self.save_col("alpha", a)
            self.save_col("beta", b)
        
  # >


# Combine options
kwutils._combine_val(AeroDataKit, rdb.DataKit._tagmap)

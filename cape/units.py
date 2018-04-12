#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.units`: Unit Conversion Module
==========================================

CAPE includes this unit-conversion module to interpret various units for inputs
and post-processing.  Many CFD solvers, and some trajectory tools that are used
to provide flight conditions, expect imperial units, which can be challenging
to interpret and difficult to use in complications.

The main function of this module is called :func:`mks`, which takes a string as
input and tries to convert it to a numerical output as multiple of meters,
kilograms, and seconds (and Kelvins and amperes or several other fundamental
units as needed). For example it will return 0.0254 if the input is ``"inch"``,
which is the length of an inch in meters.

    .. code-block:: pycon
    
        >>> mks("inch")
        0.0254

This function also takes more complexly defined units using exponents,
multiples and ratios, and SI prefixes.  For example, this converts thousand
pounds per square foot to the appropriate mks unit, which is Pascals.

    .. code-block:: pycon
    
        >>> mks("klbf/ft^2")
        4.7880e+04

The module also contains several simple :class:`float` values, such as *psf*
for the value of pounds per square feet in Pascals.  There is a :class:`dict`
called *units_dict* that contains a full dictionary of recognized unit names.

    .. code-block:: pycon
    
        >>> units_dict['psf']
        47.88025903135141

"""

# System modules
import re
import numpy as np

# List of SI prefixes
prefix_dict = {
   'Y' : 1e24 , # yotta
   'Z' : 1e21 , # zetta
   'E' : 1e18 , # exa
   'P' : 1e15 , # peta
   'T' : 1e12 , # tera
   'G' : 1e9  , # giga
   'M' : 1e6  , # mega
   'k' : 1e3  , # kilo
   'h' : 1e2  , # hecto
   'D' : 1e1  , # deca
   'da': 1e1  , # deca
   'd' : 1e-1 , # deci
   'c' : 1e-2 , # centi
   'm' : 1e-3 , # milli
   u'μ': 1e-6 , # micro
   'u' : 1e-6 , # micro
   'n' : 1e-9 , # nano
   'p' : 1e-12, # pico
   'f' : 1e-15, # femto
   'a' : 1e-18, # atto
   'z' : 1e-21, # zepto
   'y' : 1e-24, # yocto
}
# List of prefix names
prefix_name = prefix_cell.keys()

# Basis units
m    = 1.0
kg   = 1.0
s    = 1.0
K    = 1.0
A    = 1.0
# Number
mol  = 1e-3
kmol = 1e3*mol
NA   = 6.022141793e23/mol
# Angular measure
rad  = 1.0
deg  = np.pi/180*rad
rev  = 2*np.pi * rad
# Length units
inch = 0.0254*m
ft   = 12*inch
yrd  = 3*ft
mi   = 5280*ft
nmi  = 1852
# Area units
acre = mi*mi/640
ha   = 10000*m*m
# Volume
L    = 0.001*m**3
gal  = 231*inch**3
# Time
hr   = 3600*s
day  = 24*hr
yr   = 365.242199*day
a    = 365.25*day
# Speed
kn   = nmi/hr
# Astronomical units
c    = 299792458*m/s
ly   = c*a
AU   = 1.4960e11
pc   = 1/np.tan(np.pi/(3600*180))*AU/ly
# Acceleration
g0   = 9.80665*m/(s*s)
# Force
N    = kg*m/(s*s)
lbf  = 4.44822162*N
kgf  = g0*kg
# Mass
slug = lbf*s*s/ft
lbm  = 0.45359237*kg
# Pressure
Pa   = N/(m*m)
atm  = 101325*Pa
torr = atm/760
psi  = lbf/(inch*inch)
psf  = lbf/(ft*ft)
# Energy
J    = N*m
eV   = 1.602176531*J
cal  = 4.184*J
Btu  = 1054.35026444*J
# Power
W    = J/s
hp   = 550*ft*lbf/s
# Temperature (no offsets)
R    = 5./9.*K
# Charge
C    = A*s
e    = 1.6021764874e-19*C
# Electric potential
V    = J/C
# Capacitance
F    = C/V
# Resistance and conductance
ohm  = V/A
mho  = A/V

# Dictionary of basic units and their values in mks
units_dict = {
   '1'        : 1        , # number
   'm'        : m        , # meter
   'g'        : 1e-3*kg  , # gram
   's'        : s        , # second
   'K'        : K        , # Kelvin
   'A'        : A        , # Ampere
   'rad'      : rad      , # radian
   'deg'      : deg      , # degree
   'rev'      : rev      , # revolution
   'rpm'      : rev/60/s , # revolutions per minute
   'mol'      : mol      , # mol
   'kmol'     : kmol     , # kmol
   'NA'       : NA       , # Avagardro's number
   u'å'       : 1e-10*m  , # angstrom
   'in'       : inch     , # inch
   'inch'     : inch     , # inch
   'ft'       : ft       , # foot
   'yd'       : yrd      , # yard
   'yrd'      : yrd      , # yard
   'mi'       : mi       , # mile
   'nmi'      : nmi      , # nautical mile
   'furlong'  : 660*ft   , # furlong
   'acre'     : acre     , # acre
   'ha'       : ha       , # hectare
   'L'        : L        , # Litre
   'gal'      : gal      , # gallon
   'min'      : 60*s     , # minute
   'hr'       : hr       , # hour
   'day'      : day      , # day
   'days'     : day      , # days
   'fortnight': 14*day   , # fortnight
   'yr'       : yr       , # year
   'yrs'      : yr       , # year
   'year'     : yr       , # year
   'years'    : yr       , # year
   'a'        : a        , # annum
   'c'        : c        , # speed of light
   'ly'       : ly       , # light year
   'AU'       : AU       , # astronomical unit
   'pc'       : pc       , # parsec
   'mph'      : mi/hr    , # mile per hour
   'fps'      : ft/s     , # foot per second
   'kn'       : kn       , # knot (nautical mile per hour)
   'kt'       : kn       , # knot (nautical mile per hour)
   'kts'      : kn       , # knot (nautical mile per hour)
   'g0'       : g0       , # reference acceleration due to gravity
   'N'        : N        , # Newton
   'dyn'      : 1e-5*N   , # dyne
   'lb'       : lbf      , # pound (force)
   'lbf'      : lbf      , # pound force
   'oz'       : lbf/16   , # ounce (weight)
   'kgf'      : kgf      , # kilogram force
   'slug'     : slug     , # slug
   'lbm'      : lbm      , # pound mass
   'st'       : 14*lbm   , # stone
   'Pa'       : Pa       , # Pascal
   'atm'      : atm      , # atmosphere
   'torr'     : torr     , # torr
   'Torr'     : torr     , # torr
   'psi'      : psi      , # pounds per square inch
   'psf'      : psf      , # pounds per square foot
   'ksi'      : 1e3*psi  , # kilopounds per square inch
   'bar'      : 1e5*Pa   , # bar
   'J'        : J        , # Joule
   'erg'      : 1e-7*J   , # erg
   'eV'       : eV       , # electron Volt
   'cal'      : cal      , # calorie
   'Btu'      : Btu      , # British thermal unit
   'BTU'      : Btu      , # British thermal unit
   'btu'      : Btu      , # British thermal unit
   'W'        : W        , # Watt
   'Wh'       : W*hr     , # Watt-hour
   'hp'       : hp       , # horsepower
   'R'        : R        , # Rankine
   'Hz'       : 1/s      , # Herz
   'C'        : C        , # Coulomb
   'e'        : e        , # charge of an electron
   'V'        : V        , # Volt
   'F'        : F        , # Faraday
   'ohm'      : ohm      , # Ohm
   'Ohm'      : ohm      , # Ohm
   'mho'      : mho      , # Siemens
   'S'        : mho      , # Siemens
}
# Unit names
units_name = units_dict.keys()

# Primary function
def mks(unit_lbl):
    """Convert a string identifying a certain unit to numerical MKS value
    
    :Call:
        >>> val = mks(unit_lbl)
    :Examples:
        >>> mks("kg")
        1.0
        >>> mks("lbf/ft^2")
        47.880
    :Inputs:
        *unit_lbl*: :class:`str` | :class:`unicode`
            String describing units
    :Outputs:
        *val*: :class:`float`
            Units converted to numerical value assuming MKS system
    :Versions:
        * 2018-04-12 ``@dalle``: First version
    """
    # Localize input
    lbl = unicode(unit_lbl)
    # Replace exponent characters
    lbl = re.sub(u'¹', '', lbl)
    lbl = re.sub(u'²', '^2', lbl)
    lbl = re.sub(u'³', '^3', lbl)
    # Degree sign
    lbl = re.sub(u'°', " deg", lbl)
    # Replace fractions
    lbl = re.sub(u'½', '0.50', lbl)
    lbl = re.sub(u'¼', '0.25', lbl)
    lbl = re.sub(u'¾', '0.75', lbl)
    # Replace multiplication symbols with a space
    lbl = re.sub(u'[\*×]', ' ', lbl)
    # Substitute division signs (unlikely, right?)
    lbl = re.sub(u'÷', '/', lbl)
    # Replace double asterisk with '^'
    lbl = re.sub('\*\*', '^', lbl)
    
    # Remove extra spaces around '^' symbols
    lbl = re.sub('\s*\/+\s*', '/', lbl)
    # Place a space in front of each slash
    lbl = re.sub('\/', ' /', lbl)
    
    # Initialize the output value
    val = 1.0
    # Split off each chunk in the form '[/][prefix](unit)[^exponent]'
    lbls = [lbli.strip() for lbli in lbl.split()]
    # Loop through labels
    for lbli in lbls:
        # Check for an exponent character
        if '^' in lbli:
            # Split on exponent
            lbli, expi = lbli.split('^')
        else:
            # No exponent
            expi = '1'
        # Evaluate exponent
        powi = float(expi)
        
        # Check if this is a denominator
        if lbli.startswith('/'):
            # Denominator
            powi = -powi
            # Strip leading slash
            lbli = lbli[1:]
        
        # Check for numeric values (does not allow '2km', 'km2', etc.)
        try:
            # Try converting whole symbol to number
            vali = float(lbli)
            # If that worked, apply the exponent and move on
            val *= (vali**powi)
            continue
        except ValueError:
            pass
        
        # Initialize current version of symbol
        cur = lbli
        pfx = ''
        # Loop until the current symbol (w/o prefix) matches list
        while (len(cur)>0) and (cur not in units_name):
            # Move the first character into the prefix
            pfx = pfx + cur[0]
            cur = cur[1:]
            
        # Check for valid match
        if cur in units_name:
            # Get the value
            vali = units_dict[cur]
        else:
            # Unrecognized 
            print(Warning("Symbol '%s' is not a recognized unit" % lbli))
            # Assume 1.0
            vali = 1.0
            
        # Move prefix to "remaining prefix"
        rpf = pfx
        # Loop through characters of prefix
        while len(rpf) > 0:
            # Current prefix
            cpf = rpf
            # Remaining part of prefix
            rpf = ''
            # Look for matches
            while (len(cpf)>1) and (cpf not in prefix_name):
                # Move the last character out of the current prefix
                rpf = cpf[-1] + rpf
                cpf = cpf[:-1]
            # Process prefix
            if cpf in prefix_name:
                # Multiply this value into the value of the current symbol
                vali *= prefix_dict[cpf]
            else:
                # Unrecognized
                print(Warning("Prefix '%s' was not recognized" % cpf))
        
        # Multiply the final value 
        val = val * vali**powi
        
    # output
    return val
# def mks


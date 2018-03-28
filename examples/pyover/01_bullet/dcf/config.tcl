#!/usr/bin/env tclsh

source [GetIfile GlobalDefs.tcl]
source [GetIfile inputs.tcl]

# List of bullet grids
set grids "bullet/bullet_body
           bullet/bullet_cap
           bullet/bullet_base "

# List of xrays
set xrays "bullet/bullet "

# Convert variable names
set rootnames "$grids"
set xraynames "$xrays"

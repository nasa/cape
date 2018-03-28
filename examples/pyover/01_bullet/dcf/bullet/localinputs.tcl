#!/usr/bin/env tclsh

global Ovr Par 

# Body spacings
set Par(ds,bullet,cap)  [expr 0.10*$Par(ds,glb)]
set Par(ds,bullet,crn)  [expr 0.05*$Par(ds,glb)]
set Par(ds,bullet,body) [expr 0.25*$Par(ds,glb)]
set Par(ds,bullet,aft)  [expr 0.15*$Par(ds,glb)]

# Number of points around the bullet
set Par(npcirc,bullet) 73

# ==================
# Volume grid inputs
# ==================
# <
    # Front cap
    set name bullet_cap
    set ${name}(ibcja) -20
    set ${name}(ibcjb) -20
    set ${name}(ibcka) -20
    set ${name}(ibckb) -20
    
    # Volume grid inputs for back cap
    set name bullet_base
    set ${name}(ibcja) -20
    set ${name}(ibcjb) -20
    set ${name}(ibcka) -20
    set ${name}(ibckb) -20
# >

# ============
# Xray Inputs
# ============
# <
    # Offbody cutter
    set name "bullet_cuts_OffBody"
    set XRINFO($name,idxray) "bullet"
    set XRINFO($name,group)  "OffBody"
    set XRINFO($name,xdelta) 2.0
# >

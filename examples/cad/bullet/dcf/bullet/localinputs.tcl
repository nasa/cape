#!/usr/bin/env tclsh

global Ovr Par GlobalScaleFactor

# Body spacings
set Par(ds,bullet,cap)  [expr 0.10*$Par(ds,glb)]
set Par(ds,bullet,crn)  [expr 0.05*$Par(ds,glb)]
set Par(ds,bullet,body) [expr 0.15*$Par(ds,glb)]
set Par(ds,bullet,aft)  [expr 0.12*$Par(ds,glb)]


#!/usr/bin/env tclsh

global Par env File Fomo Ovr

# Load CGT scripts
lappend auto_path $env(SCRIPTLIB)

# ------
# Get global inputs
# ------
source [GetIfile GlobalDefs.tcl]
source [GetIfile inputs.tcl]
source [GetIfile localinputs.tcl]

# **********************************************************************
proc CreateBullet { } {
    
    global Par Fomo Ovr
    
    # Relevant file names
    set fcrv "bullet.crv"
    set ftri "bullet.i.tri"
    
    # Copy files
    set partdir $Par(GeomDir)
    exec /bin/cp $partdir/$ftri $ftri
    exec /bin/cp $partdir/$fcrv $fcrv
    
  # ==========
  # Body Curve
  # ==========
  # <
    # Extract curves
    ExtractSubs $fcrv a.1 [list  1 1 -1 1 1 1 1]
    ExtractSubs $fcrv a.2 [list 10 1 -1 1 1 1 1]
    ExtractSubs $fcrv a.3 [list  7 1 -1 1 1 1 1]
    
    # Redistribute
    set sr  $Par(sr)
    set ds0 $Par(ds,bullet,crn)
    set ds1 $Par(ds,bullet,cap)
    set ds2 $Par(ds,bullet,body)
    set ds3 $Par(ds,bullet,aft)
    SrapRedist a.1 c.1 1 -j 1 5 1 [list 1 -1 $sr $ds0 $ds0 $ds1]
    SrapRedist a.2 c.2 1 -j 1 5 1 [list 1 -1 $sr $ds0 $ds0 $ds2]
    SrapRedist a.3 c.3 1 -j 1 5 1 [list 1 -1 $sr $ds0 $ds0 $ds3]
    # Concatenate curves
    ConcatGridsn [list c.1 c.2 c.3] d.1 j 1
    
    exit
  # >
}

# **********************************************************************
CreateBullet

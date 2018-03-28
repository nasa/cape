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
    set fdef "bullet.lr8.crv"
    set fcrv "bullet.crv"
    set ftri "bullet.i.tri"
    
    # Copy files
    set partdir $Par(GeomDir)
    puts $partdir
    exec /bin/cp $partdir/$ftri $ftri
    exec /bin/cp $partdir/$fdef $fcrv
    
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
    # Save dimensions
    set nj1 [lindex [GetDim c.1] 0]
    set nj2 [lindex [GetDim c.2] 0]
    set nj3 [lindex [GetDim c.3] 0]
    # Concatenate curves
    ConcatGridsn [list c.1 c.2 c.3] d.1 j 1
  # >
  
  # ===============
  # Grid Generation
  # ===============
  # <
    # Revolve
    GedRevolve d.1 s.1 x -360.0 1 $Par(npcirc,bullet)
    
    # Create cap at the front
    set npcapfr 9
    set npcapbk 9
    CreateAxisCapN s.1 bullet_cap.srf ja $npcapfr full 0
    CreateAxisCapN s.1 bullet_base.srf jb $npcapbk full 0
    
    # Make volume grid slice
    set md [expr 5.0]
    ReverseInd d.1 d.2 [list revj]
    GenHypVolGrid_GU d.2 q.1 [list geouni specsr \
        $Par(sr,wall) $md \
        $Par(ds,wall) $Par(ds,glb) $Par(klayer)] \
        [list 4 4 3 3] [list 2.0 150]
    # Revolve to get volume grid
    GedRevolve q.1 v.1 x 360.0 1 $Par(npcirc,bullet)
    
    # Pull back from singular axes
    set njcapfr 3
    set njcapbk 5
    ExtractSubs v.1 v.2             [list 1 $njcapfr -$njcapbk 1 -1 1 -1]
    ExtractSubs s.1 bullet_body.srf [list 1 $njcapfr -$njcapbk 1 -1 1 -1]
    # Swap indices
    SwapInd v.2 bullet_body.vol [list revj revk]
    # Indices for family breaks in main body grid
    set njcap  [expr $nj1 - $njcapfr]
    set njbody [expr $njcap + $nj2]
    set njaft  [expr $nj3 - $njcapbk+3]
    
    # Create xray
    CreateXrayMap s.1 bullet.xry $Par(ds,xray,fine) 0.0 
  # >
  
  # ===============
  # OVERFLOW inputs
  # ===============
  # <
    set name bullet_cap
    set Ovr($name,ibtyp)   [list    5   ]
    set Ovr($name,ibdir)   [list    3   ]
    set Ovr($name,jbcs)    [list    1   ]
    set Ovr($name,jbce)    [list   -1   ]
    set Ovr($name,kbcs)    [list    1   ]
    set Ovr($name,kbce)    [list   -1   ]
    set Ovr($name,lbcs)    [list    1   ]
    set Ovr($name,lbce)    [list    1   ]
    set Ovr($name,family)  [list bullet_cap ]
    set Ovr($name,group)   [list bullet ]
    set Ovr($name,xmlcomp) [list bullet ]
    WriteOvfi $name.ovfi $name
    
    set name bullet_base
    set Ovr($name,ibtyp)   [list    5   ]
    set Ovr($name,ibdir)   [list    3   ]
    set Ovr($name,jbcs)    [list    1   ]
    set Ovr($name,jbce)    [list   -1   ]
    set Ovr($name,kbcs)    [list    1   ]
    set Ovr($name,kbce)    [list   -1   ]
    set Ovr($name,lbcs)    [list    1   ]
    set Ovr($name,lbce)    [list    1   ]
    set Ovr($name,family)  [list bullet_base ]
    set Ovr($name,group)   [list bullet ]
    set Ovr($name,xmlcomp) [list bullet ]
    WriteOvfi $name.ovfi $name
    
    set name bullet_body
    set Ovr($name,ibtyp)   [list   5        5       5     10   ]
    set Ovr($name,ibdir)   [list   3        3       3      2   ]
    set Ovr($name,jbcs)    [list   1     $njcap  -$njaft   1   ]
    set Ovr($name,jbce)    [list $njcap -$njaft    -1     -1   ]
    set Ovr($name,kbcs)    [list   1        1       1      1   ]
    set Ovr($name,kbce)    [list  -1       -1      -1      1   ]
    set Ovr($name,lbcs)    [list   1        1       1      1   ]
    set Ovr($name,lbce)    [list   1        1       1     -1   ]
    set Ovr($name,family)  [list bullet_cap bullet_body bullet_base ]
    set Ovr($name,group)   [list bullet ]
    set Ovr($name,xmlcomp) [list bullet ]
    WriteOvfi $name.ovfi $name
  # >
  
  # ========
  # Clean up
  # ========
  # <
    file delete $ftri $fcrv
    file delete {*}[glob ?.*]
    file delete -force minmax.com surf.i
    file delete -force surf.com
    file delete -force far.com
    file delete -force edges.com
    file delete -force stretch.i
  # >
}

# **********************************************************************
CreateBullet

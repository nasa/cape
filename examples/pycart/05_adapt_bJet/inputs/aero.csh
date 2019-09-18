#!/bin/csh -f
# $Id: aero.csh,v 1.13 2013/06/14 20:33:51 maftosmi Exp $

# AERO: Adjoint Error Optimization
# Script to drive adjoint-based mesh refinement

# M. Nemec, Marian.Nemec@nasa.gov
# Oct 2006, last update: April 2013

# Usage:
# ------
# % ./aero.csh
#
# or optionally: 
# % ./aero.csh restart
# use the restart flag to run more adaptation cycles, including
# abnormal exits due to machine crashes, etc.
#
# or optionally:
# % ./aero.csh jumpstart
# use the jumpstart flag to start from a given mesh. Put a Mesh.c3d.Info file
# and a Mesh file (Mesh.mg.c3d, Mesh.R.c3d, or Mesh.c3d) in the same directory
# as aero.csh (and other inputs) and the run will start from this mesh

# The script returns 0 on success and 1 if an error occurs

# Read tips, hints and documentation in $CART3D/doc/adjoint

# Set user specified options below, defaults are suggested

# Important file names:
# Components.i.tri, input.c3d, input.cntl

# control thread affinity for linux
#setenv KMP_AFFINITY compact

# choose functional error tolerance
set etol = 0.000001

# maxR for initial mesh (cubes) 
set maxR = 8

# max number of cells allowed in mesh
# if the new mesh exceeds this limit, the adaptation terminates
set max_nCells = 50000000

# number of adaptation cycles
set n_adapt_cycles = 2

# number of flowCart iters on initial mesh
set it_fc = 75
# additional flowCart iters on each new mesh
# cycle        1  2  3  4  5  6  7  8  9 10  11  12
set ws_it = ( 50 50 50 50 50 50 50 50 50 50 100 100 ) 
# number of adjointCart iters
set it_ad = 120

# cfl number: usually ~1.1 but with power may be lower, i.e. 0.8
set cfl    = 1.1
# minimum cfl number
# in case of convergence problems flowCart will try to run with cflmin 
set cflmin = 0.8

# multigrid levels
# flowCart mg levels
set mg_fc = 3
# adjointCart mg levels (usually same as flowCart)
set mg_ad = 3

# Limiter: default 5, minmod, which is most robust. In practice, accuracy is
# much better with limiter 2 (van Leer) with still excellent robustness 
set limiter = 2

# spanwise orientation (default null)
set y_is_spanwise
#set y_is_spanwise = -y_is_spanwise

# use file name preSpec.c3d.cntl for preSpec regions (either BBoxes or XLevs
# for cubes or ABoxes for adapt) 0 = no (default), 1 = yes
set use_preSpec = 0

# Specify mesh growth for each adapt cycle. Mesh growth may range between 1.01
# to 8 - specifying 8 means that you allow refinement of every cell in the
# mesh.  We found the sequence below to work well for many problems: if your
# initial mesh has roughly 10,000 cells, then after 10 adaptations it will
# surpass 10 million cells. (For 2D cases, we recommend growth factors of 1.25
# for first two cycles and 1.5 for the rest.)  If you wish to set the
# adaptation threshold manually, see the ath array in Expert Options.
# cycle               0   1   2   3   4   5   6   7   8   9  10  11  12
set mesh_growth = ( 1.5 1.5 2.0 2.0 2.0 2.0 2.0 2.5 2.5 2.5 2.5 2.5 2.5 )

# set apc: adapt or interface propagation cycle
# a = adapt 
# p = propagate interfaces (adapt with fixed maxR)
# use sparingly - we recommend switching to p cycles once your volume mesh
# over-refines your surface triangulation and using p on the first cycle to
# reduce the bias of the initial mesh
# cycle     0 1 2 3 4 5 6 7 8 9 10 11 12 
set apc = ( p a a a a a a a a a  a  a  a )

# Set extra refinement levels for final mesh.  This allows you to adapt the
# mesh multiple times with the same error map in the last adapt cycle, thereby
# bypassing the flow, adjoint, and error estimation steps. Use with caution:
# the mesh should be fine enough so that the error estimate is decreasing -
# preferably the solution should be in the Richardson region. This helps
# circumvent the memory limitations of the error estimation code. Default value
# is 0 and maximum allowed value is 3
set final_mesh_xref = 0

# Functional averaging window in terms of flowCart mg-cycles. This is useful
# for cases that do not converge to steady-state. The averaged functional is
# reported in the fourth column of fun_con.dat, and is also used to set
# relative error (default: avg_window = 1).
set avg_window = 1

# set mesh2d = 1 for 2D cases, mesh2d = 0 for 3D cases
set mesh2d = 0

# binaryIO tecplot output (default yes)
set binaryIO 

# adaptation restart 
# use command line argument or set adapt_restart = 1 for restarts
set adapt_restart = 0

# adaptation jumpstart 
# use command line argument or set adapt_jumpstart = 1 to start from an existing mesh
set adapt_jumpstart = 0

# verbose mode
set verb
# set verb = -v

# Initial mesh parameters (cubes)
set cubes_a = 10 # angle criterion (-a)
set cubes_b = 2  # buffer layers   (-b)

# Internal mesh (cubes)
set Internal
#set Internal = '-Internal'

# --------------------------------------
# Experimental options under development
# --------------------------------------

# Select relative (use_relative_etol=1) or absolute (use_relative_etol=0) error
set use_relative_etol = 0
# For relative error, etol is computed as a fraction of functional
# default 4% -> 0.04
set etol_fraction = 0.04
# smallest allowable etol value (in case your functional is close to zero) 
set etol_min = 1.e-4

# To solve adjoint equation on finest mesh, set adjoint_on_finest=1
set adjoint_on_finest = 0

# To do error analysis on finest mesh, set error_on_finest=1
set error_on_finest = 0

# Keep final error map in "./EMBED/Restart.XX.file"
# Useful for % cubes -remesh ..., default value is 0
set keep_error_maps = 0  # {Yes, No} = {1, 0}

# Refine all cells: useful for uniform mesh refinement studies. This overrides
# the error map and forces adapt to refine all cells. The adjoint correction
# term and error estimate are reported. Default value is 0
set refine_all_cells = 0  # {Yes, No} = {1, 0}

# ---------------------------------------------------
# EXPERT user options: flags below are rarely changed
# ---------------------------------------------------

# Fine tuning of mesh growth when performing extra refinements on the final
# mesh, i.e. when $final_mesh_xref>0 and $mesh_growth is being used.  The mesh
# growth for each extra refinement is given by:
# ($mesh_growth-1)*$xref_fraction+1
# The main idea is that as extra refinement cycles are performed, the
# adaptation focuses on only the highest error cells. This is where the error
# map is most accurate and most adaptation is required. The value should be
# between 0 and 1, and at most three extra refinements are allowed
set xref_fraction = ( 1.0 1.0 0.8 )

# flow solver warm starts (default 1=yes, 0=no)
set use_warm_starts = 1

# grid sequencing or multigrid (mg default)
# flowCart
set mg_gs_fc = '-gs'
set mg_gs_fc = '-mg'
# adjointCart
set mg_gs_ad = '-gs'
set mg_gs_ad = '-mg'

# full multigrid: default is to use full multigrid, except in cases with power
# boundary conditions (automatic with warm starts)
set fmg 
# set fmg = -no_fmg

# pmg: poly multigrid, default null
set pmg 
# set pmg = '-pmg'

# number of multigrid levels for initial mesh (default 2, ramps up to mg_fc/ad)
set mg_init = 2

# subcell
set subcell
#set subcell = '-subcell'

# buffer limiter: improves stability for flow with strong off-body shocks
set buffLim 
# set buffLim = '-buffLim'

# cut-cell gradients: 0=best robustness (default), 1=best accuracy
# if mesh2d=1, then we set tm=1 automatically
set tm = 0 # (0 or 1)

# Run adjoint solver in first-order mode. In hard cases where attempts to run
# second-order adjoints with or without pmg consistently fail, you can
# short-circuit this and go directly to first-order adjoints by setting
# adj_first_order = 1. The flow solution remains second-order, but all adjoints
# will be first-order accurate. Use caution: this should give a consistent set
# of error estimates, which is probably safe for relative errors and meshing,
# but may be inaccurate with respect to a second-order run.  Default value is
# adj_first_order = 0
set adj_first_order = 0 # (0 or 1)

# In 3D cases with tm=1, error estimation is done with tm=0 for robustness. If
# you want to use tm=1 in error estimation, then you need to set err_TM1=1
# below. This is recommended only for simple (academic) cases that converge
# well. In general, the default (0) setting is _strongly_ recommended.
set err_TM1 = 0 # (0 or 1)

# In 3D cases with tm=1, adjoint solver robustness is much better with tm=0. If
# you want to force tm=1 in adjoint solves, then you need to set adj_TM1=1
# below.  In general, the default (0) setting is _strongly_ recommended. Note
# that adjoint convergence is monitored, so if divergence occurs then tm=0 is
# set during runtime.
set adj_TM1 = 0 # (0 or 1)

# Set the number of multigrid levels when aero.csh drops down to pMG due to
# convergence problems. Default value is 2, which means no geometric
# multigrid. In subsonic cases, 3 multigrid levels may be better. Note that
# this flag has no effect on the pmg multigrid levels when the pmg flag is
# selected above.  It influences only the automatic run control of aero.csh.
set mg_pmg_auto = 2

# Adaptation threshold array: if you wish to set ath manually, set mesh_growth
# to null (uncomment next line) and set the ath array:
#set mesh_growth
# cycle      0 1 2 3 4 5 6 7 8 9 10 11 12
set ath = ( 16 8 4 2 1 1 1 1 1 1  1  1  1 )

# adapt buffers (default 1)
set buf = 1

# set name of user time output file
set time_file = user_time.dat

# set names of executables 
set flowCart        = flowCart
set xsensit         = xsensit
set adjointCart     = adjointCart
set adjointErrorEst = adjointErrorEst_quad

# ---------------------------------------------
# STOP: no user specified parameters below here
# ---------------------------------------------

# restart via command line
if ( 0 != $#argv ) then
  if ( $argv[1] == restart ) then
    set adapt_restart = 1
  else if ( $argv[1] == jumpstart ) then
    set adapt_jumpstart = 1
  endif
endif

limit stacksize unlimited

# check if safe to run
if ( 1 != $adapt_restart ) then
  \ls -d adapt?? >& /dev/null
  if ( $status == 0 ) then
    echo "adapt?? directories exist"
    echo "use '\rm -r adapt??' before starting a new run"
    echo "for restarts, please use 'aero.csh restart'"
    goto ERROR
  endif
endif

if ( 1 == $error_on_finest ) then
  set adjoint_on_finest = 1
endif

set is2D 
set reorder = -reorder
set mgprep = mgPrep
if ( $mesh2d == 1 ) then
  set is2D = '-mesh2d'
  set reorder 
  set mgprep = mgTree
  set pmg
  # force tm = 1 for 2d cases (best accuracy)
  set tm = 1
endif

# check codes
echo 'Using codes:'
\ls -otr `which cubes`
if ( $status != 0 ) then
  echo "ERROR: cubes not found "
  goto ERROR
endif
\ls -otr `which $mgprep`
if ( $status != 0 ) then
  echo "ERROR: $mgprep not found "
  goto ERROR
endif
\ls -otr `which $flowCart`
if ( $status != 0 ) then
  echo "ERROR: $flowCart not found "
  goto ERROR
endif
\ls -otr `which $xsensit`
if ( $status != 0 ) then
  echo "ERROR: $xsensit not found "
  goto ERROR
endif
\ls -otr `which $adjointCart`
if ( $status != 0 ) then
  echo "ERROR: $adjointCart not found "
  goto ERROR
endif
\ls -otr `which $adjointErrorEst`
if ( $status != 0 ) then
  echo "ERROR: $adjointErrorEst not found "
  goto ERROR
endif
\ls -otr `which adapt`
if ( $status != 0 ) then
  echo "ERROR: adapt not found "
  goto ERROR
endif

# report checksums
set md5sum
set md5opt
which md5sum >& /dev/null
if ( 0 == $status ) then
  set md5sum = md5sum
else
  which gmd5sum >& /dev/null
  if ( 0 == $status ) then
    set md5sum = gmd5sum
  else
    which md5 >& /dev/null
    if ( 0 == $status ) then
      set md5sum = md5
      set md5opt = '-r'
    endif
  endif
endif

if ( $md5sum != '' ) then
  echo
  echo "$md5sum checksums:"
  $md5sum $md5opt `which cubes`
  $md5sum $md5opt `which $mgprep`
  $md5sum $md5opt `which $flowCart`
  $md5sum $md5opt `which $xsensit`
  $md5sum $md5opt `which $adjointCart`  
  $md5sum $md5opt `which $adjointErrorEst`  
  $md5sum $md5opt `which adapt`
endif

echo $CART3D >& /dev/null
if ( $status != 0 ) then
  echo "ERROR: CART3D env variable not found"
  goto ERROR
endif
echo
echo "CART3D env variable set to: " $CART3D

# show which aero.csh
\ls -otr `which $0`
echo ' '

# adaptation exit flag
set error_ok = 0

# smallest allowable etol when using relative errors
set etol_cutoff = $etol_min

if ( "$mg_gs_ad" == '-mg' ) then
  set ad_multigrid = 'multigrid'
else
  set ad_multigrid = 'grid-sequencing'
endif

if ( "$mg_gs_fc" == '-mg' ) then
  set fc_multigrid = 'multigrid'
else
  set fc_multigrid = 'grid-sequencing'
endif

# use Morton order whenever mgTree is used
set sfc
if ( "$mgprep" == 'mgTree' ) then
  set sfc = '-sfc M'
endif

# buffer limiter in cut-cells
# if tm = 0, set blcc to nothing
# if tm = 1, strongly recommend using buffLimCC for stability 
# disabled 08.11.20 - see if we need it later
# if ( $tm == 0 ) then 
set blcc
# else
#    set blcc = -buffLimCC
# endif

# clean-up STOP file
if ( -e ./STOP ) then
  echo "Removing STOP"
  \rm -f STOP
endif

# clean-up ARCHIVE file
if ( -e ./AERO_FILE_ARCHIVE.txt ) then
  echo "Removing AERO_FILE_ARCHIVE.txt"
  \rm -f AERO_FILE_ARCHIVE.txt
endif

# ----------------------------------
# restarts
set x               = 0
set xx              = 00
set dirname         = adapt00
set previous
set xm1

if ( 1 == $adapt_restart ) then

  if ( -e ADAPT/ADAPT_RESTART ) then
    set x = `cat ADAPT/ADAPT_RESTART | awk '{print $1}'`
    
    if ( 0 < $x ) then
      @ xm1 = $x - 1
  
      if ( $x < 10 ) then
        set dirname  = adapt0${x}
        set xx       = 0${x}
      else
        set dirname  = adapt${x}
        set xx       = ${x}
      endif

      if ( $xm1 < 10 ) then
        set previous = adapt0${xm1}
      else
        set previous = adapt${xm1}
      endif
      
    endif
  endif
  
  # cleanup EMBED
  cd EMBED
  if ( -l Mesh.mg.c3d ) then
    unlink Mesh.mg.c3d >& /dev/null
  endif
  \rm -f Mesh.c3d.Info cutPlanesADJ.plt *.${xx}.* >& /dev/null
  \rm -f adaptedMesh.R.c3d postAdapt.ckpt aPlanes.dat >& /dev/null
  cd ../

  # cleanup ADAPT
  \rm -f ADAPT/Mesh.c3d.Info ADAPT/*.${xx}.* ADAPT/adapt.stats >& /dev/null
endif
# ----------------------------------

if ( -l ./BEST && ( 1 != $adapt_restart ) ) then
  echo "Unlinking BEST"
  unlink BEST
endif

if ( -d ./BEST && ( 1 != $adapt_restart ) ) then
  echo "BEST should not be a directory, please remove it and try again"
  goto ERROR
endif

set mydate = `date`

# ------------------------------
# functional convergence summary
set fun_con = 'fun_con.dat'
if ( -e $fun_con && 0 == $adapt_restart ) then
  \rm -f $fun_con
endif
if ( 0 == $adapt_restart ) then
  echo "# Functional convergence" >>! $fun_con
  echo "#" $mydate >>! $fun_con
  echo "# Cycle   nCells   Functional    Avg. Fun. (window=$avg_window)" >>! $fun_con
endif
# ------------------------------

# ------------------------------
# timer: keep track of user time
set timer
set timeInfo = /dev/null
set flowCartTime        = 0
set xsensitTime         = 0
set adjointCartTime     = 0
set embedTime           = 0
set adjointErrorEstTime = 0
set adaptTime           = 0
if ( -e /usr/bin/time ) then
  set timer = "/usr/bin/time -p"
  set timeInfo = TIME
  if ( -e $time_file && 1 == $adapt_restart ) then
    echo "# restart on " $mydate >> $time_file
  else
    echo "# Output of USER time from $timer" >! $time_file
    echo "#" $mydate >> $time_file
    echo "# Time rounded down to nearest second" >> $time_file
    env | grep OMP_NUM_THREADS >& /dev/null
    if ( $status == 0 ) then
      echo "# For WALLCLOCK time, divide by $OMP_NUM_THREADS CPU(s) except Embed and Adapt times " >> $time_file
    endif
    echo "# cycle flowCart xsensit adjointCart embed adjointErrorEst adapt" >> $time_file
  endif
else
  echo "NOTE: /usr/bin/time not found, user time will not be reported"
  if ( -e $time_file ) then
    \rm -f $time_file 
  endif
  set time_file = $timeInfo
endif
# ------------------------------

# set ADAPT directory, this is where new meshes are generated
if ( ! -d ADAPT ) then
  mkdir -m 0700 -p ADAPT
endif
cd ADAPT
echo "Preparing directory ADAPT"
\rm -rf XREF_* >& /dev/null
if ( -e Components.i.tri && 1 != $adapt_restart ) then
  \rm -f *.*
endif
ln -sf ../Components.i.tri
set prespec
if ( ( 1 == $use_preSpec ) && -e ../preSpec.c3d.cntl) then
  ln -sf ../preSpec.c3d.cntl
  set prespec = '-pre preSpec.c3d.cntl'
endif
cd ../  # back up to top level

# set EMBED directory, this is where embedded meshes are generated
if ( ! -d EMBED ) then
  mkdir -m 0700 -p EMBED
endif
cd EMBED
echo "Preparing directory EMBED"
if ( 1 != $adapt_restart ) then
  \rm -f *.* >& /dev/null
endif
ln -sf ../Components.i.tri     
ln -sf ../Config.xml
if ( -e ../vortexInfo.dat ) then
  ln -sf ../vortexInfo.dat
endif
if ( -e ../earthBL.Info.dat ) then
  ln -sf ../earthBL.Info.dat
endif
cd ../ # back up to top level

# set return codes for error traps
# adapt return code
set ZERO_TAGGED_HEXES = 4

# adjointErrorEst return code
set ERROR_TOL_OK = 2

# set flowCart restart flag to null
set fc_restart

# mg_start used for initial multigrid levels
set mg_start 

# initialize variables for restart
if ( 1 == $adapt_restart ) then
  echo "This is a RESTART run from $dirname"

  set nCells = 0
  
  if ( $xm1 ) then
    if ( $xm1 < 10 ) then
      set xxm1 = 0${xm1}
    else
      set xxm1 = ${xm1}
    endif
    set nCells = `grep "totNumHexes     =" ADAPT/adapt.${xxm1}.out | awk '{print $3}'`
  else
    if ( "$dirname" == 'adapt00' && -e ${dirname}/cart3d.out ) then
      set nCells = `grep "  hex cells  is:" ${dirname}/cart3d.out | awk '{print $8}'`
    endif
  endif
  
  set fun_avg = `tail -1 fun_con.dat | awk '{print $4}'`

  cd $dirname

  # check if flow solve successfully completed 
  if ( ! -e ADAPT_FLOW ) then
    # we must have a valid restart file
    if ( 1 == use_warm_starts && ! -e Restart.file ) then
      echo "ERROR: Cannot find Restart.file"
      goto ERROR
    endif
    # cleanup
    \rm -f *.dat check* *.plt 
    if ( 1 == $use_warm_starts ) then
      cp ../$previous/*.dat .
      \rm -f loads*.dat *ADJ.dat
      if ( ! -e history.dat ) then
        echo "ERROR: Cannot find history.dat file"
        goto ERROR
      endif
      # flow solve not complete so we increase cycles for warm starts
      # history file is from previous directory
      set it_fc = `tail -1 history.dat | awk '{if ("#" != $1) {print $1} else {print 0}}'`
      if ( 0 == $it_fc ) then
        echo "ERROR: Cannot parse history.dat file"
        goto ERROR
      endif
      @ it_fc += $ws_it[$x]
    endif
  else # ADAPT_FLOW present
    # set it_fc to completed cycles
    if ( 1 == $use_warm_starts ) then
      set it_fc = `tail -1 history.dat | awk '{if ("#" != $1) {print $1} else {print 0}}'`
      if ( 0 == $it_fc ) then
        echo "ERROR: Cannot parse history.dat file"
        goto ERROR
      endif
    endif
  endif # end ADAPT_FLOW
else # not a restart    
  mkdir -m 0700 -p $dirname
  cd $dirname

  ln -s ../Config.xml 
  ln -s ../Components.i.tri
  ln -s ../input.c3d

  # vortex farfield input file
  if ( -e ../vortexInfo.dat ) then
    ln -s ../vortexInfo.dat
  endif
  # earth bl input file
  if ( -e ../earthBL.Info.dat ) then
    ln -s ../earthBL.Info.dat
  endif
  
  # used to ln -s but now use cp to handle steering properly
  cp ../input.cntl .
  if ( ( 1 == $use_preSpec ) && -e ../preSpec.c3d.cntl) then
    ln -s ../preSpec.c3d.cntl
  endif

  # inverse design using target cp
  if ( -e ../target.triq ) then
    ln -s ../target.triq
  endif

  # inverse design using off-body line sensors
  \ls -1 ../target_*.dat >& /dev/null
  if ( $status == 0 ) then
    foreach target ( ../target_*.dat )
      ln -s $target
    end
  endif

  if ( 0 == $adapt_jumpstart ) then
    echo "Building initial mesh"
    if ( $cubes_b < 2 ) then
      set cubes_b = 2
    endif
    cubes -v -verify -a $cubes_a -maxR $maxR -b $cubes_b $reorder $is2D $prespec -no_est $Internal >> cntl.out
    if ($status != 0) then
      echo "==> CUBES failed"
      goto ERROR
    endif
    set nCells = `grep "  hex cells  is:" cntl.out | awk '{print $8}'`
  else
    echo "Jumpstarting from a given initial mesh"
    if ( -e ../Mesh.c3d.Info ) then
      \mv -f ../Mesh.c3d.Info .
    else
      echo "ERROR: Missing Mesh.c3d.Info"
      goto ERROR
    endif
    if ( -e ../Mesh.mg.c3d ) then
      \mv -f ../Mesh.mg.c3d Mesh.R.c3d
    else if ( -e ../Mesh.R.c3d ) then
      \mv -f ../Mesh.R.c3d Mesh.R.c3d
    else if ( -e ../Mesh.c3d ) then
      \mv -f ../Mesh.c3d Mesh.c3d
    else
      echo "ERROR: Missing Mesh.*.c3d file"
      goto ERROR
    endif
    set nCells = 0
  endif
endif

echo "  "
echo " Working in directory" $dirname
if ( $nCells > 0 ) then
  echo " Mesh contains $nCells hexes"
endif

# main loop over adaptation cycles
# exit set with break statements
while ( 1 )

  if ( 1 == $use_warm_starts && $x > 0 ) then
    set fc_restart = '-restart'
  endif         

  # on coarse meshes run fewer multigrid levels for better convergence
  # $x starts at zero ... let's try mg 2 on initial mesh
  @ mg_start = $x + $mg_init
  if ( $mg_start < $mg_fc && $n_adapt_cycles > 0 ) then
    set mg_levs = $mg_start
  else
    set mg_levs = $mg_fc
  endif

  # ADAPT_FLOW is used to flag adaptation restarts. The flag indicates if the
  # working directory contains a valid flow solution
  if ( ! -e ADAPT_FLOW ) then

    # prepare mg meshes: generate only the required number of levels
    if ( ! -e Mesh.mg.c3d ) then # restart check, skip if done
      while ( $mg_levs > 1 )
        echo "$mgprep -n $mg_levs -verifyInput $pmg" >> cntl.out
        $mgprep -n $mg_levs -verifyInput $pmg >> cntl.out
        if ( $status == 0 ) then
          break
        else
          echo "==> $mgprep failed trying to make $mg_levs levels"
          @ mg_levs -= 1
          echo "==> Trying again with $mg_levs levels"
        endif
      end
      
      if ( $mg_levs == 1 ) then
        echo "==> Caution: multigrid is not active"
        if ( $mesh2d ) then
          mv Mesh.c3d Mesh.mg.c3d
        else 
          mv Mesh.R.c3d Mesh.mg.c3d
        endif
      endif
    endif # Mesh.mg.c3d
    
    echo "   Using $mg_levs of $mg_fc levels in flowCart $fc_multigrid"

    # cleanup
    if ( -e Mesh.c3d ) then
      \rm -f Mesh.c3d
    endif

    if ( -e Mesh.R.c3d ) then
      \rm -f Mesh.R.c3d
    endif
    
    ( $timer $flowCart $verb -his -N $it_fc -T $binaryIO -clic $mg_gs_fc $mg_levs -limiter $limiter -tm $tm -cfl $cfl $y_is_spanwise $fc_restart $fmg $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
    set exit_status = $status
    if ( 0 != $exit_status ) then
      if ( 253 == $exit_status) then   
        echo "ERROR: file parsing error in flowCart "
        echo "       check flowCart output in $dirname ... exiting now"
        goto ERROR
      endif
      if ( 0 == $mesh2d ) then
        # check if using robust mode
        set isRobust = `cat input.cntl | awk 'BEGIN{ n=0; ng=0; } { if ("RK"==$1) { n++; if ("1"==$3) {ng++;} } } END{ if (n==ng) { print 1 } else { print 0 } }'`
        # try converging in robust mode if standard input file
        if ( 0 == $isRobust ) then 
          echo "==> $flowCart warm-start failed with status $exit_status ... trying robust mode"
          # cleanup failed run
          \rm -f check.* checkDT.* *.dat

          if ( $dirname == 'adapt00' ) then
            # inverse design using off-body line sensors
            \ls -1 ../target_*.dat >& /dev/null
            if ( $status == 0 ) then
              foreach target ( ../target_*.dat )
                ln -s $target
              end
            endif
          else
            \cp -f ../$previous/*.dat .
            \rm -f loads*.dat *ADJ*.dat
          endif
          
          # build robust-mode cntl file
          mv input.cntl input.not_robust.cntl
          awk '{if ("RK"==$1){print $1,"  ",$2,"  ",1}else{print $0}}' \
              input.not_robust.cntl > input.cntl            
          ( $timer $flowCart $verb -his -N $it_fc -T $binaryIO -clic $mg_gs_fc $mg_levs -limiter $limiter -tm 0 -cfl $cfl $y_is_spanwise $fc_restart $fmg $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
          set exit_status = $status
        endif
        
        if ( 0 != $exit_status ) then
          echo "==> $flowCart failed with status $exit_status ... trying cold start with CFL $cflmin"
          set gs_it = 2
          @ gs_it *= $it_ad
          ( $timer $flowCart $verb -his -N $gs_it -T $binaryIO -clic $mg_gs_fc $mg_levs -limiter $limiter -tm 0 -cfl $cflmin $y_is_spanwise -no_fmg -buffLim $subcell >> cntl.out ) >&! $timeInfo
          set exit_status = $status
        endif
      else # 2D case
        echo "==> $flowCart warm-start failed with status $exit_status ... trying cold start with CFL $cflmin"
        ( $timer $flowCart $verb -his -N $it_ad -T $binaryIO -clic $mg_gs_fc $mg_levs -limiter $limiter -tm $tm -cfl $cflmin $y_is_spanwise $subcell >> cntl.out ) >&! $timeInfo
        set exit_status = $status
      endif
      if ($exit_status != 0) then
        echo "==> $flowCart failed with status $exit_status ... giving up, check cntl.out"
        goto ERROR
      endif
    endif
    if ( -o $timeInfo ) then
      set flowCartTime = `grep -e "user " $timeInfo | awk '{printf "%d",$2 }'`
      \rm -f $timeInfo
    endif

    set fun     = `tail -1 functional.dat | awk '{printf("%18.10e",$3)}'`
    set fun_avg = `tail -$avg_window functional.dat | awk 'BEGIN{ n=0; avg=0.0;} { avg += $3; n++} END{ avg /= n; printf("%18.10e",avg)}'`
    set nCells = `grep "nCells (total number of control volumes):" functional.dat | tail -1 | awk '{print $8}'`
    echo "$x  $nCells   $fun   $fun_avg" >>! ../$fun_con

    if ( -e Restart.file ) then
      \rm -f Restart.file
    endif

    # mark as done, save mg_levs in case we have to run first-order
    echo $mg_levs > ADAPT_FLOW
  endif # ADAPT_FLOW

  # check flowCart cycles
  if ( 1 == $use_warm_starts ) then
    set it_fc = `tail -1 history.dat | awk '{if ("#" != $1) {print $1} else {print '$it_fc'}}'`
  endif
  echo "   Done $it_fc flowCart cycles" 

  if ( -l ../BEST ) then
    unlink ../BEST
  endif
  ln -s $dirname ../BEST

  # check max level of refinement in current mesh and make sure we do not
  # exceed max allowed based on 21 bits
  set maxRefLev   = `grep " # maxRef" Mesh.c3d.Info  | awk '{print $1}'`
  set maxDiv      = `grep " # initial mesh divisions" Mesh.c3d.Info  | awk '{ if ( $1 < $2 ) { $1 = $2; } if ( $1 < $3 ) { $1 = $3; } print $1}'`
  # this is eq. 3.5 on pg. 84 in VKI notes of Aftosmis
  set maxRefAllow = `echo $maxDiv | awk '{maxRef21 = 20.999999312 - log($1-1)/log(2.); print int( maxRef21 ) }'`
  if ( $maxRefLev == $maxRefAllow ) then
    echo " WARNING: Maximum refinement level reached, see STOP file"
    echo "Maximum refinement level reached - embedding this mesh would exceed 21 bits." >> ../STOP
    echo "Switch to propagation (in apc array) on next to last cycle and continue." >> ../STOP
  endif

  # check exit criteria
  if ( 0 == $adjoint_on_finest && ( $n_adapt_cycles == $x || 1 == $error_ok || -e ../STOP ) ) then
    if ( $n_adapt_cycles == $x ) then
      echo " Completed $n_adapt_cycles adaptation cycles, exiting"
    endif
    if ( -e ../STOP ) then
      echo " STOP file detected, exiting"
    endif

    # top level dir
    cd ../

    # reset timers, except flowCart
    set xsensitTime         = 0
    set adjointCartTime     = 0
    set embedTime           = 0
    set adjointErrorEstTime = 0
    set adaptTime           = 0
    echo $x $flowCartTime $xsensitTime $adjointCartTime $embedTime $adjointErrorEstTime $adaptTime >> $time_file
    
    # exit while loop
    break
  endif

  # adjust etol if using relative error
  if ( 1 == $use_relative_etol ) then
    # etol is set to fraction*functional
    # before changing etol, check that error/etol has decreased by at least 20%            
    set adjust_etol = 1
    if ( $x > 2 ) then # skip first 3 cycles
      set adjust_etol = `tail -2 ../results.dat | awk 'BEGIN{ n=0; } {ratio[n] = $7; n++;} END{if (ratio[1]*1.20 < ratio[0]) { print 1 } else { print 0}}'`
    endif

    #   ...check error/etol ratio and adjust cutoff if necessary
    if ( 1 == $adjust_etol ) then  # allowed to adjust
      # make sure etol is positive
      set etol = `echo $fun_avg | awk '{if ( 0.0 > $1 ) {printf "%e", (-1.)*'$etol_fraction'*$0} else {printf "%e", '$etol_fraction'*$0} }'`
      # also make sure etol is greater than cutoff value
      echo $etol | awk '{if ( '$etol_cutoff' > $1 ) {printf("   NOTE: ETOL set to cutoff value: %10.3e\n", '$etol_cutoff')}}'
      set etol = `echo $etol | awk '{if ( '$etol_cutoff' > $1 ) {printf "%e", '$etol_cutoff'} else {printf "%e", '$etol'} }'`
    endif

    #   ...check error/etol ratio and adjust cutoff if necessary
    # if ( 1 == $adjust_etol ) then     # allowed to adjust
    #    # make sure etol is positive
    #    #                        ...using Cleve Moller's R(J+1) formula for relative tolerance
    #    set etol = `echo $fun_avg | awk '{if ( 0.0 > $1 ) {printf "%e", (-1.)*'$etol_fraction'*($0-1.)} else {printf "%e", '$etol_fraction'*($0+1)} }'`
    #    # also make sure etol is greater than cutoff value
    # endif

    if ( $x > 1 ) then # do not trust first cycle
      # adjust cutoff if error/etol > 100
      # find order of magnitude of error/etol and power of ten multiplier
      set etol_mult = `tail -1 ../results.dat | awk '{ if ($7>100.) { printf("%d", 10^(int(log($7)/log(10.)) - 1)) } else { printf("%d", 1)} }'`
      if ( $etol_mult > 1 ) then
        # adjust etol_cutoff and etol
        set etol_cutoff = `echo $etol_cutoff | awk '{ printf("%e", $1*'$etol_mult') }'`
        set etol = $etol_cutoff
        echo "   WARNING: Adjusted ETOL and ETOL_cutoff to $etol"
      endif
    endif
  endif # use_relative_error
  
  # start of adjoint solution
  if ( ! -e ADAPT_ADJ ) then
    # cleanup in case this is a restart
    if ( -d TM0 ) then
      \rm -rf TM0
    endif
    if ( -d PMG ) then
      \rm -rf PMG
    endif
    if ( -d FIRST_ORDER ) then
      \rm -rf FIRST_ORDER
    endif

    ln -sf `\ls -1tr check.* | tail -1` Flow.file
    ln -sf `\ls -1tr checkDT.* | tail -1` DT.file

    # re-initialize in case this is a restart
    # we need mg_levs to rerun flowCart if adjoint fails
    set mg_levs_saved = `cat ADAPT_FLOW | awk '{print $1}'`
    set mg_levs       = $mg_levs_saved

    # on coarse meshes run fewer multigrid levels for better convergence
    set ad_mg_levs
    if ( $mg_start < $mg_ad && $n_adapt_cycles > 0 ) then
      set ad_mg_levs = $mg_start
    else
      set ad_mg_levs = $mg_ad
    endif
    # check that ad_mg_levs is not greater than what flowCart used
    # this can happen if mgPrep failed
    if ( $mg_levs > $ad_mg_levs ) then
      set mg_levs = $ad_mg_levs
    endif
    echo "   Using $mg_levs of $mg_ad levels in adjointCart $ad_multigrid"

    if ( ( 0 == $mesh2d && 0 == $adj_TM1 && 1 == $tm ) || ( 1 == $adj_first_order ) ) then
      # go directly into TM=0 mode or first-order mode
      set exit_status = 1
    else
      ( $timer $xsensit $verb -dQ -limiter $limiter -tm $tm $y_is_spanwise $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
      set exit_status = $status
      if ($exit_status != 0) then
        echo "==> $xsensit failed with status $exit_status"
        goto ERROR
      endif
      if ( -o $timeInfo ) then
        set xsensitTime = `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
        \rm -f $timeInfo
      endif
      
      ( $timer $adjointCart $verb -his -N $it_ad $mg_gs_ad $mg_levs -T $binaryIO -limiter $limiter -tm $tm -cfl $cfl $y_is_spanwise $fmg $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
      set exit_status = $status
      if ( -o $timeInfo ) then
        set adjointCartTime = `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
        \rm -f $timeInfo
      endif

      if ( $exit_status != 0 ) then
        echo "==> $adjointCart failed with status $exit_status"
      endif
      
      # check convergence of adjointCart via historyADJ.dat file   
      if ( $exit_status == 0 ) then
        adj_check_convergence.pl
        set exit_status = $status
        if ( $exit_status != 0 ) then
          echo "==> $adjointCart convergence problems"
        endif
      endif

      if ( ( $exit_status != 0 ) && ( 0 == $adj_first_order ) ) then
        # if we are using more than one mg level, try grid sequencing
        if ( $mg_levs > 1 && ( "$mg_gs_ad" == '-mg' ) && ( "$fmg" != '-no_fmg' ) ) then
          echo "      Running adjointCart with grid sequencing in $dirname"
          \mv -f historyADJ.dat historyADJ.mg.dat
          \rm -f checkADJ.* >& /dev/null
          set gs_it = 2 # double number of iters
          @ gs_it *= $it_ad 
          ( $timer $adjointCart $verb -his -N $gs_it -gs $mg_levs -T $binaryIO -limiter $limiter -tm $tm -cfl $cfl $y_is_spanwise $fmg $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
          set exit_status = $status
          if ( -o $timeInfo ) then
            @ adjointCartTime += `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
            \rm -f $timeInfo
          endif

          # check convergence of adjointCart via historyADJ.dat file   
          if ( $exit_status == 0 ) then
            adj_check_convergence.pl
            set exit_status = $status
          endif
        endif
      endif
    endif # end if adj_TM1

    # fail-safe strategies if adjoint convergence problems

    # try to converge with tm=0
    if ( ( $exit_status != 0 ) && ( 0 != $tm ) && ( 0 == $adj_first_order ) ) then
      echo "      Running adjointCart with tm=0 in $dirname"
      
      mkdir TM0
      cd TM0

      ln -s ../input.c3d
      ln -s ../input.cntl
      ln -s ../Config.xml
      ln -s ../Components.i.tri
      ln -s ../Mesh.mg.c3d
      ln -s ../Mesh.c3d.Info
      cp ../*.dat .
      \rm -f loads*.dat *ADJ*.dat >& /dev/null
      ln -sf `\ls -1tr ../check.* | tail -1` Restart.file

      # try for deeper convergence with flowCart via tm=0 restart
      # use half of current warm start iterations
      if ( 0 == $x ) then
	@ it_tm = $it_fc + $it_fc / 2
      else
	@ it_tm = $it_fc + $ws_it[$x] / 2
      endif
      ( $timer $flowCart $verb -his -N $it_tm -T $binaryIO -clic $mg_gs_fc $mg_levs_saved -limiter $limiter -tm 0 -cfl $cfl $y_is_spanwise -restart $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
      if ($status != 0) then
	echo "==> $flowCart failed in tm=0 mode"
	goto ERROR
      endif
      # check actual flowCart cycles
      set it_tm = `tail -1 history.dat | awk '{if ("#" != $1) {print $1} else {print '$it_tm'}}'`
      if ( -o $timeInfo ) then
	@ flowCartTime += `grep -e "user " $timeInfo | awk '{printf "%d",$2 }'`
	\rm -f $timeInfo
      endif
      
      ln -sf `\ls -1tr check.* | tail -1` Flow.file
      ln -sf `\ls -1tr checkDT.* | tail -1` DT.file

      ( $timer $xsensit $verb -dQ -limiter $limiter -tm 0 $y_is_spanwise $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
      if ($status != 0) then
	echo "==> $xsensit failed"
	goto ERROR
      endif
      if ( -o $timeInfo ) then
	@ xsensitTime += `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
	\rm -f $timeInfo
      endif
      
      ( $timer $adjointCart $verb -his -N $it_ad $mg_gs_ad $mg_levs -T $binaryIO -limiter $limiter -tm 0 -cfl $cfl $y_is_spanwise $fmg $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
      set exit_status = $status
      if ( -o $timeInfo ) then
	@ adjointCartTime += `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
	\rm -f $timeInfo
      endif

      # check convergence of adjointCart via historyADJ.dat file
      if ( $exit_status == 0 ) then
	adj_check_convergence.pl
	set exit_status = $status
      endif
      \rm -f *.q Mesh.mg.c3d 
      cd ../ # out of TM0 directory
    endif # done TM0

    # try to converge with pMG
    if ( ( $exit_status != 0 ) && ( "$pmg" != '-pmg') && ( 0 == $adj_first_order ) ) then
      echo "      Running adjointCart with pMG in ${dirname} on ${mg_pmg_auto} levels"

      # force tm=0
      @ tm_hold = $tm
      set tm = 0
      
      mkdir PMG
      cd PMG

      ln -s ../input.c3d
      ln -s ../input.cntl
      ln -s ../Config.xml
      ln -s ../Components.i.tri
      if ( $mesh2d ) then
        ln -s ../Mesh.mg.c3d Mesh.c3d
      else
        ln -s ../Mesh.mg.c3d Mesh.R.c3d
      endif
      ln -s ../Mesh.c3d.Info
      cp ../*.dat .
      \rm -f loads*.dat *ADJ*.dat >& /dev/null
      ln -sf `\ls -1tr ../check.* | tail -1` Restart.file

      if ( $mg_pmg_auto < 2 ) then
        echo "WARNING: detected mg_pmg_auto < 2, resetting to 2"
        set mg_pmg_auto = 2
      endif
      
      $mgprep -n $mg_pmg_auto -verifyInput -pmg > cntl.out
      set exit_status = $status
      if ( $exit_status != 0 ) then
        echo "==> $mgprep failed trying to make pMG mesh ... extremely rare"
        goto ERROR
      endif
      
      # try for deeper convergence with flowCart via pMG restart
      # use half of current warm start iterations
      if ( 0 == $x ) then
        @ it_pmg = $it_fc + $it_fc / 2
      else
        @ it_pmg = $it_fc + $ws_it[$x] / 2
      endif
      ( $timer $flowCart $verb -his -N $it_pmg -T $binaryIO -clic -mg $mg_pmg_auto -limiter $limiter -tm $tm -cfl $cfl $y_is_spanwise -restart $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
      if ($status != 0) then
        echo "==> $flowCart failed in pMG mode"
        goto ERROR
      endif
      # check actual flowCart cycles
      set it_pmg = `tail -1 history.dat | awk '{if ("#" != $1) {print $1} else {print '$it_pmg'}}'`
      if ( -o $timeInfo ) then
        @ flowCartTime += `grep -e "user " $timeInfo | awk '{printf "%d",$2 }'`
        \rm -f $timeInfo
      endif
      
      ln -sf `\ls -1tr check.* | tail -1` Flow.file
      ln -sf `\ls -1tr checkDT.* | tail -1` DT.file

      ( $timer $xsensit $verb -dQ -limiter $limiter -tm $tm $y_is_spanwise $blcc $buffLim $subcell >> cntl.out ) >&! $timeInfo
      if ($status != 0) then
        echo "==> $xsensit failed"
        goto ERROR
      endif
      if ( -o $timeInfo ) then
        @ xsensitTime += `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
        \rm -f $timeInfo
      endif
      
      ( $timer $adjointCart $verb -his -N $it_ad -mg $mg_pmg_auto -T $binaryIO -limiter $limiter -tm $tm -cfl $cfl $y_is_spanwise $fmg $blcc $buffLim >> cntl.out ) >&! $timeInfo
      set exit_status = $status
      if ( -o $timeInfo ) then
        @ adjointCartTime += `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
        \rm -f $timeInfo
      endif

      # check convergence of adjointCart via historyADJ.dat file   
      if ( $exit_status == 0 ) then
        adj_check_convergence.pl
        set exit_status = $status
      endif
      \rm -f *.q Mesh.mg.c3d 
      cd ../ # out of PMG directory
      @ tm = $tm_hold
    endif # done pMG

    # last resort: try first-order
    if ( $exit_status != 0 ) then
      echo "      Running adjointCart in first-order mode in $dirname"
      
      mkdir FIRST_ORDER
      cd FIRST_ORDER

      # build 1st order cntl file
      awk '{if ("RK"==$1){print $1,"  ",$2,"  ",0}else{print $0}}' \
	  ../input.cntl > input.cntl

      ln -s ../input.c3d
      ln -s ../Config.xml
      ln -s ../Components.i.tri
      ln -s ../Mesh.mg.c3d
      ln -s ../Mesh.c3d.Info

      # cold start with adjoint number of iterations
      ( $timer $flowCart $verb -his -N $it_ad -T $binaryIO $mg_gs_fc $mg_levs_saved -cfl $cfl $y_is_spanwise $fmg $subcell > cntl.out ) >&! $timeInfo
      if ($status != 0) then
	echo "==> $flowCart failed in first-order mode"
	goto ERROR
      endif
      if ( -o $timeInfo ) then
	@ flowCartTime += `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
	\rm -f $timeInfo
      endif

      # check actual flowCart cycles
      set it_o1 = `tail -1 history.dat | awk '{if ("#" != $1) {print $1} else {print '$it_ad'}}'`
      echo "       done $it_o1 flowCart cycles in first-order mode" 
      ln -sf `\ls -1tr check.* | tail -1` Flow.file
      ln -sf `\ls -1tr checkDT.* | tail -1` DT.file
      ( $timer $xsensit $verb -dQ $y_is_spanwise >> cntl.out ) >&! $timeInfo
      if ($status != 0) then
	echo "==> $xsensit failed on O1 restart"
	goto ERROR
      endif
      if ( -o $timeInfo ) then
	@ xsensitTime += `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
	\rm -f $timeInfo
      endif

      ( $timer $adjointCart $verb -his -N $it_ad $mg_gs_ad $mg_levs -T $binaryIO -cfl $cfl $fmg $y_is_spanwise $fmg $subcell >> cntl.out ) >&! $timeInfo
      set exit_status = $status
      if ( -o $timeInfo ) then
	@ adjointCartTime += `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
	\rm -f $timeInfo
      endif

      if ($exit_status == 0) then
	adj_check_convergence.pl
	set exit_status = $status
      endif

      if ($exit_status != 0 && ( "$mg_gs_ad" == '-mg' )) then
	echo "==> STILL Adjoint convergence problems: trying without multigrid"
	( $timer $adjointCart $verb -his -N $it_ad -gs $mg_levs -T $binaryIO -cfl $cfl $fmg $y_is_spanwise $fmg $subcell >> cntl.out ) >&! $timeInfo
	set exit_status = $status
	if ( -o $timeInfo ) then
	  @ adjointCartTime +=`grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
	  \rm -f $timeInfo
	endif
      endif

      if ($exit_status != 0) then
	echo "==> STILL Adjoint convergence problems: giving up"
	\rm -f *.q
	goto ERROR
      else
	echo "   adjointCart converged in first-order mode"
      endif

      \rm -f *.q
      cd ../ # out of first-order directory
    endif # end first-order

    # end adjoint convergence problems
    
    \rm -f *.q >& /dev/null

    # remove debug files 
    if ( -e adj_surf.dat ) then
      \rm -f adj_surf.dat
    endif
    if ( -e dJdQ_surf.dat ) then
      \rm -f dJdQ_surf.dat
    endif
  endif # ADAPT_ADJ
  touch ADAPT_ADJ
  echo "   Done adjointCart"
  
  cd ../ # out of adapt??, back to top level directory

  if ( 1 == $adjoint_on_finest && 0 == $error_on_finest && $n_adapt_cycles == $x ) then
    echo "   Done user-requested flow and adjoint solution on finest mesh, exiting"
    # record user times
    echo $x $flowCartTime $xsensitTime $adjointCartTime $embedTime $adjointErrorEstTime $adaptTime >> $time_file
    break
  endif
  
  # create embedded mesh and get error estimates
  cd EMBED
  echo "   ... in directory EMBED"
  
  cp ../$dirname/Mesh.c3d.Info .
  ln -sf `\ls -1tr ../$dirname/check.* | tail -1` preAdapt.ckpt
  # used to ln -s but now cp input file in case we have Steering_Info
  \cp -f ../$dirname/input.cntl . 
    
  ( $timer adapt -v -RallCells -seq -i ../$dirname/Mesh.mg.c3d $is2D -no_ckpt $sfc > adapt.${xx}.out ) >&! $timeInfo
  if ($status != 0) then
    echo "==> ADAPT failed"
    goto ERROR
  endif
  if ( -o $timeInfo ) then
    set embedTime = `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
    \rm -f $timeInfo
  endif
  echo "         Created embedded mesh"

  ln -sf `\ls -1tr ../$dirname/check.* | tail -1` Flow.file

  # check if we needed first-order adjoints, if so use them
  if ( -d ../$dirname/FIRST_ORDER ) then
    echo "         Using first-order adjoint solution"
    ln -sf `\ls -1tr ../$dirname/FIRST_ORDER/checkADJ.* | tail -1` Adjoint.file
  else if ( -d ../$dirname/PMG ) then
    ln -sf `\ls -1tr ../$dirname/PMG/checkADJ.* | tail -1` Adjoint.file
  else if ( -d ../$dirname/TM0 ) then
    ln -sf `\ls -1tr ../$dirname/TM0/checkADJ.* | tail -1` Adjoint.file
  else
    ln -sf `\ls -1tr ../$dirname/checkADJ.* | tail -1` Adjoint.file
  endif
  
  ln -s adaptedMesh.R.c3d Mesh.mg.c3d

  echo "            Computing adjoint refinement parameter"
  @ tm_hold = $tm
  if ( 0 == $mesh2d && 1 == $tm && 0 == $err_TM1 ) then
    set tm = 0
    echo "            ... using tm=0 mode"
  else if ( 0 == $mesh2d && 1 == $tm && 1 == $err_TM1 ) then
    set tm = 1
    echo "            ... using tm=1 mode"
  endif
  ( $timer $adjointErrorEst $verb -etol $etol -limiter $limiter -tm $tm $binaryIO $y_is_spanwise $blcc $buffLim $subcell > adjointErrorEst.${xx}.out ) >&! $timeInfo
  set exit_status = $status
  if ( -o $timeInfo ) then
    set adjointErrorEstTime = `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
    \rm -f $timeInfo
  endif
  if ($exit_status != 0) then
    if ( $ERROR_TOL_OK == $exit_status ) then
      echo "            Adaptation tolerance satisfied"
      if ( $x > 2 ) then
        echo "            We will build final mesh, solve and exit"
        set error_ok = 1
      else
        echo "            Ignoring tolerance for adapt cycle $x"
      endif
    else
      echo "==> $adjointErrorEst failed with status = $exit_status"
      echo "==> Trying it again"
      sleep 1
      ( $timer $adjointErrorEst $verb -etol $etol -limiter $limiter -tm $tm $binaryIO $y_is_spanwise $blcc $buffLim $subcell > adjointErrorEst.${xx}.out.2 ) >&! $timeInfo
      set exit_status = $status
      if ( -o $timeInfo ) then
        set adjointErrorEstTime = `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
        \rm -f $timeInfo
      endif
      if ($exit_status != 0) then
        echo "==> $adjointErrorEst failed again, status = $exit_status"
        goto ERROR
      endif
    endif
  endif
  @ tm = $tm_hold
  
  # cleanup
  \rm -f cutPlanesADJ.plt 
  unlink Mesh.mg.c3d
  \rm -f adaptedMesh.R.c3d
  \rm -f postAdapt.ckpt >& /dev/null
  \rm -f aPlanes.dat >& /dev/null

  # archive
  mv errorEst.ADJ.dat errorEst.ADJ.${xx}.dat
  if ( -e cutPlanesErrEst.dat ) then
    mv cutPlanesErrEst.dat cutPlanesErrEst.${xx}.dat
  endif
  if ( -e cutPlanesErrEst.plt ) then
    mv cutPlanesErrEst.plt cutPlanesErrEst.${xx}.plt
  endif

  cd ../  # out of EMBED

  # analyze results before adapting mesh
  adj_get_results.pl

  if ( -e STOP ) then
    echo " ==> STOP file detected <=="
    \rm -f ./EMBED/check.*
    goto ERROR
  endif

  if ( 1 == $error_on_finest && $n_adapt_cycles == $x ) then
    echo "   Done user-requested error analysis on finest mesh, exiting"
    \rm -f ./EMBED/check.*
    # record user times
    echo $x $flowCartTime $xsensitTime $adjointCartTime $embedTime $adjointErrorEstTime $adaptTime >> $time_file
    break
  endif

  # check error/tol ratio and warn user if > 100
  #if ( $x > 1 ) then
  #    tail -1 results.dat | awk '{ if ($7 > 100.) { print "   NOTE: ERROR/ETOL ratio exceeds 100; you may like to set higher ETOL" } }'
  #endif

  # make a new mesh for next adaptation cycle 
  @ x ++

  echo "   ... in directory ADAPT"
  cd ADAPT
  
  ln -sf ../EMBED/check.* preAdapt.ckpt
  cp ../$dirname/Mesh.c3d.Info .
  # used to ln -s but now cp input file in case we have Steering_Info
  \cp -f ../$dirname/input.cntl .

  # check if this is the last mesh, if so, set nxref to final_mesh_xref
  # if requested by user
  set nxref = 0
  if ( $n_adapt_cycles == $x || 1 == $error_ok ) then
    if ( $final_mesh_xref > 0 ) then
      set nxref = $final_mesh_xref
    endif
  endif

  # check if we are refining the mesh multiple times
  # if so do sanity checks on how many times
  if ( $nxref > 0 ) then
    set MAX_XREF = 3
    if ( $nxref > $MAX_XREF ) then
      echo "       User requested $nxref extra refinements on final mesh"
      echo "       NOTE: $nxref exceeds max allowed of $MAX_XREF, resetting to $MAX_XREF"
      @ nxref = $MAX_XREF
    endif

    # check max level of refinement in current mesh and make sure we do not
    # exceed max allowed based on 21 bits
    set maxRefLev   = `grep " # maxRef" Mesh.c3d.Info  | awk '{print $1}'`
    set maxDiv      = `grep " # initial mesh divisions" Mesh.c3d.Info  | awk '{ if ( $1 < $2 ) { $1 = $2; } if ( $1 < $3 ) { $1 = $3; } print $1}'`
    # this is eq. 3.5 on pg. 84 in VKI notes of Aftosmis
    set maxRefAllow = `echo $maxDiv | awk '{maxRef21 = 20.999999312 - log($1-1)/log(2.); print int( maxRef21 ) }'`
    @ maxRefLev += $nxref
    if ( $maxRefLev >= $maxRefAllow ) then
      echo "       User requested $nxref extra refinements on final mesh"
      echo "       WARNING: requested refinement level exceeds 21 bits"
      set maxRefLev = `grep " # maxRef" Mesh.c3d.Info  | awk '{print $1}'`
      @ nxref = $maxRefAllow - $maxRefLev
      if ( $nxref < 1 ) then
        echo "       Extra refinement not possible, solving on final mesh"
        set nxref = 0
      else
        echo "       Resetting extra refinements to $nxref"        
      endif
    endif
  endif
  
  set do_map
  set updir = '..'
  if ( $nxref > 0 ) then
    set do_map = '-prolongMap'
    set xrefdir = 'XREF_INIT'
    mkdir -m 0700 -p $xrefdir
    cd $xrefdir
    set updir = '../..'
    ln -sf ../preAdapt.ckpt
    cp ../Mesh.c3d.Info .
    ln -sf ../input.cntl
    ln -sf ../Components.i.tri
  endif
  
  # logic to support interface propagation 
  set maxRef
  if ( $apc[$x] == p ) then 
    echo "         - interface propagation"
    # adapt automatically adjusts maxRef to mesh max and holds it there
    set maxRef = "-maxRef 1"
  endif

  # set threshold value
  set athUse
  set growth
  set fg_tmp
  set check_final = 0
  set auto_ath = `echo $mesh_growth[1] | awk '{printf("%d",$1)}'`
  
  if ( $auto_ath ) then
    # check if we exceed max_nCells
    set finalMesh = `echo $mesh_growth[$x] | awk '{printf("%d",$0*'$nCells')}'`
    if ( $finalMesh > $max_nCells ) then
      set fg_tmp = `echo $max_nCells | awk '{printf("%4.2f",0.98*$0/'$nCells')}'`
      set growth = "-growth $fg_tmp"
      echo "       Requested mesh growth $fg_tmp (adjusted below max_nCells)"
      set check_final = 1
      # use p-cycle here to get as close as possible to max_nCells
      # without overshooting. a-cycles tag all split-cells so it may not
      # be possible to stay below max_nCells. also, it gives smoother
      # final meshes
      echo "       Forcing interface propagation to reach max_nCells"
      set maxRef = "-maxRef 1"
    else
      set growth = "-growth ${mesh_growth[$x]}"
      echo "       Requested mesh growth ${mesh_growth[$x]}"
    endif
  else
    # use user specified array
    echo "         Adapting mesh: threshold = " $ath[$x]
    
    # check if threshold exceeds $max_nCells, if so increase ath and exit 
    set checkTh = 1
    set nloop = 0 # safety counter
    while ( $checkTh )
      adapt -adjoint -i ${updir}/$dirname/Mesh.mg.c3d -v -t $ath[$x] -b $buf $maxRef $is2D $prespec $y_is_spanwise $sfc -stats > adapt.stats
      if ($status != 0) then
        echo "==> ADAPT failed while running in STATS mode"
        break
      endif
      @ nloop ++
      set nCVs = `grep " nCVs = " adapt.stats | awk '{printf "%d", $4 }'`
      set tagged = `grep " tagged hexes" adapt.stats | awk '{printf "%d", $3 }'`
      if ( $mesh2d ) then
        @ nCells = $nCVs + ( 3 * $tagged )
      else
        @ nCells = $nCVs + ( 7 * $tagged )
      endif
      if ( $nCells > $max_nCells ) then
        @ ath[$x] *= 2
      else
        set checkTh = 0
      endif
      # safety check against infinite loops
      if ( $nloop > 5 ) then 
        echo "NOTE: Cannot find appropriate threshold - nCells is reasonably close to max_nCells"
        set checkTh = 0
      endif
    end # while
    if ( $nloop > 1 ) then
      echo "         - adjusted threshold to $ath[$x] due to max_nCells"
      echo "Adjusted threshold to stay below $max_nCells cells" > ${updir}/STOP
    endif
    \rm -f adapt.stats >& /dev/null
    set athUse = "-t $ath[$x]"
  endif # auto
  
  # refine mesh
  if ( 0 == $refine_all_cells ) then
    ( $timer adapt -adjoint -i ${updir}/$dirname/Mesh.mg.c3d -v $athUse -b $buf $maxRef $growth $is2D $prespec $y_is_spanwise $do_map $sfc > adapt.${xx}.out ) >&! $timeInfo
    set exit_status = $status
  else
    ( $timer adapt -RallCells -i ${updir}/$dirname/Mesh.mg.c3d -v $is2D $prespec $y_is_spanwise $sfc > adapt.${xx}.out ) >&! $timeInfo
    set exit_status = $status
  endif
  if ( -o $timeInfo ) then
    set adaptTime = `grep -e "user " $timeInfo | awk '{printf "%d", $2 }'`
    \rm -f $timeInfo
  endif
  if ($exit_status != 0) then
    if ( $ZERO_TAGGED_HEXES == $exit_status ) then
      echo "      Zero cells were tagged in ADAPT"
      cp -f ${updir}/$dirname/Mesh.mg.c3d adaptedMesh.R.c3d
      cp -f ${updir}/$dirname/Mesh.c3d.Info .
      cp -f preAdapt.ckpt postAdapt.ckpt
    else
      echo "==> ADAPT failed: status = $exit_status"
      goto ERROR
    endif
  endif
  
  if ( $ZERO_TAGGED_HEXES != $exit_status ) then
    set nCells = `grep "totNumHexes     =" adapt.${xx}.out | awk '{print $3}'`
  endif
  if ( $nCells > $max_nCells ) then
    echo " ==> Number of cells $nCells exceeds user defined limit $max_nCells ... EXITING"
    \rm -f adaptedMesh.R.c3d postAdapt.ckpt ${updir}/EMBED/check.*
    break
  endif

  # check actual mesh growth and threshold
  if ( $auto_ath ) then
    if ( 0 == $refine_all_cells ) then
      set actual_growth = `grep "Mesh growth factor" adapt.${xx}.out | tail -1 | awk '{print $7}'`
      set threshold     = `grep "Mesh growth factor" adapt.${xx}.out | tail -1 | awk '{print $13}'`
      echo "       Actual mesh growth $actual_growth with threshold $threshold"
    else
      echo "       Refined all cells, mesh growth is 8"
    endif
  endif

  # check if final mesh size is close enough to max_nCells, if yes exit
  set close_enough = 0
  if ( $check_final ) then
    set close_enough = `echo $actual_growth | awk '{ if ( 1.25*$1 > '$fg_tmp' ) {print 1} else {print 0}}'`
  endif
  if ( $close_enough ) then
    echo "       Mesh contains close to max_nCells"
    date >> ${updir}/STOP
    echo "Reached final mesh size." >> ${updir}/STOP
  endif

  # actually do extra refinements
  if ( $nxref > 0 ) then
    # return out of XREF_INIT
    cd ../
    echo "       ... $nxref extra refinement(s) of final mesh"
    set cp_from = '../XREF_INIT'
    # initialize extra refinement passes
    set xa = 1
    # main loop controlling extra refinement
    while ($xa <= $nxref)
      set xa_dir = XREF_${xa}
      mkdir $xa_dir
      cd $xa_dir
      ln -s ../input.cntl
      ln -s ../Components.i.tri
      cp ${cp_from}/Mesh.c3d.Info .
      set do_map
      if ( $xa < $nxref ) then
        set do_map = '-prolongMap'
        # set do_map = '-prolongMap -Zcut 1'
      endif
      echo "           Working in ${xa_dir}"
      if ( $auto_ath ) then
        # attenuate growth by preset fraction, i.e. take the final growth,
        # subtract 1, multiply by attenuation fraction, and add 1 back
        set xref_growth = `echo $mesh_growth[$x] | awk '{printf("%.3f",($0-1.)*'$xref_fraction[$xa]'+1.)}'`
        # safety fuse => minimum growth is 1.1
        set xref_growth = `echo $xref_growth | awk '{ if ( 1.1 > $0 ) {print 1.1} else {print $0}}'`
        set growth = "-growth $xref_growth"
        echo "           Requested mesh growth $xref_growth"
      endif
      # refine mesh again
      adapt -adjoint -i ${cp_from}/adaptedMesh.R.c3d -in ${cp_from}/postAdapt.ckpt -v $athUse -b $buf $growth $is2D $prespec $y_is_spanwise $do_map $sfc >& adapt.out 
      set exit_status = $status
      if ($exit_status != 0) then
        if ( $ZERO_TAGGED_HEXES == $exit_status ) then
          echo "      Zero cells were tagged in ADAPT"
          cp -f ../../$dirname/Mesh.mg.c3d adaptedMesh.R.c3d
          cp -f ../../$dirname/Mesh.c3d.Info .
          cp -f preAdapt.ckpt postAdapt.ckpt
        else
          echo "==> ADAPT failed: status = $exit_status"
          goto ERROR
        endif
      endif
      if ( $ZERO_TAGGED_HEXES != $exit_status ) then
        set nCells = `grep "totNumHexes     =" adapt.out | awk '{print $3}'`
      endif
      # check actual mesh growth and threshold
      if ( $auto_ath ) then
        set actual_growth = `grep "Mesh growth factor" adapt.out | tail -1 | awk '{print $7}'`
        set threshold     = `grep "Mesh growth factor" adapt.out | tail -1 | awk '{print $13}'`
        echo "           Actual mesh growth ${actual_growth}, threshold ${threshold}, cells ${nCells}"
      endif
      # check that max_nCells is not exceeded, if yes use largest mesh built so far
      if ( $nCells > $max_nCells ) then
        echo "           ==> Number of cells $nCells exceeds user defined limit $max_nCells"
        \rm -f adaptedMesh.R.c3d postAdapt.ckpt
        # overwrite nxref so we can copy the best mesh to the working adapt dir
        if ( $xa == 1 ) then
          set nxref = 'INIT'
          set nCells = `grep "totNumHexes     =" ${cp_from}/adapt.${xx}.out | awk '{print $3}'`
        else
          set nxref = $xa
          set nCells = `grep "totNumHexes     =" ${cp_from}/adapt.out | awk '{print $3}'`
        endif
        echo "           ... using mesh from directory XREF_${nxref}"
        # back up to ADAPT
        cd ..
        # exit while loop
        break
      endif
      # cleanup
      \rm -f ${cp_from}/adaptedMesh.R.c3d ${cp_from}/postAdapt.ckpt ${cp_from}/Mesh.c3d.Info
      set cp_from = "../${xa_dir}"
      cd ..
      @ xa ++
    end # end while
    \mv -f XREF_${nxref}/adaptedMesh.R.c3d .
    \mv -f XREF_${nxref}/Mesh.c3d.Info .
    \mv -f XREF_${nxref}/postAdapt.ckpt .
  endif # end extra refinement
  
  # cleanup files
  \rm -f aPlanes.dat >& /dev/null
  if ( 1 != $use_warm_starts ) then
    \rm -f postAdapt.ckpt
  endif

  if (1 == $keep_error_maps) then      # ...keep the error-fortified ckpt files -MA
      ls ../EMBED/check.* >& /dev/null #    verify existence before destroying anything
      if ( 0 == $status ) then
          \rm ../EMBED/Restart.*.file >& /dev/null
          \mv  ../EMBED/check.*   ../EMBED/Restart.{$xx}.file
      endif
  else
      \rm -f  ../EMBED/check.*
  endif
  
  cd ../ # out of ADAPT, back in top level dir

  # record user times
  @ xm1 = $x - 1
  echo $xm1 $flowCartTime $xsensitTime $adjointCartTime $embedTime $adjointErrorEstTime $adaptTime >> $time_file

  # create next analysis directory and setup files
  set previous = $dirname
  if ( $x < 10 ) then
    set dirname = adapt0${x}
    set xx      = 0${x}
  else
    set dirname = adapt${x}
    set xx      = ${x}
  endif

  mkdir -m 0700 -p $dirname
  cd $dirname

  echo "  "
  echo " Working in directory" $dirname
  ln -s ../input.c3d
  # copy input.cntl to bring over any steering updates, e.g. TargetCL
  if ( -e ../input.new.cntl ) then
    echo " Switching to new input.cntl file"
    \mv ../input.new.cntl input.cntl
  else
    cp ../$previous/input.cntl .
  endif
  ln -s ../Config.xml
  ln -s ../Components.i.tri
  if ( ( 1 == $use_preSpec ) && -e ../preSpec.c3d.cntl) then
    ln -s ../preSpec.c3d.cntl
  endif

  if ( $mesh2d ) then
    mv ../ADAPT/adaptedMesh.R.c3d Mesh.c3d
  else
    mv ../ADAPT/adaptedMesh.R.c3d Mesh.R.c3d
  endif

  mv ../ADAPT/Mesh.c3d.Info .
  
  if ( 1 == $use_warm_starts ) then
    mv ../ADAPT/postAdapt.ckpt Restart.file
    cp ../$previous/*.dat .
    \rm -f loads*.dat *ADJ*.dat
    if ( -e vortexInfo.dat ) then
      \rm -f vortexInfo.dat
    endif
    if ( -e earthBL.Info.dat ) then
      \rm -f earthBL.Info.dat
    endif
    # increase cycles for warm starts
    @ it_fc += $ws_it[$x]
  endif

  # vortex farfield input file
  if ( -e ../vortexInfo.dat ) then
    ln -s ../vortexInfo.dat
  endif

  # earth bl input file
  if ( -e ../earthBL.Info.dat ) then
    ln -s ../earthBL.Info.dat
  endif
  
  echo " Mesh contains $nCells hexes"

  # mark safe-to-restart point
  echo "$x" >! ../ADAPT/ADAPT_RESTART
  
  # cleanup
  \rm -f ../$previous/ADAPT_FLOW  ../$previous/ADAPT_ADJ >& /dev/null
end # end while

# time totals
if ( -o $time_file ) then
  echo "# Totals" >> $time_file
  @ col = 2 # start at column 2 and go up to column 7
  while ( $col < 8 )
    # sum times in non-comment lines, append to end of file
    awk '{if ( "#" != $1 ) { ( times += $'$col') } }; END { { if ('$col' == 2) { printf "# %s", times } else if  ('$col' == 7) { printf " %s\n", times } else { printf " %s", times } } }' $time_file >> $time_file
    @ col ++
  end 
endif

echo " Done aero.csh"
exit 0

ERROR:
  exit 1

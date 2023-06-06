#!/bin/csh -f

# $Id: aero.csh,v 1.27 2022/10/26 13:58:40 mnemec Exp $

# AERO: Adjoint Error Optimization
# Script to drive adjoint-based mesh refinement

# ATTENTION: requires Cart3D release 1.5 or newer

# M. Nemec, Marian.Nemec@nasa.gov
# Oct 2006, last update: Oct 2021

# Help:
# ------
# % ./aero.csh help

# Read tips, hints and documentation in $CART3D/doc/adjoint
# See examples in $CART3D/cases/samples_adapt

# Set user specified options below, defaults are suggested

# -------------
# Basic options
# -------------

# Number of adaptation cycles, e.g. if you pick 8 then the run will terminate
# after the flow solve in adapt08. Min value is 0, max value is 99.
set n_adapt_cycles = 7

# maxR for initial mesh (cubes)
set maxR = 7

# Spanwise orientation (-y_is_spanwise flag in flowCart)
set y_is_spanwise = 0    # {Yes, No} = {1, 0}

# Set mesh2d = 1 for 2D cases, mesh2d = 0 for 3D cases
set mesh2d = 0

# ----------------
# Advanced options
# ----------------

# Uncomment next line to control thread affinity (improve parallel performance)
# on linux machines
#                          ...generally safer on hyperthreaded systems
#setenv KMP_AFFINITY scatter
#                     ...may be slightly faster when not hyperthreaded
#setenv KMP_AFFINITY compact

# ----- Flow and adjoint solver settings -----

# Number of fine-grid flowCart iterations on initial mesh
set it_fc = 100
# Additional flowCart iterations on each new mesh
# cycle         1   2   3   4   5   6   7   8   9  10
set ws_it = ( 100 150 150 200 200 200 200 200 250 250 )
# Number of fine-grid adjointCart iterations on each mesh
set it_ad = 150

# Number of flowCart multigrid levels (default=3)
set mg_fc = 4
# Number of adjointCart multigrid levels (usually same as flowCart)
set mg_ad = 4

# Limiter: default 1 (Barth-Jespersen) is the most accurate, 2 (van Leer) is
# smoother and may offer deeper convergence. Limiter 5 (minmod) is most robust
# and 0 means no limiter.
set limiter = 2

# Functional averaging window in terms of flowCart mg-cycles. This is useful
# for cases that do not converge to steady-state. The averaged functional is
# reported in the fourth column of fun_con.dat and in all_outputs.avg.dat
# (default: avg_window = 1).
set avg_window = 1

# ----- Mesh adaptation settings ------

# Maximum number of cells in the penultimate mesh, i.e. number of cells allowed
# in the working mesh for the final embedding.  The error estimation step
# requires ~6.9 GB per million cells.  You can use this to gauge the largest
# mesh your memory resources allow. Default value is 9M, which assumes a
# machine with 64 GB of memory.
set max_cells2embed = 9000000

# Specify mesh growth for each adaptation cycle. Mesh growth should be greater
# than 1 and less than or equal to 8. Specifying 8 means that you allow
# refinement of every cell in the mesh.  Recommended minimum growth is 1.1. We
# found the sequence below to work well for many problems: if your initial mesh
# has roughly 10,000 cells, then after 10 adaptations it will surpass 10
# million cells. (For 2D cases, we recommend growth factors of 1.2 for first
# two cycles and 1.4 for the rest.) If you enter 0 for any cycle, then the mesh
# growth is selected automatically in that cycle.
# cycle               0   1   2   3   4   5   6   7   8   9  10
set mesh_growth = ( 1.5 1.5 2.0 2.0 2.0 2.0 2.0 2.5 2.5 2.5 2.5 )

# Mesh growth can be selected automatically by enabling the auto_growth
# option. This is a new feature, where aero.csh sets the refinement threshold
# to the mean of the error distribution. This feature frequently yields better
# results than the default mesh_growth array.
set auto_growth = 0    # 0=no, 1=yes, default=0

# Set apc: adapt or interface propagation cycle
# a = adapt
# p = propagate interfaces (adapt mesh without reducing finest cell size)
# Use p-cycles sparingly. We recommend using one initial p-cycle to reduce the
# bias of the initial mesh. P-cycles should also be used once the volume mesh
# over-refines your surface triangulation.
# cycle     0 1 2 3 4 5 6 7 8 9 10
set apc = ( p a a a a a a a a a a )

# ----- Customization of initial mesh -----

# Use file name preSpec.c3d.cntl for preSpec regions (either BBoxes or XLevs
# for cubes or ABoxes for adapt) 0=no, 1=yes, default=0
set use_preSpec = 0

# Initial mesh parameters (cubes)
set cubes_a = 10    # angle criterion (-a)
set cubes_b = 3     # buffer layers   (-b)

# Remesh: use refMesh.{mg.c3d,c3d.Info} to guide cell density of initial mesh
# (cubes -remesh option). If turned on, we recommend increasing mg_init
# (initial number of multigrid levels, see flag below) to 3 or 4. Note that the
# cubes_a flag is automatically set to 20 if it is less than 20.  0=no, 1=yes,
# default=0
set use_remesh = 0

# Internal mesh (cubes): 0=no, 1=yes, default=0
set Internal = 0

# ----- Run control -----

# Do error analysis on finest mesh? 0=no, 1=yes, default=0
set error_on_finest = 0

# Exit on reaching the finest mesh, do not flow solve there. This is useful if
# you wish to run a different solver (e.g. the MPI version) on the finest mesh.
# The final adapt?? directory holds the final mesh, all the input files, as
# well as a FLOWCART file that contains the appropriate command line.
set skip_finest = 0    # 0=no, 1=yes, default=0

# Maximum level of refinement allowed in the mesh. This controls the size of
# the smallest cell in the mesh. Default value is 21, which is the maximum
# supported by Cart3D. If max_ref_level is reached before n_adapt_cycles, then
# propagation (p) cycles will be executed until n_adapt_cycles is satisfied.
set max_ref_level = 21

# Set extra refinement levels for the final mesh. This allows you to adapt the
# mesh multiple times with the same error map in the last adapt cycle, thereby
# bypassing the flow, adjoint, and error estimation steps. Use with caution:
# the mesh should be fine enough so that the error estimate is decreasing -
# preferably the solution should be in the Richardson region. This helps
# circumvent the memory limitations of the error estimation code. Default value
# is 0 and maximum allowed value is 3.
set final_mesh_xref = 0

# ---------------------------------------------------
# EXPERT user options: flags below are rarely changed
# ---------------------------------------------------

# Cut-cell gradients: 0=best robustness (default), 1=best accuracy
# If mesh2d=1, then we set tm=1 automatically
set tm = 0

# CFL number: usually ~1.1 but with power may be lower, i.e. 0.8
set cfl = 1.1
# Minimum CFL number, used in case of convergence problems with flowCart
set cflmin = 0.8

# Adaptation error tolerance. Run terminates if the error estimate falls
# below this value. At least 2 cycles will be run before terminating based on
# this tolerance to avoid triggering based on an inappropriate initial mesh.
set etol = 0.000001

# Grid sequencing (-gs) or multigrid (-mg) (-mg default)
# flowCart
set mg_gs_fc = '-mg'
# adjointCart
set mg_gs_ad = '-mg'

# Full multigrid: default is to use full multigrid, except in cases with power
# boundary conditions (automatic with warm starts)
set fmg = 1    # 0=no, 1=yes, default=1

# Polynomial multigrid, helps certain tough cases converge deeper
set pmg = 0    # 0=no, 1=yes, default=0

# Buffer limiter: improves stability for flows with strong off-body shocks
set buffLim = 0    # 0=no, 1=yes, default=0

# Number of multigrid levels for initial mesh (default 2, ramps up to mg_fc/ad)
set mg_init = 2

# Set names of executables (must be in path)
set flowCart        = flowCart
set xsensit         = xsensit
set adjointCart     = adjointCart
set adjointErrorEst = adjointErrorEst_quad

# MPI prefix: uncomment next line and also remember to set the correct flowCart
# executable above. If not running MPI, comment out next line.
# set mpi_prefix = 'mpiexec -n 16'

# Flow solver warm-starts: 0=no, 1=yes, default=1
set use_warm_starts = 1

# Subcell resolution: 0=no, 1=yes, default=0
set subcell = 0

# Run adjoint solver in 1st-order mode. In hard cases where aero.csh
# consistently falls back on 1st-order mode after trying more accurate
# settings, this setting will short-circuit the process and go directly to
# 1st-order adjoints. The flow solution remains 2nd-order. The adjoints should
# still provide a consistent set of error estimates, which is probably safe for
# _relative_ errors and adaptive meshing. However, use caution: the error
# estimates may be inaccurate with respect to a 2nd-order run.
# default 0=auto, 1=force 1st order
set adj_first_order = 0

# In 3D cases with tm=1, error estimation is still done with tm=0 for
# robustness. This flag forces aero.csh to use tm=1 in error estimation. This
# is recommended only for simple (academic) cases that converge well. In
# general, the default (0) setting is _strongly_ recommended.
set err_TM1 = 0    # default 0=off, 1=force tm 1 for error estimation

# In 3D cases with tm=1, adjoint solutions are still done with tm=0 for
# robustness. This flag forces aero.csh to use tm=1 for the adjoint solves. In
# general, the default (0) setting is _strongly_ recommended. Note that
# adjoint convergence is monitored, so if divergence occurs then tm=0 is set
# during runtime.
set adj_TM1 = 0    # default 0=off, 1=force tm 1 for adjoint solutions

# If set, the flow solve on the final mesh will use *at least* this many
# iterations. flowCart will use whichever is greater: (1) this value (2) the
# ws_it array entry. This helps when you do not know how many adaptation cycles
# will be necessary to meet the termination conditions and you would like a
# well converged answer on the final mesh. Default value is 0.
set ws_it_min_final = 0

# When running optimization with adaptive meshing, this flag allows different
# levels of adjoint convergence for the mesh adaptation functional vs. the
# design functionals. The iterations for the adjoint solutions used in gradient
# computations are it_ad + delta_it_ad. The main idea is to use fewer
# iterations when building the mesh (for speed) and go for deeper converge in
# the gradient adjoints (for better accuracy).  Default value is 0.
set delta_it_ad = 0

# Keep final error map in EMBED/Restart.XX.file, useful for cubes -remesh
set keep_error_maps = 0    # default 0=no, 1=yes

# Refine all cells: useful for uniform mesh refinement studies. This overrides
# the error map and forces adapt to refine all cells. The adjoint correction
# term and error estimate are reported.
set refine_all_cells = 0    # default 0=no, 1=yes

# adapt buffers (default 1)
set buf = 1

# Set the number of multigrid levels when aero.csh drops down to pMG due to
# convergence problems. Default value is 2, which means no geometric
# multigrid. In subsonic cases, 3 multigrid levels may be better. Note that
# this flag has no effect on the pmg multigrid levels when the pmg flag is
# selected above.  It influences only the automatic run control of aero.csh.
set mg_pmg_auto = 2

# Fine tuning of mesh growth when performing extra refinements on the final
# mesh, i.e. when $final_mesh_xref>0 and $mesh_growth are being used.
# The mesh growth for each extra refinement is given by:
# ($mesh_growth-1)*$xref_fraction+1
# The main idea is that as extra refinement cycles are performed, the
# adaptation focuses on only the highest error cells. This is where the error
# map is most accurate and most adaptation is required. Each value should be
# between 0.2 and 1, and at most three extra refinements are allowed.
set xref_fraction = ( 1.0 1.0 0.8 )

# Safety factor used in aero_getResults.pl to terminate the run if the error
# indicator value increases by more than this factor in successive
# cycles. Default value is 8.
set error_safety_factor = 8

# Adaptation restart: Alternative to command line argument 'restart'
set adapt_restart = 0

# Adaptation jumpstart from existing mesh: Alternative to command line
# argument jumpstart
set adapt_jumpstart = 0

# Write Tecplot output files in binary format
set binaryIO = 1    #  default 1=yes, 0=no

# Verbose mode for executables [flowCart/xsensit/adjointCart/adjointErrorEst]
# 0=no, 1=yes, default=0
set verb = 0

# minimum mesh growth
set min_growth = '1.1'

# Adaptation threshold array: To set ath manually, unset mesh_growth
# and set the ath array (uncomment following two lines):
# cycle      0  1 2 3 4 5 6 7 8 9 10

#set ath = ( 32 16 8 4 2 1 1 1 1 1  1 )
#unset mesh_growth

# Output cell-wise errors, useful for making histograms: 0=off, 1=yes,
# default=0
set histo = 0

# ---------------------------------------------
# STOP: no user specified parameters below here
# ---------------------------------------------

# parse command line
while ( $#argv )
  if ( "$argv[1]" == restart || "$argv[1]" == '-restart' ) then
    @ adapt_restart = 1
  else if ( "$argv[1]" == jumpstart || "$argv[1]" == '-jumpstart' ) then
    @ adapt_jumpstart = 1
  else if ( "$argv[1]" == skipfinest || "$argv[1]" == '-skipfinest' ) then
    @ skip_finest = 1
  else if ( "$argv[1]" == archive || "$argv[1]" == '-archive' ) then
    echo 'Generating run archive'
    aero_archive.csh
    exit 0
  else if ( "$argv[1]" == help || "$argv[1]" == '-help' || "$argv[1]" == '--help' || "$argv[1]" == '-' || "$argv[1]" == '-h' ) then
    cat << HELP
Usage:

 ./aero.csh [help] [restart or jumpstart] [skipfinest] [archive]

 Use 'restart' or '-restart' to run more adaptation cycles. Handles abnormal
 exits due to machine crashes, etc.  To restart from a specific 'adapt??'
 directory, delete all adapt directories that follow it. For example, to
 restart from adapt06 in a run that finished at adapt08, delete adapt07 and
 adapt08 and restart. This flag is ignored if there are no 'adapt??'
 directories and an error is reported if the file AERO_FILE_ARCHIVE.txt is
 detected.

 Use 'jumpstart' or '-jumpstart' to start from a given mesh.  Put a
 Mesh.c3d.Info file and a mesh file (Mesh.mg.c3d, Mesh.R.c3d, or Mesh.c3d) in
 the same directory as aero.csh (and other inputs) and the run will start from
 this mesh.

 Use 'skipfinest' or '-skipfinest' to skip solving on the finest mesh. This is
 used when wishing to run a different solver on the finest mesh, e.g. the MPI
 version of flowCart.  The final adapt directory contains the final mesh, all
 the input files and a FLOWCART.txt file that contains the command line. Note
 that the BEST link points to the previous 'adapt??' directory.

 Use 'archive' or '-archive' to generate a run archive.  This option deep
 cleans the run directory tree, keeping only the essential output files.  Once
 archived, restarts are not possible.  This option simply calls the
 aero_archive.csh script.

 Script returns 0 on success and 1 on error. Read tips, hints and documentation
 in \$CART3D/doc/adjoint. Must-have files: Components.i.tri, input.c3d and
 input.cntl. Use 'touch STOP' to force a stop after the next flow solve.

 Setting environment variable debug_verbose (% setenv debug_verbose) triggers
 more verbose output.

HELP
exit 0
  else
    echo "ERROR: $argv[1] unknown option"
    echo '   ... Try ./aero.csh help'
    goto ERROR
  endif
  shift
end # while

if ( 1 == $adapt_restart && 1 == $adapt_jumpstart ) then
  echo 'ERROR: conflicting options, cannot both restart and jumpstart'
  echo '   ... Try ./aero.csh help'
  goto ERROR
endif

# Remove any aliases on certain system commands
# Avoids using \ everywhere (thanks GRA)
unalias cp
unalias ls
unalias rm
unalias mv
unalias grep
unalias awk

limit stacksize unlimited

if ( 0 == $y_is_spanwise ) then
  set y_is_spanwise
else
  set y_is_spanwise = -y_is_spanwise
endif

if ( 0 == $binaryIO ) then
  set binaryIO
else
  set binaryIO = '-binaryIO'
endif

if ( 0 == $verb ) then
  set verb
else
  set verb = '-v'
endif

if ( 0 == $Internal ) then
  set Internal
else
  set Internal = '-Internal'
endif

if ( 0 == $fmg ) then
  set fmg = '-no_fmg'
else
  set fmg
endif

if ( 1 == $pmg ) then
  set pmg = '-pmg'
else
  set pmg
endif

if ( 0 == $subcell ) then
  set subcell
else
  set subcell = '-subcell'
endif

if ( 0 == $buffLim ) then
  set buffLim
else
  set buffLim = '-buffLim'
endif

if ( 0 == $histo ) then
  set histo
else
  set histo = '-histo'
endif

set run_root = `pwd`

if ( 1 == $adapt_jumpstart ) then
  if ( ! -d JUMPSTART ) then
    mkdir -m 0700 -p JUMPSTART
  endif

  if ( -e Mesh.c3d.Info ) then
    mv -f Mesh.c3d.Info JUMPSTART/.
  endif

  if ( -e Mesh.mg.c3d ) then
    mv -f Mesh.mg.c3d JUMPSTART/.
  else if ( -e Mesh.R.c3d ) then
    mv -f Mesh.R.c3d JUMPSTART/.
  else if ( -e Mesh.c3d ) then
    mv -f Mesh.c3d JUMPSTART/.
  endif
endif

@ close_enough = 0
@ max_nCells   = 7 * $max_cells2embed

# check etol value
set check = `echo $etol | awk '{ if ( $1 == $1 + 0 ) { print 0 } else { print 1 } }'`
if ( $check ) then
  echo "ERROR: etol = $etol must be a number"
  goto ERROR
endif
set check = `echo $etol | awk '{ if ( $1 < 1.e-12 ) { print 1 } else { print 0 } }'`
if ( $check ) then
  echo 'WARNING: etol too small, setting to 1.e-12'
  set etol = '1.e-12'
endif
set etol = `echo $etol | awk '{printf("%e",$1)}'`

# check if safe to run
if ( 1 != $adapt_restart ) then
  # restart is off
  ls -d adapt?? >& /dev/null
  if ( 0 == $status ) then
    echo 'adapt?? directories exist'
    echo "Use '\rm -r adapt??' before starting a new run"
    echo "For restarts, please use 'aero.csh restart'"
    echo "For help, please use 'aero.csh help'"
    goto ERROR
  endif
else
  # restart is on
  ls -d adapt?? >& /dev/null
  if ( $status ) then
    # there are no adapt dirs so start new
    if ( -e Mesh.mg.c3d && -e Mesh.c3d.Info ) then
      echo 'Detected Mesh.mg.c3d and Mesh.c3d.Info in run directory'
      echo "To jumpstart from an existing mesh, please use 'aero.csh jumpstart'"
      echo "For help, please use 'aero.csh help'"
      echo "Use '\rm Mesh.mg.c3d Mesh.c3d.Info' before starting a new run"
      goto ERROR
    endif
    @ adapt_restart = 0
    echo 'Restart requested with no adapt directories, starting new run'
  else
    # there are adapt dirs
    if ( -e AERO_FILE_ARCHIVE.txt ) then
      # danger - are we trying to restart an archived run?
      echo 'ERROR: Restart requested but AERO_FILE_ARCHIVE.txt is present'
      echo '   ... This is a run archive, restart not possible'
      echo '   ... To start a new run, please remove AERO_FILE_ARCHIVE.txt'
      echo '   ... and adapt?? directories'
      goto ERROR
    else
      if ( ! -d adapt00/FLOW ) then
        # mesh in adapt00 may not be ready so start over
        @ adapt_restart = 0
        rm -rf adapt00 >& /dev/null
      endif
    endif
  endif
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
  @ tm      = 1
  @ adj_TM1 = 1
  @ max_nCells   = 4 * $max_cells2embed
endif

# use mgTree with cubes aniso flag
if ($?C3D_ANISO_X) then
  echo "C3D_ANISO_X is set to $C3D_ANISO_X -- using mgTree"
  set mgprep = mgTree
endif

date
echo ''

# show which aero.csh
ls -otr `which $0`
echo ''

if ( $?CART3D ) then
  echo 'CART3D env variable set to:' $CART3D
else
  echo 'ERROR: CART3D env variable not found'
  goto ERROR
endif
if ( $?CART3D_ARCH ) then
  echo 'CART3D_ARCH env variable set to:' $CART3D_ARCH
else
  echo 'WARNING: CART3D_ARCH env variable not found'
endif
echo ''

# check codes
ls -otr `which aero_codeCheck.csh` >& /dev/null
if ( $status ) then
  echo 'ERROR: aero_codeCheck.csh script not found'
  echo '   ... Check that $CART3D/bin is in your path'
  goto ERROR
endif
aero_codeCheck.csh $mgprep $flowCart $xsensit $adjointCart $adjointErrorEst
if ( $status ) then
  echo 'ERROR: aero_codeCheck.csh check failed'
  goto ERROR
endif

# flowCart command with mpi
if ( $?mpi_prefix ) then
  set flowCart = "$mpi_prefix $flowCart"
endif

# timer: keep track of user time
set timer
set timeInfo = /dev/null
if ( -e /usr/bin/time ) then
  set timer = "/usr/bin/time -p"
  set timeInfo = TIME
else
  echo "NOTE: /usr/bin/time not found, user time will not be reported"
endif

# check input array length
if ( $n_adapt_cycles > $#ws_it ) then
  @ last = $ws_it[$#ws_it]
  @ len  = $#ws_it
  while ( $len < $n_adapt_cycles )
    set ws_it = ( $ws_it $last )
    @ len ++
  end
  echo "Adjusted length of ws_it array to $#ws_it"
  echo 'New array is ' $ws_it
  echo ''
endif

if ( $n_adapt_cycles > $#apc ) then
  set last = $apc[$#apc]
  @ len  = $#apc
  while ( $len < $n_adapt_cycles )
    set apc = ( $apc $last )
    @ len ++
  end
  echo "Adjusted length of apc array to $#apc"
  echo 'New array is ' $apc
  echo ''
endif

if ( 1 == $auto_growth ) then
  unset mesh_growth
  unset ath
else
  if ( $?mesh_growth && $?ath ) then
    echo 'ERROR: Detected both mesh_growth and ath arrays'
    echo '   ... Please pick one and unset the other'
    goto ERROR
  endif
  if ( $?mesh_growth ) then
    if ( $n_adapt_cycles > $#mesh_growth ) then
      set last = $mesh_growth[$#mesh_growth]
      @ len  = $#mesh_growth
      while ( $len < $n_adapt_cycles )
        set mesh_growth = ( $mesh_growth $last )
        @ len ++
      end
      echo "Adjusted length of mesh_growth array to $#mesh_growth"
      echo 'New array is' $mesh_growth
      echo ''
    endif
  else if ( $?ath ) then
    if ( $n_adapt_cycles > $#ath ) then
      set last = $ath[$#ath]
      @ len  = $#ath
      while ( $len < $n_adapt_cycles )
        set ath = ( $ath $last )
        @ len ++
      end
      echo "Adjusted length of ath array to $#ath"
      echo 'New array is' $ath
      echo ''
    endif
  endif
endif

# adaptation exit flag
@ error_ok = 0

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

# Warn if by the last adapt cycle we will not get to the full mg_levs
if ( $mg_init < 1 ) then
  echo "WARNING: mg_init $mg_init is less than 1, resetting to 1"
  @ mg_init = 1
else if ( $mg_init > $mg_fc ) then
  echo "WARNING: mg_init $mg_init exceeds mg_fc $mg_fc, resetting to $mg_fc"
  @ mg_init = $mg_fc
endif
@ mg_delta = $mg_fc - $mg_init
if ( $n_adapt_cycles > 0 && $n_adapt_cycles < $mg_delta ) then
  echo "WARNING: Cannot get from $mg_init mg cycles up to $mg_fc in only $n_adapt_cycles adapt cycles"
  echo "     ... Consider increasing mg_init in $0"
  echo ''
endif
unset mg_delta

# use Morton order whenever mgTree is used
set sfc
if ( $mgprep == 'mgTree' ) then
  set sfc = '-sfc M'
endif

# set return codes for error traps
# adapt return code
@ ZERO_TAGGED_HEXES = 4

# adjointErrorEst return code
@ ERROR_TOL_OK = 2

# set flowCart restart flag to null
set fc_restart

# mg_start used for initial multigrid levels
set mg_start

# buffer limiter in cut-cells
# if tm = 0, set blcc to nothing
# if tm = 1, strongly recommend using buffLimCC for stability
# disabled 08.11.20 - see if we need it later
# if ( $tm == 0 ) then
set blcc
# else
#    set blcc = -buffLimCC
# endif

# clean-up
rm -f results.dat results_*.dat fun_con.dat fun_con_*.dat >& /dev/null
rm -f all_outputs*.dat STOP >& /dev/null
rm -f AERO_FILE_ARCHIVE.txt case_check*.png xmgr.batch >& /dev/null

if ( -l ./BEST ) then
  echo 'Unlinking BEST'
  echo ''
  unlink BEST
endif

if ( -d ./BEST ) then
  echo 'ERROR: BEST should not be a directory, please remove it and try again'
  goto ERROR
endif

# Check for presence of the surface triangulation
if (! -e Components.i.tri ) then
  echo 'ERROR: Missing Components.i.tri'
  goto ERROR
endif

# set ADAPT directory, this is where new meshes are generated
if ( ! -d ADAPT ) then
  mkdir -m 0700 -p ADAPT
endif
cd ADAPT
rm -rf XREF_* >& /dev/null
if (0 == $adapt_restart ) then
  rm -f *.* >& /dev/null
endif
ln -sf ../Components.i.tri
set prespec
if ( ( 1 == $use_preSpec ) && -e ../preSpec.c3d.cntl) then
  ln -sf ../preSpec.c3d.cntl
  set prespec = '-pre preSpec.c3d.cntl'
endif
cd ..

# set EMBED directory, this is where embedded meshes are generated
if ( ! -d EMBED ) then
  mkdir -m 0700 -p EMBED
endif
cd EMBED
if ( 0 == $adapt_restart ) then
  rm -f *.* >& /dev/null
  rm -rf Error_in_* >& /dev/null
endif
ln -sf ../Components.i.tri
ln -sf ../Config.xml
cd ..

@ x = -1
@ nCells = 0

set xx      = 00
set dirname = adapt00
set dirnext = adapt00
set dirback = undef

# initialize variables for restart
if ( 1 == $adapt_restart ) then
  set rdir = `ls -1d adapt?? | tail -1`
  echo "This is a RESTART run from ${rdir}"

  # Start at adapt00 -- will reprocess all folders
  set file = 'adapt00/FLOW/functional.dat'
  if ( -e $file ) then
    set nCells = `grep "nCells (total number of control volumes):" $file | tail -1 | awk '{printf("%d",$8)}'`
  endif

  # how many adapt dirs are there?
  set nad = `ls -1d adapt?? | wc -l`
  @ nad --

  # prepare EMBED for restart
  cd EMBED
  # how many adapt files are there?
  set naf = 0
  ls -1 adapt.??.out >& /dev/null
  if ( ! $status ) then
    set naf = `ls -1 adapt.??.out | wc -l`
  endif

  if ( $naf > $nad ) then
    while ( $nad != $naf )
      set rx
      if ( $nad < 10 ) then
        set rx = 0${nad}
      else
        set rx = ${nad}
      endif

      rm -f *.${rx}.* >& /dev/null
      rm -f ../ADAPT/*.${rx}.* >& /dev/null

      ls -d Error_in_* >& /dev/null
      if ( $status == 0 ) then
        foreach ed ( Error_in_* )
          if (! -d $ed ) then
            continue
          endif
          rm -f ${ed}/*.${rx}.* >& /dev/null
        end
      endif

      @ nad += 1
    end
  endif
  # back to run directory
  cd ..

  if ( -e ADAPT/adaptedMesh.R.c3d ) then
    rm -f adaptedMesh.R.c3d >& /dev/null
  endif

  set restartON
  echo ''
else
  # not a restart
  mkdir -m 0700 -p $dirname
  cd $dirname

  # prepare generation of initial mesh
  ln -s ../Components.i.tri
  ln -s ../input.c3d
  if ( ( 1 == $use_preSpec ) && -e ../preSpec.c3d.cntl) then
    ln -s ../preSpec.c3d.cntl
  endif

  # handle remesh option
  set remesh
  if ( 1 == $use_remesh ) then
    if ( -e ../refMesh.mg.c3d && -e ../refMesh.c3d.Info  ) then
      echo 'Remesh option active on initial mesh'
      echo 'Relaxing angle criterion to 20 (-a 20)'
      ln -s ../refMesh.mg.c3d
      ln -s ../refMesh.c3d.Info
      set remesh = '-remesh'
      set cubes_a = 20
    else
      echo 'WARNING: remesh option requested but refMesh files missing'
      echo '         ... ignoring remesh request'
    endif
  endif

  # copy input file in case we have Steering_Info
  cp ../input.cntl .

  if ( 0 == $adapt_jumpstart ) then
    echo 'Building initial mesh'
    echo ''
    if (! -e input.c3d ) then
      echo 'ERROR: Missing input.c3d'
      cd ..
      goto ERROR
    endif
    if ( $cubes_b < 2 ) then
      set cubes_b = 2
    endif

    set flags = "-v -quiet -verify -no_est $reorder $is2D"
    echo "cubes $flags -a $cubes_a -maxR $maxR -b $cubes_b $prespec $Internal $remesh" >> cart3d.out
    cubes $flags -a $cubes_a -maxR $maxR -b $cubes_b $prespec $Internal $remesh >> cart3d.out
    if ($status != 0) then
      echo 'ERROR: CUBES failed in ' `pwd`
      if ( -e cart3d.out ) then
        grep 'ERROR:' cart3d.out
        grep 'ATTENTION:' cart3d.out
      endif
      cd ..
      goto ERROR
    endif
    set nCells = `grep "  hex cells  is:" cart3d.out | awk '{print $8}'`
    # check for timeout warnings
    grep ' until expiration' cart3d.out >& /dev/null
    if ( 0 == $status ) then
      echo 'cubes near expiration date:'
      grep ' until expiration' cart3d.out
    endif
  else
    echo 'Jumpstarting from a given mesh'
    echo ''
    if ( -e ../JUMPSTART/Mesh.c3d.Info ) then
      cp -f ../JUMPSTART/Mesh.c3d.Info .
    else
      echo 'ERROR: Missing Mesh.c3d.Info for jumpstart'
      echo '   ... Try ./aero.csh help'
      cd ..
      goto ERROR
    endif
    if ( -e ../JUMPSTART/Mesh.mg.c3d ) then
      cp -f ../JUMPSTART/Mesh.mg.c3d Mesh.R.c3d
    else if ( -e ../JUMPSTART/Mesh.R.c3d ) then
      cp -f ../JUMPSTART/Mesh.R.c3d Mesh.R.c3d
    else if ( -e ../JUMPSTART/Mesh.c3d ) then
      cp -f ../JUMPSTART/Mesh.c3d Mesh.c3d
    else
      echo 'ERROR: Missing Mesh.*.c3d file for jumpstart'
      echo '   ... Try ./aero.csh help'
      cd ..
      goto ERROR
    endif
  endif

  cd ..

  if ( -e user_time.dat ) then
    rm -f user_time.dat
  endif
endif

# Find maximum number of refinements using eq. 3.5 on pg. 84 in VKI notes of
# Aftosmis.
@ maxDiv = `grep " # initial mesh divisions" adapt00/Mesh.c3d.Info | awk '{ if ( $1 < $2 ) { $1 = $2; } if ( $1 < $3 ) { $1 = $3; } print $1}'`
# maxRef21 indicates the deepest possible refinement due to the 21-bit limit
@ maxRef21 = `echo $maxDiv | awk '{maxRef21 = 20.9999993121 - log($1-1)/log(2.); print int( maxRef21 ) }'`
unset maxDiv
# Track refinement level to make sure we do not exceed max allowed in 21 bits
# or the user-requested max-depth. Implemented suggestion / bug fix by GRA to
# track user specified max_ref_level separately from the 21 bit limit.
@ maxRefLev = `grep " # maxRef" adapt00/Mesh.c3d.Info  | awk '{print int( $1 )}'`
echo "Refinement level (maxRef) of initial mesh = $maxRefLev"
if ( $max_ref_level < $maxRef21 ) then
  echo "User-specified refinement level limit of $max_ref_level is active"
else
  echo "Max. number of refinements allowed = $maxRef21"
endif
echo ''

# Set failsafe run strategy in case adjoint diverges. Keywords are FLOW, TM0,
# PMG and FIRST_ORDER. These correspond to names of directories that hold the
# best converged flow solution.  The FLOW directory always holds the default
# flow solution (defined by user settings). The adjoints array indicates
# whether an adjoint solution should be attempted against the given flow
# solution, e.g. on may wish to skip trying to solve the adjoint against a tm=1
# flow state in challenging cases.

set failsafe = ( FLOW )
set adjoints = ( 1 )

if ( 1 == $tm ) then # tm=1 mode, highest accuracy
  set failsafe = ( $failsafe TM0 )
  if ( 0 == $adj_TM1 ) then
    set adjoints = ( 0 1 )
  else
    set adjoints = ( 1 1 )
  endif
endif

if ( "$pmg" != '-pmg' ) then
  # pmg has not been selected by the user
  set failsafe = ( $failsafe PMG )
  set adjoints = ( $adjoints 1 )
endif

if ( 0 == $adj_first_order ) then
  set failsafe = ( $failsafe FIRST_ORDER )
  set adjoints = ( $adjoints 1 )
else
  set failsafe = ( FLOW FIRST_ORDER )
  set adjoints = ( 0    1 )
endif

echo 'Adjoint failsafe strategy:' $failsafe '[' $adjoints ']'

# check if we are refining the mesh multiple times and do sanity check on how
# many times
if ( $final_mesh_xref > 0 ) then
  echo ''
  echo "User requested $final_mesh_xref extra refinements on final mesh"
  set MAX_XREF = 3
  if ( $final_mesh_xref > $MAX_XREF ) then
    echo "NOTE: $final_mesh_xref exceeds max allowed of $MAX_XREF, resetting to $MAX_XREF"
    @ final_mesh_xref = $MAX_XREF
  endif
  unset MAX_XREF
endif

@ tm_hold = $tm

# main loop over adaptation cycles
# exit set with break statements
while ( 1 )

  # update counters
  @ x ++
  @ xm1 = $x - 1
  @ xp1 = $x + 1

  if ( $x < 10 ) then
    set xx = 0${x}
  else
    set xx = ${x}
  endif

  # create next analysis directory and setup files
  set dirback = $dirname
  set dirname  = $dirnext
  if ( $xp1 < 10 ) then
    set dirnext = adapt0${xp1}
  else
    set dirnext = adapt${xp1}
  endif

  echo ''
  echo 'Working in directory' $dirname
  if ( $nCells > 0 ) then
    # comma delimited numbers, much easier to read, thanks DLR
    env LC_NUMERIC="en_US.UTF-8" printf "Mesh contains %'d hexes\n" $nCells
  endif

  if (! -d $dirname ) then
    mkdir -m 0700 $dirname
    if ( 0 != $status ) then
      echo "ERROR: Failed to create $dirname"
      goto ERROR
    endif
  endif

  cd $dirname

  if ( 1 == $adapt_restart ) then
    # short circuit the current directory if the mesh is already staged in the next directory
    if ( -e ../${dirnext}/Mesh.mg.c3d && ( ! -e ../${dirnext}/Mesh.R.c3d && ! -e ../${dirnext}/Mesh.c3d ) ) then
      # Check if we are close to cell limit on the next mesh. The adapt.??.out
      # file may not exist if there were Xrefs, in which case it is ok to skip
      # the embed-limit checking anyway.
      if ( -e ../ADAPT/adapt.${xx}.out ) then
        set nCells = `grep "totNumHexes     =" ../ADAPT/adapt.${xx}.out | awk '{print $3}'`
        if ( `echo $max_cells2embed | awk '{ if ( '$nCells' * '$min_growth' > $1 ) {print 1} else {print 0}}'` ) then
          @ close_enough += 1
        endif
      endif
      if ( -l ../BEST ) then
        unlink ../BEST
      endif
      ln -s $dirname ../BEST

      # Try to find the actual averaging window used by flowCart
      set actual_window = `grep "Tecplot stats file" FLOW/cart3d.out | tr '=)' ' ' | awk '{print $NF}'`
      if ($status) then
        set actual_window = $avg_window
      else if ("" == "$actual_window") then
        set actual_window = $avg_window
      endif

      aero_funCon.csh $x $actual_window

      cd ..
      echo "Restart OK, going to $dirnext"
      continue
    else if ( -d ../$dirnext ) then
      # restart state of dirnext is uncertain, redo it
      echo "Restart OK, resetting $dirnext"
      rm -rf ../$dirnext >& /dev/null
    endif
  endif

  # bring in adapted mesh
  if ( -e ../ADAPT/adaptedMesh.R.c3d ) then
    if ( $mesh2d ) then
      mv ../ADAPT/adaptedMesh.R.c3d Mesh.c3d
    else
      mv ../ADAPT/adaptedMesh.R.c3d Mesh.R.c3d
    endif
  endif
  if ( -e ../ADAPT/Mesh.c3d.Info ) then
    mv ../ADAPT/Mesh.c3d.Info .
  endif

  # update maxRefLev
  @ maxRefLev = `grep " # maxRef" Mesh.c3d.Info | awk '{print int( $1 )}'`

  # on coarse meshes run fewer multigrid levels for better convergence
  @ mg_start = $x + $mg_init
  if ( $mg_start < $mg_fc && $n_adapt_cycles > 0 ) then
    set mg_levs = $mg_start
  else
    set mg_levs = $mg_fc
  endif

  # prepare mg meshes: generate only the required number of levels
  if ( ! -e Mesh.mg.c3d ) then # restart check, skip if done
    while ( $mg_levs > 1 )
      echo "$mgprep -n $mg_levs -verifyInput $pmg $verb" >> cart3d.out
      $mgprep -n $mg_levs -verifyInput $pmg $verb >> cart3d.out
      if ( 0 == $status ) then
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
  else
    # mesh exists, make sure MG levels are correct
    if ( ! -e cart3d.out ) then
      echo "ERROR: missing cart3d.out in $dirname"
      goto ERROR
    endif
    # check how many times mgprep was called
    if ( `grep $mgprep cart3d.out | wc -l` > 1 ) then
      # find number of mesh levels
      @ mg_levs -= 1
      while ( $mg_levs > 1 ) then
        grep $mgprep cart3d.out | tail -1 | grep "\-n $mg_levs" >& /dev/null
        if ( 0 == $status ) then
          break
        else
          @ mg_levs -= 1
        endif
      end
    endif
  endif # Mesh.mg.c3d
  @ mg_levs_hold = $mg_levs

  # cleanup
  rm -f Mesh.c3d >& /dev/null
  rm -f Mesh.R.c3d >& /dev/null

  if ( ! -e input.cntl ) then
    # copy input.cntl to bring over any steering updates, e.g. TargetCL
    if ( -e ../input.new.cntl ) then
      echo "Switching to new input.cntl file"
      mv ../input.new.cntl input.cntl
    else
      cp ../$dirback/FLOW/input.cntl .
    endif
  endif

  # Determine ahead of time whether this is certain to be the last or
  # penultimate mesh.
  set onbest
  set onbm1
  if ( $n_adapt_cycles == $x || 1 == $error_ok || $maxRefLev == $maxRef21 || 2 == $close_enough ) then
    set onbest = '-best'
  else if ( $n_adapt_cycles == $xp1 || 1 == $close_enough ) then
    set onbm1 = '-bm1'
  endif

  # prepare flow and adjoint directories
  if ( -e ../Functionals.xml ) then
    aero_rmFun.pl
    set noerr
    set r_on
    if ( $?restartON ) then
      set r_on = '-r'
      unset restartON
    endif
    # suppress error estimates if at maxRef due to 21-bit limit (thanks GRA)
    if ( $maxRefLev == $maxRef21 ) then
      set noerr = '-no_error'
      echo "WARNING: cannot compute error estimates since 21-bit maxRef limit reached"
    endif
    if ( $?debug_verbose ) then
      echo "  aero_prepFun.pl $onbest $onbm1 $r_on $noerr"
    endif
    aero_prepFun.pl $onbest $onbm1 $r_on $noerr
    if ( $status ) then
      echo 'ERROR: aero_prepFun.pl failed'
      goto ERROR
    endif
  else
    # no functional file, set adjoint directory
    set adj_dir = AD_A_J
    if ( "$onbest" == '-best' ) then
      if ( 1 == $error_on_finest ) then
        set adj_dir = AD_E_J
        if ( $maxRefLev == $maxRef21 ) then
          echo "WARNING: cannot compute error estimates since 21-bit maxRef limit reached"
          unset adj_dir
        endif
      else
        unset adj_dir
      endif
    endif

    # correct directory labels if restarting
    if ( $?restartON && $?adj_dir ) then
      if ( "$onbest" != '-best' ) then
        # not on best
        if ( -d AD_E_J ) then
          mv AD_E_J $adj_dir
        endif
      endif
      unset restartON
    endif

    # make new AD dir
    if ( $?adj_dir ) then
      if ( ! -d $adj_dir ) then
        mkdir -m 0700 $adj_dir
      endif
    endif
    unset adj_dir
  endif

  # set exit point: 0 no exit, 1 before flow solve, 2 before adjoint, 3 after
  # adjoint, 4 after error analysis
  @ exit_point = 0
  if ( "$onbest" == '-best' ) then
    # default exit point is right after flow solve
    @ exit_point = 2
    if ( 1 == $skip_finest ) then
      # exit before flow solve
      @ exit_point = 1
    else
      ls -d AD_G_* >& /dev/null
      if ( 0 == $status ) then
        # exit after adjoint solve
        @ exit_point = 3
      endif
      ls -d AD_E*_* >& /dev/null
      if ( 0 == $status ) then
        # exit after error analysis
        @ exit_point = 4
      endif
      ls -d AD_A*_* >& /dev/null
      if ( 0 == $status ) then
        echo '  WARNING: there should be no AD_A*_* dirs'
      endif
    endif
  endif

  @ tm = $tm_hold
  # main flow solve
  foreach fs ( $failsafe )
    if ( ! -d $fs ) then
      mkdir -m 0700 $fs
      if ( 0 != $status ) then
        echo "ERROR: Failed to create $fs"
        goto ERROR
      endif
    endif

    cd $fs

    # DONE flag indicates good flow solution already done
    if ( -e DONE ) then
      echo "  Flow done"
      # re-initialize if mgprep failed
      @ mg_levs = `cat DONE | awk '{print $1}'`
    else
      # setup files for warm-starts
      if ( 1 == $use_warm_starts && $x > 0 ) then
        set fc_restart = '-restart'
        @ itf = $ws_it[$x]

        if ( -e ../../ADAPT/postAdapt.ckpt ) then
          mv ../../ADAPT/postAdapt.ckpt Restart.file
        endif
        if ( FLOW == $fs ) then
          set file = ( forces.dat history.dat moments.dat functional.dat )
          foreach f ( `grep -l "# Cart3D CLIC2 convergence monitor" ../../$dirback/FLOW/*.dat` $file )
            if ( -e ../../$dirback/FLOW/$f ) then
              cp -f ../../$dirback/FLOW/$f .
            endif
          end
        endif
      else   # Cycle 0 or cold-start -- use fixed it_fc
        @ itf = $it_fc
      endif

      # On final solve, meet the minimum iteration count (thanks GRA).
      if ("$onbest" == '-best' ) then
        if ($itf < $ws_it_min_final && FLOW == $fs ) then
          echo "  Note: user requested $ws_it_min_final iterations on final mesh"
          @ itf = $ws_it_min_final
        endif
      endif

      aero_setFlowLinks.csh

      # check failsafe and set appropriate flags
      if ( FIRST_ORDER == $fs ) then
        # build 1st order cntl file
        awk '{if ("RK"==$1){print $1,"  ",$2,"  ",0}else{print $0}}' ../input.cntl > input.cntl
        # no restart
        set fc_restart
        # set flowCart iters to adjointCart iters
        @ itf = $it_ad
        # restore mg_levs in case we did PMG
        @ mg_levs = $mg_levs_hold
        # undo pmg if pmg
        if ( "$pmg" == '-pmg' ) then
          unlink Mesh.mg.c3d
          unlink Mesh.c3d.Info
          if ( $mesh2d ) then
            ln -s ../Mesh.mg.c3d Mesh.c3d
          else
            ln -s ../Mesh.mg.c3d Mesh.R.c3d
          endif
          cp ../Mesh.c3d.Info .
          echo "$mgprep -n $mg_levs -verifyInput $verb" >> cart3d.out
          $mgprep -n $mg_levs -verifyInput $verb >> cart3d.out
          if ( $status != 0 ) then
            echo "ERROR: $mgprep failed trying to make $mg_levs levs ... extremely rare"
            goto ERROR
          endif

          if ( $mesh2d ) then
            unlink Mesh.c3d
          else
            unlink Mesh.R.c3d
          endif
        endif
      else if ( TM0 == $fs ) then
        # restart from FLOW
        ln -sf `ls -1tr ../FLOW/check.* | tail -1` Restart.file
        @ tm = 0
        @ itf /= 2
        set fc_restart = '-restart'

        set file = ( forces.dat history.dat moments.dat functional.dat input.cntl )
        foreach f ( `grep -l "# Cart3D CLIC2 convergence monitor" ../FLOW/*.dat` $file )
          if ( -e ../FLOW/$f ) then
            cp ../FLOW/$f .
          endif
        end
      else if ( PMG == $fs ) then
        # try for deeper convergence with flowCart via pMG restart
        unlink Mesh.mg.c3d
        unlink Mesh.c3d.Info
        if ( $mesh2d ) then
          ln -s ../Mesh.mg.c3d Mesh.c3d
        else
          ln -s ../Mesh.mg.c3d Mesh.R.c3d
        endif
        cp ../Mesh.c3d.Info .

        # rerun mgPrep to make a pmg level
        if ( $mg_pmg_auto < 2 ) then
          echo "WARNING: detected mg_pmg_auto < 2, resetting to 2"
          set mg_pmg_auto = 2
        endif
        echo "$mgprep -n $mg_pmg_auto -verifyInput -pmg $verb" > cart3d.out
        $mgprep -n $mg_pmg_auto -verifyInput -pmg $verb >> cart3d.out
        if ( $status != 0 ) then
          echo "ERROR: $mgprep failed trying to make pMG mesh ... extremely rare"
          goto ERROR
        endif

        if ( $mesh2d ) then
          unlink Mesh.c3d
        else
          unlink Mesh.R.c3d
        endif

        @ mg_levs = $mg_pmg_auto

        ln -sf `ls -1tr ../FLOW/check.* | tail -1` Restart.file
        @ tm = 0
        @ itf /= 2
        set fc_restart = '-restart'

        set file = ( forces.dat history.dat moments.dat functional.dat input.cntl )
        foreach f ( `grep -l "# Cart3D CLIC2 convergence monitor" ../FLOW/*.dat` $file )
          if ( -e ../FLOW/$f ) then
            cp ../FLOW/$f .
          endif
        end
      else
        cp ../input.cntl .
      endif

      # Clamp stats window to 90% of number of FC iters
      if ( $avg_window > 1 ) then
        set this_window = `echo $avg_window $itf | awk '{wmax = 0.9*$2; w= ($1 < wmax ? $1 : wmax); printf "%d",w }'`
        if ($this_window < $avg_window) then
          echo "WARNING: Reduced stats window to $this_window"
        endif
        set stats = "-stats $this_window"
      else
        set stats
        set this_window = 1
      endif

      set flags = "-fine -T -clic $verb $binaryIO $y_is_spanwise $buffLim $subcell $blcc $stats"

      # exit if not solving on the finest mesh
      if ( $fs == FLOW ) then
        if ( 1 == $exit_point ) then
          echo '  Skipfinest enabled, FLOW directory ready for finest mesh:'
          echo ' ' `pwd`
          echo "$flowCart $flags -N $itf $mg_gs_fc $mg_levs -limiter $limiter -tm $tm -cfl $cfl $fmg $fc_restart" >! FLOWCART.txt
          echo '  Exiting'
          cd ../..
          goto ALLDONE
        else
          echo "  Running $flowCart with $mg_levs of $mg_fc $fc_multigrid levs"
        endif
      else
        echo "  Starting $fs failsafe mode"
      endif

      if ( ("$fc_restart" == '-restart') && (! -e Restart.file ) ) then
        echo 'ERROR: Missing Restart.file'
        cd ../..
        goto ERROR
      endif

      ( $timer $flowCart $flags -N $itf $mg_gs_fc $mg_levs -limiter $limiter -tm $tm -cfl $cfl $fmg $fc_restart >> cart3d.out ) >&! $timeInfo
      set exit_status = $status
      echo '' >> cart3d.out

      # if flowCart failed then try to trap the error and / or try again in
      # robust mode or with a cold start and lower CFL
      if ( 0 != $exit_status ) then
        if ( 253 == $exit_status) then
          echo 'ERROR: file parsing error in flowCart '
          echo "       check flowCart output in $dirname ... exiting now"
          cd ../..
          goto ERROR
        endif
        # check for expiration notice in cart3d.out
        grep 'Cannot execute -- ' cart3d.out >& /dev/null
        if ( 0 == $status ) then
          echo 'ERROR: flowCart expired'
          grep 'ATTENTION:' cart3d.out
          grep 'ERROR:' cart3d.out
          cd ../..
          goto ERROR
        endif
        if ( 0 == $mesh2d && $fs != FIRST_ORDER ) then
          # check robust mode
          set isRobust = `cat input.cntl | awk 'BEGIN{ n=0; ng=0; } { if ("RK"==$1) { n++; if ("1"==$3) {ng++;} } } END{ if (n==ng) { print 1 } else { print 0 } }'`
          # try converging in robust mode if standard input file
          if ( 0 == $isRobust ) then
            echo "==> $flowCart failed with status $exit_status ... trying robust mode"
            # cleanup failed run and initialize files
            rm -f check.* checkDT.* loadsCC*.dat loadsTRI.dat *.plt *.trix *.triq >& /dev/null
            set file = ( forces.dat history.dat moments.dat functional.dat )
            foreach f ( `grep -l "# Cart3D CLIC2 convergence monitor" ../../$dirback/FLOW/*.dat` $file )
              if ( -e ../../$dirback/FLOW/$f ) then
                cp -f ../../$dirback/FLOW/$f . >& /dev/null
              endif
            end

            # build robust-mode cntl file
            mv input.cntl input.not_robust.cntl
            awk '{if ("RK"==$1){print $1,"  ",$2,"  ",1}else{print $0}}' ../input.cntl > input.cntl
            ( $timer $flowCart $flags -N $itf $mg_gs_fc $mg_levs -limiter $limiter -tm $tm -cfl $cfl $fmg $fc_restart >> cart3d.out ) >&! $timeInfo
            set exit_status = $status
            echo '' >> cart3d.out
            # swap back so we do not always use robust in later cycles
            mv input.cntl input.robust.cntl
            mv input.not_robust.cntl input.cntl
          endif
        endif
      endif # flowCart status

      if ( 0 != $exit_status ) then
        echo "==> $flowCart failed with status $exit_status ... trying with CFL $cflmin"
        @ gs_it = 2 * $it_ad
        if ( $mg_levs > 2 ) then
          ( $timer $flowCart $flags -N $gs_it $mg_gs_fc 2 -limiter $limiter -tm $tm -cfl $cflmin -no_fmg >> cart3d.out ) >&! $timeInfo
          set exit_status = $status
        else
          ( $timer $flowCart $flags -N $gs_it $mg_gs_fc $mg_levs -limiter $limiter -tm $tm -cfl $cflmin -no_fmg >> cart3d.out ) >&! $timeInfo
          set exit_status = $status
        endif
      endif

      if ( $exit_status != 0 ) then
        echo "ERROR: $flowCart failed with status ${exit_status}, read cart3d.out in" `pwd`
        if ( $dirname == adapt00 ) then
          echo "   ... It is likely that this is a setup issue, check that CART3D and CART3D_ARCH"
          echo "   ... are set and try running the case by hand in $dirname using the commands"
          echo "   ... from cart3d.out"
        else if ( "$buffLim" != '-buffLim' ) then
          echo '   ... This could be a setup issue or the case may require using -buffLim'
        endif
        cd ../..
        goto ERROR
      endif

      rm -f Restart.file >& /dev/null

      # mark as done, save mg_levs in case we have to run first-order
      echo $mg_levs > DONE

      if ( $fs == FLOW ) then
        set it_run = `tail -1 history.dat | awk '{if ("#" != $1) {print $1} else {print '$itf'}}'`
        echo "  Done $it_run flowCart cycles"
        if (-e cart3d.out) then
          grep WARNING cart3d.out | grep -v "end of run" | grep -v "levels changed"
        endif
      endif
    endif # end DONE branch: already done flowCart [Restart] or run flowCart

    # back to adapt?? dir level
    cd ..

    # Run any external post-processing to obtain final outputs, e.g. via
    # sBOOM. This needs to be done before running adjointCart because xsensit
    # needs gradients of these outputs.
    ls -d SBOOM_* >& /dev/null
    if ( 0 == $status ) then
      echo '  Processing sBOOM outputs'
      # run all jobs in parallel
      set nsb = `ls -d SBOOM_* | wc -l`
      c3d_parallel_runner.pl -d SBOOM_ -c $run_root/aero_sboom -j $nsb -w 2
      if ( $status ) then
        echo 'ERROR: c3d_parallel_runner.pl failed'
        goto ERROR
      endif
    endif

    if ( $fs == FLOW ) then
      # Improve the estimate of actual nCells [was nHexes from cubes/adapt]
      if ( -e FLOW/functional.dat ) then
        set nCells = `grep "nCells (total number of control volumes):" FLOW/functional.dat | tail -1 | awk '{printf("%d",$8)}'`
      endif

      if ( -l ../BEST ) then
        unlink ../BEST
      endif
      ln -s $dirname ../BEST

      # Try to find the actual averaging window used by flowCart
      set actual_window = `grep "Tecplot stats file" FLOW/cart3d.out | tr '=)' ' ' | awk '{print $NF}'`
      if ($status) then
        set actual_window = $avg_window
      else if ("" == "$actual_window") then
        set actual_window = $avg_window
      endif

      if ($?debug_verbose) then
        echo "  aero_funCon.csh $x $actual_window"
      endif
      aero_funCon.csh $x $actual_window

      # update results
      if ( $x > 0 ) then
        cd ..
        aero_getResults.pl -fos $error_safety_factor -update
        if ($status != 0) then
          echo '       aero_getResults.pl failed, exiting'
          goto ERROR
        endif
        cd $dirname
      endif

      if ( adapt00 == $dirname && ( ! -e ../Config.xml ) && ( -f FLOW/Config.xml ) ) then
        # Config.xml was generated by flowCart, copy it up so others can use it
        cp FLOW/Config.xml ../.
      endif

      # check exit criterion
      if ( 2 == $exit_point ) then
        if ( $n_adapt_cycles == $x ) then
          echo "  Completed $n_adapt_cycles adaptation cycles, exiting"
        endif
        if ( $maxRefLev == $maxRef21 ) then
          echo "  Reached $maxRefLev levels of refinement, exiting"
        endif

        # top level dir
        cd ..
        # exit main while loop
        goto ALLDONE
      endif
    endif

    if ( -e ../STOP ) then
      echo '  STOP file detected, exiting'
      # top level dir
      cd ..
      goto ALLDONE
    endif

    @ i = 1
    foreach dir ( $failsafe )
      if ( $dir == $fs ) then
        break
      else
        @ i ++
      endif
      if ( $i > $#adjoints ) then
        echo 'ERROR: adjoints vs failsafe array mismatch'
        goto ERROR
      endif
    end
    if ( 0 == $adjoints[$i] ) then
      # skipping adjoint solve
      continue
    endif

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

    # track status of adjoint solves for all outputs
    @ adj_status = 0

    # Loop over adjoint directories
    ls -d AD_*_* >& /dev/null
    if ( $status ) then
      echo "ERROR: Missing adjoint directories"
      echo "       Check functional specification for adaptation"
      goto ERROR
    endif
    foreach adir ( AD_*_* )
      if ( -e $adir/DONE ) then
        echo "    Adjoint done in $adir"
        continue
      endif

      cd $adir

      if ( FLOW == $fs ) then
        echo "  Running $adjointCart in $adir with $mg_levs of $mg_ad $ad_multigrid levs"
      else
        echo "  Running $adjointCart in $adir with $mg_levs of $mg_ad $ad_multigrid levs (<-${fs})"
      endif

      aero_setFlowLinks.csh

      if ( $?fs_prev ) then
        if ( ( -e historyADJ.dat || -e historyADJ.mg.dat ) && ( $fs != FLOW ) ) then
          mkdir DNF_${fs_prev}
          mv -f historyADJ*dat DNF_${fs_prev}/. >& /dev/null
          mv -f TIME* DNF_${fs_prev}/.
          mv -f cart3d.out DNF_${fs_prev}/. >& /dev/null
        endif
      endif

      # link in flow solution and timestep
      ln -sf `ls -1tr ../${fs}/check.* | tail -1` Flow.file
      ln -sf `ls -1tr ../${fs}/checkDT.* | tail -1` DT.file

      # bring input file from FLOW in case there was CL steering
      cp -f ../FLOW/input.cntl .

      # when using multiple functionals, cat in the correct functional section
      # for this adjoint
      if (-e ../../Functionals.xml ) then
        aero_rmFun.pl
        if ( ! -f no_design.cntl ) then
          echo 'ERROR: cannot create adjoint input file'
          cd ../..
          goto ERROR
        endif
        cat no_design.cntl functional.cntl > input.cntl
        rm -f no_design.cntl
      endif

      if ( $fs == FIRST_ORDER ) then
        # create first-order input.cntl
        mv -f input.cntl input.orig.cntl
        awk '{if ("RK"==$1){print $1,"  ",$2,"  ",0}else{print $0}}' input.orig.cntl > input.cntl
        unlink Mesh.mg.c3d   >& /dev/null
        unlink Mesh.c3d.Info >& /dev/null
        ln -s ../FIRST_ORDER/Mesh.mg.c3d
        ln -s ../FIRST_ORDER/Mesh.c3d.Info
      else if ( $fs == PMG ) then
        unlink Mesh.mg.c3d   >& /dev/null
        unlink Mesh.c3d.Info >& /dev/null
        ln -s ../PMG/Mesh.mg.c3d
        ln -s ../PMG/Mesh.c3d.Info
      endif

      # xsensit
      set flags = "$verb -dQ $y_is_spanwise $buffLim $subcell $blcc"
      ( $timer $xsensit $flags -limiter $limiter -tm $tm >> cart3d.out ) >&! $timeInfo
      set exit_status = $status
      mv -f $timeInfo ${timeInfo}.xs >& /dev/null
      if ($exit_status != 0) then
        echo "==> $xsensit failed with status $exit_status"
        grep ERROR cart3d.out | tail -1
        echo "    Check cart3d.out in $adir for more clues"
        goto ERROR
      endif

      # option to adjust adjoint iters for gradient adjoints
      @ ita = $it_ad
      if ( $adir =~ AD_*G*_* ) then
        @ ita += $delta_it_ad
      endif

      set flags = "$verb -fine -T $binaryIO $y_is_spanwise $fmg $buffLim $subcell $blcc"
      ( $timer $adjointCart $flags -N $ita $mg_gs_ad $mg_levs -limiter $limiter -tm $tm -cfl $cfl >> cart3d.out ) >&! $timeInfo
      set exit_status = $status
      if ( 0 != $exit_status ) then
        echo "WARNING: $adjointCart failed with status $exit_status"
        echo "         Check cart3d.out in $adir and ${adir}/DNF for more clues"
      endif

      # check convergence of adjointCart via historyADJ.dat file
      if ( 0 == $exit_status ) then
        adj_check_convergence.pl
        set exit_status = $status
      endif

      # if we are using more than one mg level, try grid sequencing or no multigrid
      if ( 0 != $exit_status ) then
        if ( $mg_levs > 1 && ( "$mg_gs_ad" == '-mg' ) ) then
          # double number of iters
          @ gs_it = 2 * $ita
          if ( "$fmg" != '-no_fmg' ) then
            echo "    Switching to grid sequencing in $dirname"
            mv -f historyADJ.dat historyADJ.mg.dat >& /dev/null
            rm -f checkADJ.* Com*ADJ.* *.plt >& /dev/null
            ( $timer $adjointCart $flags -N $gs_it -gs $mg_levs -limiter $limiter -tm $tm -cfl $cfl >> cart3d.out ) >&! $timeInfo
            if ( 0 == $status ) then
              adj_check_convergence.pl
              set exit_status = $status
            endif
          else
            # last attempt
            if ( $fs == FIRST_ORDER ) then
              echo "    Switching to no multigrid in $dirname"
              mv -f historyADJ.dat historyADJ.mg.dat >& /dev/null
              rm -f checkADJ.* Com*ADJ.* *.plt >& /dev/null
              ( $timer $adjointCart $flags -N $gs_it -mg 1 -limiter $limiter -tm $tm -cfl $cfl >> cart3d.out ) >&! $timeInfo
              if ( 0 == $status ) then
                adj_check_convergence.pl
                set exit_status = $status
              endif
            endif
          endif
        endif
      endif

      # track adjoint solve status for all functionals
      @ adj_status = ( $adj_status | $exit_status )

      # cleanup
      rm -f *.q adj_surf.dat dJdQ_surf.dat >& /dev/null

      if ( 0 == $exit_status ) then
        touch DONE
        rm -f functional.cntl
      endif

      cd ..
    end # end foreach AD_*

    # If any adjoint was not successfully computed, move to the next failsafe
    # and go back to the flow-solve stage.
    if ( 0 == $adj_status ) then
      break
    endif

    set fs_prev = $fs
  end # foreach failsafe

  # out of adapt??, back to top level directory
  cd ..

  # done adjoints

  # cleanup input file, the important one is in FLOW
  rm -f $dirname/input.cntl >& /dev/null

  if ( 0 != $adj_status ) then
    echo 'ERROR: Adjoint convergence problems, giving up'
    goto ERROR
  endif

  echo '  Done adjoint(s)'

  # clear restart flag if cold starting every adaptation
  if ( 0 == $use_warm_starts ) then
    set fc_restart
  endif

  if ( 3 == $exit_point ) then
    echo "  Done user-requested flow and adjoint solution on finest mesh, exiting"
    break
  endif

  cd EMBED
  echo '  Working in EMBED'

  cp -f ../$dirname/Mesh.c3d.Info .

  ( $timer adapt -v -RallCells -seq -i ../$dirname/Mesh.mg.c3d $is2D -no_ckpt $sfc > adapt.${xx}.out ) >&! $timeInfo
  if ($status != 0) then
    echo "==> ADAPT failed"
    goto ERROR
  endif

  echo '    Generated embedded mesh'

  # catch name of functional driving adaptation
  set adapt_dir = 0

  foreach adir ( ../$dirname/AD_A*_* ../$dirname/AD_E*_* )
    if ( ! -d $adir ) then
      continue
    endif

    set edir = $adir:t
    set edir = `echo $edir | sed 's/^AD_//'`
    if ( $edir =~ A*_* ) then
      set edir = `echo $edir | sed 's/^A//'`
      set adapt_dir = 1
    endif
    set edir = `echo $edir | sed 's/^E//'`
    set edir = `echo $edir | sed 's/^G//'`
    set edir = `echo $edir | sed 's/^_//'`
    echo "    Estimating error in $edir"
    set edir = Error_in_${edir}
    if ( 1 == $adapt_dir ) then
      set adapt_dir = $edir
    endif

    if (! -d $edir ) then
      mkdir $edir
    endif

    cd $edir

    ln -sf `ls -1tr ../../$dirname/FLOW/check.* | tail -1` Flow.file
    ln -sf `ls -1tr ../${adir}/checkADJ.* | tail -1` Adjoint.file

    aero_setFlowLinks.csh

    ln -sf ../adaptedMesh.R.c3d Mesh.mg.c3d
    ln -sf ../$adir/input.cntl

    if ( -e ../$adir/COLUMNS ) then
      ln -sf ../$adir/COLUMNS
    endif

    # Check for embedded value of functional in cases when it is computed
    # externally. This is specific to loudness estimates computed by sBOOM,
    # parsed from loud.dat. When adjointErrorEst executes, the value of the
    # functional on the embedded mesh is set to the parsed value, instead of
    # the line sensor integral.
    ls ../$adir/dJdP_*dat >& /dev/null
    if ( 0 == $status ) then
      foreach obj_emb ( ../$adir/dJdP_*dat )
        set lsname = $obj_emb:t
        set lsname = `echo $lsname | sed 's/^dJdP_//'`
        set lsname = `echo $lsname | sed 's/\.dat$//'`

        rm -f OBJ_EXT_${lsname}.dat >& /dev/null

        if ( -d ../../$dirname/SBOOM_EMBED_${lsname} ) then
          if ( -e ../../$dirname/SBOOM_EMBED_${lsname}/loud.dat ) then
            grep "Objective\ Function\ Value" ../../$dirname/SBOOM_EMBED_${lsname}/loud.dat | awk '{print $5}' > OBJ_EXT_${lsname}.dat
          endif
        endif
      end
    endif

    if ( 0 == $mesh2d && 1 == $tm ) then
      if ( 0 == $err_TM1 ) then
        set tm = 0
      else
        set tm = 1
        echo "            ... using tm=1 mode"
      endif
    endif

    set flags = "$verb $binaryIO $y_is_spanwise $buffLim $subcell $blcc $histo"
    ( $timer $adjointErrorEst $flags -etol $etol -limiter $limiter -tm $tm > adjointErrorEst.${xx}.out ) >&! $timeInfo
    set exit_status = $status
    if ($exit_status != 0) then
      if ( $ERROR_TOL_OK == $exit_status ) then
        echo "            Adaptation tolerance satisfied"
        if ( $x > 2 ) then
          echo "            We will build final mesh, solve and exit"
          @ error_ok = 1
        else
          echo "            Ignoring tolerance for adapt cycle $x"
        endif
      else
        echo "==> $adjointErrorEst failed with status = $exit_status"
        echo "==> Trying it again"
        sleep 1
        ( $timer $adjointErrorEst $flags -etol $etol -limiter $limiter -tm $tm > adjointErrorEst.${xx}.2.out ) >&! $timeInfo
        set exit_status = $status
        if ($exit_status != 0) then
          echo "==> $adjointErrorEst failed again, status = $exit_status"
          goto ERROR
        endif
      endif
    endif

    unlink Mesh.mg.c3d

    # archive
    mv errorEst.ADJ.dat errorEst.ADJ.${xx}.dat
    if ( -e cutPlanesErrEst.dat ) then
      mv cutPlanesErrEst.dat cutPlanesErrEst.${xx}.dat
    endif
    if ( -e cutPlanesErrEst.plt ) then
      mv cutPlanesErrEst.plt cutPlanesErrEst.${xx}.plt
    endif

    if ( -e error.allcells.dat ) then
      mv error.allcells.dat error.allcells.${xx}.dat
      mv error.cutcells.dat error.cutcells.${xx}.dat
    endif

    cd ..
  end # loop over functionals

  rm -f adaptedMesh.R.c3d aPlanes.dat >& /dev/null

  # out of EMBED
  cd ..

  # analyze results before adapting mesh
  aero_getResults.pl -fos $error_safety_factor
  if ($status != 0) then
    echo '       aero_getResults.pl failed, exiting'
    rm -f EMBED/Error_in_*/check.*
    goto ERROR
  endif

  if ( 4 == $exit_point || $close_enough > 1 ) then
    echo "Done user-requested error analysis on finest mesh, exiting"
    rm -f EMBED/Error_in_*/check.*
    break
  endif

  echo '  Working in ADAPT'
  cd ADAPT

  if ( 0 == $adapt_dir ) then
    echo 'ERROR: adaptation functional not set, exiting'
    cd ..
    goto ERROR
  endif

  ln -sf ../EMBED/${adapt_dir}/check.* preAdapt.ckpt
  cp ../$dirname/Mesh.c3d.Info .
  cp -f ../$dirname/FLOW/input.cntl .

  # check if this is the last mesh, if so, set nxref to final_mesh_xref
  # if requested by user
  set nxref = 0
  if ( $n_adapt_cycles == $xp1 || 1 == $error_ok || $close_enough > 0 ) then
    if ( $final_mesh_xref > 0 ) then
      set nxref = $final_mesh_xref
    endif
  endif

  if ( $nxref > 0 ) then
    # check max level of refinement in current mesh and make sure we do not
    # exceed max allowed based on 21 bits or user-requested depth
    @ newRefLev = $maxRefLev + $nxref
    if ( $newRefLev >= $maxRef21 ) then
      echo "       User requested $nxref extra refinements on final mesh"
      echo "       WARNING: requested refinement level exceeds 21-bit maximum"
      @ nxref = $maxRef21 - $maxRefLev
    else if ( $newRefLev >= $max_ref_level ) then
      echo "       User requested $nxref extra refinements on final mesh"
      echo "       WARNING: requested refinement level exceeds user-specified maximum"
      @ nxref = $max_ref_level - $maxRefLev
    endif
    if ( $nxref < 1 ) then
      echo "       Extra refinement not possible, solving on final mesh"
      set nxref = 0
    else
      echo "       Resetting extra refinements to $nxref"
    endif
    unset newRefLev
  endif

  set do_map
  set updir = '..'
  if ( $nxref > 0 ) then
    set do_map = '-prolongMap'
    set xrefdir = 'XREF_INIT'
    mkdir -m 0700 $xrefdir
    cd $xrefdir
    set updir = '../..'
    ln -sf ../preAdapt.ckpt
    cp ../Mesh.c3d.Info .
    ln -sf ../input.cntl
    if ( ( 1 == $use_preSpec ) && -e ../preSpec.c3d.cntl) then
      ln -s ../preSpec.c3d.cntl
    endif
    ln -sf ../Components.i.tri
  endif

  # Logic to support interface propagation

  # Try to anticipate when the adaptation will breach the 21 bit limit. Check
  # this only if the current cycle is 'a'.
  if ( a == $apc[$xp1] ) then
    @ newRefLev  = $maxRefLev + 1
    # make sure this is not the final mesh
    if ( $newRefLev == $maxRef21 && $n_adapt_cycles > $xp1 && 0 == $error_ok && 0 == $close_enough ) then
      echo "    WARNING: Current refinement level $maxRefLev is one away from max (${maxRef21})"
      echo '         ... Embedding this mesh would exceed 21 bits on the next cycle'
      echo '         ... Inserting propagate cycle into the apc array'
      set apc[$xp1] = p
      @ xp1 += 1
      set apc[$xp1] = a
      @ xp1 -= 1
      echo '            ' $apc
    else if ( $maxRefLev == $max_ref_level && $max_ref_level < 21 ) then
      # If user has specified a max ref limit, then proceed with propagate
      # cycles
      echo "    WARNING: Current refinement level $maxRefLev equals user-requested level (${max_ref_level})"
      echo '         ... Inserting propagate cycle into the apc array'
      set apc[$xp1] = p
      @ xp1 += 1
      set apc[$xp1] = a
      @ xp1 -= 1
      echo '            ' $apc
    endif
  endif

  set maxRef
  if ( p == $apc[$xp1] ) then
    echo "    Interface propagation"
    # adapt automatically adjusts maxRef to mesh max and holds it there
    set maxRef = "-maxRef 1"
  endif

  # Set adaptation threshold value

  set acrit
  if ( $?mesh_growth ) then
    # convert to int so "if" does not gripe
    if ( `echo $mesh_growth[$xp1] | awk '{printf("%d",$1)}'` ) then
      # we are using explicit mesh_growth
      set acrit = "-growth $mesh_growth[$xp1]"
      echo "    Requested mesh growth $mesh_growth[$xp1]"
      # signal explicit mesh growth
      set emg
    else
      # zero, so set automatically
      set acrit = auto
    endif
  endif
  if ( $?ath ) then
    if ( `echo $ath[$xp1] | awk '{printf("%d",$1)}'` ) then
      # we are using explicit threshold
      set acrit = "-t $ath[$xp1]"
      echo "    Requested threshold $ath[$xp1]"
    else
      # zero, so set automatically
      set acrit = auto
    endif
  endif

  if ( 1 == $auto_growth || auto == "$acrit" ) then
    # specify threshold to be the mean error per cell
    set this_ath = `grep "remaining_error_in_functional" $updir/EMBED/${adapt_dir}/errorEst.ADJ.${xx}.dat | awk '{printf("%12.4e",$2 / '$etol')}'`

    # if this is the last mesh or if propagation, shift mean one bin to the left to force a larger growth
    if ( $xp1 == $n_adapt_cycles || p == $apc[$xp1] || 1 == $error_ok || 1 == $close_enough ) then
      set this_ath = `grep "remaining_error_in_functional" $updir/EMBED/${adapt_dir}/errorEst.ADJ.${xx}.dat | awk '{printf("%12.4e",(2^(log($2)/log(2)-1))/'$etol')}'`
    endif

    # Prohibit thresholds less than 1
    set this_ath = `echo $this_ath | awk '{if ($1 > 1.0) {print $1} else {print 1.0}}'`

    set acrit = "-t $this_ath"
    echo "    Setting adaptation threshold to $this_ath"
  endif

  set flags = "$is2D $prespec $y_is_spanwise $sfc"

  # check if we exceed or are close to max_cells2embed
  # also check if we have at least the minimum mesh growth
  if ( 0 == $close_enough ) then
    # Note: The verbose flag MUST be passed, to allow the following grep
    # to find the number of CV's.
    adapt -adjoint -i ${updir}/$dirname/Mesh.mg.c3d -v $acrit -b $buf $maxRef $flags -stats > adapt.stats
    if ($status != 0) then
      echo "==> ADAPT failed while running in STATS mode"
      break
    endif
    set nCVs = `grep " nCVs = " adapt.stats | tail -1 | awk '{printf "%d", $4 }'`
    set tagged = `grep " tagged hexes" adapt.stats | tail -1 | awk '{printf "%d", $3 }'`

    rm -f adapt.stats >& /dev/null

    if ( $mesh2d ) then
      @ finalMesh = $nCVs + ( 3 * $tagged )
    else
      @ finalMesh = $nCVs + ( 7 * $tagged )
    endif

    if ( $xp1 != $n_adapt_cycles && $finalMesh > $max_cells2embed ) then
      # check if we exceed max_cells2embed
      set acrit = `echo $max_cells2embed | awk '{printf("%4.2f",0.999*$1/'$nCells')}'`
      echo "    Adjusted mesh growth to $acrit to stay below max_cells2embed"
      set acrit = "-growth $acrit"
      @ close_enough = 1
    else if ( `echo $finalMesh | awk '{ if ( '$min_growth' > $1/'$nCells' ) {print 1} else {print 0}}'` ) then
      # enforce minimum growth
      if ( $?emg ) then
        if ( `echo $mesh_growth[$xp1] | awk '{ if ( $1 < '$min_growth' ) {print 1} else {print 0}}'` ) then
          # if using explicit mesh growth, adjust the growth only if it is
          # actually less than the minimum growth
          set acrit = "-growth $min_growth"
        endif
        unset emg
      else
        echo "    Mesh growth too small ( ${finalMesh} cells )"
        echo "    Forcing growth of $min_growth"
        set acrit = "-growth $min_growth"
      endif
    endif
  else if ( 1 == $close_enough ) then
    # trigger final flow solve
    @ close_enough += 1
  endif

  # refine mesh

  if ( 0 == $refine_all_cells ) then
    ( $timer adapt -adjoint -i ${updir}/$dirname/Mesh.mg.c3d -v $acrit -b $buf $maxRef $flags $do_map > adapt.${xx}.out ) >&! $timeInfo
    set exit_status = $status
  else
    ( $timer adapt -RallCells -i ${updir}/$dirname/Mesh.mg.c3d -v $flags > adapt.${xx}.out ) >&! $timeInfo
    set exit_status = $status
  endif

  if ($exit_status != 0) then
    if ( $ZERO_TAGGED_HEXES == $exit_status ) then
      echo "    Zero cells were tagged, see ADAPT/adapt.${xx}.out"
      echo "    Try decreasing etol, increasing mesh growth or switching to 'a' cycles"
      rm -f aPlanes.dat postAdapt.ckpt >& /dev/null
    else
      echo "ERROR: ADAPT failed with status = $exit_status"
    endif
    # out of ADAPT, back in top level dir
    cd ..
    # cleanup, embedded checkpoints are big
    # comment next line if you need to debug
    rm -f EMBED/Error_in_*/check.* >& /dev/null
    goto ERROR
  endif

  set nCells = `grep "totNumHexes     =" adapt.${xx}.out | awk '{print $3}'`

  # check actual mesh growth and threshold
  if ( 0 == $refine_all_cells ) then
    if ( "$acrit" =~ "-growth*" ) then
      set actual_growth = `grep "Mesh growth factor" adapt.${xx}.out | tail -1 | awk '{print $7}'`
      set threshold     = `grep "Mesh growth factor" adapt.${xx}.out | tail -1 | awk '{print $13}'`
      echo "    Actual mesh growth $actual_growth with threshold $threshold"
    else
      set init_hex  = `grep "| Initial number of   hex cells  is:" adapt.${xx}.out | awk '{print $8}'`
      set final_hex = `grep "| Final   number of   hex cells  is:" adapt.${xx}.out | awk '{print $8}'`
      set actual_growth = `echo $init_hex | awk '{print '$final_hex' / $1 }'`
      echo "    Mesh growth $actual_growth"
    endif
  else
    echo "    Refined all cells"
  endif

  # safety fuse on number of cells
  if ( $nCells > $max_nCells ) then
    echo "    WARNING: Number of cells $nCells exceeds max limit $max_nCells"
    echo "             Enabling skipfinest to setup the final flow solve directory"
    @ skip_finest = 1
  endif

  if ( $close_enough ) then
    # check if mesh size is close enough to max_cells2embed, if not reset
    # close_enough to allow more adaptation cycles
    if ( `echo $max_cells2embed | awk '{ if ( '$nCells' * '$min_growth' <= $1 ) {print 1} else {print 0}}'`  ) then
      echo "   Small mesh growth, continuing adaptation"
      @ close_enough -= 1
    endif
  else
    # check if this new mesh will exceed max_cells2embed with minimum growth,
    # if yes this must be the final adaptation cycle
    if ( `echo $max_cells2embed | awk '{ if ( '$nCells' * '$min_growth' > $1 ) {print 1} else {print 0}}'` ) then
      @ close_enough += 1
    endif
  endif

  # skip if this is the final cycle
  if ( $xp1 != $n_adapt_cycles ) then
    if ( 1 == $close_enough ) then
      echo "    Mesh size is close to the embedding limit (reached ${nCells} of $max_cells2embed cells)"
      echo "    Performing final adaptation cycle"
    else if ( 2 == $close_enough && 0 == $skip_finest ) then
      echo "    Mesh size exceeds the embedding limit (reached ${nCells} of $max_cells2embed cells)"
      echo "    Performing final flow analysis"
    endif
  endif

  # actually do extra refinements
  if ( $nxref > 0 ) then
    # return out of XREF_INIT
    cd ..
    echo "    Generating $nxref extra mesh refinement(s)"
    set cp_from = '../XREF_INIT'
    # initialize extra refinement passes
    set xa = 1

    if ("$acrit" =~ "-growth*" ) then
      set base_growth = $mesh_growth[$xp1]
    else
      set base_growth = $actual_growth
    endif

    # main loop controlling extra refinement
    while ($xa <= $nxref)
      set xa_dir = XREF_${xa}
      mkdir $xa_dir
      cd $xa_dir
      ln -s ../input.cntl
      ln -s ../Components.i.tri
      if ( ( 1 == $use_preSpec ) && -e ../preSpec.c3d.cntl) then
        ln -s ../preSpec.c3d.cntl
      endif
      cp ${cp_from}/Mesh.c3d.Info .
      set do_map
      if ( $xa < $nxref ) then
        set do_map = '-prolongMap'
        # set do_map = '-prolongMap -Zcut 1'
      endif
      echo "      Working in ${xa_dir}"
      # attenuate growth by preset fraction, i.e. take the final growth,
      # subtract 1, multiply by attenuation fraction, and add 1 back
      set xref_growth = `echo $base_growth | awk '{printf("%.3f",($0-1.)*'$xref_fraction[$xa]'+1.)}'`
      # safety fuse => minimum growth
      set xref_growth = `echo $xref_growth | awk '{ if ( '$min_growth' > $0 ) {print '$min_growth'} else {print $0}}'`
      set acrit = "-growth $xref_growth"
      echo "      Requested mesh growth $xref_growth"
      # refine mesh again
      ( $timer adapt -adjoint -i ${cp_from}/adaptedMesh.R.c3d -in ${cp_from}/postAdapt.ckpt -v $acrit -b $buf $flags $do_map >& adapt.out ) >&! $timeInfo
      set exit_status = $status
      if ($exit_status != 0) then
        if ( $ZERO_TAGGED_HEXES == $exit_status ) then
          if ( $xa == 1 ) then
            set nxref = 'INIT'
          else
            @ nxref = $xa - 1
          endif
          echo "      Zero cells were tagged in ADAPT, using mesh from directory XREF_${nxref}"
          # back up to ADAPT
          cd ..
          # exit while loop
          break
        else
          echo "ERROR: ADAPT failed with status = $exit_status"
          goto ERROR
        endif
      endif
      set nCells = `grep "totNumHexes     =" adapt.out | awk '{print $3}'`
      # check actual mesh growth and threshold
      set actual_growth = `grep "Mesh growth factor" adapt.out | tail -1 | awk '{print $7}'`
      set threshold     = `grep "Mesh growth factor" adapt.out | tail -1 | awk '{print $13}'`
      echo "      Actual mesh growth ${actual_growth}, threshold ${threshold}, cells ${nCells}"
      # check that max_nCells is not exceeded, if yes use largest mesh built so far
      if ( $nCells > $max_nCells ) then
        echo "      WARNING: Number of cells $nCells exceeds max limit $max_nCells"
        rm -f adaptedMesh.R.c3d postAdapt.ckpt
        # overwrite nxref so we can copy the best mesh to the working adapt dir
        if ( $xa == 1 ) then
          set nxref = 'INIT'
          set nCells = `grep "totNumHexes     =" ${cp_from}/adapt.${xx}.out | awk '{print $3}'`
        else
          @ nxref = $xa - 1
          set nCells = `grep "totNumHexes     =" ${cp_from}/adapt.out | awk '{print $3}'`
        endif
        echo "               using mesh from directory XREF_${nxref}"
        # back up to ADAPT
        cd ..
        # exit while loop
        break
      endif
      # cleanup
      rm -f ${cp_from}/adaptedMesh.R.c3d ${cp_from}/postAdapt.ckpt ${cp_from}/Mesh.c3d.Info
      set cp_from = "../${xa_dir}"
      cd ..
      @ xa ++
    end # end while
    mv -f XREF_${nxref}/adaptedMesh.R.c3d .
    mv -f XREF_${nxref}/Mesh.c3d.Info .
    mv -f XREF_${nxref}/postAdapt.ckpt .
  endif # end extra refinement

  # cleanup files
  rm -f aPlanes.dat >& /dev/null
  if ( 1 != $use_warm_starts ) then
    rm -f postAdapt.ckpt
  endif

  if (1 == $keep_error_maps) then      # ...keep the error-fortified ckpt files -MA
    ls ../EMBED/${adapt_dir}/check.* >& /dev/null #    verify existence before destroying anything
    if ( 0 == $status ) then
      rm ../EMBED/${adapt_dir}/Restart.*.file >& /dev/null
      mv ../EMBED/${adapt_dir}/check.* ../EMBED/${adapt_dir}/Restart.{$xx}.file
    endif
  endif

  # update maxRefLev
  @ maxRefLev = `grep " # maxRef" Mesh.c3d.Info | awk '{print int( $1 )}'`

  # out of ADAPT, back in top level dir
  cd ..

  # cleanup, embedded checkpoints are big
  rm -f EMBED/Error_in_*/check.* >& /dev/null

  # collect time info
  if ( $timeInfo != '/dev/null' ) then
    aero_timer.csh $x $dirname $timeInfo $timer
  endif

end # end main while loop

ALLDONE:

  # sum user time
  if ( $timeInfo != '/dev/null' ) then
    aero_timer.csh $x $dirname $timeInfo sum
  endif

  # check for expiration notice in cart3d.out
  if ( -e BEST/FLOW/cart3d.out ) then
    grep ' until expiration' BEST/FLOW/cart3d.out
  endif

  echo 'Done aero.csh'
  exit 0

ERROR:
  exit 1

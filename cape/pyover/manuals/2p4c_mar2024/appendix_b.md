# **Appendix B. NAMELIST Input**

This section describes the input variables for OVERFLOW. The entries in the input file are in the form of NAMELISTs. The default name for the NAMELIST file is **over.namelist**. The variables are described below, and default values for each of the inputs are given in brackets at the end of the description. If the values in any NAMELIST variables are omitted in the input file, they are automatically set to the default value. A number of the NAMELISTs are repeated for each grid. If an input variable is set for one grid, generally it becomes the default for all following grids, until the variable is set again. This is true for all variables (such as numerical scheme, smoothing parameters, etc.) that are not dependent on specific grid topology. Notable exceptions are boundary conditions, turbulent regions, or enabling viscous terms in specific coordinate directions. The user must define the beginning and end of each NAMELIST (e.g., **&NAMELIST ... /**) even if only the default values are required.

| &GLOBAL (Global inputs for OVERFLOW)                                               |                                                               |
|------------------------------------------------------------------------------------|---------------------------------------------------------------|
| NSTEPS Number of (fine-grid) steps to advance solution. Use zero for input         |                                                               |
| RESTRT TRUE—Read restart flowfield from file                                       | q.restart                                                     |
| NSAVE ≥0—Save the overall solution to file                                         | q.save every how many steps.                                  |
| <0—Save the solution to file                                                       | q.step# every how many steps.                                 |
| Note that files are saved as                                                       | q.step# for dynamic or adaption cases                         |
| TRUE—Globally solve SSOR schemes                                                   | with Chimera updates. Interacts                               |
| with &METPRM                                                                       | ILHS: see README.gls for more information.                    |
| SAVE_HIORDER Controls whether Q                                                    | n-1 data for 2 nd                                             |
|                                                                                    | -order restarts is written to q.save and                      |
| q.step# files:                                                                     |                                                               |
| -1—Always include Q                                                                | n-1                                                           |
| 0—Never include Q                                                                  | n-1                                                           |
| 1—Only include Q                                                                   | n-1                                                           |
|                                                                                    | for final q.save                                              |
| 2—Always include Q                                                                 | n-1                                                           |
|                                                                                    | for q.save , never for q.step# if NSAVE<0. [2]                |
| SAVE_SPLIT TRUE—Save                                                               | additional x.split and q.split files for restart to avoid     |
| residual spikes due to                                                             | auto-split grid boundaries. (Only works when                  |
| FALSE—Don’t save                                                                   | x.split, q.split files. [FALSE]                               |
| ISTART_QAVG 0—Do not save Q average/perturbation data.                             |                                                               |
| >0—Start saving Q average and                                                      | (rho,u,v,w,p) perturbation data at step                       |
| ISTART_QAVG. Write to file                                                         | q.avg whenever q.save is written. Note                        |
| NFLUSH Flush output history files (                                                | fomoco.out, resid.out, turb.out, species.out,                 |
| rpmin.out, sixdof.out, animate.out                                                 | ) every how many steps. [100]                                 |
| NFOMO Compute aerodynamic forces and moments every how many steps. For             |                                                               |
| negative values, compute Newton/dual subiteration forces and                       | moments                                                       |
| NVIS Save visualization files                                                      | x.iw.step# and q.iw.step# every how many steps,               |
| where iw                                                                           | is the multiple writer number. Visualization files always use |
| NWRITERS Number of MPI ranks used for writing visualization files (on the order of |                                                               |
| 8 or 16). See utility code                                                         | mwmerge for merging multiple writer files                     |

| NQT            | Global turbulence model type declaration:                                 |
|----------------|---------------------------------------------------------------------------|
|                | 0—Baldwin-Lomax (algebraic) or no turbulence model.                       |
|                | specification (DDADI left-hand side).                                     |
|                | 102—Spalart-Allmaras (SA-neg) model (DDADI left-hand side).               |
|                | 302—Spalart-Allmaras (SA-neg) model with Medida-Baeder γ- Re θ -SA        |
|                | 405—SST-2003 model with Langtry-Menter (LM2009 or LM2015)                 |
| NQC            | Variable γ model type declaration (number of species):                    |
| MULTIG         | Flag to enable/disable multigrid acceleration. [FALSE]                    |
| FMG            | Flag to enable/disable grid sequencing. [FALSE]                           |
| FMGCYC(level#) | Number of steps to take on coarser grid levels during grid sequencing.    |
| NGLVL          | Number of multigrid and/or grid sequencing levels to use. [3]             |
| TPHYS          | Starting physical time (overrides value from q.restart ). [not specified] |
| DTPHYS         | Physical time-step (based on V ref ). [0]                                 |
| NITNWT         | Maximum number of Newton/dual subiterations per physical time-step        |
| FSONWT         | 1.0—First-order (BDF1) time-advance for Newton/dual subiteration.         |
| ORDNWT         | 0—Do not limit number of Newton/dual subiterations.                       |
|                | >0—Order of convergence to limit subiterations (use L 2 -norm(RHS)).      |
|                | subiterations (use L 2 -norm(RHS)). [0]                                   |

| NDIRK Select multi-stage time advance scheme.                                     |                                                  |
|-----------------------------------------------------------------------------------|--------------------------------------------------|
| 0—Default                                                                         | (Euler implicit, BDF1 or BDF2 based on FSONWT)   |
| 1—Euler explicit (1-stage, 1                                                      | st                                               |
| 2—Euler implicit BDF1 or BDF2 (1-stage, 1                                         | st                                               |
|                                                                                   | or 2 nd                                          |
| 3—Runge-Kutta explicit RK3 (3-stage, 3                                            | rd                                               |
| 4—Runge-Kutta explicit RK4 (4-stage, 4                                            | th                                               |
| 5—ESDIRK3 implicit (3-stage, 3                                                    | rd                                               |
| 6—ESDIRK4 implicit (5-stage, 4                                                    | th                                               |
| RF Global coordinate system rotation speed (rad/time, based on                    | V ref ). [0.]                                    |
| RFAXIS Global coordinate system rotation axis (1/2/3/4                            | corresponds to                                   |
| Rotating frame arbitrary-axis                                                     | (RFAXIS=4) direction vector. [0.,0.,1.]          |
| CDISC TRUE—Expect to                                                              | read a NAMELIST input file overdisc.con ,        |
| GRDWTS TRUE—Use grid timing information in                                        | grdwghts.restart for MPI load                   |
| MAX_GRID_SIZE 0—Use automatic grid splitting algorithm for load-balancing.        |                                                  |
| <0—Do not split grids. [0]                                                        |                                                  |
| NOBOMB Inhibit writing                                                            | q.bomb file if solution procedure fails. [FALSE] |
| CONSERVE_MEM Conserve memory by recomputing metrics and regenerating coarse-level |                                                  |
| WALLDIST 0—Read precomputed wall distance from file                               | walldist.dat (PLOT3D                             |
| ±2—Global wall distance                                                           | calculation.                                     |
| If WALLDIST is negative, write wall distance file                                 | walldist.save . [2]                              |
| NWALL Recompute global wall distance every NWALL steps                            | (dynamic                                         |
| DEBUG 0—Normal run.                                                               |                                                  |
| 1—Write turbulence model debug file                                               | q.turb and quit.                                 |
| 2—Write timestep debug                                                            | file q.time and quit.                            |
| 3—Write residual debug file                                                       | q.resid and quit.                                |
| 4—Write grid adaption debug file                                                  | q.errest and quit.                               |
| 5—Write Reynolds stress debug file                                                | q.reystr and quit. [0]                           |
| STOPTIME_STEPX PBS time limit reserve factor (multiplies max step time). [1.5]    |                                                  |
| STOPTIME_SEC PBS time limit reserve time (sec). [0.]                              |                                                  |

**&OMIGLB** (Global inputs for OVERFLOW-D) (OVERFLOW-D only)

| IRUN          | 0—Do a complete run.                                                     |
|---------------|--------------------------------------------------------------------------|
| I6DOF         | 0—Body motion is defined by user-defined USER6 routine.                  |
|               | 2—Body motion is defined by GMP interface (files Config.xml and          |
|               | Scenario.xml ). [0]                                                      |
| DYNMCS        | Enable/disable body motion. [FALSE]                                      |
| NADAPT        | 0—Do not regenerate off-body grids.                                      |
|               | <0—Regenerate off-body grids every –NADAPT steps, based on               |
| NREFINE       | Number of off-body grid refinement levels allowed for solution adaption  |
| NBREFINE      | Number of near-body grid refinement levels allowed for solution adaption |
|               | (NADAPT>0). NBREFINE  0 turns off near-body adaption. [NREFINE]         |
| ETYPE         | Sensor function for grid adaption error estimate. [0]                    |
|               | nd difference (squared) of flow variables Q(1-5).                        |
|               | nd difference of pressure (normalized).                                  |
|               | nd difference of pressure/density/temperature (normalized).              |
|               | 7—Undivided 2 nd difference (absolute value) of flow variables Q(1-5).   |
| SIGERR        | Solution error levels for adaption. [5.0]                                |
| EREFINE       | Solution error estimate level above which the grid will be refined.      |
|               | [(1/8) SIGERR ]                                                          |
| ECOARSEN      | Solution error estimate level below which the grid will be coarsened.    |
|               | [(1/8) SIGERR+2 ]                                                        |
| MAX_SIZE      | Maximum grid system size during solution adaption (0 for no limit).      |
|               | [0.95*IGSIZE*number of groups]                                           |
| MAX_GROWTH    | Maximum growth factor for grid system per adapt cycle. [1.3]             |
| DISTWT_LENGTH | Length scale representing one refinement level decay of the error sensor |
| DISTWT_OFFSET | Offset distance from surface to begin decay. [0.0]                       |
| R_COEF        | Coefficient of restitution for collisions. [1.0]                         |
| LFRINGE       | LFRINGE  is the number of fringe points for near-body grids and hole     |
|               | boundaries. If LFRINGE<0, do not revert double- and higher-fringe        |
|               | orphan points to field points. [Determined from numerical scheme (all    |
| IBXMIN,IBXMAX | Boundary condition type for X min , X max far-field boundaries. [47]     |
| IBYMIN,IBYMAX | Boundary condition type for Y min , Y max far-field boundaries. [47]     |
| IBZMIN,IBZMAX | Boundary condition type for Z min , Z max far-field boundaries. [47]     |
| LAMINAR_OB    | Force laminar flow in off-body grids. (Applicable to all 1- and 2-eq     |

## **&GBRICK** (Off-body grid generation inputs) (OVERFLOW-D only)

| OBGRIDS        |    |   |     |                       |                          | Allow or inhibit off-body grids. [TRUE]                               |
|----------------|----|---|-----|-----------------------|--------------------------|-----------------------------------------------------------------------|
| MAX_BRICK_SIZE |    |   |     |                       |                          | >0—Maximum off-body grid size.                                        |
| DS             |    |   |     |                       |                          | Spacing of level-1 (finest) off-body grids. [must be specified]       |
| DFAR           |    |   |     |                       |                          | Distance to far-field boundaries. [5.]                                |
| I_XMIN,I_XMAX  | 0— | X | min | , X                   | max                      | far-field boundary will be determined by DFAR.                        |
|                | 1— | X | min | , X                   | max                      | boundary will be specified by P_XMIN, P_XMAX, resp.[0]                |
| I_YMIN,I_YMAX  | 0— | Y | min | , Y                   | max                      | far-field boundary will be determined by DFAR.                        |
|                | 1— | Y | min | , Y                   | max                      | boundary will be specified by P_YMIN, P_YMAX, resp. [0]               |
| I_ZMIN,I_ZMAX  | 0— | Z | min | , Z                   | max                      | far-field boundary will be determined by DFAR.                        |
|                | 1— | Z | min | , Z                   | max                      | boundary will be specified by P_ZMIN, P_ZMAX, resp. [0]               |
| P_XMIN,P_XMAX  |    |   |     | Physical location for |                          | X min , X max off-body grid boundary, if corresponding                |
| P_YMIN,P_YMAX  |    |   |     | Physical location for |                          | Y min , Y max off-body grid boundary, if corresponding                |
| P_ZMIN,P_ZMAX  |    |   |     | Physical location for |                          | Z min , Z max off-body grid boundary, if corresponding                |
| MINBUF         |    |   |     |                       |                          | Minimum buffer width of points at each level. [4]                     |
| OFRINGE        |    |   |     |                       |                          | Number of fringe points for off-body grids. [Determined from off-body |
|                |    |   |     |                       | numerical scheme or from | brkset.restart file.]                                                 |

## **&NBINP** (User-specified near-body refinement regions) (OVERFLOW-D only)

| REFLVL_DEFAULT    | Default refinement level limit for any near-body grid that does not have |
|-------------------|--------------------------------------------------------------------------|
| IGBRK(brick#)     | Original near-body grid that this is region applies to.                  |
| BRKLVL(brick#)    | Grid level for proximity region, in the range [-1,-NBREFINE]. [-1]       |
|                   | J-range of user-specified proximity region. [1,J max from original grid] |
|                   | K-range of user-specified proximity region. [1,K max from original grid] |
|                   | L-range of user-specified proximity region. [1,L max from original grid] |
| IGREF(region#)    | Original near-body grid that this limit region applies to.               |
| REFLVL(region#)   | Minimum grid level for limit region, in the range [0,-NBREFINE].         |
| REFINOUT(region#) | “INSIDE”—Grid level will be limited inside this region.                  |
|                   | J-range of error adaption limit region. [1,J max from original grid]     |
|                   | K-range of error adaption limit region. [1,K max from original grid]     |

| &BRKINP           | (User-specified proximity and refinement regions) (OVERFLOW-D only) |
|-------------------|---------------------------------------------------------------------|
| NBRICK            | Number of user-specified proximity regions. If NBRICK<0, user must  |
| BRKLVL(brick#)    | Grid level of proximity region, in the range [1,-NREFINE]. [1]      |
| IBDYTAG(brick#)   | 0—Proximity region will have no body transformations.               |
| DELTAS(brick#)    | Distance to expand proximity region (in all directions). [0.]       |
| REFLVL(region#)   | Minimum grid level for limit region, in the range [2,-NREFINE].     |
| IBDYREF(region#)  | 0—Limit region will have no body transformations.                   |
| REFINOUT(region#) | “INSIDE”—Grid level will be limited inside this region.             |

## **&GROUPS** (Load balance input) (OVERFLOW-D only)

| R UP   |                                                                        |
|--------|------------------------------------------------------------------------|
| USEFLE | TRUE—Use grid timing information in grdwghts.restart for MPI load     |
| MAXNB  | 0—Use automatic splitting algorithm for near-body grid load-balancing. |
|        | (Can be set by MAX_GRID_SIZE in \$GLOBAL.)                             |
| MAXGRD | 0—Use automatic splitting algorithm for off-body grid load-balancing.  |
| WGHTNB | Weight-factor for near-body grids vs. off-body grids in normal load   |
| IGSIZE | Maximum group size during grid adaption. [10,000,000]                  |

| DQUAL  | Acceptable “quality” of donor interpolation stencils. [1.]        |
|--------|-------------------------------------------------------------------|
| MORFAN | 1/0—Enable/disable wall region stencil repair. [0]                |
| NORFAN | Number of points above solid walls subject to stencil repair. [5] |

## **&XRINFO** (X-ray input, repeat per X-ray cutter) (OVERFLOW-D only)

| &XRINFO       | (X-ray input, repeat per X-ray cutter) (OVERFLOW-D only)                        |
|---------------|---------------------------------------------------------------------------------|
| IDXRAY        | X-ray to be used for this cutter. (Note that X-rays may be used in multiple     |
| IGXLIST       | Specify a list of grids to be cut by this cutter (a grid number of -1 refers to |
| IGXBEG,IGXEND | Or specify beginning and ending grids to be cut by this cutter. [none]          |
| XDELTA        | Hole will extend XDELTA from the X-rayed surface. [0.]                          |

**&SPLITM** (Write SPLITMX/Q grid and/or Q files (PLOT3D format), or Cart3D triq files. Use multiple namelists as needed.)

| XFILE             | Write grid file to <XFILE>.step# . If XFILE is blank, don’t write grid    |
|-------------------|---------------------------------------------------------------------------|
| QFILE             | Write Q file to <QFILE>.step# . If QFILE is blank, don’t write Q file.    |
| QAVGFILE          | Write Q-average file to <QAVGFILE>.step# . If QAVGFILE is blank,          |
|                   | don’t write Q-average file. [“ ”]                                         |
| TRIQFILE          | Write triq file to < TRIQFILE >. nnnnnn.triq , where nnnnnn is a 6-digit, |
| FOMOFILE          | Write triq file of COMPNAME FOMOCO component(s) to                        |
|                   | <FOMOFILE>.nnnnnn.triq , where nnnnnn is a 6-digit,zero-padded            |
| NSTART            | Start writing files at step NSTART (use –1 for last). [1]                 |
| NSTOP             | Stop writing files at step NSTOP (use –1 for last). [-1]                  |
| NSAVE             | Save files every NSAVE steps. [1]                                         |
| IPRECIS           | 0—Write files in default-precision (same as grid.in ).                    |
| IG(subset#)       | -2—Cut all grids using CUT and VALUE.                                     |
|                   | >0—Specify grid numbers for subsets or cuts. Grid subsets can be          |
|                   | specified using J,K,L indices; grid cuts can be specified using CUT and   |
| JS,JE,JI(subset#) | J-index start/end/increment for subsets. [Increment default is 1]         |
| KS,KE,KI(subset#) | K-index start/end/increment for subsets. [Increment default is 1]         |
| LS,LE,LI(subset#) | L-index start/end/increment for subsets. [Increment default is 1]         |
| INCREF(subset#)   | For near-body subsets, include associated refinement regions. [TRUE]      |
| CUT(subset#)      | Grid cut type (“X”, “Y”, “Z”, or “J”, “K”, “L”).                          |
| VALUE(subset#)    | Coordinate value for grid cut.                                            |
| COMPNAME(comp#)   | FOMOCO component name(s) to write to FOMOFILE                             |

**&ADSNML** (Acoustic Data Surface parameters)

| PGRID   | List of 2D grids to save. Grids will be written in the same order as    |
|---------|-------------------------------------------------------------------------|
| PSAVE   | >0—Save ADS files ads_files/ads.save.[xq] every how many steps.         |
|         | <0—Save ADS files ads_files/ads#####.[xq] every how many steps.         |
| PS,PE   | Specify starting and ending step numbers for ADS file saves. PS=0 means |
| IPRECIS | 0—Write ADS files in default-precision (same as grid.in ).              |

## **&CNDTNS** (Rotor global parameters; *rotorcraft coupling only*)

| NDTN         | rotorcrat couping ony                                        |
|--------------|--------------------------------------------------------------|
| IFORMAT      | 1—Use standard rotorcraft interface files.                   |
| IQCFORMAT    | 0—Use chord and twist from CFD grid.                         |
| IMANVR       | 0—No vehicle motion.                                         |
| CFDORIGIN(3) | Origin of CFD coordinate system, in vehicle frame. [(0,0,0)] |
| CFDXAXIS(3)  | X-axis of CFD coordinate system, in vehicle frame. [(1,0,0)] |
| CFDYAXIS(3)  | Y-axis of CFD coordinate system, in vehicle frame. [(0,1,0)] |

# **&EROTOR** (Rotor parameters, one namelist per rotor; *rotorcraft coupling only*)

| &EROTOR    | (Rotor parameters, one namelist per rotor; rotorcraft coupling only ) |
|------------|-----------------------------------------------------------------------|
| ICOUPLING  | <0—No aeroelastic coupling and no rotor force reporting.              |
|            | 1—Aeroelastic coupling using Euler angles.                            |
| DYMORE     | Enable DYMORE structures code interface (not used). [FALSE]           |
| TIPEXTRAP  | Extrapolate aeroelastic deformations to the tip, if not defined.      |
| ROOTEXTRAP | Extrapolate aeroelastic deformations to the root, if not defined.     |
| IBS        | Body ID (or component ID) for first blade. [1]                        |
| IBE        | Body ID for last blade (blades are assumed to be consecutive). [1]    |
| IGS        | Main grid (used for force and moment calculations) for first blade.   |
| IGINC      | Increment between main blade grids (assumed constant for all          |
| NPSI       | Number of azimuth stations for saving sectional blade forces to       |
|            | <FORCEFILE>.onerev.txt . [72]                                         |
| IVISC      | 1/0—Do/don’t include viscous forces. [1]                              |
| ITINC      | Save rotor forces every ITINC steps. Sectional and total forces are   |
|            | written to <FORCEFILE>.forces.txt ; total forces are written to       |
|            | <FORCEFILE>.hist.txt . [100]                                          |
| IEXCH      | >0—Exchange data with comprehensive code every IEXCH steps.           |

| NMAP               | Inverse-maps updated every NMAP steps. [25]                           |
|--------------------|-----------------------------------------------------------------------|
| NXRAY              | X-rays updated every NXRAY steps. [NMAP]                              |
| RTIP               | Rotor tip radius (in grid units). [1.]                                |
| MTIP               | Tip Mach number (for blade force non-dimensionalization).             |
| CREF               | >0—Reference chord (for blade force non-dimensionalization).          |
| RRCMREF(station#)  | Radial stations for blade moment reference line, normalized by        |
| XCCMREF(station#)  | Chordwise offset from the leading edge of the blade moment            |
| YCCMREF(station#)  | Normal offset from the leading edge of the blade moment reference     |
| SCALEMOTION(1:6)   | Scale factor for each component of blade deformation (dx,dy,dz,       |
| CAMRAD2_ROTATE     | Reverse definition of azimuth angle psi (use +1 for counter          |
| BASENAME           | Base name for all rotor files. [ rotor_n ]                            |
| MOTIONFILE         | Motion file for reading blade deformations.                           |
|                    | [ <BASENAME>.motion.txt ]                                             |
| ONEREVFILE         | Output file for blade sectional force map (airfoil frame, coefs non  |
|                    | dimensionalized by speed-of-sound). [ <BASENAME>.onerev.txt ]         |
| ONEREVINERTIALFILE | Output file for blade sectional force map (inertial frame, coefs non |
|                    | dimensionalized by V tip ). [ <BASENAME>.onerev_inertial.txt ]        |
| FORCEFILE          | Output file for blade sectional forces. [ <BASENAME>.forces.txt ]     |
| HISTFILE           | Rotor total force history. [ <BASENAME>.hist.txt ]                    |
| QCFILE             | Quarter-chord diagnostic file. [ <BASENAME>.qc.txt ]                  |

### **&FLOINP** (Flow parameters)

| FSMACH   | Freestream Mach number (              | M              | ∞                            | ). [0.0]                                          |
|----------|---------------------------------------|----------------|------------------------------|---------------------------------------------------|
| REFMACH  | Reference Mach number (               | M ref          |                              | ). [FSMACH]                                       |
| ALPHA    | Angle-of-attack (α), deg. [0.0]       |                |                              |                                                   |
| BETA     | Sideslip angle (β), deg. [0.0]        |                |                              |                                                   |
| REY      | Reynolds number (                     | Re ) (based on |                              | V ref and grid length unit). [0.0]                |
| TINF     | Freestream static temperature (       |                |                              | T ∞ ), deg. Rankine. [518.7]                      |
| GAMINF   | Freestream ratio of specific heats (γ |                |                              | ∞ ). [1.4]                                        |
| PR       | Prandtl number (                      | Pr ). [0.72]   |                              |                                                   |
| PRT      | Turbulent Prandtl number (            | Pr             | t                            | ). [0.9]                                          |
| MUTINF   | Freestream turbulence level (         |                | μ                            | t /μ l ) ∞ for 1- or 2-eq turbulence models. [1.0 |
|          |                                       |                | for NQT=101,103,301; 0.1 for | all other                                         |
| (RETINF) | (Replaced by MUTINF. If used,         |                |                              | MUTINF will be set to RETINF.)                    |
| XKINF    | Freestream turbulent kinetic energy ( |                |                              | k ∞ /V ref2                                       |
|          | [10 -6                                |                |                              |                                                   |

| FSTI      | Freestream turbulence intensity (percent) (=100 sqrt (2/3 XKINF)). (Enter  |
|-----------|----------------------------------------------------------------------------|
| TARGCL    | Enable the target C L -driver option. [FALSE]                              |
| CLTARG    | Value of C L the code will try to match. [0.0]                             |
| CLALPH    | Fixed value of dC L /dα used to update ALPHA. [0.1]                        |
| NTARG     | Number of steps between ALPHA corrections (corrections are not done on     |
| CTP       | Rotor thrust coefficient (for BC type 37). [0.0]                           |
| ASPCTR    | Rotor radius (for BC type 37). [1.0]                                       |
| FROUDE    | Froude number (gravity term), Fr=V ref /sqrt(gL) (based on V ref and grid  |
| GVEC(1:3) | Unit up-vector for FROUDE gravity term. (Note that this vector is taken    |
|           | verbatim—it is not modified internally by the angle-of-attack, since other |

## **&VARGAM** (Variable γ input)

| &VARGAM     | (Variable γ input)                      |                                                |     |       |                                             |                                   |                            |                |
|-------------|-----------------------------------------|------------------------------------------------|-----|-------|---------------------------------------------|-----------------------------------|----------------------------|----------------|
| IGAM        |                                         | Options for specifying calculation of γ when   |     | not   |                                             |                                   | solving species continuity |                |
| HT1         | Total enthalpy ratio h                  | 0 /h 0∞                                        |     |       | below which the mixture is all gas 1. [10.] |                                   |                            |                |
| HT2         | Total enthalpy ratio h                  | 0 /h 0∞ above which the                        |     |       |                                             | mixture is all gas 2. [10.]       |                            |                |
| SCINF(gas#) | Freestream species mass fraction        | c i∞                                           |     |       |                                             | . [1 for gas 1, 0 for all others] |                            |                |
| SMW(gas#)   | Species molecular weight                | MW i                                           |     |       | , or normalized molecular weight            |                                   |                            | MW i /MW ∞     |
| ALT0(gas#)  |                                         | Lower temperature range polynomial coefficient |     |       | a 0                                         | (540                             | R<T<1800                   |  R).          |
|             | [γ ∞ /( γ ∞ -1)]                        |                                                |     |       |                                             |                                   |                            |                |
| ALT1(gas#)  | Lower temp range polynomial coefficient |                                                | a 1 | (540  |                                            | R<T<1800                          |                           | R). [0.0]      |
| ALT2(gas#)  | Lower temp range polynomial coefficient |                                                | a 2 | (540  |                                            | R<T<1800                          |                           | R). [0.0]      |
| ALT3(gas#)  |                                         | Lower temp range polynomial coefficient        | a 3 | (540  |                                            | R<T<1800                          |                           | R). [0.0]      |
| ALT4(gas#)  | Lower temp range polynomial coefficient |                                                | a 4 | (540  |                                            | R<T<1800                          |                           | R). [0.0]      |
| AUT0(gas#)  |                                         | Upper temperature range polynomial coefficient |     |       | a 0                                         | (1800                             |                           | R<T<9000  R). |
| AUT1(gas#)  | Upper temp range                        | polynomial coefficient a                       | 1   | (1800 |                                             |                                  | R<T<9000                  | R). [0.0]      |
| AUT2(gas#)  | Upper temp range polynomial coefficient | a                                              | 2   | (1800 |                                             |                                  | R<T<9000                  | R). [0.0]      |
| AUT3(gas#)  | Upper temp range polynomial coefficient | a                                              | 3   | (1800 |                                             |                                  | R<T<9000                  | R). [0.0]      |
| AUT4(gas#)  | Upper temp range polynomial coefficient | a                                              | 4   | (1800 |                                             |                                  | R<T<9000                  | R). [0.0]      |
| SIGL(gas#)  | Laminar diffusion coefficient           |  l . [1.0]                                    |     |       |                                             |                                   |                            |                |
| SIGT(gas#)  | Turbulent diffusion coefficient         |  t . [1.0]                                    |     |       |                                             |                                   |                            |                |

The following NAMELISTs are repeated per grid.

| NAME    | Grid name (not used internally). [blank]                               |
|---------|------------------------------------------------------------------------|
| &NITERS | (Subiterations per grid)                                               |
| ITER    | Number of flow solver iterations per step. (Each flow solver iteration |

| &METPRM    | (Numerical method selection)                                               |
|------------|----------------------------------------------------------------------------|
| IRHS       | 0—Central difference Euler terms.                                          |
| ILHS       | 0—ARC3D Beam-Warming block tridiagonal scheme.                             |
|            | 6/7/8—SSOR (Jacobi in J/L/K) algorithm (with subiteration), 32-bit         |
|            | 16/17/18—Improved SSOR (Jacobi in J/L/K), 64-bit arithmetic.               |
|            | 26/27/28—Improved SSOR (Jacobi in J/L/K), 32-bit arithmetic. [2]           |
| ILHSIT     | Number of subiterations for D3ADI or SSOR.                                 |
|            | [10 for ILHS=6-8,16-18,26-28; 3 for ILHS=4]                                |
| IDISS      | 2—ARC3D dissipation scheme (2 nd                                           |
|            | -, 4 th                                                                    |
|            | 3—TLNS3D dissipation scheme (same as IDISS=2, but smooth ρh 0 instead      |
|            | of ρe 0 ).                                                                 |
| ILIMIT     | Limiter for upwind Euler terms (IRHS=3-6):                                 |
|            | See DELTA for further control. [1]                                         |
| BIMIN      | 1.0—Disable low-Mach preconditioning.                                      |
|            | -1.0—Enable low-Mach preconditioning; reset BIMIN to 3x M ref2             |
| SSOR_RELAX | Relaxation factor for SSOR schemes (flow eqns, turb models, species eqns). |
| Q_LIMIT    | TRUE—Limit Q update procedure to try to keep density and energy from       |
|            | FALSE—Use simple Q update procedure (from OVERFLOW 1.8).                   |

| MULTIG | Local flag to enable/disable multigrid acceleration for this grid. [Default is |
|--------|--------------------------------------------------------------------------------|
| SMOOP  | Smoothing coefficient for prolongation of coarse-grid solution onto next      |
| SMOOC  | Smoothing coefficient for multigrid correction before interpolation onto       |
| SMOOR  | Smoothing coefficient for multigrid residual before restricting to the next   |
| CORSVI | Enable/disable computation of viscous terms on coarse grid levels. [TRUE]      |
| RECMUT | Recompute μ t on finest level during multigrid. [FALSE]                        |

## **&TIMACU** (Time accuracy)

| ITIME        | Time-step scaling flag:                                             |
|--------------|---------------------------------------------------------------------|
|              | 0—Constant time-step, no scaling (used for simple time-stepping or  |
| DT           | Time-step factor. [0.5]                                             |
| CFLMIN       | Minimum CFL number. [0.0]                                           |
| CFLMAX       | Maximum CFL number. [0.0]                                           |
| RAMP_CFL     | Global flag to turn on CFL ramping for ITIME=3 or 4. [.FALSE.]      |
| CFLMIN_LIMIT | Global minimum CFL number during ramping. [CFLMIN]                  |
| CFLMAX_LIMIT | Global maximum CFL number during ramping. [CFLMAX]                  |
| TFOSO        | Order of time-accuracy, when using simple time-stepping (NITNWT=0): |
|              | -order time-accuracy (trapezoidal scheme).                          |

### **&SMOACU** (Smoothing parameters)

| ISPEC | Dissipation scaling flag; single value to specify ISPECJ,ISPECK,ISPECL:   |
|-------|---------------------------------------------------------------------------|
| SMOO  | 0.0—Spectral radius is computed normally, as  U +kc.                      |
|       | 1.0—Sound speed c is replaced by   V  /M ref , reducing smoothing in low |
| DIS2  | 2                                                                         |

|        | th                                                                                                                                      |
|--------|-----------------------------------------------------------------------------------------------------------------------------------------|
| DIS4   | 4 -order smoothing coefficient. [0.04]                                                                                                  |
| FSO    | Order of accuracy for spatial differencing of Euler terms. FSO=[1,6]; non integer values allowed.                                      |
|        | For IRHS=0, values of [2,6] are implemented: FSO=2 gives 2 nd -order with                                                               |
|        | 4/2 dissipation; FSO=3 gives 4 th                                                                                                       |
|        | -order with 4/2; FSO=4 gives 4 th -order with                                                                                           |
|        | 6/2; FSO=5 gives 6 th                                                                                                                   |
|        | -order with 6/2; and FSO=6 gives 6 th -order with 8/2. For IRHS=2, values of [1,2] are implemented. WENO5M. [3.0 (2.0 for IRHS=2)]      |
| DELTA  | MUSCL scheme flux limiter flag:                                                                                                         |
|        | For ILIMIT=1 (Koren limiter): <0.0—Turn off limiter. 0.0—Koren limiter. <0.0—Turn off limiter. 0.0-1.0—Standard limiter implementation. |
| FILTER | 0—No Q filtering. 3—3rd-order (5-point) Q filtering. 5—5th-order (7-point) Q filtering.                                                 |
| EPSSGS | LU-SGS left-hand side spectral radius epsilon term (ILHS=3 only). [0.02]                                                                |
| VEPSL  | Matrix dissipation minimum limit on linear eigenvalues. [0.0]                                                                           |
| VEPSN  | Matrix dissipation minimum limit on nonlinear eigenvalues. [0.0]                                                                        |
| ROEAVG | Matrix dissipation flag to use Roe averaging for half-grid point flow quantities. [FALSE]                                               |

# **&VISINP** (Viscous and turbulence modeling input)

| VISC    | TRUE—Include all viscous terms including cross terms. This overrides   |
|---------|------------------------------------------------------------------------|
| VISCJ   | TRUE—Include viscous thin-layer terms in J.                            |
| VISCK   | TRUE—Include viscous thin-layer terms in K.                            |
| VISCL   | TRUE—Include viscous thin-layer terms in L.                            |
| VISCX   | TRUE—Include viscous cross terms between coordinate directions that    |
| WALLFUN | TRUE—Use wall function formulation for all viscous walls in this grid. |

| CFLT           | Turbulence model time-step is CFLT times the flow solver time-step. [1.0]         |
|----------------|-----------------------------------------------------------------------------------|
| ITERT          | Number of turbulence model iterations per flow solver iteration (ITER); or        |
| ITLHIT         | Number of subiterations for DDADI or SSOR scheme. [3 for NQT=100-                 |
| FSOT           | 1.0-1st-order differencing for turbulence convection terms.                       |
| MUT_LIMIT      | =0.0—No limit on turbulent eddy viscosity.                                        |
| IQCR           | 0—No Quadratic Constitutive Relation (QCR).                                       |
| IDES           | 0—No Detached Eddy Simulation (DES).                                              |
|                | 4—Use DDES with modified Scotti length scale ( max(Δ 1 ,Δ 2 ,Δ 3 )f(a 1 ,a 2 ) ). |
| IRC            | 0—No rotation/curvature correction term for turbulence model.                     |
|                | 1—Use SA-RC/SST-RC rotation/curvature correction term (with dS ij /dt             |
|                | May be applied to any 1- or 2-equation turbulence model. [0]                      |
| ICC            | 0—No compressibility correction.                                                  |
|                | 1—Use Secundov (SA model) or Sarkar (SST model) compressibility                   |
| ITC            | 0—No temperature correction.                                                      |
| ICF            | 0—No crossflow transition option.                                                 |
| H_CF           | RMS surface roughness height, to be used with ICF=1 crossflow transition          |
| ISTRAIN        | 0—Use strain as calculated.                                                       |
| ISUST          | 0—No sustaining terms.                                                            |
| ILMG           | 0—Standard Langtry-Menter transition model.                                       |
|                | 1—Langtry-Menter Galilean invariance modification (from Rohit Jain). [0]          |
| ITTYP(region#) | Turbulence modeling region type.                                                  |

| ITDIR(region#)  | Turbulence model region coordinate direction (away from wall or shear |
|-----------------|-----------------------------------------------------------------------|
| JTLS(region#)   | Starting J index.                                                     |
| JTLE(region#)   | Ending J index.                                                       |
| KTLS(region#)   | Starting K index.                                                     |
| KTLE(region#)   | Ending K index.                                                       |
| LTLS(region#)   | Starting L index.                                                     |
| LTLE(region#)   | Ending L index.                                                       |
| TLPAR1(region#) | Turbulence model region parameter (usage depends on region type).     |

## **&BCINP** (Boundary condition input)

| IBTYP(region#)  | Boundary condition type.                                              |
|-----------------|-----------------------------------------------------------------------|
| IBDIR(region#)  | Boundary condition coordinate direction (away from boundary surface). |
| JBCS(region#)   | Starting J index.                                                     |
| JBCE(region#)   | Ending J index.                                                       |
| KBCS(region#)   | Starting K index.                                                     |
| KBCE(region#)   | Ending K index.                                                       |
| LBCS(region#)   | Starting L index.                                                     |
| LBCE(region#)   | Ending L index.                                                       |
| BCPAR1(region#) | Boundary condition parameter (usage depends on boundary type).        |
| BCPAR2(region#) | Boundary condition parameter (usage depends on boundary type).        |
| BCFILE(region#) | File name for reading boundary data (usage depends on boundary type). |

### **&SCEINP** (Species continuity input)

| CFLC   | Species continuity equation time-step is CFLC times the flow solver time  |
|--------|----------------------------------------------------------------------------|
| ITERC  | Number of species continuity equation iterations per flow solver iteration |
| ITLHIC | Number of species equation left-hand side subiterations:                   |
| IUPC   | 0—Central differencing for species convection terms.                       |
|        | 1—Upwind differencing for species convection terms.                        |
| FSOC   | Order of accuracy for spatial differencing of species convection terms.    |
| DIS2C  | 2                                                                          |
| DIS4C  | 4                                                                          |

# **&SIXINP** (6-DOF input) (OVERFLOW-D only; only for I6DOF≠2)

| IGMOVE      | 0—Body does not move (even if DYNMCS=TRUE).                                |
|-------------|----------------------------------------------------------------------------|
| IDFORM      | 0—Grid does not deform.                                                    |
| NMAP        | Update inverse-maps for deforming body every NMAP steps. [1]               |
| NXRAY       | Update X-rays for deforming body every NXRAY steps. [1]                    |
| BMASS       | Body mass. [1.0]                                                           |
| TJJ,TKK,TLL | Body moments of inertia, about the principal axes (assumed to be body      |
| WEIGHT      | Body weight. [0.0]                                                         |
| ISHIFT      | Starting step number for applied loads (time=0). [0]                       |
| FX,FY,FZ    | Body applied forces (in global x,y,z directions). [0,0,0]                  |
| FMX,FMY,FMZ | Body applied moments (about global x,y,z axes). [0,0,0]                    |
| STROKT      | Time duration for applied loads to be active. [0.]                         |
| FREER       | Enable/disable (all 3) body rotational degrees-of-freedom, while applied   |
| FREE        | Enable/disable all body degrees-of-freedom, while applied loads are active |
| X00,Y00,Z00 | Body CG location in body coordinates. [0,0,0]                              |
| X0,Y0,Z0    | Initial body CG location in global coordinates. [X00,Y00,Z00]              |
| E1,E2,E3,E4 | Initial body Euler parameters in global coordinates. [0,0,0,1]             |
| UR,VR,WR    | Initial velocity of CG in global coordinates. [0,0,0]                      |
| WX,WY,WZ    | Initial angular velocity about CG in global coordinates. [0,0,0]           |
| WJ,WK,WL    | Initial angular velocity about CG in body coordinates. [0,0,0]             |
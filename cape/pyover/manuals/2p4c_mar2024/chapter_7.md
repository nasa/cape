## **7. Utility Codes**

Several utility codes are provided for pre-processing, running, monitoring, and post-processing solutions with OVERFLOW. These utility codes are compiled using the **Maketools** make file or the **makeall** script. The following codes are provided:

## **Preprocessing tools:**

| addbb ( RETINF addgam addke field variables for the | ~  PLOT3D Q file, corresponding to , the field variable for   in the NAMELIST input). Q file suitable to use as a restart or BCFILE file. grid) PLOT3D Q file, corresponding to rho*k and rho*  , the k  two-equation turbulence model. k |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                     | 0 .0001 2                                                                                                                                                                                                                                      |
|                                                     | V f                                                                                                                                                                                                                                            |
| Values are initialized to                           | , and  such that = Re                                                                                                                                                                                                                         |
|                                                     | 0 .1                                                                                                                                                                                                                                           |
|                                                     |                                                                                                                                                                                                                                              |
| the default free stream values.                     |  t the eddy viscosity at infinity is set to . These are =                                                                                                                                                                                     |
| changeq                                             | The changeq program reads in a OVERFLOW Q file and allows                                                                                                                                                                                      |
| boundary conditions.                                | This is useful to set modify existing Q data for BCFILE                                                                                                                                                                                        |
| endian_convert                                      | Converts Fortran unformatted files between big- and little-endian.                                                                                                                                                                             |
| Applies to grid, Q, and                             | XINTOUT files.                                                                                                                                                                                                                                 |
| find_y                                              | Program to find initial spacing in the normal (viscous) direction,                                                                                                                                                                             |
| + y                                                 | =1 is desired. This program uses the simple form of flat plate                                                                                                                                                                                 |
| effects.                                            | It is appropriate for subsonic through transonic (air) in grid units. We assume normal AIR properties.                                                                                                                                         |
| find_y2                                             | Program to find initial spacing in the normal (viscous) direction,                                                                                                                                                                             |
| + =1 is desired. y                                  | Also required are the Mach number and free                                                                                                                                                                                                     |
| stream temperature (deg R).                         | This program uses the more                                                                                                                                                                                                                     |
|                                                     | Sommer and Short, which includes temperature effects. It is                                                                                                                                                                                    |
| appropriate for supersonic flows.                   | It is assumed that Re and the                                                                                                                                                                                                                  |
|                                                     | distance downstream (x) are both supplied in "grid units." The                                                                                                                                                                                 |

returned value of y is then also in grid units. We assume normal

air properties.

gaschem The program GASCHEM computes the Cp/R polynomial

coefficients required to run the variable gamma options of the OVERFLOW code. The program reads the following input for

each gas:

Chemical species symbol

 Molecular weight of chemical species Mass fraction of chemical species

> The code will use the chemical species symbol to search the database (fort.4) for a match. Thus it is IMPERATIVE that the symbols match down to the character. This database is derived from the NASA Lewis CEC-80 (Chemical Equilibrium Chemistry - 1980) database. Program output is a file containing input echo and polynomial coefficients for the variable gamma options, and a gamma vs. temperature table for plotting purposes.

grid32\_to\_64/grid64\_to\_32 Converts a single- or multiple-grid PLOT3D grid file from

REAL\*4 to REAL\*8 and vice versa (integers remain \*4).

intout\_to\_xintout Convert a DCF (OVERFLOW-D style) **INTOUT** file to a

PEGASUS style **XINTOUT** file.

q32\_to\_64/q64\_to\_32 Converts a single- or multiple-grid OVERFLOW Q file from

REAL\*4 to REAL\*8 and vice versa (integers remain \*4).

turb\_init Convert a Q file from a 1-equation turbulence model to a 2-

equation turbulence model, or vice-versa.

xintout32\_to\_64/xintout64\_to\_32 Converts a PEGASUS **XINTOUT** file from REAL\*4 to

REAL\*8 and vice versa (integers remain \*4).

xrays32\_to\_64/xrays64\_to\_32 Converts a DCF X-Ray file from REAL\*4 to REAL\*8 and vice

versa (integers remain \*4).

## **Execution tools:**

livePlot\_p3d Script to view the **resid.out** or **turb.out** file as overflow is running

overrun Unix script to run OVERFLOW and keep a history of run

information, residuals, force and moments, and minimum density

and pressure.

overrunmpi Similar to **overrun**, but runs the **overflowmpi** executable on

multiple machines using MPI.

tellme Unix script for echoing a message from a batch job to an

interactive terminal of the same user.

| bbplot     | Generate “fake Q file” for plotting, containing ( Re t ,1,1,1,1) for a |
|------------|------------------------------------------------------------------------|
|            | 1-eq turbulence model, or ( k , ω ,1,1,1) for a 2-eq model. (These     |
| cfwf       | Calculates surface skin friction and heat transfer coefficients.       |
| checkq     | Reads q.save and prints a summary of min/max density, pressure,        |
| fvbnd      | Creates an fvbnd file for FieldView or Tecplot from a grid.in file     |
| listq      | This lists values of Q in a specified subset.                          |
| mergem     | Produce simple min rho/p history plotting file from OVERFLOW           |
|            | rpmin.out -type file. (See OVERPOST.)                                  |
| merger     | Produce simple flow solver residual history plotting file from         |
|            | OVERFLOW resid.out -type file. (See OVERPOST.)                         |
| merges     | Produce simple species continuity residual history plotting file       |
|            | from OVERFLOW species.out -type file. (See OVERPOST.)                  |
| merget     | Produce simple turbulence residual history plotting file from          |
|            | OVERFLOW turb.out -type file (1- and 2-equation turbulence             |
| overpost   | Unix script to summarize residual, min ρ/p , turbulence residual,      |
|            | the (unsupported) utility xyplot . (Local plotting utilities ought to  |
|            | Fortran utilities merge { m,r,s,t }.                                   |
| outline_ob | Program to generate a PLOT3D command file to outline off-body          |
| overclean  | Unix script to clean up (delete) log, resid, fomoco, rpmin, turb,      |
| subs_ob    | Program to generate a PLOT3D command file to set off-body              |
|            | grid subsets for a given x , y , or z =constant plane.                 |
| vgplot     | The vgplot program reads in OVERFLOW solution files from               |

xysift Program to "thin" the input history plotting file.

xysplit Program to split the input history plotting file into files with no more than 10 curves per file.
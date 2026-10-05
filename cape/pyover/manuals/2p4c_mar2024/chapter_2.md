# **2. Requesting and Compiling OVERFLOW 2.4**

### **2.1 Obtaining OVERFLOW 2.4**

OVERFLOW 2.4 is a NASA research code, and is available only within the United States, to U.S. companies, universities, and individuals under an appropriate Software Usage Agreement. OVERFLOW 2.4 is distributed as source code. The code can be requested from the NASA Software Catalog at https://software.nasa.gov/software/LAR-20095-1. **Note:** Only U.S. Persons are authorized to request and use OVERFLOW Version 2.4.

Once a Software Usage Agreement form has been signed and approval granted by NASA, instructions on how to access the code will be provided.

## **2.2 Compiling OVERFLOW 2.4**

The code comes packaged as a gzipped tar file containing source code, documentation, and test cases. Uncompress the tar file:

#### **tar xvfz over2.4.tar.gz**

The code will be uncompressed and placed into a directory named over2.4. The OVERFLOW code is stored in a collection of subdirectories, generally with each subroutine in a separate file. The upper level subdirectories are:

| chimera    | Basic overhead and control subroutines for OVERFLOW                                        |
|------------|--------------------------------------------------------------------------------------------|
| doc        | Collection of documentation for options in OVERFLOW                                        |
| grid       | Grid-related subroutines                                                                   |
| helios     | Interface to the U.S. Army AFDD HELIOS software framework                                  |
| include    | Top-level include files                                                                    |
| ke         | Two-equation turbulence transport model subroutines                                        |
| linear     | Matrix solution routines                                                                   |
| modules    | Fortran 90 modules                                                                         |
| ns         | Navier-Stokes subroutines                                                                  |
| omisoft    | Overset grid assembly (DCF), near-body grid refinement, off-body grid generation, Geometry |
| ret        | One-equation turbulence transport model subroutines                                        |
| rotor      | Rotorcraft interface subroutines                                                           |
| rotor_disk | Rotor or propeller actuator disk subroutines                                               |
| sce        | Species transport equation subroutines                                                     |
| spline     | Spline routines                                                                            |
| test       | Group of test cases for OVERFLOW 2.4                                                       |
| tools      | Various utility codes for pre- and post-processing as well as running OVERFLOW             |
| unix       | Interfaces to Unix routines used by OVERFLOW                                               |
| utilities  | Various utility routines used by OVERFLOW                                                  |
| walldist   | Wall distance calculation routines                                                         |

Each of these directories has further subdirectories beneath them. The test and doc subdirectories are not used in compiling the code.

A Unix "Makefile" is supplied to automatically compile and link these subroutines together. Makefiles are also supplied for compiling several utility programs. Appropriate system and compiler flags for several types of computer systems are included in a file called **Make.sys**. System flags differ in such details as Fortran compiler name and options. Instructions for adding other systems are included at the top of this file. To get a list of currently implemented systems, type **make** with no arguments, in the main OVERFLOW directory. Note that Linux systems using MPI may require substantial modification of the **Make.sys** file. The environment variable **MPI\_ROOT** should be set for compilation of the code using MPI. This environment variable should be set to the directory that contains the MPI include files and libraries as subdirectories that will be used to compile and to run the code. The following is an example of setting the environment for csh and tcsh Unix shells

#### **setenv MPI\_ROOT /usr/local/mpich**

The following makefiles are included:

Makefile Serial version of OVERFLOW 2.4 Makefilempi MPI parallel version of OVERFLOW 2.4 Maketools Utility codes for use with OVERFLOW 2.4

A **makeall** script is also included that compiles and installs *overflow*, OVERFLOW tools, and the MPI version of OVERFLOW (*overflowmpi*), for both single- and double-precision. Single-precision executables are placed in the directory **over2.4/bin**, and double-precision executables are placed in **over2.4/bin\_dp**. Here precision refers to floating-point precision, and must match the precision of the grid and restart files. Be sure to include the proper directory in your path when running OVERFLOW. The **makeall** script may also be used as an example for using the Makefile system for compiling individual codes.

To compile and link the serial version of OVERFLOW manually, type

**make <machine> CMD=<command>**

The parallel code can be compiled and linked manually by typing

**make –f Makefilempi <machine> CMD=<command>**

where **<machine>** (required) is one of the following:

| crayx1        | Cray x1 (ftn)                                    |
|---------------|--------------------------------------------------|
| cray          | Cray (f90)                                       |
| generic       | Generic (no special optimizations)               |
| gfortran      | GNU gfortran/gcc                                 |
| ibm           | IBM power series processors (xlf)                |
| mac           | Macintosh OSX (xlf)                              |
| intel         | Intel ifort/icc                                  |
| intel_omp     | Intel ifort/icc (with OpenMP)                    |
| altix         | SGI Altix with Intel ifort/icc                   |
| altix_ice     | SGI Altix ICE with Intel ifort/icc               |
| altix_ice_omp | SGI Altix ICE with Intel ifort/icc (with OpenMP) |
| lahey         | Lahey lf95/gcc                                   |
| nagware       | Linux Mandrake OS with NAGWare f95/gcc           |
| pgf           | Portland Group compilers                         |
| pgf_omp       | Portland Group (with OpenMP)                     |
| sicortex      | SiCortex recommended (no OpenMP)                 |
| sicortex_omp  | SiCortex (MPI/OpenMP hybrid)                     |
| sun           | Sun Ultra with multitasking                      |

and **<command>** (optional) is one of:

| all (default) | compile and link                      |
|---------------|---------------------------------------|
| update        | compile only                          |
| install       | compile, link, move executable to bin |
| clean         | remove all intermediate files         |
| deinstall     | clean, also remove executable         |

The defaults in the Makefiles are for single-precision compilation. The **make** command must be modified if you are compiling the code manually, and you would like the double-precision version of the code; see the **makeall** script for examples.

## **2.3 Recommended Companion Software**

The Chimera Grid Tools<sup>1</sup> (CGT) software suite developed at NASA Ames includes many useful utility codes wrapped in a graphical user interface called **overgrid**. CGT includes utilities for building and modifying grids, overset grid diagnosis, building code input files, generating X-rays, constructing and viewing Geometry Manipulation Protocol<sup>2</sup> (GMP) moving-body XML files, FOMOCO<sup>3</sup> and USURP<sup>4</sup> code for overset force and moment calculations, and much more. CGT is required for running many of the OVERFLOW test cases, as well as for any aerodynamic force and moment integration in OVERFLOW. More information on CGT is available at [https://www.nas.nasa.gov/publications/software/docs/chimera/index.html.](https://www.nas.nasa.gov/publications/software/docs/chimera/index.html)

The PEGASUS 5<sup>5</sup> grid assembly code may be used to generate the overset communication files required for OVERFLOW 2.4. PEGASUS 5 is particularly effective for steady-state problems, and is used for a number of the OVERFLOW test cases. More information on PEGASUS 5 is available at [https://www.nas.nasa.gov/publications/software/docs/pegasus5/index.html.](https://www.nas.nasa.gov/publications/software/docs/pegasus5/index.html)

PLOT3D<sup>6</sup> is an old-style (command-driven) CFD visualization program. Many of the OVERFLOW test cases include command files for PLOT3D visualization; Chimera Grid Tools also uses PLOT3D for illustrating results.

Chimera Grid Tools, PEGASUS 5, and PLOT3D may all be requested from the NASA Software Catalog at [https://software.nasa.gov/.](https://software.nasa.gov/)

The SUGGAR<sup>7</sup> grid assembly code may also be used to generate the overset communication files required for OVERFLOW 2.4. The code may be obtained from the developer Ralph Noack at Celeritas Simulation Technology [\(https://sites.google.com/view/celeritas-sim-tech\)](https://sites.google.com/view/celeritas-sim-tech).

## **2.4 OVERFLOW Mailing List**

Users may join the OVERFLOW mailing list at [https://overflow.larc.nasa.gov.](https://overflow.larc.nasa.gov/) This will provide email updates from the developers and problems and solutions from OVERFLOW users.

### **References**

 Chan, W.M., "The OVERGRID Interface for Computational Simulations on Overset Grids," AIAA-2002- 3188, June 2002. Murman, S., Chan, W.M., Aftosmis, M.J., Meakin, R.L., "An Interface for Specifying Rigid-Body Motions for CFD Applications", AIAA-2003-1237, Jan. 2003. Chan, W.M., and Buning, P.G., "User's Manual for FOMOCO Utilities – Force and Moment Computation Tools for Overset Grids," NASA TM 110408, July 1996. Boger, D., and Dreyer, J., "Prediction of Hydrodynamic Forces and Moments for Underwater Vehicles Using Overset Grids," AIAA-2006-1148, Jan. 2006. Suhs, N.E., Rogers, S.E., and Dietz, W.E., "PEGASUS 5: An Automated Pre-Processor for Overset-Grid CFD," AIAA-2002-3186, June 2002. Walatka, P.P., Buning, P.G., Pierce, L., and Elson, P.A., "PLOT3D User's Manual," NASA TM 101067, Mar. 1990. Noack, R., "SUGGAR: A General Capability for Moving Body Overset Grid Assembly," AIAA-2005-5118, June 2005.
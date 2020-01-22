
.. _pyover-example-powered-nacelle:

--------------------------------
OVERFLOW Powered Nacelle Example
--------------------------------

This pyOver example shows how to use pyover to run one of the simple test cases
that come with the OVERFLOW source code. This example starts with the grids and
inputs files that are created within the OVERFLOW examples, and documents how
to create the pyOver setup, run matrix, how to run several OVERFLOW cases,
covers some post-processing.  This example is located in 

    * ``$CAPE/examples/pyover/02_powered_nacelle/``

This example shows how to use pyOver for a test case with two related
configurations, a flow-through axisymmetric nacelle, and a powered axisymmetric
nacelle.  The example comes with the grids and input files ready to run
OVERFLOW. However, if one desires to generate these files locally, here are the
commands that were used to create the grid and input files in the OVERFLOW
source.  This assumes that the OVERFLOW source bundle has been installed in a
directory whose absolute path is given by the environment variable
``$OVERHOME``.  The following generates double-precision, little-endian
unformatted versions of the grid files.

  .. code-block:: bash

    export GFORTRAN_CONVERT_UNIT=little_endian
    export FC=gfortran
    export FFLAGS=-fdefault-real-8
    cd $OVERHOME/test/powered_nacelle/grids_ft
    ./makegrids
    cd ../run_ft
    rsync -av Config.xml xrays.in grid.in \
       $CAPE/example/pyover/02_powered_nacelle/common_flowthrough/.
    rsync -av mixsur.{inp,fmp} grid.{ibi,ib,map,i.tri,nsf} \
       $CAPE/example/pyover/02_powered_nacelle/common_flowthrough/fomo/.
    rsync -av m0.80.1.inp
       $CAPE/example/pyover/02_powered_nacelle/common_flowthrough/overflow.inp
    cd ../grids
    ./makegrids
    cd ../run
    rsync -av Config.xml xrays.in grid.in \
       $CAPE/example/pyover/02_powered_nacelle/common_powered/.
    rsync -av pr136.1.inp \
       $CAPE/example/pyover/02_powered_nacelle/common_powered/overflow.inp
    rsync -av mixsur.{inp,fmp} grid.{ibi,ib,map,i.tri,nsf} \
       $CAPE/example/pyover/02_powered_nacelle/common_powered


Flow-Through Nacelle Case
-------------------------

Given the grid and input files created by the OVERFLOW scripts, the only two
files that are required to run the flow-through example using pyOver are
``flowthrough.json`` and ``inputs/matrix/flowthrough.csv``.

To execute and duplicate the OVERFLOW Mach 0.8 flow-through example, simply run
the command ``pyover -f flowthrough.json -I 1``. This simple problem
will run in a matter of seconds:

    .. code-block:: console

        pyover -f flowthrough.json -I 1
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    flowthrough/m0.8      ---     /           .            
          Case name: 'flowthrough/m0.8' (index 1)
             Starting case 'flowthrough/m0.8'
         > overrunmpi -np 8 run 01
             (PWD = '/examples/pyover/02_powered_nacelle/flowthrough/m0.8')
             (STDOUT = 'overrun.out')
           Wall time used: 0.00 hrs (phase 0)
           Wall time used: 0.00 hrs
           Previous phase: 0.00 hrs
         > overrunmpi -np 8 run 02
             (PWD = '/examples/pyover/02_powered_nacelle/flowthrough/m0.8')
             (STDOUT = 'overrun.out')
           Wall time used: 0.00 hrs (phase 1)
        
        Submitted or ran 1 job(s).
        
        ---=1, 

This ran the MPI version of OVERFLOW locally, (without submitting a PBS job or
similar) using 8 processors (cores). The actions that pyOver takes are
summarized here:

    #. Create the directory ``flowthrough/m0.8/``
    #. Create symbolic links pointing to some files in ``common_flowthrough``
    #. Create two OVERFLOW inputs files ``run.01.inp`` and ``run.02.inp``
    #. Execute ``overrunmpi -np 8 run 01``
    #. Execute ``overrunmpi -np 8 run 02``


The sections in ``flowthrough.json`` that control the execution of
OVERFLOW are shown here:

    .. code-block:: javascript


        // Namelist template
        "OverNamelist": "common_flowthrough/overflow.inp",
        // Options for overall run control and command-line inputs
        "RunControl": {
            // Run sequence
            "PhaseSequence": [0,   1],
            "PhaseIters":    [600, 1400],
            // Operation modes
            "Prefix": "run",
            "MPI": true,
            "qsub": false,
            "Resubmit": [false, true],
            "Continue": true,
            "mpicmd": null,
            "nProc": 8,
            // Dictionary of environment variables
            "Environ": {
                "F_UFMTENDIAN": "little"
            },

            // OVERFLOW command-line interface
            "overrun": {
                "cmd": "overrunmpi",
                "aux": null
            }
        },


The *PhaseSequence* and *PhaseIters* specify how many times and how long the
code is run. The first specifies that OVERFLOW will run for phase ``0`` and
phase ``1``, (which are labeled as ``01`` and ``02`` for ``overrunmpi``
execution). These phases run until there are 600 and 1400 total global steps
in OVERFLOW. For these and other inputs in the .json file, the sequential
list of arguments are applied to sequentially to each phase. Note that
if only one value is given, that value is applied for all phases. Also note
that if the number of phases are greater than the number of inputs in a
sequential list, the latter phases will use the last value given in the list.

Setting *MPI* to ``true`` instructs pyOver to use the MPI version
of OVERFLOW, but setting *mpicmd* to ``null`` is required because we want
pyOver to use the ``overrunmpi`` script, as specified by the *cmd* value in
the *overrun* section.

Note that the actual number of iterations in one run of each phase is not set in the
*RunControl* section above. These are controlled by the OVERFLOW input
variable *NSTEPS* in the *GLOBAL* namelist. In the first phase we are also
running full-multi-grid (FMG) iterations with FMGCYC = [[300,300]] and
*NSTEPS[0]* = 0, thus 600 total iterations in the first phase. 

Here are the sections in ``flowthrough.json`` that control the *GLOBAL*
and *OMIGLB* namelists:

    .. code-block:: javascript

        // Namelist inputs
        "Overflow": {
            "GLOBAL": {
                "NQT": 102,
                "NSTEPS": [0,   800],
                "NSAVE":  [0,  2000],
                "FMG": [true, false],
                "FMGCYC": [[300,300]],
                "NGLVL": 3,
                "ISTART_QAVG": 15000,
                "WALLDIST": [2],
                "DTPHYS": [0.0, 0.0, 0.0, 0.0, 1.0],
                "NITNWT": [0,   0,   0,     0,   5]
            },
            "OMIGLB": {
                "IRUN": 0
            }
        },

Here are the sections in ``flowthrough.json`` that control the namelists
for each individual mesh.  The *"ALL":* section is applied to all grids. 
If one wants to specify different input values for a single grid, duplicate
this section and replace *"ALL"* with the name of that grid in double quotes.

    .. code-block:: javascript

        // Namelist parameters for each grid
        "Grids": {
            // Settings applied to all grids
            "ALL": {
                // Solver parameters
                "METPRM": {
                    "IRHS": 0,
                    "ILHS": 2
                },
                "TIMACU": {
                    "ITIME": 1,
                    "DT": 0.10,
                    "CFLMIN": 5.0,
                    "CFLMAX": 0.0
                },
                "SMOACU": {
                    "DIS2": 2.0,
                    "DIS4": 0.04,
                    "DELTA": 1.0
                }
            }
        },
        
Here is the *MESH* section, which tells pyOver which files to copy and which
files to create symbolic links for.

    .. code-block:: javascript

        // Mesh
        "Mesh": {
            // Folder containing definition files
            "ConfigDir": "common_flowthrough",
            // Grid type, dcf or peg5
            "Type": "dcf",
            // List or dictionary of files to link
            "LinkFiles": [
                "grid.in",
                "xrays.in",
                "fomo/grid.ib",
                "fomo/grid.ibi",
                "fomo/grid.nsf",
                "fomo/grid.map"
            ],
            // List of files to copy instead of linking
            "CopyFiles": [
                "Config.xml",
                "fomo/mixsur.fmp"
            ]
        },



One very important section of ``flowthrough.json`` is the *RunMatrix*
section, shown here:

    .. code-block:: javascript

        // RunMatrix description
        "RunMatrix": {
            // If a file is specified, and it exists, trajectory values will be
            // read from it.  RunMatrix values can also be specified locally.
            "File": "inputs/matrix/flowthrough.csv",
            "Keys": ["mach"],
            // Copy the mesh
            "GroupMesh": true,
            // Configuration name [default]
            "GroupPrefix": "flowthrough"
        }

This describes an extremely simple run matrix file, whose only primary input
variable (listed in the *Keys* input) is *mach*. Because the flow-through
nacelle is an axisymmetric flow problem, one cannot run different angles of
incidence, therefore *alpha* and *beta* are not listed as input variables.

Run Mach Sweep
--------------

Having defined the *RunMatrix* section in the json file, we can see that the
run matrix given in the ``inputs/matrix/flowthrough.csv`` file looks
like this:

  .. code-block:: console

    # mach, config, Label
      0.75, flowthrough, 
      0.80, flowthrough, 
      0.85, flowthrough, 
      0.90, flowthrough, 

The run matrix consists of four cases with different Mach numbers. These cases
can all be run using just the command ``pyover``.  Doing this will execute the
three remaining cases (since we ran case 1 in the beginning).  Afterwards, 
check the status of the cases using ``pyover -c``, which should produce a list
showing all the cases with a status of ``DONE``:

  .. code-block:: console

    Case Config/Run Directory  Status  Iterations  Que CPU Time 
    ---- --------------------- ------- ----------- --- --------
    0    flowthrough/m0.75     DONE    1400/1400   .        0.0 
    1    flowthrough/m0.8      DONE    1400/1400   .        0.0 
    2    flowthrough/m0.85     DONE    1400/1400   .        0.0 
    3    flowthrough/m0.9      DONE    1400/1400   .        0.0 
    
    DONE=4, 


Report Generation
-----------------

After running all four cases in the run matrix, the next thing to do is
examine the convergence and view the flow. This can be accomplished for our
case using the command:

    .. code-block:: console

        pyover --report -I 0:4

This will create the report in the file ``report/report-flowthrough.pdf``.
There should be two pages for each case, one page with a table of aerodynamic
data and several convergence plots, and one page with two flow-visualization
figures.

Convergence Plots
^^^^^^^^^^^^^^^^^

Nine different convergence plots are shown on the first page of the report for
each case.  In addition to plotting the history of the three force coefficients
and the three moment coefficients, the plot of the residual history, two
different views are added zooming into the tail end of the axial force
coefficient convergence.  The *force_CAzoom1* and *force_CAzoom2* subfigures
show the last 800 and last 400 iterations of the convergence history. 
The definition of the subfigures used to view the convergence is relatively
straightforward. The following shows the these subfigure definitions in
``flowthrough.json``:


    .. code-block:: javascript

        // Definitions for subfigures
        "Subfigures": {
            ...
            ...
            // Iterative history
            "force": {
                "Type": "PlotCoeff",
                "Component": "TOTAL FORCE",
                "nPlotFirst": 0,
                "FigWidth": 4.5,
                "FigHeight": 3.4,
                "Width": 0.33,
                "StandardDeviation": 1.0
            },
            "force_CA": {"Type": "force", "Coefficient": "CA"},
            "force_CY": {"Type": "force", "Coefficient": "CY"},
            "force_CN": {"Type": "force", "Coefficient": "CN"},
            "force_CLL": {"Type": "force", "Coefficient": "CLL"},
            "force_CLM": {"Type": "force", "Coefficient": "CLM"},
            "force_CLN": {"Type": "force", "Coefficient": "CLN"},
            "force_CAzoom1": {
                "Type": "force", 
                "Coefficient": "CA",
                "nPlotFirst": -800
            },
            "force_CAzoom2": {
                "Type": "force", 
                "Coefficient": "CA",
                "nPlotFirst": -400
            },
            // Residual history
            "L2": {
                "Type": "PlotL2",
                "FigWidth": 5.5,
                "FigHeight": 6,
                "Width": 0.33,
                "nPlotFirst": 1,
                "Caption": "$L_2$ Density Residual"
            }
        }

When viewing the convergence and showing the entire history it can appear that
the forces are very tightly converged. But when viewing the tail end, one can
see that the axial force is still dropping slightly. The following figures show
four of the convergence plots illustrating the three views of *CA* as well
as the history of the L2 norm of the residual of the mean-flow quantities.

    .. _tab-pyover-nacelle-01:
    .. table:: Convergence plots for the m0.75 case

        +-----------------------------+-----------------------------+
        |.. image:: force_CA.*        |.. image:: force_CAzoom1.*   |
        |     :width: 3.2in           |     :width: 3.2in           |
        |                             |                             |
        |TOTAL FORCE/*CA*             |TOTAL FORCE/*CA*             |
        +-----------------------------+-----------------------------+
        |.. image:: force_CAzoom2.*   |.. image:: L2.*              |
        |     :width: 3.2in           |     :width: 3.2in           |
        |                             |                             |
        |TOTAL FORCE/*CA*             |*L2* Residual                |
        +-----------------------------+-----------------------------+




Flow Visualization
^^^^^^^^^^^^^^^^^^

In the *Report* section of ``flowthrough.json``, the subfigures for the 
flow visualization use Tecplot® subfigures. Here we re-use the contour and 
color map settings from the ``01-bullet`` pyover example. The *MachSlice*
subfigure uses tecplot and the supplied layout file in 
``inputs/flowthrough-mach.lay`` to create Mach contours in the *Y=0* plane
of the nacelle. Note that the *MaxLevel* for the contours is dependant
upon the freestream Mach number. The color map break points are also a function
of the freestream Mach. 

At the end of this section, the *MachSlice-mesh* subfigure is defined. This
subfigure inherits all of the settings from the *MachSlice* subfigure, but
uses a different layout file. The only difference between the two layout
files is that the addition of the mesh overlay on the Mach contours.


    .. code-block:: javascript

        // Definitions for subfigures
        "Subfigures": {
            // Tecplot figures
            "MachSlice": {
                "Type": "Tecplot",
                "Layout": "inputs/flowthrough-mach.lay",
                "FigWidth": 1024,
                "Width": 0.65,
                "Caption": "Mach slice $y=0$",
                "ContourLevels": [
                    {
                        "NContour": 1,
                        "MinLevel": 0,
                        "MaxLevel": "max(1.4, 1.4*$mach)",
                        "Delta": 0.05
                    }
                ],
                "ColorMaps": [
                    {
                        "Name": "Diverging - Purple/Green modified",
                        "NContour": 2,
                        "ColorMap": {
                            "0.0": "purple",
                            "$mach": "white",
                            "1.0": ["green", "orange"],
                            "max(1.4,1.4*$mach)": "red"
                        }
                    }
                ],
                "Keys": {
                    "GLOBALCONTOUR": {
                        "LABELS": {
                            "Value": {
                                "AUTOLEVELSKIP": 2,
                                "NUMFORMAT": {
                                    "FORMATTING": "'FIXEDFLOAT'",
                                    "PRECISION": 1,
                                    "TIMEDATEFORMAT": "''"
                                }
                            },
                            "Parameter": 1
                        }
                    }
                }
            },
            "MachSlice-mesh": {
                "Type": "MachSlice",
                "Layout": "inputs/flowthrough-mach-mesh.lay"
            },
          ...
          ...
        }

The resulting *MachSlice* subfigures for each of the four cases are shown here:

    .. _tab-pyover-nacelle-02:
    .. table:: Tecplot® Mach contour plots for each case

        +------------------------------+------------------------------+
        |.. image:: MachSlice_m075.png |.. image:: MachSlice_m080.png |
        |    :width: 3.2in             |    :width: 3.2in             |
        |                              |                              |
        |Mach slice m0.75              |Mach slice m0.80              |
        +------------------------------+------------------------------+
        |.. image:: MachSlice_m080.png |.. image:: MachSlice_m090.png |
        |    :width: 3.2in             |    :width: 3.2in             |
        |                              |                              |
        |Mach slice m0.80              |Mach slice m0.90              |
        +------------------------------+------------------------------+



Other pyOver Actions
--------------------


    .. code-block:: console

        pyover -c
        pyover -I 0,2,3
        pyover --PASS -I 0:4
        pyover --report
        pyover --aero
        pyover --archive
        pyover --clean
        pyover --skeleton



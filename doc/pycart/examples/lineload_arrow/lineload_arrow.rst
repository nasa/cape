
.. _pycart-ex-lineload-arrow:

Demo 6: Line Loads on the Arrow Example
=======================================

NOTE: This example requires `Chimera Grid Tools 
<https://www.nas.nasa.gov/publications/software/docs/chimera/index.html>`_ to
calculate sectional loads.  Specifically, the `triload
<https://www.nas.nasa.gov/publications/software/docs/chimera/pages/triload.html>`_
command is used.  To acquire Chimera Grid Tools, free software from NASA, use
the `NASA Software Catalog <https://software.nasa.gov/software/ARC-16025-1A>`_.

Using the geometry from :ref:`Example 2 <pycart-ex-arrow>`, this case
continues the analysis and adds computation of sectional loads.  The example is
located in ``$PYCART/examples/pycart/06_lineload_arrow``.

The geometry used for this shape is a capped cylinder with four fins and 9216
faces and seven components.  The surface triangulation, :file:`arrow.tri`, is
shown below.

    .. figure:: ../arrow01.png
        :width: 4in
        
        Simple bullet shape triangulation with four fins
        
This example is set up with a small run matrix to demonstrate line loads on a
few related cases.

    .. code-block:: console
    
        $ pycart -c
        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m1.25a0.0b0.0 ---     /           .            
        1    poweroff/m1.25a2.0b0.0 ---     /           .            
        2    poweroff/m1.25a0.0b2.0 ---     /           .            
        
Running Cases
-------------
When doing post-processing of Cart3D results, it is often desirable to perform
time-averaging or iteration-averaging before doing analysis.  When Cart3D exits
(either ``flowCart`` or ``mpix_flowCart`` called with the ``-clic`` option), it
writes a ``Components.i.triq`` file.  This is a surface triangulation with some
results from the last iteration.  It contains the pressure coefficient and the
five native Cart3D state variables at each node of the triangulation.

To get a ``triq`` file with averaged results, we have to run ``flowCart`` a few
iterations at a time and manually perform averaging.  The ``-stats`` option
performs a similar task, but it is not quite consistent with what's needed for
an averaged line load.  To get pyCart to perform this unusual task, we have the
following ``"RunControl"`` section in :file:`pyCart.json`.

    .. code-block:: javascript
    
        // Iteration control and command-line inputs
        "RunControl": {
            // Run sequence
            "PhaseSequece": [0],
            "PhaseIters": [200],
            // System configuration
            "nProc": 4,
            "MPI": 0,
            "Adaptive": 0,
            // Options for ``flowCart``
            "flowCart": {
                "it_fc": 200,
                "it_avg": 10,
                "it_start": 100,
                "cfl": 1.1,
                "mg_fc": 3,
                "y_is_spanwise": true
            },
            // Defines the flow domain automatically
            "autoInputs": {"r": 6},
            // Volume mesh options
            "cubes": {
                "maxR": 8,
                "pre": "preSpec.c3d.cntl",
                "cubes_a": 8,
                "cubes_b": 2,
                "reorder": true
            }
        }

As previously, the *RunControl>flowCart>it_fc* option controls how many
iterations ``flowCart`` runs for.  The *it_avg* and *it_start* are new options.
The idea is that Cart3D will be run for *it_avg* iterations at a time.  pyCart
then calculates a cumulative average ``triq`` file that updates after each
*it_avg* iterations.  However, it first runs *it_start* iterations before
initiating this start-stop behavior.  This prevents initial iterations from
corrupting the average.

If we run one case, there is a lot of output printed to STDOUT, and it looks
something like this.  The output has been truncated.  

**Note:** This is set up to run on four threads and take
less than one minute.

    .. code-block:: console
    
        $ pycart -I 0
        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m1.25a0.0b0.0 ---     /           .            
          Group name: 'poweroff' (index 0)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
         > autoInputs -r 6 -t Components.i.tri -maxR 8 -nDiv 4
         > cubes -pre preSpec.c3d.cntl -maxR 8 -reorder -a 8 -b 2
         > mgPrep -n 3
             Starting case 'poweroff/m1.25a0.0b0.0'
         > flowCart -his -clic -N 100 ...
         > flowCart -his -clic -restart -N 110 ...
         > flowCart -his -clic -restart -N 120 ...
         > flowCart -his -clic -restart -N 130 ...
         > flowCart -his -clic -restart -N 140 ...
         > flowCart -his -clic -restart -N 150 ...
         > flowCart -his -clic -restart -N 160 ...
         > flowCart -his -clic -restart -N 170 ...
         > flowCart -his -clic -restart -N 180 ...
         > flowCart -his -clic -restart -N 190 ...
         > flowCart -his -clic -restart -N 200 ...
             Writing triangulation: 'Components.11.100.200.triq'
        
        Submitted or ran 1 job(s).
        
        ---=1, 
        
This lengthy output explains more clearly what is meant by running ``flowCart``
10 iterations at a time.  The iteration-averaged surface file that gets created
at the end, ``Components.11.100.200.triq``, explains the contents of the file. 
Specifically, it says that the file contains input from 11 iterations between
100 and 200.

Let's run the last two cases in the run matrix, too.

    .. code-block:: console
    
        $ pycart -n 2
        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m1.25a0.0b0.0 DONE    200/200     .        0.0 
        1    poweroff/m1.25a2.0b0.0 ---     /           .            
             Starting case 'poweroff/m1.25a2.0b0.0'
         > flowCart -his -clic -N 100 ...
         > flowCart -his -clic -restart -N 110 ...
         ...
         > flowCart -his -clic -restart -N 200 ...
             Writing triangulation: 'Components.11.100.200.triq'
        2    poweroff/m1.25a0.0b2.0 ---     /           .            
             Starting case 'poweroff/m1.25a0.0b2.0'
         > flowCart -his -clic -N 100 ...
         > flowCart -his -clic -restart -N 110 ...
         ...
         > flowCart -his -clic -restart -N 200 ...
             Writing triangulation: 'Components.11.100.200.triq'
        
        Submitted or ran 2 job(s).
        
        ---=2, DONE=1, 
        
Calculating Line Loads
----------------------
The purpose of this example was to create line loads, so let's investigate that
part.  To instruct pyCart which components on which to compute line loads, we
go to the ``"DataBook"`` section of :file:`pyCart.json`.

    .. code-block:: javascript
    
        // Database info
        "DataBook": {
            // List of data book components
            "Components": ["arrow_no_base", "ll_arrow"],
            // Location of data book
            "Folder": "data/",
            // Parameters for collecting data
            "nFirst": 0,
            "nStats": 100,
            "nMin": 100,
            // Basic component
            "bullet_no_base": {
                "Type": "FM"
            },
            // Line load
            "ll_arrow": {
                "Type": "LineLoad",
                "CompID": "arrow_no_base",
                "nCut": 100
            }
        }

This specifies that the databook contains two "Components".  One of them is the
the statistically averaged forces and moments on the ``arrow_no_base`` CompID,
and the other is the sectional load on the same.  Recall from :ref:`Example 2
<pycart-ex-arrow>` that the ``arrow_no_base`` component includes all the
surfaces except the base.

The ``"ll_arrow"`` databook component is defined as a ``"LineLoad"`` component
on the ``arrow_no_base`` CompID, and it is instructed to calculate the
sectional loads on 100 slices of that component.  By default, these slices will
be at constant-*x* planes.

This ``"CompID"`` option allows users to calculate line loads on parts of the
vehicle (for example a wing) and also have multiple line load databooks for the
same vehicle.

Adding this little section to the ``"DataBook"`` is all that's needed to set up
a line load computation.  To actually calculate the line loads, run the
following commands.

**Note:** This command should take less than five seconds to run.

    .. code-block:: console
    
        $ pycart --ll
        Updating line load data book 'll_arrow' ...
        poweroff/m1.25a0.0b0.0
          Adding new databook entry at iteration 200.
            triloadCmd < triload.ll_arrow.i > triload.ll_arrow.o
        poweroff/m1.25a2.0b0.0
          Adding new databook entry at iteration 200.
            triloadCmd < triload.ll_arrow.i > triload.ll_arrow.o
        poweroff/m1.25a0.0b2.0
          Adding new databook entry at iteration 200.
            triloadCmd < triload.ll_arrow.i > triload.ll_arrow.o

This command creates a collection of files.  First, we will note the creation
of a ``lineload`` folder in each case directory.  In the
``poweroff/m1.25a0.0b0.0/lineload`` folder, there are several files used in the
raw computation of line loads created by the Chimera Grid Tools utility
``triloadCmd``.

The file :file:`triload.ll_arrow.i` is the input to ``triloadCmd`` that is
automatically created by pyCart.  The main output file is
:file:`LineLoad_ll_arrow.slds`, which contains the non-dimensionalized forces
on each of the 100 slices.

These raw files are then read by pyCart and processed into a databook in the
``data/`` folder (locations specified by the *DataBook>Folder* option in
:file:`pyCart.json`).  Below is a file tree of the ``06_lineload_arrow/data``
folder.

    .. code-block:: none
    
        data/
            ll_ll_arrow.csv
            lineload/
                LineLoad_ll_arrow.smy
                LineLoad_ll_arrow.smz
                poweroff/
                    m1.25a0.0b0.0/
                        LineLoad_ll_arrow.csv
                    m1.25a2.0b0.0/
                        LineLoad_ll_arrow.csv
                    m1.25a0.0b2.0/
                        LineLoad_ll_arrow.csv

The top-level ``ll_ll_arrow.csv`` file is a status file that stores which cases
have computed line loads and what iteration at which they have been computed.
It looks a lot like a force and moment databook file (e.g.
:file:`aero_arrow_no_base.csv`) except that there are no data columns (since
those are stored in the line load folders.

In the ``data/lineload/`` directory, there are two files with unusual file
extensions.  These are just text files that give the outline of the body
intersected by the :math:`y{=}0` plane (``.smy``) and :math:`z{=}0` plane
(``.smz``).  They are used to make the line load plots more convenient, and
which will make more sense in the next subsection.

Within the ``data/lineload/`` folder, there is a whole file tree that mirrors
that of the run cases.  The actual sectional loads from
``poweroff/m1.25a.0.b0.0`` are stored in
``data/lineload/poweroff/m1.25a0.0b0.0``, etc.  In this case, each line load
case folder contains only one file, but if there were more line load
components, there would be one for each line load. Each is a very simple file
containing seven columns: *x/Lref*, and then one for each of the six
coefficients (*CA*, *CY*, *CN*, *CLL*, *CLM*, *CLN*).  The coefficient data is
stored in a seemingly strange format of
:math:`\mathrm{d}C_A/\mathrm{d}(x/L_\mathit{ref})`.  Using this form keeps
results nondimensional but also removes dependence on the number of cuts.

Creating Plots and Automated Reports
------------------------------------
Line load plots are fairly easy to set up.  First let's just create the report
and then describe the ``"Report"`` section of :file:`pyCart.json`.  

    .. code-block:: console
    
        $ pycart --report
        poweroff/m1.25a0.0b0.0
          CaseConds: New subfig at iteration 200.0
          FMTable: New subfig at iteration 200.0
        /usr/lib/python2.7/dist-packages/numpy/core/_methods.py:105: RuntimeWarning: overflow encountered in multiply
          x = um.multiply(x, x, out=x)
          arrow_CA: New subfig at iteration 200.0
          arrow_CY: New subfig at iteration 200.0
          arrow_CN: New subfig at iteration 200.0
          arrow_CLL: New subfig at iteration 200.0
          arrow_CLN: New subfig at iteration 200.0
          arrow_CLM: New subfig at iteration 200.0
          L1: New subfig at iteration 200.0
          LL_CY: New subfig at iteration 200.0
          LL_CN: New subfig at iteration 200.0
        poweroff/m1.25a2.0b0.0
          CaseConds: New subfig at iteration 200.0
          FMTable: New subfig at iteration 200.0
          arrow_CA: New subfig at iteration 200.0
          arrow_CY: New subfig at iteration 200.0
          arrow_CN: New subfig at iteration 200.0
          arrow_CLL: New subfig at iteration 200.0
          arrow_CLN: New subfig at iteration 200.0
          arrow_CLM: New subfig at iteration 200.0
          L1: New subfig at iteration 200.0
          LL_CY: New subfig at iteration 200.0
          LL_CN: New subfig at iteration 200.0
        poweroff/m1.25a0.0b2.0
          CaseConds: New subfig at iteration 200.0
          FMTable: New subfig at iteration 200.0
          arrow_CA: New subfig at iteration 200.0
          arrow_CY: New subfig at iteration 200.0
          arrow_CN: New subfig at iteration 200.0
          arrow_CLL: New subfig at iteration 200.0
          arrow_CLN: New subfig at iteration 200.0
          arrow_CLM: New subfig at iteration 200.0
          L1: New subfig at iteration 200.0
          LL_CY: New subfig at iteration 200.0
          LL_CN: New subfig at iteration 200.0
        Compiling...
        Compiling...
        Cleaning up...
        
This creates a multipage PDF (in this case one title page and three more pages
with one page dedicated to each case) that contains selected analysis tables
and plots.  In this case we have set up the report to show one table
identifying the case in more detail, one table of basic force coefficient
results, and nine plots.

    .. figure:: report-case-p3.*
        :width: 5.5in
        
        Automatically generated report for ``poweroff/m1.25a2.0b0.0``
        
The first six plots are of each force or moment coefficient on the
``arrow_no_base`` component.  There is an obvious problem with the *CLL* and
*CLN* plots, which has to do with some confusion due to the symmetry of the
arrow shape.  (This bug may go away in future versions of Cart3D).  We will
discuss how to make these two figures look a little better shortly, but let's
move on to the other three plots.  The first is a plot of the global
:math:`L_1` norm of density residuals (which is the main residual reported by
Cart3D).

The last two plots are line load plots.  Let's discuss the JSON syntax to set
up each of these plots and also how these subfigures are assembled into a
report.  The basic skeleton of the ``"Report"`` section of :file:`pyCart.json`
is shown below.

    .. code-block:: javascript
    
        "Report": {
            // List of reports
            "Reports": ["case"],
            // Define the report
            "case": {
                "Title": "Automated Cart3D Report with Line Load Plots",
                "Subtitle": "Example \\texttt{06\\_lineload\\_arrow}",
                "Restriction": "pyCart Example - Distribution Unlimited",
                "Figures": ["CaseTables", "CasePlots"]
            },
            // Define the figures
            "Figures": {
                "CaseTables": {
                    "Alignment": "left",
                    "Subfigures": ["CaseConds", "FMTable"]
                },
                "CasePlots": {
                    "Header": "Iterative analysis and sectional loads",
                    "Alignment": "center",
                    "Subfigures": [
                        "arrow_CA",  "arrow_CY",  "arrow_CN",
                        "arrow_CLL", "arrow_CLN", "arrow_CLM",
                        "L1",        "LL_CY",     "LL_CN"
                    ]
                }
            },
            // Definitions for subfigures
            "Subfigures" {
                ...
            }
        }

The overall structure is relatively simple: there is a list of reports (the
same JSON file can have many different reports defined), an overall definition
for the report including a list of figures, a section defining each figure, and
a section defining the subfigures.  A figure is a collection of subfigures plus
an alignment option and optional header.

Creating this report creates a file called :file:`report-case.pdf` in the
``report/`` folder.  The individual plots created for the report are stored in
folders such as ``report/poweroff/m1.25a2.0/a0.0/``, with each subfigure having
a file name corresponding to the title of the subfigure (e.g.
:file:`arrow_CA.pdf`).

Line Load Subfigures
^^^^^^^^^^^^^^^^^^^^
The focus of this section is on the subfigures, and in particular the plots.
To learn more about the two tables, the actual example :file:`pyCart.json` file
is relatively easy to understand.  Defining syntax for the line load plots is
shown below.

    .. code-block:: javascript
    
        "Subfigures": {
            ...
            "LL_arrow": {
                "Type": "PlotLineLoad",
                "Component": "ll_arrow",
                "FigWidth": 5.5,
                "FigHeight": 6,
                "Width": 0.33,
                "SeamCurves": "smy",
                "SeamLocation": "bottom"
            },
            "LL_CY": {
                "Type": "LL_arrow",
                "Caption": "arrow\\_no\\_base/CY",
                "Coefficient": "CY"
            },
            "LL_CN": {
                "Type": "LL_arrow",
                "Caption": "arrow\\_no\\_base/CN",
                "Coefficient": "CN"
            }
        }

We have two line load plots that share many common options defined in
``"LL_arrow"``.  This demonstrates the concept of cascading options and can
save time, effort, and number of lines in the JSON file.  The *LL_arrow>Type*
option is set to ``"PlotLineLoad"``, which is the basic pyCart line load
subfigure type.  The *Component* is set to the name of the line load component
as listed in the ``"DataBook"`` section, and the *Width* setting determines
what percentage of the available text width in the final PDF document is taken
up by the figure.

The *FigWidth* and *FigHeight* obviously set an aspect ratio for the figure,
but the absolute scale of *FigWidth* also determines the size at which the
figure is rendered.  A larger *FigWidth* will make the labels appear to be in a
smaller font size since the size in the document is set by *Width*.

Finally, the *SeamCurves* option list which slice of the geometry (if any) to
plot to help the reader anchor what part of the line load corresponds to what
geometrical features.  The *SeamLocation* plot sets where to put this slice;
``"bottom"`` is the usual choice.

**Waring**: The seam curve plots have automatically adjusted aspect ratio to
avoid distorting the seam curve. As a result, geometry with inconvenient actual
aspect ratios will lead to problematic seam curve plots.

    .. figure:: LL_CN.*
        :width: 4in
        
        Normal sectional loads at 2 degrees angle of attack

Residual History Subfigure
---------------------------
Cart3D residual plots almost always have the same JSON inputs.  The version for
this plot uses a different *FigHeight* in order to match the aspect ratio of
the neighboring line load plots.

    .. code-block:: javascript
    
        "Subfigures": {
            ...
            "L1": {
                "Type": "PlotL1",
                "FigWidth": 5.5,
                "FigHeight": 6,
                "Width": 0.33,
                "Caption": "$L_1$ Density Residual"
            },
            ...
        }
        
Force & Moment Plots
--------------------
The iterative history plots are relatively simple for this case since we are
only plotting one component.

    .. code-block:: javascript
    
        "Subfigures": {
            ...
            "arrow": {
                "Type": "PlotCoeff",
                "Component": "arrow_no_base",
                "FigWidth": 5.5,
                "FigHeight": 4.2,
                "Width": 0.33
            },
            "arrow_CA": {"Type": "arrow", "Coefficient": "CA"},
            "arrow_CY": {"Type": "arrow", "Coefficient": "CY"},
            "arrow_CN": {"Type": "arrow", "Coefficient": "CN"},
            "arrow_CLL": {"Type": "arrow", "Coefficient": "CLL"},
            "arrow_CLM": {"Type": "arrow", "Coefficient": "CLM"},
            "arrow_CLN": {"Type": "arrow", "Coefficient": "CLN"},
            ...
        }
        
As we saw above, this simulation results in very poor results for *CLL* and
*CLN* due to the symmetry of the configuration (among other things).  We can at
least make the figures look readable by using scientific notation for the mean
value and removing the standard deviation.

    .. code-block:: javascript
    
        "Subfigures": {
            ...
            "arrow_CLL": {
                "Type": "arrow",
                "Coefficient": "CLL",
                "MuFormat": "%.2e",
                "ShowSigma": false
            },
            "arrow_CLN": {
                "Type": "arrow",
                "Coefficient": "CLN",
                "MuFormat": "%.2e",
                "ShowSigma": false
            },
            ...
        }
        
The updated *CLN* plot is shown below.

    .. figure:: arrow_CLN.*
        :width: 3.5 in
        
        Problematic yawing moment coefficient with slightly improved formatting

There are also many different options for each of these plots, and it is also
possible to plot line loads from other databases on top of those of the most
recent case for comparison.  See the :ref:`JSON page <pycart-json-Report>` for
a thorough description of options.

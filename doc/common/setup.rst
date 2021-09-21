
General Case Setup
==================

Some guidelines for setting up a new case apply to all configurations
independent of the particular CFD solver and whether the case is simple or very
complex.  As mentioned several times before, the primary JSON file is mandatory
and controls most of the settings of a run configuration.

This master JSON file points to several other files, such as meshes, template
input files, or a list of input conditions.  In addition, users may wish to
define some of the settings in a Python script instead of typing them out in
JSON.  This enables the use of loops, etc., and also it provides input settings
that are dependent on conditions.

Two of the sections of the JSON file, particularly ``"DataBook"`` and
``"Report"`` can be filled in after getting the cases running.  The other
sections of the JSON file are generally required to be configured correctly
before starting a case.  The most important section for determining the basic
structure of a run setup is the ``"RunMatrix"`` :ref:`section
<cape-json-RunMatrix>`.

Matrix Setup
------------
The first step is to pick the list of run matrix variables, which is a list set
in *RunMatrix* |>| *Keys*. For example, this could be as simple as ``["mach",
"alpha"]`` for a simple run matrix, or there could be may more variables such
as fin deflection angles or thrust settings. Common variables such as
``"mach"``, ``"alpha"``, ``"beta"``, etc. do not need definitions, but special
variables such as thrust settings need to have further explanation in the
*RunMatrix* |>| *Definitions* section.

In addition, there are three other pre-declared variable types that do not
directly impact any CFD input settings.  These variables are ``"config"``, which
sets the name of the group folder; ``"Label"``, which appends an extra string to
the end of the run folder name; and ``"tag"``, which does not affect either
folder name but can be used to store other markers.  For example, consider the
following run matrix CSV file:

    .. code-block:: none
    
        # mach, alpha, config, Label, tag
        2.50, 2.0, hi-q, ,
        2.50, 2.0, hi-q, b, 
        2.50, 2.0, lo-q, , 
       
The folder names for these three cases are below:

    .. code-block:: none
    
        hi-q/m2.50a2.0
        hi-q/m2.50a2.0_b
        lo-q/m2.50a2.0
        
Quite often the ``"Label"`` key is used to redo cases that failed on the first
attempt but users can also use for other purposes, such as denoting different
grid resolutions, etc.

As hinted above, another file that is needed independently of which particular
solver is being used is the run matrix file.  This is a simple comma-separated
or white-space-separated file where each row is a separate CFD case, and each
column is a separate variable.  **Note:** commas are recommended over spaces
because space-separated values don't work properly with empty columns.

We generally recommend leaving two spaces at the start of each row so that
marking a case as completed (PASS) doesn't mess with the alignment.  The run
matrix file shown above with the first two cases marked as PASs would look like
the following.

    .. code-block:: none
    
        # mach, alpha, config, Label, tag
        p 2.50, 2.0, hi-q, ,
        p 2.50, 2.0, hi-q, b, 
          2.50, 2.0, lo-q, , 

Pre-run JSON Setup
-------------------
Most of the sections of the JSON file other than *Report*, *DataBook*, and
*RunControl* |>| *Archive* affect the way the CFD solver is run, and thus they
should generally be determined before starting the first case.

For each version of the code, the *RunControl* section contains some key
parameters. For example, *RunControl* |>| *PhaseSequence* and *RunControl* |>|
*PhaseIters* are mandatory. Some of the other *RunControl* parameters are very
dependent on which solver is being used, for example command-line options to
``flowCart`` (for Cart3D) ``nodet`` (for FUN3D).

Users should take special consideration of the *Config* section prior to
starting the actual runs.  This section can be used to request certain outputs
from a solver, and it is important that the solver is reporting all the needed
information.  It is possible to have the template input files (such as
``input.cntl`` for Cart3D) already set up to request the right outputs, but if
the user is relying on the convenience of pyCart to request the information,
*Config* is the section to do it.  Most commonly, this description refers to
asking for iterative force and moment histories, but other parameters such as
point sensors are occasionally set here, too.

Mesh Files and Other Templates
------------------------------
There are certain pointers to mesh files in each version of the JSON file, and
each code needs several other template input files as well. Most of these input
files have default templates provided in ``$CAPE/templates/``, but some files
like OVERFLOW's ``overflow.inp`` namelist depend on the number of grids and
cannot be reduced to a global default template.

Below is a list of the required or semi-required input templates for each
solver.

    * Cart3D
        * ``input.cntl``: Main Cart3D input file (global template provided)
        * ``aero.csh``: Cart3D adaptation script (global template provided)
        
    * FUN3D
        * ``fun3d.nml``: Template namelist (global template provided)
        * ``pyfun.mapbc``: Sets boundary condition for each surface in mesh
        
    * OVERFLOW
        * ``overflow.inp``: Template input namelist
        
Data Book Setup
---------------
The *DataBook* |>| *Components* parameter sets the list of data components that
are considered final post-processing data products from the run. With no other
specifications, a "Component" is assumed to be a force and moment (``"Type":
"FM"``) taken from an iterative history. The exception is Cart3D, where the
default type is ``"Force"`` since users must separately request forces and
moments in that particular solver.

However, there are other data book types. The following example requests
iterative force and moments on the components called ``"left_wing"`` and
``"right_wing"``, a line load on ``"fuselage"``, and a protuberance patch load
taken from the final surface solution on the ``"cockpit"``. There is also an FM
component called ``"wings"`` in which pyCart adds the two wings' forces
together for each iteration.

    .. code-block:: javascript
    
        "DataBook": {
            "Components": ["left_wing", "right_wing", "fuselage",
                "cockpit", "wings"],
            "fuselage": {
                "Type": "LineLoad",
                "CompID": "fuselage"
            },
            "cockpit": {
                "Type": "TriqFM",
                "MapTri": "inputs/cockpit.patch.uh3d",
                "Patches": ["front", "left", "top", "right", "back"]
            }
            "wings": {
                "Type": "FM",
                "CompID": ["left_wing", "right_wing"]
            }
        }
        
Because this is a pure post-processing step, these parameters can be filled in
after starting or even finishing some of the cases. Other *DataBook*
parameters, such as *DataBook* |>| *nStats* and *DataBook* |>| *nFirst* are
also important; see the :ref:`appropriate subsection of the JSON settings
description <cape-json-DataBook>` for more information.

Collecting the data into a database, which is kept in a separate folder outside
the run folders (so that the run folders can be deleted when appropriate
without affecting the databases), is performed via several commands:

    ================   =========================
    *Type*             *Command*
    ================   =========================
    ``"FM"``           ``pycart --fm``
    ``"LineLoad"``     ``pycart --ll``
    ``"TriqFM"``       ``pycart --triqfm``
    ``"TriqPoint"``    ``pycart --pt``
    ================   =========================

    
Case Disposition and Archiving
-------------------------------
Once a case has been marked as PASS using a ``p`` in the first column of the
run matrix file, it can be archived. (Note: cases marked with the ``p`` but
that have not run the appropriate number of iterations or are still running
have the status ``PASS*`` and will not be archived.) Archiving a case is
performed using a command that conforms to the following template.

    .. code-block:: bash
    
        $ pycart -I 140 --archive
        
This will save some of the important files to a backup location and also delete
files if requested. It can be useful for keeping below file size and file count
quotas while running large databases.

The ``--clean`` command performs a subset of the ``--archive`` actions and can
be run at any time. Any files identified in the *RunControl* |>| *Archive* |>|
*ProgressDeleteFiles* as noncritical files will be deleted at any time this
``--clean`` command is run.

The command

    .. code-block:: bash
    
        $ pycart -I 140 --skeleton
        
performs even more cleanup tasks. Users may distinguish between ``--archive``
and ``--skeleton`` for various tasks of post-processing.  Typically it is
useful to leave the solution folder in such a state that all necessary
post-processing can still be performed after ``--archive`` has been run, but
the remaining files after ``--skeleton`` are only sufficient for ``pycart -c``
to report the correct number of iterations run.



********************
Changelog
********************


Release 2.1.1
=============================

New Features
-----------------

*   The interface for ``py{x} -c`` now has more options. For example you can
    show the values of additional columns from the run matrix explicitly.

    This example will show the value of run matrix keys called *user* and
    *arch* for each row of the status table, provided they exist.

    .. code-block:: console

        $ pyfun -I 3:7 --add-cols "user,arch" -c

    This feature is not fully documented yet and will be discussed more in the
    release of CAPE 2.2.0.

Bugs Fixed
-------------

*   When running Cart3D, CAPE will now avoid trying to restart ``aero.csh`` in
    an infinite loop for certain cases. This seemed to be caused by Cart3D
    terminating early when a target residual is reached (see the *nOrders*
    option for ``input.cntl``). ``pycart`` has now been set to lower the
    *PhaseIters* option if an adaptive phase looks to have run successfully but
    with fewer iterations than expected.

    
Release 2.1.0
=============================

New Features
-----------------

*   CAPE now includes an unstructured mesh module called ``cape.gruvoc``. It
    contains a mesh conversion capability, for example to convert a UGRID
    volume mesh to a Tecplot volume PLT file, you could run

    .. code-block:: console

        $ python3 -m cape.gruvoc pyfun.lb8.ugrid pyfun.plt --mapbc pyfun.mapbc
    
    It also creates a ``gruvoc`` executable so that you could just run

    .. code-block:: console

        $ gruvoc pyfun.lb8.ugrid pyfun.plt --mapbc pyfun.mapbc
    
    Another useful tool is ``gruvoc report``, which summarizes the number of
    nodes, surface tris, tets, etc. in a grid.

    .. code-block:: console

        $ gruvoc report pyfun.lb8.ugrid -h
    
    You can call ``gruvoc report`` without the ``-h`` to see the raw numbers.
    The ``-h`` flag means "human-readable", so for example it will abbreviate
    1234567 to "1.2M".

*   The ``pyfun`` module now contains the ability to directly read FUN3D
    solutions directly, both instantaneous restart files (``.flow``) and
    time-averaged solutions (``_TAVG.1``). An example usage is to convert a
    time-averaged solution to a volume Tecplot PLT file:

    .. code-block:: console

        $ gruvoc convert pyfun.lb8.ugrid pyfun_volume_tavg_timestep1000.plt \
            --tavg pyfun_TAVG.1 --add-mach --add-cp --mapbc pyfun.mapbc
    
    The ``cape.pyfun.casecntl`` module conveniently provides some built-in
    "workers" (See the notes on the CAPE 2.0.3 release) to make these
    conversions while FUN3D is running. A common use case is to add the
    following to the ``"RunControl"`` section for a ``pyfun`` JSON file:

    .. code-block:: javascript

        "WorkerPythonFuncs": [
            {
                "name": "flow2plt",
                "type": "runner"
            },
            {
                "name": "tavg2plt",
                "type": "runner"
            },
            {
                "name": "clean",
                "type": "runner"
            }
        ],
        "PostPythonFuncs": [
            {
                "name": "flow2plt",
                "type": "runner"
            },
            {
                "name": "tavg2plt",
                "type": "runner"
            }
        ]
    
    This code will automatically convert each new ``pyfun.flow`` file to
    ``pyfun_volume_timestep{n}.plt`` and each new ``pyfun_TAVG.1`` file to
    ``pyfun_volume_tavg_timestep{n}.plt`` without requiring FUN3D to do any of
    the output except for writing its usual restart files.

*   The ``cape.gruvoc`` module also contains tools for creating cut planes and
    performing other data sampling routines by using PyVista
    (https://pyvista.org/). The ``tavg2x`` runner for ``pyfun`` can be used for
    more data sampling routines; see the documentation of that function for
    more information.

*   Checking PBS status is now more efficient and effective. CAPE can now
    automatically detect which user's queue to check based on the owner of a
    job (useful if you care calling ``cape -c`` from another user's folder) and
    call to multiple PBS servers if appropriate. (This last example has become
    quite usefule on NASA HPC systems, for example, where the CPU and GPU jobs
    are controlled by distinct and separate PBS servers.)

*   Several upgrades have been made to the MPI interface in
    ``cape.cfdx.cmdgen``. These changes allow support for mixed GPU/CPU
    workflows (for example when running ``refine/three`` on a GPU job with
    FUN3D) and support additional environments that require more command-line
    arguments (for example the new Grace Hopper systems on the NASA Advanced
    Supercomputing facility).

*   Fun3D namelist control in the ``pyfun`` JSON file now supports setting
    multiple indices of the same option in an efficient manner. Here's an
    example setting the boundary conditions for a variety of surfaces:

    .. code-block:: javascript

        "Fun3D": {
            "boundary_conditions": {
                "wall_temp_flag": {
                    "1-38": true
                },
                "wall_temperature": {
                    "1-38": -1
                },
                "static_pressure_ratio": {
                    "8": 0.5,
                    "15": 0.75,
                    "18": 0.52
                }
        }

    It will lead to a namelist such as this:

    .. code-block:: none

        &boundary_conditions
            wall_temp_flag(1:38) = .true.
            wall_temperature(1:38) = -1
            static_pressure_ratio(8) = 0.5
            static_pressure_ratio(15) = 0.75
            static_pressure_ratio(18) = 0.52
        /
    
    This is generally much more practical than making a list of 38 ``true``
    values for *wall_temp_flag* and is especially convenient for the
    *static_pressure_ratio* in this example.

Behavior Changes
-----------------------

*   Many of the "DataBook" classes in ``cape.cfdx.databook`` have been renamed.
    These may cause issues for advanced users who have custom "hooks" or other
    Python modules.

*   The default data type in ``cape.dkit.textdata`` has been changed to
    ``int32``. This minor change makes it much easier to read CAPE run matrix
    files as DataKits without the need for extra keyword arguments.

*   Job status (e.g. when running ``pyover -c``) is now computed by
    ``CaseRunner`` instead of ``Cntl``. This is generally more efficient (users
    may notice the difference) and allows stand-alone cases to be aware of
    their status without being part of a run matrix.

Bug Fixes
-----------------

*   The CAPE 2.0.3 ``pycart`` modules contained several bugs that prevented
    adaptive runs (even the published CAPE examples) from running properly. All
    CAPE test cases run properly now.


Release 2.0.3
=============================

New Features
------------------

*   CAPE now includes *PreShellCmds* to go alongise *PostShellCmds*. This is an
    option in the ``"RunControl"`` section that allows the user to run one or
    more BASH (or whatever your shell of choice is) commands prior to running
    the primary CFD solver executables.

*   There is an exciting new feature called *WorkerShellCmds* in the
    ``"RunControl"`` section. It allows you to specify 0 or more BASH commands
    that you run every *WorkerSleepTime* seconds (default=``10.0``) while your
    case is running. It has working clean-up after the main executable is
    finished, allowing up to ``"WorkerTimeout"`` (default=``600.0``) seconds
    for the last instance of the worker to complete.

*   These run hooks also have Python function versions, in the form of options
    *PrePythonFuncs*, *PostPythonFuncs*, and *WorkerPythonFuncs*. If these are
    defined as a simple string, CAPE will import any modules implied by the
    function name and then call that function with no arguments. However, users
    may also specify more details for Python functions by defining the function
    in a ``dict``.

    .. code-block:: javascript

        "RunControl": {
            "WorkerPythonFuncs": [
                "mymod.mufunc",
                {
                    "name": "clean",
                    "type": "runner"
                },
                {
                    "name": "mymod.otherfunc",
                    "args": [
                        "$runner",
                        "$mach"
                    ]
                }
            ]
        }

*   Users of FUN3D and Kestrel can now link the mesh file into folders instead
    of copying it. Set ``"LinkMesh"`` to ``true`` in the ``"Mesh"`` section.

*   ``cape.pyfun`` in particular changes how it uses XML or JSON configuration
    files (which is specified in the ``"Config"`` > ``"File"`` setting). In
    previous versions of CAPE, the face labels or component ID numbers in that
    file had to match your actual grid, which had to match your ``.mapbc``
    file. Now CAPE only uses the text names in the *ConfigFile*, and it's ok to
    include components that aren't actually present in your grid. If your case
    worked as expected before, it will still work now, but for new cases it
    might be much easier to set up. The (new) recommended process is to use a
    ConfigJSON file and only specify a ``"Tree"`` section. See
    :ref:`configjson-syntax`.

Behavior Changes
------------------------

*   Binary files storing iterative histories are no longer saved automatically
*   Calculation of job status, especially for FUN3D, is much faster. This
    change should not cause any functional changes for users.
*   Python modules used to define hooks are no longer universally imported
    during ``Cntl`` instantiation. Modules are imported dynamically if needed
    to execute a hook. The ``"Modules"`` setting is still present in the JSON
    file but has no effect.

Bugs Fixed
------------------------

*   Fix bug in area-weighted node normal calculation,
    :func:`cape.trifile.TriBase.GetNodeNormals`.
*   The ``refine/three`` capability with FUN3D now works more reliably.

Release 2.0.2
=============================

New Features
-----------------

*   New capabilities in *nProc*, the number of MPI processes to use. Instead of
    requiring a positive integer, there are now four ways to interpret this
    setting:

    -   **positive integer**: `"nProc": 128` will continue to work in the
        obvious way that it always has
    -   **negative integer**: `"nProc": -2` on a node with 128 cores will mean
        using 126 cores
    -   **fraction**: `"nProc": 0.5` will mean using 50% (rounded down), so
        `mpiexec -np 64` on a 128-core node
    -   **blank**: `"nProc": null` (or leaving out entirely) means use all the
        MPI procs available

Behavior Changes
------------------

*   Don't write *Archive* settings to each case folder


Bugs Fixed
--------------

All of the tutorials at

https://github.com/nasa-ddalle/

now work properly with this version. Most of the updates were to the tutorials
themselves, but some CAPE bugs were fixed, too.


Release 2.0.1
=============================

New Features
---------------

*   GPU options in *RunControl* section of options
*   ``CaseRunner`` system calls now allow piping lines of a file to STDIN


Behavior Changes
------------------

*   Archiving uses ``tar -u`` if using the standard ``.tar`` archive format
*   Fix ``-e`` option to execute commands in case folders, and allow it to run
    regular system commands (not just local scripts)


Bugs Fixed
--------------

*   Add several missing options to *RunMatrix* definitions
*   Fix zone type when reading Tecplot file from Cart3D ``.tri[q]`` format
*   Improve handling of different-sized iterative histories in ``CaseFM``
*   Add PyYAML and colorama to install requirements



Release 2.0.0
=============================

New Features
---------------

*   Added a command ``cape --1to2`` to help update Python files written against
    the CAPE 1.2 API to the newer module names mentioned below.
*   The main input file can now be a YAML file in addition to the standard
    JSON. However, there is no "include" statement like the ``JSONFile()``
    directive supported in CAPE JSON files.
*   New command-line interface. The CLI supports the commands that would have
    worked for CAPE 1 but also support a new method that allows the user to be
    more explicit about the primary purpose of the command. For example

    .. code-block:: console

        $ pyfun --re "m1.2" --report

    is the same as

    .. code-block:: console

        $ pyfun report --re "m1.2"

    The new CLI also implements checks so that misspelled or unrecognized
    options will result in an error instead of just ignoring those options.

*   Created a new executable ``cape-tec`` that takes a Tecplot(R) layout file
    as input and exports a PNG from that layout.
*   Rewritten interface to *RunControl* > *Archive*. Users may now prescribe
    "only keep the most recent file of this set" of multiple patterns in a
    single line. For example ...

    .. code-block:: javascript

        "Archive": {
            "SearchMethod": "regex",
            "clean": {
                "PreDeleteFiles": {
                    "pyfun[0-9]+_([a-z][a-z0-9_-]+)_timestep[0-9]+\\.plt": 1
                }
            }
        }

    This will delete most Tecplot ``.plt`` files but keep the most recent ``1``
    matches. The new feature is that it will collect all the files that match
    this regular expression but divide them into separate lists for all the
    unique values of the regular expression group (the part inside
    parentheses). So if you have the following files:

        *   ``pyfun00_plane-y0_timestep1000.plt``
        *   ``pyfun00_tec_boundary_timestep1000.plt``
        *   ``pyfun01_plane-y0_timestep2000.plt``
        *   ``pyfun01_tec_boundary_timestep2000.plt``
        *   ``pyfun02_plane-y0_timestep3000.plt``
        *   ``pyfun02_plane-y0_timestep4000.plt``
        *   ``pyfun02_tec_boundary_timestep3000.plt``
        *   ``pyfun02_tec_boundary_timestep4000.plt``

    Then it would delete most of these files but only keep

        *   ``pyfun02_plane-y0_timestep4000.plt``
        *   ``pyfun02_tec_boundary_timestep4000.plt``

    This would not have been possible in CAPE 1; users would need to provide
    two separate instructions.

*   A *RunMatrix* key with the type ``"translation"`` can now use two named
    points as the ``"Vector"``. This means that the direction that a component
    is translated can be affected by prior *RunMatrix* keys


Behavior Changes
------------------

*   Many modules have been renamed, including renaming the ``case`` modules to
    the less-confusing name ``casecntl``. In addition, the main ``cntl`` module
    has been moved into the ``cape.cfdx`` folder.

Bugs Fixed
--------------

*   Determination of number of available MPI ranks on Slurm jobs


Release 1.2.1
=============================

New Features
-----------------

*   Each case now generates logs, which are helpful for debugging or just
    understanding the sequence of actions CAPE takes. The two log files within
    each case are ``cape/cape-main.log`` and ``cape/cape-verbose.log``).
*   PBS/Slurm job names are now longer (32 chars instead of 15), and the length
    is configurable (*RunMatrix* > *MaxJobNameLength*).

Behavior Changes
-------------------

*   PBS/Slurm job IDs are now saved as the full string instead of just the
    job number (often something like ``123456.pbspl1``)
*   The extensions are now build against NumPy version 2.0+ for Python 3.10
    and later. The Python 3.9 extension is still build against NumPy 1.x.

Bugs Fixed
------------

*   Better support of newer ``aero.csh`` script for Cart3D
*   Various compatibility issues with NumPy 2.0 release

Release 1.2.0
=============================

CAPE 1.2 is a smaller change than CAPE 1.1 and focuses on improving the quality
of CAPE's underlying code. Many modules have been de-linted, and some of the
older modules have been rewritten. Test coverage is also significantly
improved.

New Features
----------------

*   The iterative histories (both ``CaseFM`` and ``CaseResid``) now create a
    cache file so that CAPE can read them in much faster after the first read.
    It also creates a uniform file format for users who might be interested in
    saving iterative histories.
*   Add ``TSVTecDatFile`` class to read Tecplot-style column-data into
    ``DataKit``. See
    https://nasa.github.io/cape-doc/1.2/api/attdb/ftypes/tecdatfile.html
*   Add a ``--incremental`` option (or set *RunControl* |>| *StartNextPhase* to
    ``False``) option to run one phase at a time. See
    https://nasa.github.io/cape-doc/1.2/common/json/RunControl.html for the
    *StartNextPhase* option and/or
    https://nasa.github.io/cape-doc/1.2/bin/pyfun.html for ``--incremental``.

Behavior Changes
-------------------

*   The iterative history modules, ``CaseFM`` and ``CaseResid``, are now
    subclasses of ``DataKit``. Among other things, this means that what used to
    be ``fm.CN`` is now ``fm["CN"]``. This is a major improvement to making
    those classes extensible for histories of things other than forces &
    moments.
*   The ``cape.filecntl.filecntl`` module, which is critical to how CAPE
    reads and modifies CFD input files, was rewritten and tested to 100%
    coverage.
*   Rename some *RunControl* options to more understandable

    -   *Resubmit* |->| *ResubmitNextPhase*
    -   *Continue* |->| opposite of *ResubmitSamePhase*

    (See https://nasa.github.io/cape-doc/1.2/common/json/RunControl.html)


Bugs Fixed
--------------

*   The documentation now builds without warnings.


Release 1.1.1.post2
====================

Bugs Fixed
------------

*   Add (back) default ``"MuFormat"`` for coefficient table subfigures, which
    was causing tables full of the text "*None*" in some cases
*   Fix ``nmlfile`` when saving a long string in an existing array
*   Fix default formatting of ``user`` and ``tag`` run matrix keys in
    conditions table subfigures


Release 1.1.1.post1
====================

That's a weird-looking version number...

This post-release fixes some issues that the testing suite did not catch
regarding the previous CAPE 1.1 releases.

Bugs Fixed
------------

*   The ``TriRotate`` and ``TriTranslate`` run matrix keys now work properly
    again; they were not getting noticed as the correct key type in previous
    1.1 releases.
*   Using a ``list`` inside a ``@map`` ``dict`` now works with phase numbers in
    ``cape.optdict``
*   Fixes to flow initializations for FUN3D for new ``nmlfile`` Fortran
    namelist manipulation module
*   The ``cape.nmlfile`` namelist module now supports *N*-dimensional arrays,
    whereas the ``set_opt()`` method didn't support this before.


Release 1.1.1
====================

CAPE 1.1.1 introduces the optional ``"NJob"`` option, which can be placed in
the ``"RunControl"`` section. If you set this parameter to a positive integer,
CAPE will automatically keep that many jobs running. When one case finishes, it
will submit the appropriate number of new jobs until the total number of jobs
(not counting the one that is finishing) equals ``NJob``. Using this option,
users can start a run matrix and keep a roughly fixed number of cases running
for long periods of time without having to manually check and/or submit new
jobs.

Features added
----------------

*   ``"RunControl"`` > ``"NJob"`` option

Bugs Fixed
------------
(Same as Release 1.0.4)

*   Allow spaces in strings when reading tab-delimited files using ``DataKit``
    or ``TextDataFile``.
*   Fix some ``matplotlib`` imports to work with more ``matplotlib`` versions.
*   Switch order of ``CaseFunction()`` hook and ``WriteCaseJSON()`` in
    ``cape.pycart`` so that ``case.json`` reflects options changes from all
    hooks.


Release 1.1.0
====================

CAPE 1.1 incorporates an entirely new interface to how it reads the JSON files
that define most of the CAPE inputs. See :mod:`cape.optdict` for details about
the new options package and :mod:`cape.cfdx.options` for an gateway to the
CAPE-specific options for each section.

CAPE 1.1 removes support for Python 2.7. It supports Python 3.6+ (because
that's the version available on standard Red Hat Enterprise Linux versions 7
and 8), but testing is performed in Python 3.9.

This change is meant to be backwards-compatible with CAPE 1.0 with respect to
the JSON files, so the same JSON file that worked with CAPE 1.0 *should* work
with CAPE 1.1. However, the API is not fully backward-compatible, so some user
scripts and any hooks may need to be modified for CAPE 1.1. Also, although CAPE
1.0 JSON files should be compatible with CAPE 1.1, there may be many warnings
when using CAPE 1.1.

CAPE 1.1 adds support for a fourth CFD solver, namely
Kestrel from the Department of Defense's
`CREATE-AV <https://centers.hpc.mil/CREATE/CREATE-AV.html>`_ program.

There are three key features for CAPE 1.1 that all come from the incorporation
of :mod:`cape.optdict`:

*   Option names, types, and values are checked and validated throughout the
    JSON file. This contrasts with the CAPE 1.0 behavior where unrecognized
    options (e.g. a spelling error) were silently ignored, and invalid values
    (e.g. a :class:`str` instead of an :class:`int`) may or may not result in
    an Exception later.
*   JSON syntax errors generate much more helpful messages, especially if the
    error is in a nested file using the ``JSONFile()`` directive.
*   All or nearly all settings in the JSON file (except in the ``"RunMatrix"``
    section) can vary with run matrix conditions using one of three methods.

Related to the third bullet, you can use ``@cons`` (constraints), ``@map``,
and ``@expr``. For example to set a CFL number equal to 2 times the Mach
number, assuming the ``"RunMatrix"`` > ``"Keys"`` includes a key called
``"mach"``, set

.. code-block:: javascript

    "CFL": {
        "@epxr": "2*$mach"
    }

The next example demonstrates how to use a separate grid for supersonic and
subsonic conditions.

.. code-block:: javascript

    "Mesh": {
        "File": {
            "@cons": {
                "$mach < 1": "subsonic.ugrid",
                "$mach >= 1": "supersonic.ugrid"
            }
        }
    }

The third method is ``@map``, which might be used to use specific values based
on the value of some run matrix key. This example creates a map of how many PBS
nodes to use based on a run matrix key called ``"arch"``.

.. code-block:: javascript

    "PBS": {
        "select": {
            "@map": {
                "model1": 10,
                "model2": 20
            },
            "key": "arch"
        }
    }

You can also nest these features, with the most common example having an
``@expr`` inside a ``cons`` set.

Features added
----------------

*   Better error messages for JSON syntax errors
*   Explicit checks for option names and option values in most of JSON file
*   Ability to easily vary almost any JSON parameter as a function of run
    matrix conditions
*   Add support for Kestrel as fourth CFD solver (:mod:`cape.pykes`)

Bugs fixed
-----------

*   Raise an exception if component list not found during ``py{x} --ll``
    (previously wrote invalid triload input files and encountered an error
    later)

Behavior changes
-----------------

*   Drop support for Python 2.7.
*   FUN3D namelists no longer preserve text of template file; instead
    :class:`cape.nmlfile.NmlFile` reads a namelist into a :class:`dict`.
*   Options modules and classes renamed to more reasonable convention, e.g.
    :class:`cape.cfdx.options.runctlopts.RunControlOpts`.
*   More readable :func:`cape.pyfun.case.run_fun3d` and other main loop runner
    functions.


Release 1.0.4
====================
The test suite now runs with three Python versions: Python 2.7, 3.6, and 3.11.
We also found a way to create wheels with the ``_cape2`` or ``_cape3``
extension module in more Python versions.

Bugs Fixed
------------

*   Allow spaces in strings when reading tab-delimited files using ``DataKit``
    or ``TextDataFile``.
*   Fix some ``matplotlib`` imports to work with more ``matplotlib`` versions.
*   Switch order of ``CaseFunction()`` hook and ``WriteCaseJSON()`` in
    ``cape.pycart`` so that ``case.json`` reflects options changes from all
    hooks.


Release 1.0.3
====================


Features added
---------------

*   Add ``"Config"`` > ``"KeepTemplateComponents"`` for pyfun, which tells
    pyfun to add components to the ``'component_parameters'`` section rather
    than replacing it.
*   Support FUN3D 14.0 (a change to the STDOUT used to measure progress
    in ``pyfun``)

Bugs fixed
-----------

*   Properly tests if ``grid.i.tri`` is already present using ``usurp`` for
    ``pyover --ll``
*   Raise an exception if component list not found during ``py{x} --ll``
    (previously wrote invalid triload input files and ecnountered an error
    later)

Release 1.0.2.post1
====================

Bugs fixed
------------

*   Restore previous support for dictionaries like

    .. code-block:: python

        {
            "sampling_parameters": {
                "plane_center(1:3, 2)": [0.0, 1.0, 0.0],
                "label(2)": "plane-y1",
            }
        }

    as inputs to :mod:`cape.filecntl.namelist.Namelist.ApplyDict`. This is
    related to GitHub issues #4 and #19.

Release 1.0.2
====================

Features added
--------------

*   Add ``"PostShellCmds"`` to ``"RunControl"`` for :mod:`cape.pyover`;
    allows users to add a list of commands that run after every call to
    OVERFLOW
*   Support more recent versions of ``aero.csh`` in :mod:`cape.pycart`
*   Add command-line options to ``py{x} --report``:

    --report RP
        Update report named *RP* (default: first report in JSON file)

    --report RP --force
        Update report and ignore cache for all subfigures

    --report RP --no-compile
        Create images for a report but don't compile into PDF

    --report RP --rm
        Delete existing caches of report subfigure images instead of
        creating them

*   Add support for commas within strings in DataBooks and run matrices
*   Add ``"A"`` option in ``"PBS"`` section
*   Allow ``nodet_mpi`` to set ``"nProc"`` automatically with Slurm
*   Add options ``"YLim"``, ``"YMin"``, ``"YMax"``, ``"YLimMin"`` and likewise
    for ``"PlotCoeff"`` subfigures.

    - ``"YLim"``: list of explicit min and explicit max to use for *y*-axis
    - ``"YMin"``: explicit min to use for *y*-axis
    - ``"YMax"``: explicit max to use for *y*-axis
    - ``"YLimMax"``: outer bounds for *ymin* and *ymax*; CAPE will not plot a
      *y*-value below ``YLimMax[0]`` but may have a min *y*-axis value greater
      than that, and CAPE will not plot a *y*-value above ``YLimMax[1]``. Also
      supports using None (in Python) or null (in JSON) to use one of the
      bounds. E.g. ``"YLimMax": [0.0, null]`` will guarantee only positive
      *y*-values are shown but not set an upper bound.
    - The same options, replacing ``Y`` with ``X``


Release 1.0.1
====================

Features added
---------------

*   Warm-start capability for :mod:`cape.pyfun`, adds options *WarmStart* and
    *WarmStartDir* to ``"RunControl"``  section

Behavior changes
--------------------

*   Use :func:`os.mkdir` instead of :func:`cape.cfdx.options.Options.mkdir`
    during archiving (affects resulting file permissions of new folders)
*   Write binary (``lr4``) instead of ASCII ``.triq`` files when using *it_avg*
    in :mod:`cape.pycart`; speeds up ``pycart --ll`` significantly
*   Allow users to write PNG or JPG files during ``--report`` commands w/o also
    creating PDFs; also ability to include PNG or JPG into compiled report

Bug fixes
----------

*   Better control of force & moment requests in :mod:`cape.pycart`
*   Fix bug in reading some OVERFLOW iterative residual histories
*   Support columns with all ``np.nan`` in
    ``cape.attdb.rdb.DataKit.write_csv()``
*   Allow adding two ``cape.pycart.dataBook.CaseFM`` instances with
    different iteration counts

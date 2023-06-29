
********************
Changelog
********************

Release 1.1.0
====================

CAPE 1.1 incorporates an entirely new interface to how it reads the JSON files
that define most of the CAPE inputs. See :mod:`cape.optdict` for details about
the new options package and :mod:`cape.cfdx.options` for an gateway to the
CAPE-sepcific options for each section.

This change is meant to be backwards-compatible with CAPE 1.0 with respect to
the JSON files, so the same JSON file that worked with CAPE 1.0 *should* work
with CAPE 1.1. However, the API is not fully backward-compatible, so some user
scripts and any hooks may need to be modified for CAPE 1.1. Also, although CAPE
1.0 JSON files should be compatible with CAPE 1.1, there may be many warnings
when using CAPE 1.1.

There are three key features for CAPE 1.1 that all come from the incorporation
of :mod:`cape.optdict`:

*   Option names, types, and values are checked and validated throughout the
    JSON file. This contrasts with the CAPE 1.0 behavior where uncrecognized
    options (e.g. a spelling error) were silently ignored, and invalid values
    (e.g. a :class:`str` instead of an :class:`int`) may or may not result in
    an Exception later.
*   JSON syntax errors generate much more helpful messages, especially if the
    error is in a nested file using the ``JSONFile()`` directive.
*   All or nearly all settings in the JSON file (except in the ``"RunMatrix"``
    section) can vary with run matrix conditions using one of three methods.

Related to the third bullet, you can use ``@cons`` (consttraings), ``@map``,
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
            }
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
*   Add support for Kestrel as fourth CFD solver (:mod:`cape,pykes`)

Behavior changes
-----------------

*   FUN3D namelists no longer preserve text of template file; instead
    :class:`cape.nmlfile.NmlFile` reads a namelist into a :class:`dict`.
*   Options modules and classes renamed to more reasonable convention, e.g.
    :class:`cape.cfdx.options.runctlopts.RunControlOpts`.
*   More readable :func:`cape.pyfun.case.run_fun3d` and other main loop runner
    functions.

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
    :func:`cape.attdb.rdb.DataKit.write_csv`
*   Allow adding two :mod:`cape.pycart.dataBook.CaseFM` instances with
    different iteration counts

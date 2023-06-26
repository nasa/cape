
********************
Changelog
********************

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


********************
Changelog
********************

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
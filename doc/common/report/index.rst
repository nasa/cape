
.. _report:

Automated Post-Processing Reports
==================================

pyCart has an extensive system for creating PDF reports using ``pdflatex``.
These are defined in the ``"Report"`` section of the master JSON file and often
because the longest section of that JSON file.

These reports are created using the following template of commands

    .. code-block:: console
    
        $ pycart --report $REP --cons $CONS
        
This will create a report named *REP*, and the PDF will be
``report/report-$REP.pdf``.

For normal reports, there will be one or more pages in this PDF dedicated to
each case that means the constraints *CONS*.  Each case will have its own
folder in the ``report/`` folder with a name that is the same as the folder
name that contains the CFD results.  For example, the images for a case called
``poweroff/m2.5a0.0`` will be in ``report/poweroff/m2.5a0.0`` or
``report/poweroff/m2.5a0.0.tar``.

If the report *REP* contains so-called sweeps, there will be a set of sweeps
defined in the JSON report.  For example, it is very common to have a sweep of
forces and moments as a function of Mach number.  The results in this case
(assuming the user called this sweep ``"mach"``) would be in
``report/sweep-mach/poweroff/m0.5a0.0`` if ``poweroff/m0.5a0.0`` is the first
case in the sweep.


.. toctree::
    :maxdepth: 2

    main
    figure
    tecplot


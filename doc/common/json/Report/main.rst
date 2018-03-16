        
.. _cape-json-ReportReport:

Report Definitions
==================

Each report is defined with a :class:`dict` containing several options.  The
name of the key is the name of the report, so for example ``pycart --report
case`` will look for a definition under ``"case"`` in the ``"Report"`` section. 
If ``pycart --report`` is called without a report name, cape will update the
first report definition it finds.  Reports named any of ``"Reports"``,
``"Archive"``, ``"Sweeps"``, ``"Figures"``, or ``"Subfigures"`` (case-sensitive)
are not allowed.

The options used to describe a single report are listed below.

    *case*: :class:`dict`
        Definition of report named ``"case"``
        
        *Title*: :class:`str`
            Title placed on title page and PDF title
            
        *Subtitle*: :class:`str`
            Subtitle placed on title page
            
        *Author*: :class:`str`
            LaTeX string of author(s) printed on title page
            
        *Affiliation*: :class:`str`
            Name of institution or otherwise to be placed below author
            
        *Logo*: :class:`str`
            File name (relative to ``report/`` folder) of logo to place on
            bottom left of each report page
            
        *Frontispiece*: :class:`str`
            File name (relative to ``report/`` folder) of image to be placed on
            title page
            
        *Restriction*: {``""``} | ``"SBU - ITAR"`` | :class:`str`
            Data release restriction to place in center footnote if applicable
            
        *ShowCaseNumber*: ``True`` | {``False``}
            Whether or not to print the case number on each page
            
        *Sweeps*: {``[]``} | :class:`list` (:class:`str`)
            List of names of sweeps (plots of run matrix subsets) to include
            
        *Figures*: {``[]``} | :class:`list` (:class:`str`)
            List of figures for analysis of each case that has been run at
            least one iteration
        
        *ErrorFigures*: {``[]``} | :class:`list` (:class:`str`)
            List of figures for each case with ``"ERROR"`` status, defaults to
            value of *Figures*
            
        *ZeroFigures*: {``[]``} | :class:`list` (:class:`str`)
            List of figures for each cases that have a folder but have not run
            for any iterations yet and are not marked ``"ERROR"``, defaults to
            value of *Figures*

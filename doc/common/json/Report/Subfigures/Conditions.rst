

.. _cape-json-ReportConditions:

Run Conditions Table Subfigure
------------------------------
The ``"Conditions"`` subfigure creates a table of conditions for the
independent variables. The primary purpose is to list the run conditions for
each case for the observer to quickly reference which case is being analyzed.
It creates a table with three three columns: name of trajectory key,
abbreviation for the key, and the value of that key for the case being
reported.

If the subfigure is used as part of a sweep report, then the "Value" column
will show either the value of the first case in the sweep (if all cases in the
sweep have the same value) or an entry with the format ``v0, [vmin, vmax]``
where *v0* is the value at the first point in the sweep, *vmin* is the minimum
value for that independent variable for each point in the sweep, and *vmax* is
the maximum value.

The options are listed below.
    
    *C*: :class:`dict`
        Dictionary of settings for *Conditions* type subfigure
        
        *Type*: {``"Conditions"``} | :class:`str`
            Subfigure type
        
        *Header*: {``"Conditions"``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: {``"t"``} | ``"c"`` | ``"b"``
            Vertical position in row of subfigures
            
        *Alignment*: {``"left"``} | ``"center"``
            Horizontal alignment
            
        *Width*: {``0.4``} | :class:`float`
            Width of subfigure as a fraction of page text width
        
        *SkipVars*: {``[]``} | :class:`list` (:class:`str`)
            List of trajectory keys to not include in conditions table


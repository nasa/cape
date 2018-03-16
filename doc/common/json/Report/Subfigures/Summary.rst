            
            
.. _cape-json-ReportSummary:

Tabular Force & Moment Results
------------------------------
The ``"FMTable"`` subfigure class presents a table of textual force and/or
moment coefficients for an individual case.  The user can specify a list of
components and a list of coefficients.  For each coefficient, the user may
choose to display the mean value, the iterative standard deviation, and/or the
iterative uncertainty estimate.

Aliases for this subfigure are ``"ForceTable`` and ``"Summary"``.

Each component (for example left wing, right wing, fuselage) has its own
column, and the coefficients form rows. This subfigure class is only available
for case reports; it cannot be used on a sweep.

    *S*: :class:`dict`
        Dictionary of settings for *Summary* subfigures
        
        *Type*: {``"Summary"``} | :class:`str`
            Subfigure type
            
        *Header*: {``"Force \\& moment summary"``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: {``"t"``} | ``"c"`` | ``"b"``
            Vertical alignment of subfigure
            
        *Alignment*: {``"left"``} | ``"center"``
            Horizontal alignment
            
        *Width*: {``0.6``} | :class:`float`
            Width of subfigure as a fraction of page text width
            
        *Iteration*: {``0``} | :class:`int`
            If nonzero, display results from specified iteration number
            
        *Components*: {``["entire"]``} | :class:`list` (:class:`str`)
            List of components
            
        *Coefficients*: {``["CA", "CY", "CN"]``} | :class:`list` (:class:`str`)
            List of coefficients to display
            
        *CA*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CA*; mean, standard deviation, and error
            
        *CY*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CY*; mean, standard deviation, and error
            
        *CN*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CN*; mean, standard deviation, and error
            
        *CLL*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CLL*; mean, standard deviation, and error
            
        *CLM*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CLM*; mean, standard deviation, and error
            
        *CLN*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CLN*; mean, standard deviation, and
            error


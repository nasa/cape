

.. _cape-json-ReportSweepConditions:

Sweep Conditions Table Subfigure
--------------------------------
The ``"SweepConditions"`` subfigure class, which is only available for sweeps
(i.e. cannot be included in reports for individual cases), shows the list of
constraints that define a sweep. It creates a three-column table with the first
column the name of the variable, the second column the value of the variable
(i.e. trajectory key or derived key such as ``k%10``) for the first case in the
sweep, and the third column a description of the constraint. The constraint
description is either ``=``, meaning that all cases in the sweep have the same
value for that variable, or ``±tol`` if all the cases in the sweep are
constrained to be within a tolerance *tol* of the first point in the sweep.
            
    *C*: :class:`dict`
        Dictionary of settings for *SweepConditions* type subfigure
        
        *Type*: {``"SweepConditions"``} | :class:`str`
            Subfigure type
            
        *Header*: {``"Sweep Constraints"``} | :class:`str`
            Heading placed above subfigure (bold, italic)
        
        *Position*: {``"t"``} | ``"c"`` | ``"b"``
            Vertical alignment of subfigure
            
        *Alignment*: {``"left"``} | ``"center"``
            Horizontal alignment
            
        *Width*: {``0.4``} | :class:`float`
            Width of subfigure as a fraction of page text width


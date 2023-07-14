-------------------
"intersect" section
-------------------

*overlap*: {``None``} | :class:`int`
    perform boolean intersection of this comp number
*triged*: {``True``} | ``False``
    option to use CGT ``triged`` to clean output file
*o*: {``'Components.i.tri'``} | :class:`str`
    output file for ``intersect``
*T*: ``True`` | {``False``}
    option to also write Tecplot file ``Components.i.plt``
*run*: {``None``} | ``True`` | ``False``
    whether to execute program
*rm*: ``True`` | {``False``}
    option to remove small triangles from results
*fast*: ``True`` | {``False``}
    also write unformatted FAST file ``Components.i.fast``
*v*: ``True`` | {``False``}
    verbose mode
*smalltri*: {``0.0001``} | :class:`float`
    cutoff size for small triangles with *rm*
*i*: {``'Components.tri'``} | :class:`str`
    input file to ``intersect``
*cutout*: {``None``} | :class:`int`
    number of component to subtract
*intersections*: ``True`` | {``False``}
    option to write intersections to ``intersect.dat``
*ascii*: {``None``} | ``True`` | ``False``
    flag that input file is ASCII


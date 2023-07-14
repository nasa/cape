-------------------
"intersect" section
-------------------

*cutout*: {``None``} | :class:`int`
    number of component to subtract
*smalltri*: {``0.0001``} | :class:`float`
    cutoff size for small triangles with *rm*
*rm*: ``True`` | {``False``}
    option to remove small triangles from results
*overlap*: {``None``} | :class:`int`
    perform boolean intersection of this comp number
*v*: ``True`` | {``False``}
    verbose mode
*triged*: {``True``} | ``False``
    option to use CGT ``triged`` to clean output file
*fast*: ``True`` | {``False``}
    also write unformatted FAST file ``Components.i.fast``
*intersections*: ``True`` | {``False``}
    option to write intersections to ``intersect.dat``
*ascii*: {``None``} | ``True`` | ``False``
    flag that input file is ASCII
*i*: {``'Components.tri'``} | :class:`str`
    input file to ``intersect``
*o*: {``'Components.i.tri'``} | :class:`str`
    output file for ``intersect``
*T*: ``True`` | {``False``}
    option to also write Tecplot file ``Components.i.plt``
*run*: {``None``} | ``True`` | ``False``
    whether to execute program


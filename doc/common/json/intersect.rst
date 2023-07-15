-------------
IntersectOpts
-------------

*ascii*: {``None``} | ``True`` | ``False``
    flag that input file is ASCII
*v*: ``True`` | {``False``}
    verbose mode
*i*: {``'Components.tri'``} | :class:`str`
    input file to ``intersect``
*o*: {``'Components.i.tri'``} | :class:`str`
    output file for ``intersect``
*intersections*: ``True`` | {``False``}
    option to write intersections to ``intersect.dat``
*T*: ``True`` | {``False``}
    option to also write Tecplot file ``Components.i.plt``
*rm*: ``True`` | {``False``}
    option to remove small triangles from results
*triged*: {``True``} | ``False``
    option to use CGT ``triged`` to clean output file
*cutout*: {``None``} | :class:`int`
    number of component to subtract
*run*: {``None``} | ``True`` | ``False``
    whether to execute program
*fast*: ``True`` | {``False``}
    also write unformatted FAST file ``Components.i.fast``
*overlap*: {``None``} | :class:`int`
    perform boolean intersection of this comp number
*smalltri*: {``0.0001``} | :class:`float`
    cutoff size for small triangles with *rm*


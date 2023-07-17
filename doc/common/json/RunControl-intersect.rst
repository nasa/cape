---------------------------------
Options for ``intersect`` section
---------------------------------

**Recognized options:**

*T*: {``False``} | ``True``
    option to also write Tecplot file ``Components.i.plt``
*ascii*: {``None``} | ``True`` | ``False``
    flag that input file is ASCII
*cutout*: {``None``} | :class:`int`
    number of component to subtract
*fast*: {``False``} | ``True``
    also write unformatted FAST file ``Components.i.fast``
*i*: {``'Components.tri'``} | :class:`str`
    input file to ``intersect``
*intersections*: {``False``} | ``True``
    option to write intersections to ``intersect.dat``
*o*: {``'Components.i.tri'``} | :class:`str`
    output file for ``intersect``
*overlap*: {``None``} | :class:`int`
    perform boolean intersection of this comp number
*rm*: {``False``} | ``True``
    option to remove small triangles from results
*run*: {``None``} | ``True`` | ``False``
    whether to execute program
*smalltri*: {``0.0001``} | :class:`float`
    cutoff size for small triangles with *rm*
*triged*: {``True``} | ``False``
    option to use CGT ``triged`` to clean output file
*v*: {``False``} | ``True``
    verbose mode


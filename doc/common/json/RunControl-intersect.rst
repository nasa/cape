-------------
``intersect``
-------------

**Option aliases:**

* *clean* ? *pv-clean*
* *pvclean* ? *pv-clean*
* *pyvista-clean* ? *pv-clean*
* *cleantol* ? *clean-tol*
* *pv-tol* ? *clean-tol*
* *tol* ? *clean-tol*

**Recognized options:**

*T*: {``False``} | ``True``
    option to also write Tecplot file ``Components.i.plt``
*ascii*: {``None``} | ``True`` | ``False``
    flag that input file is ASCII
*cleantol*: {``1e-06``} | :class:`float`
    tolerance for PyVista surface cleaning with *pv-clean*
*cutout*: {``None``} | :class:`int`
    number of component to subtract
*fast*: {``False``} | ``True``
    also write unformatted FAST file ``Components.i.fast``
*groups*: {``None``} | :class:`list`\ [:class:`str`]
    list of families to treat as groups for `intersect`
*i*: {``'Components.tri'``} | :class:`str`
    input file to ``intersect``
*intersections*: {``False``} | ``True``
    option to write intersections to ``intersect.dat``
*o*: {``'Components.i.tri'``} | :class:`str`
    output file for ``intersect``
*overlap*: {``None``} | :class:`int`
    perform boolean intersection of this comp number
*pvclean*: {``True``} | ``False``
    option to clean intersected surface using PyVista
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


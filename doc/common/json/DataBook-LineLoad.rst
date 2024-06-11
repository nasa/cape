----------------------------------
Options for ``LineLoad`` component
----------------------------------

**Option aliases:**

* *Cut* -> *CutPlaneNormal*
* *CutDir* -> *CutPlaneNormal*
* *CutPlane* -> *CutPlaneNormal*
* *SlicePlane* -> *CutPlaneNormal*
* *nCut* -> *NCut*
* *Coeffs* -> *Cols*
* *Coefficients* -> *Cols*
* *Component* -> *CompID*
* *NAvg* -> *nStats*
* *NFirst* -> *NMin*
* *NLast* -> *NLastStats*
* *NMax* -> *NLastStats*
* *NStatsMax* -> *NMaxStats*
* *coeffs* -> *Cols*
* *cols* -> *Cols*
* *dnStats* -> *DNStats*
* *nAvg* -> *NStats*
* *nFirst* -> *NMin*
* *nLast* -> *NLastStats*
* *nLastStats* -> *NLastStats*
* *nMax* -> *NLastStats*
* *nMaxStats* -> *NMaxStats*
* *nMin* -> *NMin*
* *nStats* -> *NStats*
* *nStatsMax* -> *NMaxStats*
* *tagets* -> *Targets*

**Recognized options:**

*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*CutPlaneNormal*: {``'x'``} | ``'y'`` | ``'z'``
    direction to step between each cut
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Gauge*: {``True``} | ``False``
    option to use gauge pressures in computations
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*Momentum*: {``False``} | ``True``
    whether to use momentum flux in line load computations
*NCut*: {``200``} | :class:`int`
    number of cuts to make using ``triload`` (-> +1 slice)
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*SectionType*: ``'clds'`` | {``'dlds'``} | ``'slds'``
    line load section type
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Trim*: {``1``} | :class:`int`
    *trim* flag to ``triload``
*TriqFormat*: ``''`` | ``'b4'`` | {``'lb4'``} | ``'lr4'`` | ``'r4'``
    format for any ``.triq`` files written
*Type*: {``'FM'``} | :class:`str`
    databook component type


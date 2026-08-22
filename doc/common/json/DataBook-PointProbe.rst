---------------------------------------------
Options for ``PointProbe`` databook component
---------------------------------------------

**Option aliases:**

* *Coeffs* ? *Cols*
* *Coefficients* ? *Cols*
* *Component* ? *CompID*
* *NAvg* ? *nStats*
* *NFirst* ? *NMin*
* *NLast* ? *NLastStats*
* *NMax* ? *NLastStats*
* *NStatsMax* ? *NMaxStats*
* *body* ? *Body*
* *coeffs* ? *Cols*
* *cols* ? *Cols*
* *dnStats* ? *DNStats*
* *nAvg* ? *NStats*
* *nFirst* ? *NMin*
* *nLast* ? *NLastStats*
* *nLastStats* ? *NLastStats*
* *nMax* ? *NLastStats*
* *nMaxStats* ? *NMaxStats*
* *nMin* ? *NMin*
* *nStats* ? *NStats*
* *nStatsMax* ? *NMaxStats*
* *tagets* ? *Targets*
* *CTUMin* ? *MinCTU*
* *CTUmin* ? *MinCTU*
* *MinTime* ? *MinT*
* *TMin* ? *MinT*
* *TimeMin* ? *MinT*
* *ctumin* ? *MinCTU*
* *tmin* ? *MinT*

**Recognized options:**

*Body*: {``None``} | :class:`int` | :class:`str`
    reference body name for motion of this component
*Cols*: {``'cp'``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Index*: {``None``} | :class:`int`
    index of point probe in list, if necessary
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*MinCTU*: {``None``} | :class:`float`
    discard history before *MinCTU* char. time units
*MinT*: {``None``} | :class:`float`
    discard history before *MinT* seconds
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Point*: {``None``} | :class:`str`
    name of point probe
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type


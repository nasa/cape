-----------------------------------
Options for ``TriqPoint`` component
-----------------------------------

**Option aliases:**

* *Coeffs* -> *Cols*
* *Coefficients* -> *Cols*
* *Component* -> *CompID*
* *NAvg* -> *nStats*
* *NFirst* -> *NMin*
* *NLast* -> *nLastStats*
* *NMax* -> *nLastStats*
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
* *tagets* -> *Targets*

**Recognized options:**

*Cols*: {``'x'``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type


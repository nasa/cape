.. _cape-json-databook:

------------
``DataBook``
------------

**Option aliases:**

* *CTUMin* ? *MinCTU*
* *CTUmin* ? *MinCTU*
* *Dir* ? *Folder*
* *MinTime* ? *MinT*
* *NAvg* ? *nStats*
* *NFirst* ? *NMin*
* *NLast* ? *NLastStats*
* *NMax* ? *NLastStats*
* *TMin* ? *MinT*
* *ctumin* ? *MinCTU*
* *delim* ? *Delimiter*
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
* *tmin* ? *MinT*

**Recognized options:**

*Components*: {``None``} | :class:`str`
    list of databook components
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files
*Folder*: {``'data'``} | :class:`str`
    folder for root of databook
*MinCTU*: {``None``} | :class:`float`
    value of option "MinCTU"
*MinT*: {``None``} | :class:`float`
    value of option "MinT"
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NMin*: {``0``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``0``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Type*: ``'CaseProp'`` | {``'FM'``} | ``'IterPoint'`` | ``'LineLoad'`` | ``'PyFunc'`` | ``'TriqFM'`` | ``'TriqPoint'``
    Default component type

**Subsections:**

.. toctree::
    :maxdepth: 1

    DataBook-Targets
    DataBook-_default_
    DataBook-FM
    DataBook-IterFM
    DataBook-IterPoint
    DataBook-LineLoad
    DataBook-PointProbe
    DataBook-PyFunc
    DataBook-SurfCp
    DataBook-TriqFM
    DataBook-TriqPoint

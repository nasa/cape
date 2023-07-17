.. _cape-json-databook:

--------------------------------
Options for ``DataBook`` section
--------------------------------

**Option aliases:**

* *Dir* -> *Folder*
* *NAvg* -> *nStats*
* *NFirst* -> *NMin*
* *NLast* -> *nLastStats*
* *NMax* -> *nLastStats*
* *delim* -> *Delimiter*
* *dnStats* -> *DNStats*
* *nAvg* -> *NStats*
* *nFirst* -> *NMin*
* *nLast* -> *NLastStats*
* *nLastStats* -> *NLastStats*
* *nMax* -> *NLastStats*
* *nMaxStats* -> *NMaxStats*
* *nMin* -> *NMin*
* *nStats* -> *NStats*

**Recognized options:**

*Components*: {``None``} | :class:`str`
    list of databook components
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files
*Folder*: {``'data'``} | :class:`str`
    folder for root of databook
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NMin*: {``0``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``0``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Type*: {``'FM'``} | ``'IterPoint'`` | ``'LineLoad'`` | ``'PyFunc'`` | ``'TriqFM'`` | ``'TriqPoint'``
    Default component type

**Subsections:**

.. toctree::
    :maxdepth: 1

    DataBook-Targets
    DataBook-FM
    DataBook-IterPoint
    DataBook-LineLoad
    DataBook-PyFunc
    DataBook-TriqFM
    DataBook-TriqPoint

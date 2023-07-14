------------------
"DataBook" section
------------------

*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files
*NStats*: {``0``} | :class:`int`
    iterations to use in averaging window [for a comp]
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Components*: {``None``} | :class:`str`
    list of databook components
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: ``'IterPoint'`` | ``'TriqFM'`` | ``'LineLoad'`` | {``'FM'``} | ``'PyFunc'`` | ``'TriqPoint'``
    Default component type
*Targets*: {``None``} | :class:`object`
    value of option "Targets"
*Folder*: {``'data'``} | :class:`str`
    folder for root of databook
*NMin*: {``0``} | :class:`int`
    first iter to consider for use in databook [for a comp]


.. toctree::
    :maxdepth: 1

    Targets

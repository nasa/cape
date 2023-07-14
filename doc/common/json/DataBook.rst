------------------
"DataBook" section
------------------

*NStats*: {``0``} | :class:`int`
    iterations to use in averaging window [for a comp]
*NMin*: {``0``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files
*Targets*: {``None``} | :class:`object`
    value of option "Targets"
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Components*: {``None``} | :class:`str`
    list of databook components
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Type*: ``'PyFunc'`` | ``'TriqFM'`` | ``'LineLoad'`` | {``'FM'``} | ``'TriqPoint'`` | ``'IterPoint'``
    Default component type
*Folder*: {``'data'``} | :class:`str`
    folder for root of databook

-------------------------
Options for *Type*\ ="FM"
-------------------------

*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Type*: {``'FM'``} | :class:`str`
    databook component type
*Cols*: {``'CA'``} | :class:`str`
    list of primary solver output variables to include


--------------------------------
Options for *Type*\ ="IterPoint"
--------------------------------

*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Type*: {``'FM'``} | :class:`str`
    databook component type
*Cols*: {``'cp'``} | :class:`str`
    list of primary solver output variables to include


-------------------------------
Options for *Type*\ ="LineLoad"
-------------------------------

*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*Trim*: {``1``} | :class:`int`
    *trim* flag to ``triload``
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Gauge*: {``True``} | ``False``
    option to use gauge pressures in computations
*Momentum*: ``True`` | {``False``}
    whether to use momentum flux in line load computations
*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include
*NCut*: {``200``} | :class:`int`
    number of cuts to make using ``triload`` (-> +1 slice)
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*SectionType*: ``'clds'`` | {``'dlds'``} | ``'slds'``
    line load section type
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Type*: {``'FM'``} | :class:`str`
    databook component type


-----------------------------
Options for *Type*\ ="PyFunc"
-----------------------------

*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Function*: {``None``} | :class:`str`
    Python function name
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Type*: {``'FM'``} | :class:`str`
    databook component type
*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include


-----------------------------
Options for *Type*\ ="TriqFM"
-----------------------------

*CompTol*: {``None``} | :class:`float`
    tangent tolerance relative to component
*ConfigFile*: {``None``} | :class:`str`
    configuration file for surface groups
*AbsTol*: {``None``} | :class:`float`
    absolute tangent tolerance for surface mapping
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*MapTri*: {``None``} | :class:`str`
    name of a tri file to use for remapping CFD surface comps
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*CompProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of component
*Cols*: {``'CA'``} | :class:`str`
    list of primary solver output variables to include
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*RelTol*: {``None``} | :class:`float`
    relative tangent tolerance for surface mapping
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*OutputFormat*: {``'plt'``} | ``'dat'``
    output format for component surface files
*RelProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of geometry
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*AbsProjTol*: {``None``} | :class:`float`
    absolute projection tolerance
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Type*: {``'FM'``} | :class:`str`
    databook component type
*Patches*: {``None``} | :class:`list`\ [:class:`str`]
    list of patches for a databook component


--------------------------------
Options for *Type*\ ="TriqPoint"
--------------------------------

*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Type*: {``'FM'``} | :class:`str`
    databook component type
*Cols*: {``'x'``} | :class:`str`
    list of primary solver output variables to include



.. toctree::
    :maxdepth: 1

    Targets

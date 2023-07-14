------------------
"DataBook" section
------------------

*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*Targets*: {``None``} | :class:`object`
    value of option "Targets"
*Folder*: {``'data'``} | :class:`str`
    folder for root of databook
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: ``'PyFunc'`` | ``'LineLoad'`` | ``'TriqPoint'`` | {``'FM'``} | ``'IterPoint'`` | ``'TriqFM'``
    Default component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Components*: {``None``} | :class:`str`
    list of databook components
*NMin*: {``0``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``0``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files

Options for *Type*\ ="FM"
=========================

*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Cols*: {``'CA'``} | :class:`str`
    list of primary solver output variables to include
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component


Options for *Type*\ ="IterPoint"
================================

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*Cols*: {``'cp'``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component


Options for *Type*\ ="LineLoad"
===============================

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Gauge*: {``True``} | ``False``
    option to use gauge pressures in computations
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Momentum*: ``True`` | {``False``}
    whether to use momentum flux in line load computations
*Trim*: {``1``} | :class:`int`
    *trim* flag to ``triload``
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NCut*: {``200``} | :class:`int`
    number of cuts to make using ``triload`` (-> +1 slice)
*SectionType*: ``'slds'`` | {``'dlds'``} | ``'clds'``
    line load section type
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component


Options for *Type*\ ="PyFunc"
=============================

*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*Function*: {``None``} | :class:`str`
    Python function name
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component


Options for *Type*\ ="TriqFM"
=============================

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*MapTri*: {``None``} | :class:`str`
    name of a tri file to use for remapping CFD surface comps
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Patches*: {``None``} | :class:`list`\ [:class:`str`]
    list of patches for a databook component
*AbsProjTol*: {``None``} | :class:`float`
    absolute projection tolerance
*CompProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of component
*ConfigFile*: {``None``} | :class:`str`
    configuration file for surface groups
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*RelProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of geometry
*Cols*: {``'CA'``} | :class:`str`
    list of primary solver output variables to include
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*AbsTol*: {``None``} | :class:`float`
    absolute tangent tolerance for surface mapping
*RelTol*: {``None``} | :class:`float`
    relative tangent tolerance for surface mapping
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*CompTol*: {``None``} | :class:`float`
    tangent tolerance relative to component
*OutputFormat*: ``'dat'`` | {``'plt'``}
    output format for component surface files
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component


Options for *Type*\ ="TriqPoint"
================================

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*Cols*: {``'x'``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component



.. toctree::
    :maxdepth: 1

    Targets

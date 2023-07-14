------------------
"DataBook" section
------------------

*NMin*: {``0``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*Type*: ``'LineLoad'`` | {``'FM'``} | ``'TriqPoint'`` | ``'IterPoint'`` | ``'TriqFM'`` | ``'PyFunc'``
    Default component type
*Folder*: {``'data'``} | :class:`str`
    folder for root of databook
*Targets*: {``None``} | :class:`object`
    value of option "Targets"
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NStats*: {``0``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Components*: {``None``} | :class:`str`
    list of databook components

Options for *Type*\ ="FM"
=========================

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: {``'FM'``} | :class:`str`
    databook component type
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Cols*: {``'CA'``} | :class:`str`
    list of primary solver output variables to include


Options for *Type*\ ="IterPoint"
================================

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*Type*: {``'FM'``} | :class:`str`
    databook component type
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Cols*: {``'cp'``} | :class:`str`
    list of primary solver output variables to include


Options for *Type*\ ="LineLoad"
===============================

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NCut*: {``200``} | :class:`int`
    number of cuts to make using ``triload`` (-> +1 slice)
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Trim*: {``1``} | :class:`int`
    *trim* flag to ``triload``
*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include
*SectionType*: ``'slds'`` | ``'clds'`` | {``'dlds'``}
    line load section type
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*Gauge*: {``True``} | ``False``
    option to use gauge pressures in computations
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Momentum*: ``True`` | {``False``}
    whether to use momentum flux in line load computations


Options for *Type*\ ="PyFunc"
=============================

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*Type*: {``'FM'``} | :class:`str`
    databook component type
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Function*: {``None``} | :class:`str`
    Python function name
*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include


Options for *Type*\ ="TriqFM"
=============================

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*ConfigFile*: {``None``} | :class:`str`
    configuration file for surface groups
*OutputFormat*: ``'dat'`` | {``'plt'``}
    output format for component surface files
*Type*: {``'FM'``} | :class:`str`
    databook component type
*CompTol*: {``None``} | :class:`float`
    tangent tolerance relative to component
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*AbsTol*: {``None``} | :class:`float`
    absolute tangent tolerance for surface mapping
*Patches*: {``None``} | :class:`list`\ [:class:`str`]
    list of patches for a databook component
*CompProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of component
*Cols*: {``'CA'``} | :class:`str`
    list of primary solver output variables to include
*RelTol*: {``None``} | :class:`float`
    relative tangent tolerance for surface mapping
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*AbsProjTol*: {``None``} | :class:`float`
    absolute projection tolerance
*MapTri*: {``None``} | :class:`str`
    name of a tri file to use for remapping CFD surface comps
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*RelProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of geometry


Options for *Type*\ ="TriqPoint"
================================

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*Type*: {``'FM'``} | :class:`str`
    databook component type
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Cols*: {``'x'``} | :class:`str`
    list of primary solver output variables to include



.. toctree::
    :maxdepth: 1

    Targets

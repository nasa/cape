------------
DataBookOpts
------------

*Components*: {``None``} | :class:`str`
    list of databook components
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: ``'IterPoint'`` | ``'PyFunc'`` | ``'LineLoad'`` | ``'TriqPoint'`` | ``'TriqFM'`` | {``'FM'``}
    Default component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files
*NStats*: {``0``} | :class:`int`
    iterations to use in averaging window [for a comp]
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*Folder*: {``'data'``} | :class:`str`
    folder for root of databook
*Targets*: {``None``} | :class:`object`
    value of option "Targets"
*NMin*: {``0``} | :class:`int`
    first iter to consider for use in databook [for a comp]

Options for *Type*\ ="FM"
=========================

*Cols*: {``'CA'``} | :class:`str`
    list of primary solver output variables to include
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values


Options for *Type*\ ="IterPoint"
================================

*Cols*: {``'cp'``} | :class:`str`
    list of primary solver output variables to include
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values


Options for *Type*\ ="LineLoad"
===============================

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Trim*: {``1``} | :class:`int`
    *trim* flag to ``triload``
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Gauge*: {``True``} | ``False``
    option to use gauge pressures in computations
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*Momentum*: ``True`` | {``False``}
    whether to use momentum flux in line load computations
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*NCut*: {``200``} | :class:`int`
    number of cuts to make using ``triload`` (-> +1 slice)
*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include
*SectionType*: ``'clds'`` | ``'slds'`` | {``'dlds'``}
    line load section type
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values


Options for *Type*\ ="PyFunc"
=============================

*Function*: {``None``} | :class:`str`
    Python function name
*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values


Options for *Type*\ ="TriqFM"
=============================

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Patches*: {``None``} | :class:`list`\ [:class:`str`]
    list of patches for a databook component
*MapTri*: {``None``} | :class:`str`
    name of a tri file to use for remapping CFD surface comps
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*OutputFormat*: {``'plt'``} | ``'dat'``
    output format for component surface files
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*AbsTol*: {``None``} | :class:`float`
    absolute tangent tolerance for surface mapping
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*CompProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of component
*AbsProjTol*: {``None``} | :class:`float`
    absolute projection tolerance
*Cols*: {``'CA'``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*CompTol*: {``None``} | :class:`float`
    tangent tolerance relative to component
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*ConfigFile*: {``None``} | :class:`str`
    configuration file for surface groups
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*RelTol*: {``None``} | :class:`float`
    relative tangent tolerance for surface mapping
*RelProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of geometry


Options for *Type*\ ="TriqPoint"
================================

*Cols*: {``'x'``} | :class:`str`
    list of primary solver output variables to include
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*Type*: {``'FM'``} | :class:`str`
    databook component type
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values



.. toctree::
    :maxdepth: 1

    Targets

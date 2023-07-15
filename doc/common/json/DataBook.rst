--------------------------------
Options for ``DataBook`` section
--------------------------------

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
*Targets*: {``None``} | :class:`object`
    value of option "Targets"
*Type*: {``'FM'``} | ``'IterPoint'`` | ``'LineLoad'`` | ``'PyFunc'`` | ``'TriqFM'`` | ``'TriqPoint'``
    Default component type

Options for *Type*\ ="FM"
=========================

*Cols*: {``'CA'``} | :class:`str`
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
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type


Options for *Type*\ ="IterPoint"
================================

*Cols*: {``'cp'``} | :class:`str`
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


Options for *Type*\ ="LineLoad"
===============================

*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Gauge*: {``True``} | ``False``
    option to use gauge pressures in computations
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*Momentum*: ``True`` | {``False``}
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
*Type*: {``'FM'``} | :class:`str`
    databook component type


Options for *Type*\ ="PyFunc"
=============================

*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*Function*: {``None``} | :class:`str`
    Python function name
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
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type


Options for *Type*\ ="TriqFM"
=============================

*AbsProjTol*: {``None``} | :class:`float`
    absolute projection tolerance
*AbsTol*: {``None``} | :class:`float`
    absolute tangent tolerance for surface mapping
*Cols*: {``'CA'``} | :class:`str`
    list of primary solver output variables to include
*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component
*CompProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of component
*CompTol*: {``None``} | :class:`float`
    tangent tolerance relative to component
*ConfigFile*: {``None``} | :class:`str`
    configuration file for surface groups
*DNStats*: {``None``} | :class:`int`
    increment for candidate window sizes
*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values
*IntCols*: {``'nIter'``} | :class:`str`
    additional databook cols with integer values
*MapTri*: {``None``} | :class:`str`
    name of a tri file to use for remapping CFD surface comps
*NLastStats*: {``None``} | :class:`int`
    specific iteration at which to extract stats
*NMaxStats*: {``None``} | :class:`int`
    max number of iters to include in averaging window
*NMin*: {``None``} | :class:`int`
    first iter to consider for use in databook [for a comp]
*NStats*: {``None``} | :class:`int`
    iterations to use in averaging window [for a comp]
*OutputFormat*: ``'dat'`` | {``'plt'``}
    output format for component surface files
*Patches*: {``None``} | :class:`list`\ [:class:`str`]
    list of patches for a databook component
*RelProjTol*: {``None``} | :class:`float`
    projection tolerance relative to size of geometry
*RelTol*: {``None``} | :class:`float`
    relative tangent tolerance for surface mapping
*Targets*: {``{}``} | :class:`dict`
    targets for this databook component
*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component
*Type*: {``'FM'``} | :class:`str`
    databook component type


Options for *Type*\ ="TriqPoint"
================================

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


**Subsections:**

.. toctree::
    :maxdepth: 1

    Targets

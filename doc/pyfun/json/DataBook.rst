
.. _pyfun-json-databook:

********************************
Options for ``DataBook`` Section
********************************
The options below are the available options in the ``DataBook`` Section of the ``pyfun.json`` control file


*Folder*: {``'data'``} | :class:`str`
    folder for root of databook



*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window



*Type*: {``'FM'``} | ``'PyFunc'`` | ``'TriqFM'`` | ``'TriqPoint'`` | ``'IterPoint'`` | ``'LineLoad'``
    Default component type



*Components*: {``None``} | :class:`str`
    list of databook components



*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files



*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats



*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes



*NMin*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]



*NStats*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]



*Targets*: {``None``} | :class:`object`
    value of option "Targets"


Options for all ``Targets``
===========================

*Folder*: {``'data'``} | :class:`str`
    Name of folder from which to read data



*Translations*: {``None``} | :class:`dict`
    value of option "Translations"



*Type*: {``'generic'``} | ``'databook'``
    DataBook Target type



*Label*: {``None``} | :class:`str`
    Label to use when plotting this target



*Delimiter*: {``','``} | :class:`str`
    Delimiter in databook target data file



*Components*: {``None``} | :class:`list`\ [:class:`str`]
    List of databook components with data from this target



*Tolerances*: {``None``} | :class:`dict`
    Dictionary of tolerances for run matrix keys



*CommentChar*: {``'#'``} | :class:`str`
    value of option "CommentChar"



*Name*: {``None``} | :class:`str`
    Internal *name* to use for target



*File*: {``None``} | :class:`str`
    Name of file from which to read data


FM Options
==========

*FloatCols*: {``['nOrders']``} | :class:`str`
    additional databook cols with floating-point values



*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component



*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window



*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component



*Type*: {``'FM'``} | :class:`str`
    databook component type



*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats



*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes



*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values



*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]



*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]



*Cols*: {``['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']``} | :class:`str`
    list of primary solver output variables to include



*Targets*: {``{}``} | :class:`dict`
    targets for this databook component


IterPoint Options
=================

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values



*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component



*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window



*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component



*Type*: {``'FM'``} | :class:`str`
    databook component type



*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats



*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes



*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values



*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]



*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]



*Targets*: {``{}``} | :class:`dict`
    targets for this databook component



*Cols*: {``['cp']``} | :class:`str`
    list of primary solver output variables to include



*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors


LineLoad Options
================

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values



*Gauge*: {``True``} | :class:`bool` | :class:`bool_`
    option to use gauge pressures in computations



*SectionType*: {``'dlds'``} | ``'clds'`` | ``'slds'``
    line load section type



*Type*: {``'FM'``} | :class:`str`
    databook component type



*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats



*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes



*Trim*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    *trim* flag to ``triload``



*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values



*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]



*NCut*: {``200``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cuts to make using ``triload`` (-> +1 slice)



*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include



*Targets*: {``{}``} | :class:`dict`
    targets for this databook component



*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component



*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window



*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component



*Momentum*: {``False``} | :class:`bool` | :class:`bool_`
    whether to use momentum flux in line load computations



*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]


PyFunc Options
==============

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values



*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component



*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window



*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component



*Type*: {``'FM'``} | :class:`str`
    databook component type



*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats



*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes



*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values



*Function*: {``None``} | :class:`str`
    Python function name



*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]



*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]



*Targets*: {``{}``} | :class:`dict`
    targets for this databook component



*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include


TriqFM Options
==============

*ConfigFile*: {``None``} | :class:`str`
    configuration file for surface groups



*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values



*Type*: {``'FM'``} | :class:`str`
    databook component type



*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats



*CompTol*: {``None``} | :class:`float` | :class:`float32`
    tangent tolerance relative to component



*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes



*IntCols*: {``['nIter']``} | :class:`str`
    additional databook cols with integer values



*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]



*AbsProjTol*: {``None``} | :class:`float` | :class:`float32`
    absolute projection tolerance



*MapTri*: {``None``} | :class:`str`
    name of a tri file to use for remapping CFD surface comps



*Cols*: {``['CA', 'CY', 'CN', 'CAv', 'CYv', 'CNv', 'Cp_min', 'Cp_max', 'Ax', 'Ay', 'Az']``} | :class:`str`
    list of primary solver output variables to include



*Targets*: {``{}``} | :class:`dict`
    targets for this databook component



*AbsTol*: {``None``} | :class:`float` | :class:`float32`
    absolute tangent tolerance for surface mapping



*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component



*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window



*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component



*CompProjTol*: {``None``} | :class:`float` | :class:`float32`
    projection tolerance relative to size of component



*OutputFormat*: {``'plt'``} | ``'dat'``
    output format for component surface files



*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]



*Patches*: {``None``} | :class:`list`\ [:class:`str`]
    list of patches for a databook component



*RelProjTol*: {``None``} | :class:`float` | :class:`float32`
    projection tolerance relative to size of geometry



*RelTol*: {``None``} | :class:`float` | :class:`float32`
    relative tangent tolerance for surface mapping


TriqPoint Options
=================

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values



*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component



*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window



*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component



*Type*: {``'FM'``} | :class:`str`
    databook component type



*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats



*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes



*IntCols*: {``['nIter']``} | :class:`str`
    additional databook cols with integer values



*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]



*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]



*Targets*: {``{}``} | :class:`dict`
    targets for this databook component



*Cols*: {``['x', 'y', 'z', 'cp']``} | :class:`str`
    list of primary solver output variables to include



*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors



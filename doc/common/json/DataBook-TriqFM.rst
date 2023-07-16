--------------------------------
Options for ``TriqFM`` component
--------------------------------

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


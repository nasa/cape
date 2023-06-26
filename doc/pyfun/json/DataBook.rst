
.. _pyfun-json-databook:

************************
DataBook Section Options
************************
The options below are the available options in the DataBook Section of the ``pyfun.json`` control file

..
    start-DataBook-nmin

*NMin*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-DataBook-nmin

..
    start-DataBook-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-DataBook-dnstats

..
    start-DataBook-type

*Type*: ``'LineLoad'`` | ``'IterPoint'`` | ``'TriqPoint'`` | {``'FM'``} | ``'TriqFM'`` | ``'PyFunc'``
    Default component type

..
    end-DataBook-type

..
    start-DataBook-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-DataBook-nmaxstats

..
    start-DataBook-components

*Components*: {``None``} | :class:`str`
    list of databook components

..
    end-DataBook-components

..
    start-DataBook-folder

*Folder*: {``'data'``} | :class:`str`
    folder for root of databook

..
    end-DataBook-folder

..
    start-DataBook-nstats

*NStats*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-DataBook-nstats

..
    start-DataBook-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-DataBook-nlaststats

..
    start-DataBook-targets

*Targets*: {``None``} | :class:`object`
    value of option "Targets"

..
    end-DataBook-targets

..
    start-DataBook-delimiter

*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files

..
    end-DataBook-delimiter

Targets Options
===============
DBTarget Options
----------------
..
    start-DBTarget-file

*File*: {``None``} | :class:`str`
    Name of file from which to read data

..
    end-DBTarget-file

..
    start-DBTarget-label

*Label*: {``None``} | :class:`str`
    Label to use when plotting this target

..
    end-DBTarget-label

..
    start-DBTarget-type

*Type*: {``'generic'``} | ``'databook'``
    DataBook Target type

..
    end-DBTarget-type

..
    start-DBTarget-components

*Components*: {``None``} | :class:`list`\ [:class:`str`]
    List of databook components with data from this target

..
    end-DBTarget-components

..
    start-DBTarget-folder

*Folder*: {``'data'``} | :class:`str`
    Name of folder from which to read data

..
    end-DBTarget-folder

..
    start-DBTarget-commentchar

*CommentChar*: {``'#'``} | :class:`str`
    value of option "CommentChar"

..
    end-DBTarget-commentchar

..
    start-DBTarget-tolerances

*Tolerances*: {``None``} | :class:`dict`
    Dictionary of tolerances for run matrix keys

..
    end-DBTarget-tolerances

..
    start-DBTarget-name

*Name*: {``None``} | :class:`str`
    Internal *name* to use for target

..
    end-DBTarget-name

..
    start-DBTarget-delimiter

*Delimiter*: {``','``} | :class:`str`
    Delimiter in databook target data file

..
    end-DBTarget-delimiter

..
    start-DBTarget-translations

*Translations*: {``None``} | :class:`dict`
    value of option "Translations"

..
    end-DBTarget-translations

FM Options
----------
..
    start-FM-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-FM-nmin

..
    start-FM-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-FM-intcols

..
    start-FM-cols

*Cols*: {``['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']``} | :class:`str`
    list of primary solver output variables to include

..
    end-FM-cols

..
    start-FM-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-FM-dnstats

..
    start-FM-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-FM-type

..
    start-FM-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-FM-compid

..
    start-FM-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-FM-nmaxstats

..
    start-FM-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-FM-transformations

..
    start-FM-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-FM-nstats

..
    start-FM-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-FM-nlaststats

..
    start-FM-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-FM-targets

..
    start-FM-floatcols

*FloatCols*: {``['nOrders']``} | :class:`str`
    additional databook cols with floating-point values

..
    end-FM-floatcols

IterPoint Options
-----------------
..
    start-IterPoint-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-IterPoint-nmin

..
    start-IterPoint-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-IterPoint-intcols

..
    start-IterPoint-cols

*Cols*: {``['cp']``} | :class:`str`
    list of primary solver output variables to include

..
    end-IterPoint-cols

..
    start-IterPoint-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-IterPoint-dnstats

..
    start-IterPoint-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-IterPoint-type

..
    start-IterPoint-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-IterPoint-compid

..
    start-IterPoint-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-IterPoint-nmaxstats

..
    start-IterPoint-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-IterPoint-transformations

..
    start-IterPoint-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-IterPoint-nstats

..
    start-IterPoint-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-IterPoint-nlaststats

..
    start-IterPoint-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-IterPoint-targets

..
    start-IterPoint-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-IterPoint-floatcols

..
    start-IterPoint-points

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors

..
    end-IterPoint-points

LineLoad Options
----------------
..
    start-LineLoad-ncut

*NCut*: {``200``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cuts to make using ``triload`` (-> +1 slice)

..
    end-LineLoad-ncut

..
    start-LineLoad-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-LineLoad-intcols

..
    start-LineLoad-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-LineLoad-type

..
    start-LineLoad-momentum

*Momentum*: {``False``} | :class:`bool` | :class:`bool_`
    whether to use momentum flux in line load computations

..
    end-LineLoad-momentum

..
    start-LineLoad-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-LineLoad-nstats

..
    start-LineLoad-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-LineLoad-nlaststats

..
    start-LineLoad-sectiontype

*SectionType*: {``'dlds'``} | ``'slds'`` | ``'clds'``
    line load section type

..
    end-LineLoad-sectiontype

..
    start-LineLoad-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-LineLoad-targets

..
    start-LineLoad-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-LineLoad-floatcols

..
    start-LineLoad-gauge

*Gauge*: {``True``} | :class:`bool` | :class:`bool_`
    option to use gauge pressures in computations

..
    end-LineLoad-gauge

..
    start-LineLoad-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-LineLoad-nmin

..
    start-LineLoad-cols

*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include

..
    end-LineLoad-cols

..
    start-LineLoad-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-LineLoad-dnstats

..
    start-LineLoad-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-LineLoad-compid

..
    start-LineLoad-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-LineLoad-nmaxstats

..
    start-LineLoad-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-LineLoad-transformations

..
    start-LineLoad-trim

*Trim*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    *trim* flag to ``triload``

..
    end-LineLoad-trim

PyFunc Options
--------------
..
    start-PyFunc-function

*Function*: {``None``} | :class:`str`
    Python function name

..
    end-PyFunc-function

..
    start-PyFunc-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-PyFunc-nmin

..
    start-PyFunc-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-PyFunc-intcols

..
    start-PyFunc-cols

*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include

..
    end-PyFunc-cols

..
    start-PyFunc-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-PyFunc-dnstats

..
    start-PyFunc-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-PyFunc-type

..
    start-PyFunc-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-PyFunc-compid

..
    start-PyFunc-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-PyFunc-nmaxstats

..
    start-PyFunc-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-PyFunc-transformations

..
    start-PyFunc-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-PyFunc-nstats

..
    start-PyFunc-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-PyFunc-nlaststats

..
    start-PyFunc-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-PyFunc-targets

..
    start-PyFunc-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-PyFunc-floatcols

TriqFM Options
--------------
..
    start-TriqFM-intcols

*IntCols*: {``['nIter']``} | :class:`str`
    additional databook cols with integer values

..
    end-TriqFM-intcols

..
    start-TriqFM-comptol

*CompTol*: {``None``} | :class:`float` | :class:`float32`
    tangent tolerance relative to component

..
    end-TriqFM-comptol

..
    start-TriqFM-relprojtol

*RelProjTol*: {``None``} | :class:`float` | :class:`float32`
    projection tolerance relative to size of geometry

..
    end-TriqFM-relprojtol

..
    start-TriqFM-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-TriqFM-type

..
    start-TriqFM-reltol

*RelTol*: {``None``} | :class:`float` | :class:`float32`
    relative tangent tolerance for surface mapping

..
    end-TriqFM-reltol

..
    start-TriqFM-patches

*Patches*: {``None``} | :class:`list`\ [:class:`str`]
    list of patches for a databook component

..
    end-TriqFM-patches

..
    start-TriqFM-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-TriqFM-nstats

..
    start-TriqFM-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-TriqFM-nlaststats

..
    start-TriqFM-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-TriqFM-targets

..
    start-TriqFM-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-TriqFM-floatcols

..
    start-TriqFM-outputformat

*OutputFormat*: ``'dat'`` | {``'plt'``}
    output format for component surface files

..
    end-TriqFM-outputformat

..
    start-TriqFM-abstol

*AbsTol*: {``None``} | :class:`float` | :class:`float32`
    absolute tangent tolerance for surface mapping

..
    end-TriqFM-abstol

..
    start-TriqFM-maptri

*MapTri*: {``None``} | :class:`str`
    name of a tri file to use for remapping CFD surface comps

..
    end-TriqFM-maptri

..
    start-TriqFM-configfile

*ConfigFile*: {``None``} | :class:`str`
    configuration file for surface groups

..
    end-TriqFM-configfile

..
    start-TriqFM-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-TriqFM-nmin

..
    start-TriqFM-cols

*Cols*: {``['CA', 'CY', 'CN', 'CAv', 'CYv', 'CNv', 'Cp_min', 'Cp_max', 'Ax', 'Ay', 'Az']``} | :class:`str`
    list of primary solver output variables to include

..
    end-TriqFM-cols

..
    start-TriqFM-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-TriqFM-dnstats

..
    start-TriqFM-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-TriqFM-compid

..
    start-TriqFM-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-TriqFM-nmaxstats

..
    start-TriqFM-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-TriqFM-transformations

..
    start-TriqFM-absprojtol

*AbsProjTol*: {``None``} | :class:`float` | :class:`float32`
    absolute projection tolerance

..
    end-TriqFM-absprojtol

..
    start-TriqFM-compprojtol

*CompProjTol*: {``None``} | :class:`float` | :class:`float32`
    projection tolerance relative to size of component

..
    end-TriqFM-compprojtol

TriqPoint Options
-----------------
..
    start-TriqPoint-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-TriqPoint-nmin

..
    start-TriqPoint-intcols

*IntCols*: {``['nIter']``} | :class:`str`
    additional databook cols with integer values

..
    end-TriqPoint-intcols

..
    start-TriqPoint-cols

*Cols*: {``['x', 'y', 'z', 'cp']``} | :class:`str`
    list of primary solver output variables to include

..
    end-TriqPoint-cols

..
    start-TriqPoint-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-TriqPoint-dnstats

..
    start-TriqPoint-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-TriqPoint-type

..
    start-TriqPoint-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-TriqPoint-compid

..
    start-TriqPoint-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-TriqPoint-nmaxstats

..
    start-TriqPoint-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-TriqPoint-transformations

..
    start-TriqPoint-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-TriqPoint-nstats

..
    start-TriqPoint-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-TriqPoint-nlaststats

..
    start-TriqPoint-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-TriqPoint-targets

..
    start-TriqPoint-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-TriqPoint-floatcols

..
    start-TriqPoint-points

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors

..
    end-TriqPoint-points


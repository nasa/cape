
.. _pycart-json-databook:

************************
DataBook Section Options
************************
The options below are the available options in the DataBook Section of the ``pycart.json`` control file

..
    start-DataBook-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-DataBook-nmaxstats

..
    start-DataBook-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-DataBook-dnstats

..
    start-DataBook-nmin

*NMin*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-DataBook-nmin

..
    start-DataBook-components

*Components*: {``None``} | :class:`str`
    list of databook components

..
    end-DataBook-components

..
    start-DataBook-type

*Type*: ``'TriqPoint'`` | ``'TriqFM'`` | ``'LineLoad'`` | {``'FM'``} | ``'PyFunc'`` | ``'PointSensor'`` | ``'LineSensor'``
    Default component type

..
    end-DataBook-type

..
    start-DataBook-targets

*Targets*: {``None``} | :class:`object`
    value of option "Targets"

..
    end-DataBook-targets

..
    start-DataBook-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-DataBook-nlaststats

..
    start-DataBook-delimiter

*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files

..
    end-DataBook-delimiter

..
    start-DataBook-nstats

*NStats*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-DataBook-nstats

..
    start-DataBook-folder

*Folder*: {``'data'``} | :class:`str`
    folder for root of databook

..
    end-DataBook-folder

Targets Options
===============
DBTarget Options
----------------
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
    start-DBTarget-file

*File*: {``None``} | :class:`str`
    Name of file from which to read data

..
    end-DBTarget-file

..
    start-DBTarget-components

*Components*: {``None``} | :class:`list`\ [:class:`str`]
    List of databook components with data from this target

..
    end-DBTarget-components

..
    start-DBTarget-commentchar

*CommentChar*: {``'#'``} | :class:`str`
    value of option "CommentChar"

..
    end-DBTarget-commentchar

..
    start-DBTarget-type

*Type*: {``'generic'``} | ``'databook'``
    DataBook Target type

..
    end-DBTarget-type

..
    start-DBTarget-translations

*Translations*: {``None``} | :class:`dict`
    value of option "Translations"

..
    end-DBTarget-translations

..
    start-DBTarget-tolerances

*Tolerances*: {``None``} | :class:`dict`
    Dictionary of tolerances for run matrix keys

..
    end-DBTarget-tolerances

..
    start-DBTarget-label

*Label*: {``None``} | :class:`str`
    Label to use when plotting this target

..
    end-DBTarget-label

..
    start-DBTarget-folder

*Folder*: {``'data'``} | :class:`str`
    Name of folder from which to read data

..
    end-DBTarget-folder

FM Options
----------
..
    start-FM-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-FM-intcols

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
    start-FM-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-FM-dnstats

..
    start-FM-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-FM-floatcols

..
    start-FM-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-FM-transformations

..
    start-FM-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-FM-nmin

..
    start-FM-cols

*Cols*: {``['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']``} | :class:`str`
    list of primary solver output variables to include

..
    end-FM-cols

..
    start-FM-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-FM-type

..
    start-FM-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-FM-targets

..
    start-FM-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-FM-nlaststats

..
    start-FM-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-FM-nstats

PointSensor Options
-------------------
..
    start-PointSensor-intcols

*IntCols*: {``['RefLev']``} | :class:`str`
    additional databook cols with integer values

..
    end-PointSensor-intcols

..
    start-PointSensor-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-PointSensor-compid

..
    start-PointSensor-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-PointSensor-nmaxstats

..
    start-PointSensor-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-PointSensor-dnstats

..
    start-PointSensor-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-PointSensor-floatcols

..
    start-PointSensor-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-PointSensor-transformations

..
    start-PointSensor-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-PointSensor-nmin

..
    start-PointSensor-cols

*Cols*: {``['x', 'y', 'z', 'cp', 'dp', 'rho', 'u', 'v', 'w', 'p']``} | :class:`str`
    list of primary solver output variables to include

..
    end-PointSensor-cols

..
    start-PointSensor-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-PointSensor-type

..
    start-PointSensor-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-PointSensor-targets

..
    start-PointSensor-points

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors

..
    end-PointSensor-points

..
    start-PointSensor-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-PointSensor-nlaststats

..
    start-PointSensor-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-PointSensor-nstats

LineLoad Options
----------------
..
    start-LineLoad-trim

*Trim*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    *trim* flag to ``triload``

..
    end-LineLoad-trim

..
    start-LineLoad-momentum

*Momentum*: {``False``} | :class:`bool` | :class:`bool_`
    whether to use momentum flux in line load computations

..
    end-LineLoad-momentum

..
    start-LineLoad-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-LineLoad-nmaxstats

..
    start-LineLoad-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-LineLoad-floatcols

..
    start-LineLoad-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-LineLoad-transformations

..
    start-LineLoad-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-LineLoad-nmin

..
    start-LineLoad-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-LineLoad-targets

..
    start-LineLoad-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-LineLoad-nstats

..
    start-LineLoad-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-LineLoad-intcols

..
    start-LineLoad-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-LineLoad-compid

..
    start-LineLoad-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-LineLoad-dnstats

..
    start-LineLoad-cols

*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include

..
    end-LineLoad-cols

..
    start-LineLoad-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-LineLoad-type

..
    start-LineLoad-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-LineLoad-nlaststats

..
    start-LineLoad-ncut

*NCut*: {``200``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cuts to make using ``triload`` (-> +1 slice)

..
    end-LineLoad-ncut

..
    start-LineLoad-gauge

*Gauge*: {``True``} | :class:`bool` | :class:`bool_`
    option to use gauge pressures in computations

..
    end-LineLoad-gauge

..
    start-LineLoad-sectiontype

*SectionType*: {``'dlds'``} | ``'clds'`` | ``'slds'``
    line load section type

..
    end-LineLoad-sectiontype

LineSensor Options
------------------
..
    start-LineSensor-intcols

*IntCols*: {``['RefLev']``} | :class:`str`
    additional databook cols with integer values

..
    end-LineSensor-intcols

..
    start-LineSensor-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-LineSensor-compid

..
    start-LineSensor-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-LineSensor-nmaxstats

..
    start-LineSensor-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-LineSensor-dnstats

..
    start-LineSensor-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-LineSensor-floatcols

..
    start-LineSensor-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-LineSensor-transformations

..
    start-LineSensor-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-LineSensor-nmin

..
    start-LineSensor-cols

*Cols*: {``['x', 'y', 'z', 'cp', 'dp', 'rho', 'u', 'v', 'w', 'p']``} | :class:`str`
    list of primary solver output variables to include

..
    end-LineSensor-cols

..
    start-LineSensor-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-LineSensor-type

..
    start-LineSensor-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-LineSensor-targets

..
    start-LineSensor-points

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors

..
    end-LineSensor-points

..
    start-LineSensor-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-LineSensor-nlaststats

..
    start-LineSensor-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-LineSensor-nstats

PyFunc Options
--------------
..
    start-PyFunc-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-PyFunc-intcols

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
    start-PyFunc-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-PyFunc-dnstats

..
    start-PyFunc-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-PyFunc-floatcols

..
    start-PyFunc-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-PyFunc-transformations

..
    start-PyFunc-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-PyFunc-nmin

..
    start-PyFunc-cols

*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include

..
    end-PyFunc-cols

..
    start-PyFunc-function

*Function*: {``None``} | :class:`str`
    Python function name

..
    end-PyFunc-function

..
    start-PyFunc-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-PyFunc-type

..
    start-PyFunc-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-PyFunc-targets

..
    start-PyFunc-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-PyFunc-nlaststats

..
    start-PyFunc-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-PyFunc-nstats

TriqFM Options
--------------
..
    start-TriqFM-absprojtol

*AbsProjTol*: {``None``} | :class:`float` | :class:`float32`
    absolute projection tolerance

..
    end-TriqFM-absprojtol

..
    start-TriqFM-outputformat

*OutputFormat*: ``'dat'`` | {``'plt'``}
    output format for component surface files

..
    end-TriqFM-outputformat

..
    start-TriqFM-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-TriqFM-nmaxstats

..
    start-TriqFM-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-TriqFM-floatcols

..
    start-TriqFM-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-TriqFM-transformations

..
    start-TriqFM-compprojtol

*CompProjTol*: {``None``} | :class:`float` | :class:`float32`
    projection tolerance relative to size of component

..
    end-TriqFM-compprojtol

..
    start-TriqFM-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-TriqFM-nmin

..
    start-TriqFM-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-TriqFM-targets

..
    start-TriqFM-configfile

*ConfigFile*: {``None``} | :class:`str`
    configuration file for surface groups

..
    end-TriqFM-configfile

..
    start-TriqFM-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-TriqFM-nstats

..
    start-TriqFM-patches

*Patches*: {``None``} | :class:`list`\ [:class:`str`]
    list of patches for a databook component

..
    end-TriqFM-patches

..
    start-TriqFM-abstol

*AbsTol*: {``None``} | :class:`float` | :class:`float32`
    absolute tangent tolerance for surface mapping

..
    end-TriqFM-abstol

..
    start-TriqFM-intcols

*IntCols*: {``['nIter']``} | :class:`str`
    additional databook cols with integer values

..
    end-TriqFM-intcols

..
    start-TriqFM-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-TriqFM-compid

..
    start-TriqFM-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-TriqFM-dnstats

..
    start-TriqFM-relprojtol

*RelProjTol*: {``None``} | :class:`float` | :class:`float32`
    projection tolerance relative to size of geometry

..
    end-TriqFM-relprojtol

..
    start-TriqFM-reltol

*RelTol*: {``None``} | :class:`float` | :class:`float32`
    relative tangent tolerance for surface mapping

..
    end-TriqFM-reltol

..
    start-TriqFM-cols

*Cols*: {``['CA', 'CY', 'CN', 'CAv', 'CYv', 'CNv', 'Cp_min', 'Cp_max', 'Ax', 'Ay', 'Az']``} | :class:`str`
    list of primary solver output variables to include

..
    end-TriqFM-cols

..
    start-TriqFM-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-TriqFM-type

..
    start-TriqFM-maptri

*MapTri*: {``None``} | :class:`str`
    name of a tri file to use for remapping CFD surface comps

..
    end-TriqFM-maptri

..
    start-TriqFM-comptol

*CompTol*: {``None``} | :class:`float` | :class:`float32`
    tangent tolerance relative to component

..
    end-TriqFM-comptol

..
    start-TriqFM-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-TriqFM-nlaststats

TriqPoint Options
-----------------
..
    start-TriqPoint-intcols

*IntCols*: {``['nIter']``} | :class:`str`
    additional databook cols with integer values

..
    end-TriqPoint-intcols

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
    start-TriqPoint-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-TriqPoint-dnstats

..
    start-TriqPoint-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-TriqPoint-floatcols

..
    start-TriqPoint-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-TriqPoint-transformations

..
    start-TriqPoint-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-TriqPoint-nmin

..
    start-TriqPoint-cols

*Cols*: {``['x', 'y', 'z', 'cp']``} | :class:`str`
    list of primary solver output variables to include

..
    end-TriqPoint-cols

..
    start-TriqPoint-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-TriqPoint-type

..
    start-TriqPoint-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-TriqPoint-targets

..
    start-TriqPoint-points

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors

..
    end-TriqPoint-points

..
    start-TriqPoint-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-TriqPoint-nlaststats

..
    start-TriqPoint-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-TriqPoint-nstats

IterPoint Options
-----------------
..
    start-IterPoint-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-IterPoint-intcols

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
    start-IterPoint-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-IterPoint-dnstats

..
    start-IterPoint-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-IterPoint-floatcols

..
    start-IterPoint-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-IterPoint-transformations

..
    start-IterPoint-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-IterPoint-nmin

..
    start-IterPoint-cols

*Cols*: {``['cp']``} | :class:`str`
    list of primary solver output variables to include

..
    end-IterPoint-cols

..
    start-IterPoint-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-IterPoint-type

..
    start-IterPoint-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-IterPoint-targets

..
    start-IterPoint-points

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors

..
    end-IterPoint-points

..
    start-IterPoint-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-IterPoint-nlaststats

..
    start-IterPoint-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-IterPoint-nstats


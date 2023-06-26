
.. _pyover-json-databook:

************************
DataBook Section Options
************************
The options below are the available options in the DataBook Section of the ``pyover.json`` control file

..
    start-DataBook-fomo

*fomo*: {``None``} | :class:`str`
    path to ``mixsur`` output files

        If each of the following files is found, there is no need to run
        ``mixsur``, and files are linked instead.

            * ``grid.i.tri``
            * ``grid.bnd``
            * ``grid.ib``
            * ``grid.ibi``
            * ``grid.map``
            * ``grid.nsf``
            * ``grid.ptv``
            * ``mixsur.fmp``

..
    end-DataBook-fomo

..
    start-DataBook-mixsur

*mixsur*: {``None``} | :class:`str`
    input file for ``mixsur``, ``overint``, or ``usurp``

..
    end-DataBook-mixsur

..
    start-DataBook-targets

*Targets*: {``None``} | :class:`object`
    value of option "Targets"

..
    end-DataBook-targets

..
    start-DataBook-nstats

*NStats*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-DataBook-nstats

..
    start-DataBook-xsurf

*XSurf*: {``None``} | :class:`str`
    preprocessed ``x.srf`` file name

..
    end-DataBook-xsurf

..
    start-DataBook-usurp

*usurp*: {``None``} | :class:`str`
    input file for ``usurp``

..
    end-DataBook-usurp

..
    start-DataBook-type

*Type*: ``'TriqFM'`` | ``'IterPoint'`` | {``'FM'``} | ``'LineLoad'`` | ``'PyFunc'`` | ``'TriqPoint'``
    Default component type

..
    end-DataBook-type

..
    start-DataBook-splitmq

*splitmq*: {``None``} | :class:`str`
    input file for ``splitmq``

..
    end-DataBook-splitmq

..
    start-DataBook-qsurf

*QSurf*: {``None``} | :class:`str`
    preprocessed ``q.srf`` file name

..
    end-DataBook-qsurf

..
    start-DataBook-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-DataBook-dnstats

..
    start-DataBook-folder

*Folder*: {``'data'``} | :class:`str`
    folder for root of databook

..
    end-DataBook-folder

..
    start-DataBook-qout

*QOut*: {``None``} | :class:`str`
    preprocessed ``q`` file for a databook component

..
    end-DataBook-qout

..
    start-DataBook-qin

*QIn*: {``None``} | :class:`str`
    input ``q`` file

..
    end-DataBook-qin

..
    start-DataBook-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-DataBook-nmaxstats

..
    start-DataBook-delimiter

*Delimiter*: {``','``} | :class:`str`
    delimiter to use in databook files

..
    end-DataBook-delimiter

..
    start-DataBook-xout

*XOut*: {``None``} | :class:`str`
    preprocessed ``x`` file for a databook component

..
    end-DataBook-xout

..
    start-DataBook-components

*Components*: {``None``} | :class:`str`
    list of databook components

..
    end-DataBook-components

..
    start-DataBook-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-DataBook-nlaststats

..
    start-DataBook-xin

*XIn*: {``None``} | :class:`str`
    input ``x`` file

..
    end-DataBook-xin

..
    start-DataBook-nmin

*NMin*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-DataBook-nmin

Targets Options
===============
DBTarget Options
----------------
..
    start-DBTarget-tolerances

*Tolerances*: {``None``} | :class:`dict`
    Dictionary of tolerances for run matrix keys

..
    end-DBTarget-tolerances

..
    start-DBTarget-file

*File*: {``None``} | :class:`str`
    Name of file from which to read data

..
    end-DBTarget-file

..
    start-DBTarget-name

*Name*: {``None``} | :class:`str`
    Internal *name* to use for target

..
    end-DBTarget-name

..
    start-DBTarget-label

*Label*: {``None``} | :class:`str`
    Label to use when plotting this target

..
    end-DBTarget-label

..
    start-DBTarget-delimiter

*Delimiter*: {``','``} | :class:`str`
    Delimiter in databook target data file

..
    end-DBTarget-delimiter

..
    start-DBTarget-type

*Type*: {``'generic'``} | ``'databook'``
    DataBook Target type

..
    end-DBTarget-type

..
    start-DBTarget-commentchar

*CommentChar*: {``'#'``} | :class:`str`
    value of option "CommentChar"

..
    end-DBTarget-commentchar

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
    start-DBTarget-translations

*Translations*: {``None``} | :class:`dict`
    value of option "Translations"

..
    end-DBTarget-translations

FM Options
----------
..
    start-FM-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-FM-targets

..
    start-FM-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-FM-nstats

..
    start-FM-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-FM-floatcols

..
    start-FM-cols

*Cols*: {``['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']``} | :class:`str`
    list of primary solver output variables to include

..
    end-FM-cols

..
    start-FM-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-FM-transformations

..
    start-FM-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-FM-nmaxstats

..
    start-FM-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-FM-compid

..
    start-FM-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-FM-type

..
    start-FM-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-FM-nlaststats

..
    start-FM-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-FM-nmin

..
    start-FM-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-FM-dnstats

..
    start-FM-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-FM-intcols

IterPoint Options
-----------------
..
    start-IterPoint-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-IterPoint-intcols

..
    start-IterPoint-points

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors

..
    end-IterPoint-points

..
    start-IterPoint-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-IterPoint-targets

..
    start-IterPoint-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-IterPoint-nstats

..
    start-IterPoint-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-IterPoint-floatcols

..
    start-IterPoint-cols

*Cols*: {``['cp']``} | :class:`str`
    list of primary solver output variables to include

..
    end-IterPoint-cols

..
    start-IterPoint-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-IterPoint-transformations

..
    start-IterPoint-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-IterPoint-nmaxstats

..
    start-IterPoint-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-IterPoint-compid

..
    start-IterPoint-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-IterPoint-type

..
    start-IterPoint-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-IterPoint-nlaststats

..
    start-IterPoint-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-IterPoint-dnstats

..
    start-IterPoint-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-IterPoint-nmin

LineLoad Options
----------------
..
    start-LineLoad-fomo

*fomo*: {``None``} | :class:`str`
    path to ``mixsur`` output files

        If each of the following files is found, there is no need to run
        ``mixsur``, and files are linked instead.

            * ``grid.i.tri``
            * ``grid.bnd``
            * ``grid.ib``
            * ``grid.ibi``
            * ``grid.map``
            * ``grid.nsf``
            * ``grid.ptv``
            * ``mixsur.fmp``

..
    end-LineLoad-fomo

..
    start-LineLoad-mixsur

*mixsur*: {``'mixsur.i'``} | :class:`str`
    input file for ``mixsur``, ``overint``, or ``usurp``

..
    end-LineLoad-mixsur

..
    start-LineLoad-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-LineLoad-targets

..
    start-LineLoad-trim

*Trim*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    *trim* flag to ``triload``

..
    end-LineLoad-trim

..
    start-LineLoad-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-LineLoad-nstats

..
    start-LineLoad-xsurf

*XSurf*: {``'x.pyover.surf'``} | :class:`str`
    preprocessed ``x.srf`` file name

..
    end-LineLoad-xsurf

..
    start-LineLoad-gauge

*Gauge*: {``True``} | :class:`bool` | :class:`bool_`
    option to use gauge pressures in computations

..
    end-LineLoad-gauge

..
    start-LineLoad-usurp

*usurp*: {``''``} | :class:`str`
    input file for ``usurp``

..
    end-LineLoad-usurp

..
    start-LineLoad-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-LineLoad-compid

..
    start-LineLoad-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-LineLoad-type

..
    start-LineLoad-splitmq

*splitmq*: {``'splitmq.i'``} | :class:`str`
    input file for ``splitmq``

..
    end-LineLoad-splitmq

..
    start-LineLoad-momentum

*Momentum*: {``False``} | :class:`bool` | :class:`bool_`
    whether to use momentum flux in line load computations

..
    end-LineLoad-momentum

..
    start-LineLoad-qsurf

*QSurf*: {``'q.pyover.surf'``} | :class:`str`
    preprocessed ``q.srf`` file name

..
    end-LineLoad-qsurf

..
    start-LineLoad-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-LineLoad-dnstats

..
    start-LineLoad-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-LineLoad-nmin

..
    start-LineLoad-qout

*QOut*: {``None``} | :class:`str`
    preprocessed ``q`` file for a databook component

..
    end-LineLoad-qout

..
    start-LineLoad-qin

*QIn*: {``'q.pyover.p3d'``} | :class:`str`
    input ``q`` file

..
    end-LineLoad-qin

..
    start-LineLoad-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-LineLoad-floatcols

..
    start-LineLoad-cols

*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include

..
    end-LineLoad-cols

..
    start-LineLoad-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-LineLoad-transformations

..
    start-LineLoad-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-LineLoad-nmaxstats

..
    start-LineLoad-sectiontype

*SectionType*: {``'dlds'``} | ``'clds'`` | ``'slds'``
    line load section type

..
    end-LineLoad-sectiontype

..
    start-LineLoad-xout

*XOut*: {``None``} | :class:`str`
    preprocessed ``x`` file for a databook component

..
    end-LineLoad-xout

..
    start-LineLoad-ncut

*NCut*: {``200``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cuts to make using ``triload`` (-> +1 slice)

..
    end-LineLoad-ncut

..
    start-LineLoad-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-LineLoad-nlaststats

..
    start-LineLoad-xin

*XIn*: {``'x.pyover.p3d'``} | :class:`str`
    input ``x`` file

..
    end-LineLoad-xin

..
    start-LineLoad-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-LineLoad-intcols

PyFunc Options
--------------
..
    start-PyFunc-intcols

*IntCols*: {``['nIter', 'nStats']``} | :class:`str`
    additional databook cols with integer values

..
    end-PyFunc-intcols

..
    start-PyFunc-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-PyFunc-targets

..
    start-PyFunc-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-PyFunc-nstats

..
    start-PyFunc-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-PyFunc-floatcols

..
    start-PyFunc-cols

*Cols*: {``[]``} | :class:`str`
    list of primary solver output variables to include

..
    end-PyFunc-cols

..
    start-PyFunc-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-PyFunc-transformations

..
    start-PyFunc-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-PyFunc-nmaxstats

..
    start-PyFunc-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-PyFunc-compid

..
    start-PyFunc-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-PyFunc-type

..
    start-PyFunc-function

*Function*: {``None``} | :class:`str`
    Python function name

..
    end-PyFunc-function

..
    start-PyFunc-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-PyFunc-nlaststats

..
    start-PyFunc-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-PyFunc-dnstats

..
    start-PyFunc-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-PyFunc-nmin

TriqFM Options
--------------
..
    start-TriqFM-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-TriqFM-targets

..
    start-TriqFM-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-TriqFM-nstats

..
    start-TriqFM-usurp

*usurp*: {``''``} | :class:`str`
    input file for ``usurp``

..
    end-TriqFM-usurp

..
    start-TriqFM-compprojtol

*CompProjTol*: {``None``} | :class:`float` | :class:`float32`
    projection tolerance relative to size of component

..
    end-TriqFM-compprojtol

..
    start-TriqFM-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-TriqFM-compid

..
    start-TriqFM-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-TriqFM-type

..
    start-TriqFM-splitmq

*splitmq*: {``'splitmq.i'``} | :class:`str`
    input file for ``splitmq``

..
    end-TriqFM-splitmq

..
    start-TriqFM-qsurf

*QSurf*: {``'q.pyover.surf'``} | :class:`str`
    preprocessed ``q.srf`` file name

..
    end-TriqFM-qsurf

..
    start-TriqFM-absprojtol

*AbsProjTol*: {``None``} | :class:`float` | :class:`float32`
    absolute projection tolerance

..
    end-TriqFM-absprojtol

..
    start-TriqFM-comptol

*CompTol*: {``None``} | :class:`float` | :class:`float32`
    tangent tolerance relative to component

..
    end-TriqFM-comptol

..
    start-TriqFM-qout

*QOut*: {``None``} | :class:`str`
    preprocessed ``q`` file for a databook component

..
    end-TriqFM-qout

..
    start-TriqFM-qin

*QIn*: {``'q.pyover.p3d'``} | :class:`str`
    input ``q`` file

..
    end-TriqFM-qin

..
    start-TriqFM-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-TriqFM-transformations

..
    start-TriqFM-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-TriqFM-nmaxstats

..
    start-TriqFM-abstol

*AbsTol*: {``None``} | :class:`float` | :class:`float32`
    absolute tangent tolerance for surface mapping

..
    end-TriqFM-abstol

..
    start-TriqFM-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-TriqFM-nlaststats

..
    start-TriqFM-xin

*XIn*: {``'x.pyover.p3d'``} | :class:`str`
    input ``x`` file

..
    end-TriqFM-xin

..
    start-TriqFM-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-TriqFM-nmin

..
    start-TriqFM-patches

*Patches*: {``None``} | :class:`list`\ [:class:`str`]
    list of patches for a databook component

..
    end-TriqFM-patches

..
    start-TriqFM-fomo

*fomo*: {``None``} | :class:`str`
    path to ``mixsur`` output files

        If each of the following files is found, there is no need to run
        ``mixsur``, and files are linked instead.

            * ``grid.i.tri``
            * ``grid.bnd``
            * ``grid.ib``
            * ``grid.ibi``
            * ``grid.map``
            * ``grid.nsf``
            * ``grid.ptv``
            * ``mixsur.fmp``

..
    end-TriqFM-fomo

..
    start-TriqFM-mixsur

*mixsur*: {``'mixsur.i'``} | :class:`str`
    input file for ``mixsur``, ``overint``, or ``usurp``

..
    end-TriqFM-mixsur

..
    start-TriqFM-xsurf

*XSurf*: {``'x.pyover.surf'``} | :class:`str`
    preprocessed ``x.srf`` file name

..
    end-TriqFM-xsurf

..
    start-TriqFM-maptri

*MapTri*: {``None``} | :class:`str`
    name of a tri file to use for remapping CFD surface comps

..
    end-TriqFM-maptri

..
    start-TriqFM-relprojtol

*RelProjTol*: {``None``} | :class:`float` | :class:`float32`
    projection tolerance relative to size of geometry

..
    end-TriqFM-relprojtol

..
    start-TriqFM-outputformat

*OutputFormat*: ``'dat'`` | {``'plt'``}
    output format for component surface files

..
    end-TriqFM-outputformat

..
    start-TriqFM-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-TriqFM-dnstats

..
    start-TriqFM-reltol

*RelTol*: {``None``} | :class:`float` | :class:`float32`
    relative tangent tolerance for surface mapping

..
    end-TriqFM-reltol

..
    start-TriqFM-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-TriqFM-floatcols

..
    start-TriqFM-cols

*Cols*: {``['CA', 'CY', 'CN', 'CAv', 'CYv', 'CNv', 'Cp_min', 'Cp_max', 'Ax', 'Ay', 'Az']``} | :class:`str`
    list of primary solver output variables to include

..
    end-TriqFM-cols

..
    start-TriqFM-xout

*XOut*: {``None``} | :class:`str`
    preprocessed ``x`` file for a databook component

..
    end-TriqFM-xout

..
    start-TriqFM-intcols

*IntCols*: {``['nIter']``} | :class:`str`
    additional databook cols with integer values

..
    end-TriqFM-intcols

..
    start-TriqFM-configfile

*ConfigFile*: {``None``} | :class:`str`
    configuration file for surface groups

..
    end-TriqFM-configfile

TriqPoint Options
-----------------
..
    start-TriqPoint-intcols

*IntCols*: {``['nIter']``} | :class:`str`
    additional databook cols with integer values

..
    end-TriqPoint-intcols

..
    start-TriqPoint-points

*Points*: {``None``} | :class:`list`\ [:class:`str`]
    list of individual point sensors

..
    end-TriqPoint-points

..
    start-TriqPoint-targets

*Targets*: {``{}``} | :class:`dict`
    targets for this databook component

..
    end-TriqPoint-targets

..
    start-TriqPoint-nstats

*NStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    iterations to use in averaging window [for a comp]

..
    end-TriqPoint-nstats

..
    start-TriqPoint-floatcols

*FloatCols*: {``[]``} | :class:`str`
    additional databook cols with floating-point values

..
    end-TriqPoint-floatcols

..
    start-TriqPoint-cols

*Cols*: {``['x', 'y', 'z', 'cp']``} | :class:`str`
    list of primary solver output variables to include

..
    end-TriqPoint-cols

..
    start-TriqPoint-transformations

*Transformations*: {``[]``} | :class:`dict`
    list of transformations applied to component

..
    end-TriqPoint-transformations

..
    start-TriqPoint-nmaxstats

*NMaxStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max number of iters to include in averaging window

..
    end-TriqPoint-nmaxstats

..
    start-TriqPoint-compid

*CompID*: {``None``} | :class:`object`
    surface componet(s) to use for this databook component

..
    end-TriqPoint-compid

..
    start-TriqPoint-type

*Type*: {``'FM'``} | :class:`str`
    databook component type

..
    end-TriqPoint-type

..
    start-TriqPoint-nlaststats

*NLastStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to extract stats

..
    end-TriqPoint-nlaststats

..
    start-TriqPoint-dnstats

*DNStats*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    increment for candidate window sizes

..
    end-TriqPoint-dnstats

..
    start-TriqPoint-nmin

*NMin*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    first iter to consider for use in databook [for a comp]

..
    end-TriqPoint-nmin


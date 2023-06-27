
.. _pycart-json-runcontrol:

**********************************
Options for ``RunControl`` Section
**********************************
The options below are the available options in the ``RunControl`` Section of the ``pycart.json`` control file


*WarmStart*: {``False``} | :class:`bool` | :class:`bool_`
    whether to warm start a case



*intersect*: {``None``} | :class:`object`
    value of option "intersect"



*ulimit*: {``None``} | :class:`object`
    value of option "ulimit"



*cubes*: {``None``} | :class:`object`
    value of option "cubes"



*nProc*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cores/threads to run



*MPI*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to run MPI in phase



*Environ*: {``None``} | :class:`object`
    value of option "Environ"



*flowCart*: {``None``} | :class:`object`
    value of option "flowCart"



*slurm*: {``False``} | :class:`bool` | :class:`bool_`
    wheter or not to submit jobs with Slurm



*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name



*Adaptive*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to use ``aero.csh`` in phase



*Continue*: {``True``} | :class:`bool` | :class:`bool_`
    whether restarts of same phase can use same job



*PhaseSequence*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    list of phase indices to run



*PhaseIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    check-point iterations for phase *j*



*verify*: {``None``} | :class:`object`
    value of option "verify"



*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file



*autoInputs*: {``None``} | :class:`object`
    value of option "autoInputs"



*Verbose*: {``False``} | :class:`bool` | :class:`bool_`
    "RunControl" verbosity flag



*Archive*: {``None``} | :class:`object`
    value of option "Archive"



*Adaptation*: {``None``} | :class:`object`
    value of option "Adaptation"



*Resubmit*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to submit new job at end of phase *j*



*adjointCart*: {``None``} | :class:`object`
    value of option "adjointCart"



*qsub*: {``True``} | :class:`bool` | :class:`bool_`
    wheter or not to submit jobs with PBS



*aflr3*: {``None``} | :class:`object`
    value of option "aflr3"



*PreMesh*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to generate volume mesh before submitting



*nIter*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of iterations to run in phase *j*


Adaptation Options
------------------

*jumpstart*: {``False``} | :class:`bool` | :class:`bool_`
    whether to create meshes b4 running ``aero.csh``



*max_nCells*: {``5000000.0``} | :class:`int` | :class:`int32` | :class:`int64`
    maximum cell count



*final_mesh_xref*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    num. of additional adapts using final error map



*ws_it*: {``50``} | :class:`int` | :class:`int32` | :class:`int64`
    number of ``flowCart`` iters for ``aero.csh`` cycles



*etol*: {``1e-06``} | :class:`float` | :class:`float32`
    target output error tolerance



*apc*: {``'a'``} | ``'p'``
    adaptation cycle type (adapt/propagate)



*buf*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    number of buffer layers



*mesh_growth*: {``1.5``} | :class:`float` | :class:`float32`
    mesh growth ratio between cycles of ``aero.csh``



*n_adapt_cycles*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    number of Cart3D adaptation cycles in phase


Archive Options
---------------

*ArchiveTemplate*: {``'full'``} | :class:`str`
    template for default archive settings



*PreTarDirs*: {``[]``} | :class:`object`
    folders to tar before archiving



*PostDeleteFiles*: {``[]``} | :class:`object`
    list of files to delete after archiving



*nCheckPoint*: {``2``} | :class:`int` | :class:`int32` | :class:`int64`
    number of check point files to keep



*ArchiveAction*: ``''`` | {``'archive'``} | ``'rm'`` | ``'skeleton'``
    action to take after finishing a case



*ArchiveExtension*: {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    archive file extension



*ArchiveFiles*: {``[]``} | :class:`object`
    files to copy to archive



*ProgressTarGroups*: {``[]``} | :class:`object`
    list of file groups to tar while running



*PostDeleteDirs*: {``[]``} | :class:`object`
    list of folders to delete after archiving



*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root



*ProgressDeleteFiles*: {``[]``} | :class:`object`
    files to delete while still running



*ArchiveFormat*: ``''`` | {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    format for case archives



*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar after archiving



*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving



*SkeletonDirs*: {``None``} | :class:`object`
    folders to **keep** during skeleton action



*ProgressUpdateFiles*: {``[]``} | :class:`object`
    files to delete old versions while running



*RemoteCopy*: {``'scp'``} | :class:`str`
    command for archive remote copies



*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving



*TarAdapt*: ``'full'`` | {``'restart'``} | ``'none'``
    archive option for adapt folders



*PreUpdateFiles*: {``[]``} | :class:`object`
    files to keep *n* and delete older, b4 archiving



*PostUpdateFiles*: {``[]``} | :class:`object`
    globs: keep *n* and rm older, after archiving



*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files



*PostTarGroups*: {``[]``} | :class:`object`
    groups of files to tar after archiving



*ProgressDeleteDirs*: {``[]``} | :class:`object`
    folders to delete while still running



*PreTarGroups*: {``[]``} | :class:`object`
    file groups to tar before archiving



*TarViz*: ``'full'`` | {``'restart'``} | ``'none'``
    archive option for visualization files



*SkeletonFiles*: {``['case.json']``} | :class:`object`
    files to **keep** during skeleton action



*ProgressArchiveFiles*: {``[]``} | :class:`object`
    files to archive at any time



*SkeletonTarDirs*: {``[]``} | :class:`object`
    folders to tar before deletion during skeleton



*ProgressTarDirs*: {``[]``} | :class:`object`
    folders to tar while running



*SkeletonTailFiles*: {``[]``} | :class:`object`
    files to tail before deletion during skeleton


adjointCart Options
-------------------

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program



*it_ad*: {``120``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "it_ad"



*mg_ad*: {``3``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "mg_ad"



*adj_first_order*: {``False``} | :class:`bool` | :class:`bool_`
    value of option "adj_first_order"


autoInputs Options
------------------

*maxR*: {``10``} | :class:`int` | :class:`int32` | :class:`int64`
    maximum number of cell refinements



*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program



*r*: {``30.0``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64`
    nominal ``autoInputs`` mesh radius



*nDiv*: {``4``} | :class:`int` | :class:`int32` | :class:`int64`
    number of divisions in background mesh


cubes Options
-------------

*maxR*: {``11``} | :class:`int` | :class:`int32` | :class:`int64`
    maximum number of refinements in ``cubes`` mesh



*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program



*b*: {``2``} | :class:`int` | :class:`int32` | :class:`int64`
    number of layers of buffer cells



*pre*: {``'preSpec.c3d.cntl'``} | :class:`str`
    value of option "pre"



*a*: {``10.0``} | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    angle threshold [deg] for geom refinement



*sf*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    additional levels at sharp edges



*reorder*: {``True``} | :class:`bool` | :class:`bool_`
    whether to reorder output mesh


flowCart Options
----------------

*mg_fc*: {``3``} | :class:`int` | :class:`int32` | :class:`int64`
    multigrid levels for ``flowCart``



*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program



*robust_mode*: {``False``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` should be run in robust mode



*it_fc*: {``200``} | :class:`int` | :class:`int32` | :class:`int64`
    number of ``flowCart`` iterations



*it_sub*: {``10``} | :class:`int` | :class:`int32` | :class:`int64`
    number of subiters for each ``flowCart`` time step



*limiter*: {``2``} | :class:`int` | :class:`int32` | :class:`int64`
    limiter for ``flowCart``



*cfl*: {``1.1``} | :class:`float` | :class:`float32`
    nominal CFL number for ``flowCart``



*y_is_spanwise*: {``True``} | :class:`bool` | :class:`bool_`
    whether *y* is spanwise axis for ``flowCart``



*binaryIO*: {``True``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` is set for binary I/O



*fc_stats*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    number of iters for iterative or time averaging



*clic*: {``True``} | :class:`bool` | :class:`bool_`
    whether to write ``Components.i.triq``



*first_order*: {``False``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` should be run first-order



*tm*: {``False``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` is set for cut cell gradient



*fmg*: {``True``} | :class:`bool` | :class:`bool_`
    whether to run ``flowCart`` w/ full multigrid



*buffLim*: {``False``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` will use buffer limits



*nOrders*: {``12``} | :class:`int` | :class:`int32` | :class:`int64`
    convergence drop orders of magnitude for early exit



*it_avg*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    number of ``flowCart`` iters b/w ``.triq`` outputs



*it_start*: {``100``} | :class:`int` | :class:`int32` | :class:`int64`
    number of ``flowCart`` iters b4 ``.triq`` outputs



*checkptTD*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    steps between unsteady ``flowCart`` checkpoints



*dt*: {``0.1``} | :class:`float` | :class:`float32`
    nondimensional physical time step



*fc_clean*: {``False``} | :class:`bool` | :class:`bool_`
    whether to run relaxation step before time-accurate step



*vizTD*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    steps between ``flowCart`` visualization outputs



*tecO*: {``True``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` dumps Tecplot triangulations



*RKScheme*: {``None``} | :class:`str` | :class:`list`
    the Runge-Kutta scheme for a phase



*pmg*: {``False``} | :class:`bool` | :class:`bool_`
    whether to run ``flowCart`` w/ poly multigrid



*mpi_fc*: {``False``} | :class:`bool` | :class:`bool_` | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64`
    whether or not to run ``mpi_flowCart``



*unsteady*: {``False``} | :class:`bool` | :class:`bool_`
    whether to run time-accurate ``flowCart``



*cflmin*: {``0.8``} | :class:`float` | :class:`float32`
    min CFL number for ``flowCart``


Environ Options
---------------
aflr3 Options
-------------

*mdsblf*: ``0`` | {``1``} | ``2``
    AFLR3 BL spacing thickness factor option



*bli*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of AFLR3 prism layers



*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether or not to run AFLR3



*cdfs*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64`
    AFLR3 geometric growth exclusion zone size



*flags*: {``{}``} | :class:`dict`
    AFLR3 options using ``-flag val`` format



*BCFile*: {``None``} | :class:`str`
    AFLR3 boundary condition file



*grow*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 off-body growth rate



*mdf*: ``1`` | {``2``}
    AFLR3 volume grid distribution flag



*blr*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 boundary layer stretching ratio



*angqbf*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 max angle on surface triangles



*nqual*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    number of AFLR3 mesh quality passes



*angblisimx*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 max angle b/w BL intersecting faces



*i*: {``None``} | :class:`str`
    input file for AFLR3



*blds*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 initial boundary-layer spacing



*blc*: {``None``} | :class:`bool` | :class:`bool_`
    AFLR3 prism layer option



*cdfr*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 max geometric growth rate



*o*: {``None``} | :class:`str`
    output file for AFLR3



*keys*: {``{}``} | :class:`dict`
    AFLR3 options using ``key=val`` format


intersect Options
-----------------

*T*: {``False``} | :class:`bool` | :class:`bool_`
    option to also write Tecplot file ``Components.i.plt``



*v*: {``False``} | :class:`bool` | :class:`bool_`
    verbose mode



*triged*: {``True``} | :class:`bool` | :class:`bool_`
    option to use CGT ``triged`` to clean output file



*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program



*rm*: {``False``} | :class:`bool` | :class:`bool_`
    option to remove small triangles from results



*intersections*: {``False``} | :class:`bool` | :class:`bool_`
    option to write intersections to ``intersect.dat``



*fast*: {``False``} | :class:`bool` | :class:`bool_`
    also write unformatted FAST file ``Components.i.fast``



*ascii*: {``None``} | :class:`bool` | :class:`bool_`
    flag that input file is ASCII



*cutout*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of component to subtract



*smalltri*: {``0.0001``} | :class:`float` | :class:`float32`
    cutoff size for small triangles with *rm*



*o*: {``'Components.i.tri'``} | :class:`str`
    output file for ``intersect``



*overlap*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    perform boolean intersection of this comp number



*i*: {``'Components.tri'``} | :class:`str`
    input file to ``intersect``


ulimit Options
--------------

*r*: {``0``} | :class:`object`
    max real-time scheduling priority, ``ulimit -r``



*q*: {``819200``} | :class:`object`
    max bytes in POSIX message queues, ``ulimit -q``



*c*: {``0``} | :class:`object`
    core file size limit, ``ulimit -c``



*i*: {``127556``} | :class:`object`
    max number of pending signals, ``ulimit -i``



*l*: {``64``} | :class:`object`
    max size that may be locked into memory, ``ulimit -l``



*d*: {``'unlimited'``} | :class:`object`
    process data segment limit, ``ulimit -d``



*f*: {``'unlimited'``} | :class:`object`
    max size of files written by shell, ``ulimit -f``



*s*: {``4194304``} | :class:`object`
    stack size limit, ``ulimit -s``



*x*: {``'unlimited'``} | :class:`object`
    max number of file locks, ``ulimit -x``



*v*: {``'unlimited'``} | :class:`object`
    max virtual memory avail to shell, ``ulimit -v``



*n*: {``1024``} | :class:`object`
    max number of open files, ``ulimit -n``



*m*: {``'unlimited'``} | :class:`object`
    max resident set size, ``ulimit -m``



*t*: {``'unlimited'``} | :class:`object`
    max amount of cpu time in s, ``ulimit -t``



*u*: {``127812``} | :class:`object`
    max number of procs avail to one user, ``ulimit -u``



*p*: {``8``} | :class:`object`
    pipe size in 512-byte blocks, ``ulimit -p``



*e*: {``0``} | :class:`object`
    max scheduling priority, ``ulimit -e``


verify Options
--------------

*i*: {``None``} | :class:`str`
    input file for ``verify``



*ascii*: {``True``} | :class:`bool` | :class:`bool_`
    option for ASCII input file to ``verify``



*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program



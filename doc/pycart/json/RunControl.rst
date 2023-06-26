
.. _pycart-json-runcontrol:

**************************
RunControl Section Options
**************************
The options below are the available options in the RunControl Section of the ``pycart.json`` control file

..
    start-RunControl-nproc

*nProc*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cores/threads to run

..
    end-RunControl-nproc

..
    start-RunControl-ulimit

*ulimit*: {``None``} | :class:`object`
    value of option "ulimit"

..
    end-RunControl-ulimit

..
    start-RunControl-slurm

*slurm*: {``False``} | :class:`bool` | :class:`bool_`
    wheter or not to submit jobs with Slurm

..
    end-RunControl-slurm

..
    start-RunControl-mpicmd

*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name

..
    end-RunControl-mpicmd

..
    start-RunControl-warmstart

*WarmStart*: {``False``} | :class:`bool` | :class:`bool_`
    whether to warm start a case

..
    end-RunControl-warmstart

..
    start-RunControl-aflr3

*aflr3*: {``None``} | :class:`object`
    value of option "aflr3"

..
    end-RunControl-aflr3

..
    start-RunControl-continue

*Continue*: {``True``} | :class:`bool` | :class:`bool_`
    whether restarts of same phase can use same job

..
    end-RunControl-continue

..
    start-RunControl-intersect

*intersect*: {``None``} | :class:`object`
    value of option "intersect"

..
    end-RunControl-intersect

..
    start-RunControl-warmstartfolder

*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file

..
    end-RunControl-warmstartfolder

..
    start-RunControl-adaptation

*Adaptation*: {``None``} | :class:`object`
    value of option "Adaptation"

..
    end-RunControl-adaptation

..
    start-RunControl-adjointcart

*adjointCart*: {``None``} | :class:`object`
    value of option "adjointCart"

..
    end-RunControl-adjointcart

..
    start-RunControl-verify

*verify*: {``None``} | :class:`object`
    value of option "verify"

..
    end-RunControl-verify

..
    start-RunControl-adaptive

*Adaptive*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to use ``aero.csh`` in phase

..
    end-RunControl-adaptive

..
    start-RunControl-environ

*Environ*: {``None``} | :class:`object`
    value of option "Environ"

..
    end-RunControl-environ

..
    start-RunControl-flowcart

*flowCart*: {``None``} | :class:`object`
    value of option "flowCart"

..
    end-RunControl-flowcart

..
    start-RunControl-phasesequence

*PhaseSequence*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    list of phase indices to run

..
    end-RunControl-phasesequence

..
    start-RunControl-qsub

*qsub*: {``True``} | :class:`bool` | :class:`bool_`
    wheter or not to submit jobs with PBS

..
    end-RunControl-qsub

..
    start-RunControl-verbose

*Verbose*: {``False``} | :class:`bool` | :class:`bool_`
    "RunControl" verbosity flag

..
    end-RunControl-verbose

..
    start-RunControl-phaseiters

*PhaseIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    check-point iterations for phase *j*

..
    end-RunControl-phaseiters

..
    start-RunControl-resubmit

*Resubmit*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to submit new job at end of phase *j*

..
    end-RunControl-resubmit

..
    start-RunControl-mpi

*MPI*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to run MPI in phase

..
    end-RunControl-mpi

..
    start-RunControl-premesh

*PreMesh*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to generate volume mesh before submitting

..
    end-RunControl-premesh

..
    start-RunControl-archive

*Archive*: {``None``} | :class:`object`
    value of option "Archive"

..
    end-RunControl-archive

..
    start-RunControl-niter

*nIter*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of iterations to run in phase *j*

..
    end-RunControl-niter

..
    start-RunControl-autoinputs

*autoInputs*: {``None``} | :class:`object`
    value of option "autoInputs"

..
    end-RunControl-autoinputs

..
    start-RunControl-cubes

*cubes*: {``None``} | :class:`object`
    value of option "cubes"

..
    end-RunControl-cubes

Adaptation Options
------------------
..
    start-Adaptation-ws_it

*ws_it*: {``50``} | :class:`int` | :class:`int32` | :class:`int64`
    number of ``flowCart`` iters for ``aero.csh`` cycles

..
    end-Adaptation-ws_it

..
    start-Adaptation-etol

*etol*: {``1e-06``} | :class:`float` | :class:`float32`
    target output error tolerance

..
    end-Adaptation-etol

..
    start-Adaptation-apc

*apc*: {``'a'``} | ``'p'``
    adaptation cycle type (adapt/propagate)

..
    end-Adaptation-apc

..
    start-Adaptation-final_mesh_xref

*final_mesh_xref*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    num. of additional adapts using final error map

..
    end-Adaptation-final_mesh_xref

..
    start-Adaptation-n_adapt_cycles

*n_adapt_cycles*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    number of Cart3D adaptation cycles in phase

..
    end-Adaptation-n_adapt_cycles

..
    start-Adaptation-mesh_growth

*mesh_growth*: {``1.5``} | :class:`float` | :class:`float32`
    mesh growth ratio between cycles of ``aero.csh``

..
    end-Adaptation-mesh_growth

..
    start-Adaptation-jumpstart

*jumpstart*: {``False``} | :class:`bool` | :class:`bool_`
    whether to create meshes b4 running ``aero.csh``

..
    end-Adaptation-jumpstart

..
    start-Adaptation-buf

*buf*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    number of buffer layers

..
    end-Adaptation-buf

..
    start-Adaptation-max_ncells

*max_nCells*: {``5000000.0``} | :class:`int` | :class:`int32` | :class:`int64`
    maximum cell count

..
    end-Adaptation-max_ncells

Archive Options
---------------
..
    start-Archive-remotecopy

*RemoteCopy*: {``'scp'``} | :class:`str`
    command for archive remote copies

..
    end-Archive-remotecopy

..
    start-Archive-progressarchivefiles

*ProgressArchiveFiles*: {``[]``} | :class:`object`
    files to archive at any time

..
    end-Archive-progressarchivefiles

..
    start-Archive-progressdeletedirs

*ProgressDeleteDirs*: {``[]``} | :class:`object`
    folders to delete while still running

..
    end-Archive-progressdeletedirs

..
    start-Archive-archivetype

*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files

..
    end-Archive-archivetype

..
    start-Archive-posttargroups

*PostTarGroups*: {``[]``} | :class:`object`
    groups of files to tar after archiving

..
    end-Archive-posttargroups

..
    start-Archive-ncheckpoint

*nCheckPoint*: {``2``} | :class:`int` | :class:`int32` | :class:`int64`
    number of check point files to keep

..
    end-Archive-ncheckpoint

..
    start-Archive-archivefiles

*ArchiveFiles*: {``[]``} | :class:`object`
    files to copy to archive

..
    end-Archive-archivefiles

..
    start-Archive-postdeletedirs

*PostDeleteDirs*: {``[]``} | :class:`object`
    list of folders to delete after archiving

..
    end-Archive-postdeletedirs

..
    start-Archive-progresstargroups

*ProgressTarGroups*: {``[]``} | :class:`object`
    list of file groups to tar while running

..
    end-Archive-progresstargroups

..
    start-Archive-archivetemplate

*ArchiveTemplate*: {``'full'``} | :class:`str`
    template for default archive settings

..
    end-Archive-archivetemplate

..
    start-Archive-archivefolder

*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root

..
    end-Archive-archivefolder

..
    start-Archive-progresstardirs

*ProgressTarDirs*: {``[]``} | :class:`object`
    folders to tar while running

..
    end-Archive-progresstardirs

..
    start-Archive-predeletedirs

*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving

..
    end-Archive-predeletedirs

..
    start-Archive-pretargroups

*PreTarGroups*: {``[]``} | :class:`object`
    file groups to tar before archiving

..
    end-Archive-pretargroups

..
    start-Archive-archiveaction

*ArchiveAction*: ``''`` | {``'archive'``} | ``'rm'`` | ``'skeleton'``
    action to take after finishing a case

..
    end-Archive-archiveaction

..
    start-Archive-predeletefiles

*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving

..
    end-Archive-predeletefiles

..
    start-Archive-tarviz

*TarViz*: ``'full'`` | {``'restart'``} | ``'none'``
    archive option for visualization files

..
    end-Archive-tarviz

..
    start-Archive-postupdatefiles

*PostUpdateFiles*: {``[]``} | :class:`object`
    globs: keep *n* and rm older, after archiving

..
    end-Archive-postupdatefiles

..
    start-Archive-progressupdatefiles

*ProgressUpdateFiles*: {``[]``} | :class:`object`
    files to delete old versions while running

..
    end-Archive-progressupdatefiles

..
    start-Archive-skeletonfiles

*SkeletonFiles*: {``['case.json']``} | :class:`object`
    files to **keep** during skeleton action

..
    end-Archive-skeletonfiles

..
    start-Archive-postdeletefiles

*PostDeleteFiles*: {``[]``} | :class:`object`
    list of files to delete after archiving

..
    end-Archive-postdeletefiles

..
    start-Archive-progressdeletefiles

*ProgressDeleteFiles*: {``[]``} | :class:`object`
    files to delete while still running

..
    end-Archive-progressdeletefiles

..
    start-Archive-skeletontailfiles

*SkeletonTailFiles*: {``[]``} | :class:`object`
    files to tail before deletion during skeleton

..
    end-Archive-skeletontailfiles

..
    start-Archive-preupdatefiles

*PreUpdateFiles*: {``[]``} | :class:`object`
    files to keep *n* and delete older, b4 archiving

..
    end-Archive-preupdatefiles

..
    start-Archive-posttardirs

*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar after archiving

..
    end-Archive-posttardirs

..
    start-Archive-archiveformat

*ArchiveFormat*: ``''`` | {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    format for case archives

..
    end-Archive-archiveformat

..
    start-Archive-skeletontardirs

*SkeletonTarDirs*: {``[]``} | :class:`object`
    folders to tar before deletion during skeleton

..
    end-Archive-skeletontardirs

..
    start-Archive-pretardirs

*PreTarDirs*: {``[]``} | :class:`object`
    folders to tar before archiving

..
    end-Archive-pretardirs

..
    start-Archive-taradapt

*TarAdapt*: ``'full'`` | {``'restart'``} | ``'none'``
    archive option for adapt folders

..
    end-Archive-taradapt

..
    start-Archive-archiveextension

*ArchiveExtension*: {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    archive file extension

..
    end-Archive-archiveextension

..
    start-Archive-skeletondirs

*SkeletonDirs*: {``None``} | :class:`object`
    folders to **keep** during skeleton action

..
    end-Archive-skeletondirs

adjointCart Options
-------------------
..
    start-adjointCart-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program

..
    end-adjointCart-run

..
    start-adjointCart-adj_first_order

*adj_first_order*: {``False``} | :class:`bool` | :class:`bool_`
    value of option "adj_first_order"

..
    end-adjointCart-adj_first_order

..
    start-adjointCart-mg_ad

*mg_ad*: {``3``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "mg_ad"

..
    end-adjointCart-mg_ad

..
    start-adjointCart-it_ad

*it_ad*: {``120``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "it_ad"

..
    end-adjointCart-it_ad

autoInputs Options
------------------
..
    start-autoInputs-ndiv

*nDiv*: {``4``} | :class:`int` | :class:`int32` | :class:`int64`
    number of divisions in background mesh

..
    end-autoInputs-ndiv

..
    start-autoInputs-maxr

*maxR*: {``10``} | :class:`int` | :class:`int32` | :class:`int64`
    maximum number of cell refinements

..
    end-autoInputs-maxr

..
    start-autoInputs-r

*r*: {``30.0``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64`
    nominal ``autoInputs`` mesh radius

..
    end-autoInputs-r

..
    start-autoInputs-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program

..
    end-autoInputs-run

cubes Options
-------------
..
    start-cubes-a

*a*: {``10.0``} | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    angle threshold [deg] for geom refinement

..
    end-cubes-a

..
    start-cubes-maxr

*maxR*: {``11``} | :class:`int` | :class:`int32` | :class:`int64`
    maximum number of refinements in ``cubes`` mesh

..
    end-cubes-maxr

..
    start-cubes-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program

..
    end-cubes-run

..
    start-cubes-pre

*pre*: {``'preSpec.c3d.cntl'``} | :class:`str`
    value of option "pre"

..
    end-cubes-pre

..
    start-cubes-reorder

*reorder*: {``True``} | :class:`bool` | :class:`bool_`
    whether to reorder output mesh

..
    end-cubes-reorder

..
    start-cubes-sf

*sf*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    additional levels at sharp edges

..
    end-cubes-sf

..
    start-cubes-b

*b*: {``2``} | :class:`int` | :class:`int32` | :class:`int64`
    number of layers of buffer cells

..
    end-cubes-b

flowCart Options
----------------
..
    start-flowCart-it_fc

*it_fc*: {``200``} | :class:`int` | :class:`int32` | :class:`int64`
    number of ``flowCart`` iterations

..
    end-flowCart-it_fc

..
    start-flowCart-clic

*clic*: {``True``} | :class:`bool` | :class:`bool_`
    whether to write ``Components.i.triq``

..
    end-flowCart-clic

..
    start-flowCart-fc_stats

*fc_stats*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    number of iters for iterative or time averaging

..
    end-flowCart-fc_stats

..
    start-flowCart-fc_clean

*fc_clean*: {``False``} | :class:`bool` | :class:`bool_`
    whether to run relaxation step before time-accurate step

..
    end-flowCart-fc_clean

..
    start-flowCart-fmg

*fmg*: {``True``} | :class:`bool` | :class:`bool_`
    whether to run ``flowCart`` w/ full multigrid

..
    end-flowCart-fmg

..
    start-flowCart-mg_fc

*mg_fc*: {``3``} | :class:`int` | :class:`int32` | :class:`int64`
    multigrid levels for ``flowCart``

..
    end-flowCart-mg_fc

..
    start-flowCart-pmg

*pmg*: {``False``} | :class:`bool` | :class:`bool_`
    whether to run ``flowCart`` w/ poly multigrid

..
    end-flowCart-pmg

..
    start-flowCart-rkscheme

*RKScheme*: {``None``} | :class:`str` | :class:`list`
    the Runge-Kutta scheme for a phase

..
    end-flowCart-rkscheme

..
    start-flowCart-y_is_spanwise

*y_is_spanwise*: {``True``} | :class:`bool` | :class:`bool_`
    whether *y* is spanwise axis for ``flowCart``

..
    end-flowCart-y_is_spanwise

..
    start-flowCart-checkpttd

*checkptTD*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    steps between unsteady ``flowCart`` checkpoints

..
    end-flowCart-checkpttd

..
    start-flowCart-viztd

*vizTD*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    steps between ``flowCart`` visualization outputs

..
    end-flowCart-viztd

..
    start-flowCart-mpi_fc

*mpi_fc*: {``False``} | :class:`bool` | :class:`bool_` | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64`
    whether or not to run ``mpi_flowCart``

..
    end-flowCart-mpi_fc

..
    start-flowCart-limiter

*limiter*: {``2``} | :class:`int` | :class:`int32` | :class:`int64`
    limiter for ``flowCart``

..
    end-flowCart-limiter

..
    start-flowCart-cfl

*cfl*: {``1.1``} | :class:`float` | :class:`float32`
    nominal CFL number for ``flowCart``

..
    end-flowCart-cfl

..
    start-flowCart-tm

*tm*: {``False``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` is set for cut cell gradient

..
    end-flowCart-tm

..
    start-flowCart-it_avg

*it_avg*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    number of ``flowCart`` iters b/w ``.triq`` outputs

..
    end-flowCart-it_avg

..
    start-flowCart-it_start

*it_start*: {``100``} | :class:`int` | :class:`int32` | :class:`int64`
    number of ``flowCart`` iters b4 ``.triq`` outputs

..
    end-flowCart-it_start

..
    start-flowCart-binaryio

*binaryIO*: {``True``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` is set for binary I/O

..
    end-flowCart-binaryio

..
    start-flowCart-cflmin

*cflmin*: {``0.8``} | :class:`float` | :class:`float32`
    min CFL number for ``flowCart``

..
    end-flowCart-cflmin

..
    start-flowCart-norders

*nOrders*: {``12``} | :class:`int` | :class:`int32` | :class:`int64`
    convergence drop orders of magnitude for early exit

..
    end-flowCart-norders

..
    start-flowCart-dt

*dt*: {``0.1``} | :class:`float` | :class:`float32`
    nondimensional physical time step

..
    end-flowCart-dt

..
    start-flowCart-first_order

*first_order*: {``False``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` should be run first-order

..
    end-flowCart-first_order

..
    start-flowCart-it_sub

*it_sub*: {``10``} | :class:`int` | :class:`int32` | :class:`int64`
    number of subiters for each ``flowCart`` time step

..
    end-flowCart-it_sub

..
    start-flowCart-bufflim

*buffLim*: {``False``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` will use buffer limits

..
    end-flowCart-bufflim

..
    start-flowCart-unsteady

*unsteady*: {``False``} | :class:`bool` | :class:`bool_`
    whether to run time-accurate ``flowCart``

..
    end-flowCart-unsteady

..
    start-flowCart-robust_mode

*robust_mode*: {``False``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` should be run in robust mode

..
    end-flowCart-robust_mode

..
    start-flowCart-teco

*tecO*: {``True``} | :class:`bool` | :class:`bool_`
    whether ``flowCart`` dumps Tecplot triangulations

..
    end-flowCart-teco

..
    start-flowCart-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program

..
    end-flowCart-run

Environ Options
---------------
aflr3 Options
-------------
..
    start-aflr3-blds

*blds*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 initial boundary-layer spacing

..
    end-aflr3-blds

..
    start-aflr3-angblisimx

*angblisimx*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 max angle b/w BL intersecting faces

..
    end-aflr3-angblisimx

..
    start-aflr3-blr

*blr*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 boundary layer stretching ratio

..
    end-aflr3-blr

..
    start-aflr3-grow

*grow*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 off-body growth rate

..
    end-aflr3-grow

..
    start-aflr3-cdfr

*cdfr*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 max geometric growth rate

..
    end-aflr3-cdfr

..
    start-aflr3-keys

*keys*: {``{}``} | :class:`dict`
    AFLR3 options using ``key=val`` format

..
    end-aflr3-keys

..
    start-aflr3-cdfs

*cdfs*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64`
    AFLR3 geometric growth exclusion zone size

..
    end-aflr3-cdfs

..
    start-aflr3-mdsblf

*mdsblf*: ``0`` | {``1``} | ``2``
    AFLR3 BL spacing thickness factor option

..
    end-aflr3-mdsblf

..
    start-aflr3-bcfile

*BCFile*: {``None``} | :class:`str`
    AFLR3 boundary condition file

..
    end-aflr3-bcfile

..
    start-aflr3-bli

*bli*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of AFLR3 prism layers

..
    end-aflr3-bli

..
    start-aflr3-i

*i*: {``None``} | :class:`str`
    input file for AFLR3

..
    end-aflr3-i

..
    start-aflr3-flags

*flags*: {``{}``} | :class:`dict`
    AFLR3 options using ``-flag val`` format

..
    end-aflr3-flags

..
    start-aflr3-nqual

*nqual*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    number of AFLR3 mesh quality passes

..
    end-aflr3-nqual

..
    start-aflr3-blc

*blc*: {``None``} | :class:`bool` | :class:`bool_`
    AFLR3 prism layer option

..
    end-aflr3-blc

..
    start-aflr3-angqbf

*angqbf*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 max angle on surface triangles

..
    end-aflr3-angqbf

..
    start-aflr3-o

*o*: {``None``} | :class:`str`
    output file for AFLR3

..
    end-aflr3-o

..
    start-aflr3-mdf

*mdf*: ``1`` | {``2``}
    AFLR3 volume grid distribution flag

..
    end-aflr3-mdf

..
    start-aflr3-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether or not to run AFLR3

..
    end-aflr3-run

intersect Options
-----------------
..
    start-intersect-ascii

*ascii*: {``None``} | :class:`bool` | :class:`bool_`
    flag that input file is ASCII

..
    end-intersect-ascii

..
    start-intersect-v

*v*: {``False``} | :class:`bool` | :class:`bool_`
    verbose mode

..
    end-intersect-v

..
    start-intersect-t

*T*: {``False``} | :class:`bool` | :class:`bool_`
    option to also write Tecplot file ``Components.i.plt``

..
    end-intersect-t

..
    start-intersect-intersections

*intersections*: {``False``} | :class:`bool` | :class:`bool_`
    option to write intersections to ``intersect.dat``

..
    end-intersect-intersections

..
    start-intersect-i

*i*: {``'Components.tri'``} | :class:`str`
    input file to ``intersect``

..
    end-intersect-i

..
    start-intersect-overlap

*overlap*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    perform boolean intersection of this comp number

..
    end-intersect-overlap

..
    start-intersect-rm

*rm*: {``False``} | :class:`bool` | :class:`bool_`
    option to remove small triangles from results

..
    end-intersect-rm

..
    start-intersect-fast

*fast*: {``False``} | :class:`bool` | :class:`bool_`
    also write unformatted FAST file ``Components.i.fast``

..
    end-intersect-fast

..
    start-intersect-triged

*triged*: {``True``} | :class:`bool` | :class:`bool_`
    option to use CGT ``triged`` to clean output file

..
    end-intersect-triged

..
    start-intersect-smalltri

*smalltri*: {``0.0001``} | :class:`float` | :class:`float32`
    cutoff size for small triangles with *rm*

..
    end-intersect-smalltri

..
    start-intersect-o

*o*: {``'Components.i.tri'``} | :class:`str`
    output file for ``intersect``

..
    end-intersect-o

..
    start-intersect-cutout

*cutout*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of component to subtract

..
    end-intersect-cutout

..
    start-intersect-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program

..
    end-intersect-run

ulimit Options
--------------
..
    start-ulimit-q

*q*: {``819200``} | :class:`object`
    max bytes in POSIX message queues, ``ulimit -q``

..
    end-ulimit-q

..
    start-ulimit-r

*r*: {``0``} | :class:`object`
    max real-time scheduling priority, ``ulimit -r``

..
    end-ulimit-r

..
    start-ulimit-t

*t*: {``'unlimited'``} | :class:`object`
    max amount of cpu time in s, ``ulimit -t``

..
    end-ulimit-t

..
    start-ulimit-m

*m*: {``'unlimited'``} | :class:`object`
    max resident set size, ``ulimit -m``

..
    end-ulimit-m

..
    start-ulimit-c

*c*: {``0``} | :class:`object`
    core file size limit, ``ulimit -c``

..
    end-ulimit-c

..
    start-ulimit-p

*p*: {``8``} | :class:`object`
    pipe size in 512-byte blocks, ``ulimit -p``

..
    end-ulimit-p

..
    start-ulimit-l

*l*: {``64``} | :class:`object`
    max size that may be locked into memory, ``ulimit -l``

..
    end-ulimit-l

..
    start-ulimit-d

*d*: {``'unlimited'``} | :class:`object`
    process data segment limit, ``ulimit -d``

..
    end-ulimit-d

..
    start-ulimit-x

*x*: {``'unlimited'``} | :class:`object`
    max number of file locks, ``ulimit -x``

..
    end-ulimit-x

..
    start-ulimit-v

*v*: {``'unlimited'``} | :class:`object`
    max virtual memory avail to shell, ``ulimit -v``

..
    end-ulimit-v

..
    start-ulimit-u

*u*: {``127812``} | :class:`object`
    max number of procs avail to one user, ``ulimit -u``

..
    end-ulimit-u

..
    start-ulimit-i

*i*: {``127556``} | :class:`object`
    max number of pending signals, ``ulimit -i``

..
    end-ulimit-i

..
    start-ulimit-e

*e*: {``0``} | :class:`object`
    max scheduling priority, ``ulimit -e``

..
    end-ulimit-e

..
    start-ulimit-f

*f*: {``'unlimited'``} | :class:`object`
    max size of files written by shell, ``ulimit -f``

..
    end-ulimit-f

..
    start-ulimit-n

*n*: {``1024``} | :class:`object`
    max number of open files, ``ulimit -n``

..
    end-ulimit-n

..
    start-ulimit-s

*s*: {``4194304``} | :class:`object`
    stack size limit, ``ulimit -s``

..
    end-ulimit-s

verify Options
--------------
..
    start-verify-ascii

*ascii*: {``True``} | :class:`bool` | :class:`bool_`
    option for ASCII input file to ``verify``

..
    end-verify-ascii

..
    start-verify-i

*i*: {``None``} | :class:`str`
    input file for ``verify``

..
    end-verify-i

..
    start-verify-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program

..
    end-verify-run


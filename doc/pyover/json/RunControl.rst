
.. _pyover-json-runcontrol:

**************************
RunControl Section Options
**************************
The options below are the available options in the RunControl Section of the ``pyover.json`` control file

..
    start-RunControl-premesh

*PreMesh*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to generate volume mesh before submitting

..
    end-RunControl-premesh

..
    start-RunControl-mpi

*MPI*: {``True``} | :class:`bool` | :class:`bool_`
    whether or not to run MPI in phase

..
    end-RunControl-mpi

..
    start-RunControl-niter

*nIter*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of iterations to run in phase *j*

..
    end-RunControl-niter

..
    start-RunControl-resubmit

*Resubmit*: {``False``} | :class:`bool` | :class:`bool_`
    whether or not to submit new job at end of phase *j*

..
    end-RunControl-resubmit

..
    start-RunControl-aflr3

*aflr3*: {``None``} | :class:`object`
    value of option "aflr3"

..
    end-RunControl-aflr3

..
    start-RunControl-prefix

*Prefix*: {``'run'``} | :class:`str`
    project root name, or file prefix

..
    end-RunControl-prefix

..
    start-RunControl-overrun

*overrun*: {``None``} | :class:`object`
    value of option "overrun"

..
    end-RunControl-overrun

..
    start-RunControl-mpicmd

*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name

..
    end-RunControl-mpicmd

..
    start-RunControl-nproc

*nProc*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cores/threads to run

..
    end-RunControl-nproc

..
    start-RunControl-verbose

*Verbose*: {``False``} | :class:`bool` | :class:`bool_`
    "RunControl" verbosity flag

..
    end-RunControl-verbose

..
    start-RunControl-archive

*Archive*: {``None``} | :class:`object`
    value of option "Archive"

..
    end-RunControl-archive

..
    start-RunControl-warmstartfolder

*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file

..
    end-RunControl-warmstartfolder

..
    start-RunControl-environ

*Environ*: {``None``} | :class:`object`
    value of option "Environ"

..
    end-RunControl-environ

..
    start-RunControl-intersect

*intersect*: {``None``} | :class:`object`
    value of option "intersect"

..
    end-RunControl-intersect

..
    start-RunControl-slurm

*slurm*: {``False``} | :class:`bool` | :class:`bool_`
    wheter or not to submit jobs with Slurm

..
    end-RunControl-slurm

..
    start-RunControl-phasesequence

*PhaseSequence*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    list of phase indices to run

..
    end-RunControl-phasesequence

..
    start-RunControl-phaseiters

*PhaseIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    check-point iterations for phase *j*

..
    end-RunControl-phaseiters

..
    start-RunControl-warmstart

*WarmStart*: {``False``} | :class:`bool` | :class:`bool_`
    whether to warm start a case

..
    end-RunControl-warmstart

..
    start-RunControl-verify

*verify*: {``None``} | :class:`object`
    value of option "verify"

..
    end-RunControl-verify

..
    start-RunControl-ulimit

*ulimit*: {``None``} | :class:`object`
    value of option "ulimit"

..
    end-RunControl-ulimit

..
    start-RunControl-qsub

*qsub*: {``True``} | :class:`bool` | :class:`bool_`
    wheter or not to submit jobs with PBS

..
    end-RunControl-qsub

..
    start-RunControl-continue

*Continue*: {``True``} | :class:`bool` | :class:`bool_`
    whether restarts of same phase can use same job

..
    end-RunControl-continue

Archive Options
---------------
..
    start-Archive-preupdatefiles

*PreUpdateFiles*: {``[]``} | :class:`object`
    files to keep *n* and delete older, b4 archiving

..
    end-Archive-preupdatefiles

..
    start-Archive-skeletontailfiles

*SkeletonTailFiles*: {``[]``} | :class:`object`
    files to tail before deletion during skeleton

..
    end-Archive-skeletontailfiles

..
    start-Archive-predeletefiles

*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving

..
    end-Archive-predeletefiles

..
    start-Archive-postdeletefiles

*PostDeleteFiles*: {``[]``} | :class:`object`
    list of files to delete after archiving

..
    end-Archive-postdeletefiles

..
    start-Archive-archivefolder

*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root

..
    end-Archive-archivefolder

..
    start-Archive-archivetemplate

*ArchiveTemplate*: {``'full'``} | :class:`str`
    template for default archive settings

..
    end-Archive-archivetemplate

..
    start-Archive-progressdeletedirs

*ProgressDeleteDirs*: {``[]``} | :class:`object`
    folders to delete while still running

..
    end-Archive-progressdeletedirs

..
    start-Archive-archiveextension

*ArchiveExtension*: {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    archive file extension

..
    end-Archive-archiveextension

..
    start-Archive-remotecopy

*RemoteCopy*: {``'scp'``} | :class:`str`
    command for archive remote copies

..
    end-Archive-remotecopy

..
    start-Archive-archiveformat

*ArchiveFormat*: ``''`` | {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    format for case archives

..
    end-Archive-archiveformat

..
    start-Archive-progressdeletefiles

*ProgressDeleteFiles*: {``[]``} | :class:`object`
    files to delete while still running

..
    end-Archive-progressdeletefiles

..
    start-Archive-skeletondirs

*SkeletonDirs*: {``None``} | :class:`object`
    folders to **keep** during skeleton action

..
    end-Archive-skeletondirs

..
    start-Archive-postupdatefiles

*PostUpdateFiles*: {``[]``} | :class:`object`
    globs: keep *n* and rm older, after archiving

..
    end-Archive-postupdatefiles

..
    start-Archive-posttargroups

*PostTarGroups*: {``[]``} | :class:`object`
    groups of files to tar after archiving

..
    end-Archive-posttargroups

..
    start-Archive-pretardirs

*PreTarDirs*: {``[]``} | :class:`object`
    folders to tar before archiving

..
    end-Archive-pretardirs

..
    start-Archive-skeletonfiles

*SkeletonFiles*: {``['case.json']``} | :class:`object`
    files to **keep** during skeleton action

..
    end-Archive-skeletonfiles

..
    start-Archive-progresstardirs

*ProgressTarDirs*: {``[]``} | :class:`object`
    folders to tar while running

..
    end-Archive-progresstardirs

..
    start-Archive-progresstargroups

*ProgressTarGroups*: {``[]``} | :class:`object`
    list of file groups to tar while running

..
    end-Archive-progresstargroups

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
    start-Archive-progressupdatefiles

*ProgressUpdateFiles*: {``[]``} | :class:`object`
    files to delete old versions while running

..
    end-Archive-progressupdatefiles

..
    start-Archive-posttardirs

*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar after archiving

..
    end-Archive-posttardirs

..
    start-Archive-predeletedirs

*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving

..
    end-Archive-predeletedirs

..
    start-Archive-archiveaction

*ArchiveAction*: ``''`` | {``'archive'``} | ``'rm'`` | ``'skeleton'``
    action to take after finishing a case

..
    end-Archive-archiveaction

..
    start-Archive-pretargroups

*PreTarGroups*: {``[]``} | :class:`object`
    file groups to tar before archiving

..
    end-Archive-pretargroups

..
    start-Archive-progressarchivefiles

*ProgressArchiveFiles*: {``[]``} | :class:`object`
    files to archive at any time

..
    end-Archive-progressarchivefiles

..
    start-Archive-archivetype

*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files

..
    end-Archive-archivetype

..
    start-Archive-skeletontardirs

*SkeletonTarDirs*: {``[]``} | :class:`object`
    folders to tar before deletion during skeleton

..
    end-Archive-skeletontardirs

overrun Options
---------------
..
    start-overrun-nthreads

*nthreads*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of OpenMP threads

..
    end-overrun-nthreads

..
    start-overrun-args

*args*: {``''``} | :class:`str`
    string of extra args to ``overrun``

..
    end-overrun-args

..
    start-overrun-aux

*aux*: {``'"-v pcachem -- dplace -s1"'``} | :class:`str`
    auxiliary CLI args to ``overrun``

..
    end-overrun-aux

..
    start-overrun-cmd

*cmd*: {``'overrunmpi'``} | :class:`str`
    name of OVERFLOW executable to use

..
    end-overrun-cmd

Environ Options
---------------
aflr3 Options
-------------
..
    start-aflr3-blc

*blc*: {``None``} | :class:`bool` | :class:`bool_`
    AFLR3 prism layer option

..
    end-aflr3-blc

..
    start-aflr3-cdfr

*cdfr*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 max geometric growth rate

..
    end-aflr3-cdfr

..
    start-aflr3-bli

*bli*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of AFLR3 prism layers

..
    end-aflr3-bli

..
    start-aflr3-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether or not to run AFLR3

..
    end-aflr3-run

..
    start-aflr3-grow

*grow*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 off-body growth rate

..
    end-aflr3-grow

..
    start-aflr3-o

*o*: {``None``} | :class:`str`
    output file for AFLR3

..
    end-aflr3-o

..
    start-aflr3-cdfs

*cdfs*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64`
    AFLR3 geometric growth exclusion zone size

..
    end-aflr3-cdfs

..
    start-aflr3-keys

*keys*: {``{}``} | :class:`dict`
    AFLR3 options using ``key=val`` format

..
    end-aflr3-keys

..
    start-aflr3-angqbf

*angqbf*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 max angle on surface triangles

..
    end-aflr3-angqbf

..
    start-aflr3-mdf

*mdf*: ``1`` | {``2``}
    AFLR3 volume grid distribution flag

..
    end-aflr3-mdf

..
    start-aflr3-blds

*blds*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 initial boundary-layer spacing

..
    end-aflr3-blds

..
    start-aflr3-flags

*flags*: {``{}``} | :class:`dict`
    AFLR3 options using ``-flag val`` format

..
    end-aflr3-flags

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
    start-aflr3-blr

*blr*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 boundary layer stretching ratio

..
    end-aflr3-blr

..
    start-aflr3-i

*i*: {``None``} | :class:`str`
    input file for AFLR3

..
    end-aflr3-i

..
    start-aflr3-nqual

*nqual*: {``0``} | :class:`int` | :class:`int32` | :class:`int64`
    number of AFLR3 mesh quality passes

..
    end-aflr3-nqual

..
    start-aflr3-angblisimx

*angblisimx*: {``None``} | :class:`float` | :class:`float32`
    AFLR3 max angle b/w BL intersecting faces

..
    end-aflr3-angblisimx

intersect Options
-----------------
..
    start-intersect-v

*v*: {``False``} | :class:`bool` | :class:`bool_`
    verbose mode

..
    end-intersect-v

..
    start-intersect-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program

..
    end-intersect-run

..
    start-intersect-triged

*triged*: {``True``} | :class:`bool` | :class:`bool_`
    option to use CGT ``triged`` to clean output file

..
    end-intersect-triged

..
    start-intersect-overlap

*overlap*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    perform boolean intersection of this comp number

..
    end-intersect-overlap

..
    start-intersect-ascii

*ascii*: {``None``} | :class:`bool` | :class:`bool_`
    flag that input file is ASCII

..
    end-intersect-ascii

..
    start-intersect-fast

*fast*: {``False``} | :class:`bool` | :class:`bool_`
    also write unformatted FAST file ``Components.i.fast``

..
    end-intersect-fast

..
    start-intersect-o

*o*: {``'Components.i.tri'``} | :class:`str`
    output file for ``intersect``

..
    end-intersect-o

..
    start-intersect-intersections

*intersections*: {``False``} | :class:`bool` | :class:`bool_`
    option to write intersections to ``intersect.dat``

..
    end-intersect-intersections

..
    start-intersect-smalltri

*smalltri*: {``0.0001``} | :class:`float` | :class:`float32`
    cutoff size for small triangles with *rm*

..
    end-intersect-smalltri

..
    start-intersect-t

*T*: {``False``} | :class:`bool` | :class:`bool_`
    option to also write Tecplot file ``Components.i.plt``

..
    end-intersect-t

..
    start-intersect-i

*i*: {``'Components.tri'``} | :class:`str`
    input file to ``intersect``

..
    end-intersect-i

..
    start-intersect-rm

*rm*: {``False``} | :class:`bool` | :class:`bool_`
    option to remove small triangles from results

..
    end-intersect-rm

..
    start-intersect-cutout

*cutout*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of component to subtract

..
    end-intersect-cutout

ulimit Options
--------------
..
    start-ulimit-v

*v*: {``'unlimited'``} | :class:`object`
    max virtual memory avail to shell, ``ulimit -v``

..
    end-ulimit-v

..
    start-ulimit-q

*q*: {``819200``} | :class:`object`
    max bytes in POSIX message queues, ``ulimit -q``

..
    end-ulimit-q

..
    start-ulimit-p

*p*: {``8``} | :class:`object`
    pipe size in 512-byte blocks, ``ulimit -p``

..
    end-ulimit-p

..
    start-ulimit-m

*m*: {``'unlimited'``} | :class:`object`
    max resident set size, ``ulimit -m``

..
    end-ulimit-m

..
    start-ulimit-f

*f*: {``'unlimited'``} | :class:`object`
    max size of files written by shell, ``ulimit -f``

..
    end-ulimit-f

..
    start-ulimit-t

*t*: {``'unlimited'``} | :class:`object`
    max amount of cpu time in s, ``ulimit -t``

..
    end-ulimit-t

..
    start-ulimit-l

*l*: {``64``} | :class:`object`
    max size that may be locked into memory, ``ulimit -l``

..
    end-ulimit-l

..
    start-ulimit-u

*u*: {``127812``} | :class:`object`
    max number of procs avail to one user, ``ulimit -u``

..
    end-ulimit-u

..
    start-ulimit-s

*s*: {``4194304``} | :class:`object`
    stack size limit, ``ulimit -s``

..
    end-ulimit-s

..
    start-ulimit-x

*x*: {``'unlimited'``} | :class:`object`
    max number of file locks, ``ulimit -x``

..
    end-ulimit-x

..
    start-ulimit-r

*r*: {``0``} | :class:`object`
    max real-time scheduling priority, ``ulimit -r``

..
    end-ulimit-r

..
    start-ulimit-d

*d*: {``'unlimited'``} | :class:`object`
    process data segment limit, ``ulimit -d``

..
    end-ulimit-d

..
    start-ulimit-n

*n*: {``1024``} | :class:`object`
    max number of open files, ``ulimit -n``

..
    end-ulimit-n

..
    start-ulimit-e

*e*: {``0``} | :class:`object`
    max scheduling priority, ``ulimit -e``

..
    end-ulimit-e

..
    start-ulimit-i

*i*: {``127556``} | :class:`object`
    max number of pending signals, ``ulimit -i``

..
    end-ulimit-i

..
    start-ulimit-c

*c*: {``0``} | :class:`object`
    core file size limit, ``ulimit -c``

..
    end-ulimit-c

verify Options
--------------
..
    start-verify-run

*run*: {``None``} | :class:`bool` | :class:`bool_`
    whether to execute program

..
    end-verify-run

..
    start-verify-i

*i*: {``None``} | :class:`str`
    input file for ``verify``

..
    end-verify-i

..
    start-verify-ascii

*ascii*: {``True``} | :class:`bool` | :class:`bool_`
    option for ASCII input file to ``verify``

..
    end-verify-ascii


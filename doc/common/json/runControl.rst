
.. _cape-json-RunControl:

--------------------
Run Control Settings
--------------------

The ``"RunControl"`` section of a Cape JSON file contains basic settings for the
number of iterations to run of a particular solver and the manner in which to
run it.  The entire contents of this section, with all default options expanded
where applicable, is written to a file called :file:``case.json`` in the folder
created for each case.

The derivative modules (i.e. pyCart, pyOver, etc.) contain additional sections
that specify command-line options for particular binaries.  For example, this
section in :file:`pyCart.json` contains a subsection called ``"flowCart"``, and
it also contains subsections for the various volume meshing commands associated
with Cart3D.  Finally, the :ref:`archiving section <cape-json-Archive>` is
described separately.

The universal options for this section have the following JSON syntax.


    .. code-block:: javascript
    
        "RunControl": {
            // Phase description
            "PhaseSequence": [0, 1],
            "PhaseIters": [200, 400]
            // Job type
            "MPI": false,
            "qsub": false,
            "sbatch": false,
            "Resubmit": false,
            "Continue": true,
            "nProc": 1,
            // Environment variable settings
            "Environ": { },
            // System-wide resource settings
            "ulimit": {
                "s": 4194304,
                "c": 0,
                "d": "unlimited",
                "f": "unlimited",
                "l": 64,
                "n": 1024,
                "t": "unlimited",
                "u": 127812,
                "v": "unlimited"
            },
            // Surface mesh preparation
            "verify": false,
            "intersect": { },
            // Archive settings
            "Archive": {
                "ArchiveFolder": "",
                "ArchiveFormat": "tar",
                "ArchiveType": "full",
                "ArchiveTemplate": "viz",
                "RemoteCopy": "scp",
                "PreDeleteFiles": [],
                "PreDeleteDirs": [],
                "PreUpdateFiles": [],
                "PreArchiveGroups": [],
                "PreArchiveDirs": [],
                "PostDeleteFiles": [],
                "PostDeleteDirs": [],
                "PostUpdateFiles": [],
                "PostArchiveGroups": [],
                "PostArchiveDirs": []
            }
        }
        
Links to additional options for each specific solver are found below.

    * :ref:`Cart3D <pycart-json-RunControl>`
    * :ref:`FUN3D <pyfun-json-RunControl>`
    * :ref:`OVERFLOW <pyover-json-RunControl>`
        
.. _cape-json-PhaseControl:

Phase Control Options
=====================

Two key options define the critical aspect of how many options to run whichever
solver is being run.  These are the *PhaseSequence* and *PhaseIters* settings.
Cape modules use a concept called a "phase," which refers to running the solver
with one set of inputs.  In many occasions it is necessary or at least practical
to run a CFD solver in multiple phases.

The phases are numbered from ``0``, since that is the Python standard.  Almost
every Cape setting can be defined as a list.  For example if *nProc* is 
``[8, 24, 120]``, then phase 0 runs with 8 processors, phase 1 runs with 24, and
phase 2 runs with 120.  If there is a phase 3, it will also use 120 processors.
It is possible to define a phase and then not use it, for example by defining
*PhaseSequence* as ``[0, 1, 3]``.

Finally, each phase is run until the total number of iterations is at least the
value of the corresponding entry of *PhaseIters*.

The list of these options and their available options is shown below.

    *PhaseSequence*: {``[0]``} | ``[0, 1]`` | :class:`list` (:class:`int`)
        List of input phases to use and order in which to use them.  These
        do not have to be in numerical order.  For example, ``[0, 1, 3]`` is an
        allowable sequence that may become useful if it's later determined that
        phase 2 is unnecessary or unproductive.
        
    *PhaseIters*: {``[200]``} | :class:`int` | :class:`list` (:class:`int`)
        List of minimum number of total iterations after each run.  This list
        must have the same length as *PhaseIters*, and a ``0`` tells Cape to
        continue to the next run regardless of the current iteration count.

.. _cape-json-job-type:
        
Job Type Options
================

Several options are system-oriented and applicable to almost any solver. 
        
The *Resubmit* option for phase *i* controls whether or not the PBS job has to
end and be resubmitted as a new job before starting phase *i+1*.  If one run of
phase *i* ends without reaching the required minimum iteration count, it is
restarted, and the *Continue* setting determines whether or not this is run
within the same PBS job.

The dictionary of options is explained below.

    *MPI*: ``true`` | {``false``} | :class:`list` (:class:`bool`)
        Whether or not to use MPI version of solver (if applicable)
        
    *qsub*: {``true``} | ``false`` | :class:`list` (:class:`bool`)
        Whether or not to submit job PBS queue
        
    *sbatch*: ``true`` | {``false``} | :class:`list` (:class:`bool`)
        Whether or not to submit job to SLURM queue
        
    *Resubmit*: ``true`` | {``false``} | :class:`list` (:class:`bool`)
        Whether or not to terminate a job and resubmit new one between phases
        
    *Continue*: {``true``} | ``false`` | :class:`list` (:class:`bool`)
        Whether or not to continue a job when rerunning a phase
        
    *PreMesh*: ``true`` | {``false``}
        Whether or not to create mesh before submitting a job (e.g. ``cubes``)
        
    *mpicmd*: {``"mpiexec"``} | ``"mpirun"`` | :class:`str`
        Name of shell command to invoke MPI
        
    *nProc*: {``12``} | :class:`int` | :class:`list` (:class:`int`)
        Number of processors (for each phase)

        
.. _cape-json-Environ:

Environment Variables
=====================

The ``"Environ"`` section allows the user to specify a dictionary of
environment variables and values with which to set them.  This is a simple
section with very flexible syntax.  The following example is better than an
options dictionary.

    .. code-block:: javascript
    
        "Environ": {
            "CFDPATH": "~/usr/CFD/bin",
            "NUMTHREADS": 10,
            "PATH": "+~/usr/CFD/bin"
        }

If the environment variable value starts with ``"+"``, the value is appended to
the existing environment variable.  Also, numbers can be specified, but they
are converted to strings before being applied.  The environment variables are
set using the built-in ``os.environ``.


.. _cape-json-ulimit:

System Resources
================

The ``"ulimit"`` subsection alters certain system resources. Methods are
provided for all of the inputs to the system command ``ulimit``, but currently
only the settings that can be altered through the standard Python module
:mod:`resource` have an effect. The dictionary of options, their default value,
and the possible values is given below.

    *s*: {``4194304``} | nonnegative :class:`int` | ``"unlimited"``
        Stack size in kbytes, important parameter for many solvers
        
    *c*: {``0``} | nonnegative :class:`int` | ``"unlimited"``
        Maximum core size in blocks
        
    *d*: nonnegative :class:`int` | {``"unlimited"``}
        Maximum data segment size in kbytes

    *f*: nonnegative :class:`int` | {``"unlimited"``}
        Maximum file size in blocks
        
    *l*: {``64``} | nonnegative :class:`int` | ``"unlimited"``
        Maximum locked memory in kbytes
        
    *n*: {``1024``} | nonnegative :class:`int` | ``"unlimited"``
        Maximum number of open files
        
    *t*: nonnegative :class:`int` | {``"unlimited"``}
        Maximum time for a process in seconds
        
    *u*: nonnegative :class:`int` | {``"unlimited"``}
        Maximum number of processes launched
        
    *v*: nonnegative :class:`int` | {``"unlimited"``}
        Maximum virtual memory in kilobytes
        
.. _cape-json-intersect:

Surface Triangulation Intersection
====================================

The ``"intersect"`` subsection alters settings for Boolean operations on
surface triangulations (used in coordination with either ``cubes`` for Cart3D
or ``aflr3`` for any unstructured solver).  For the most part, making this a
nonempty value (either a :class:`dict` with at least one setting or ``True``)
will instruct the various CAPE modules to perform intersection.  This is often
done in a special phase 0 with no iterations so that the intersection (and
possibly mesh generation) can be performed in a sort of preprocessing job with
only one node before submitting the compute job as a subsequent multi-node job.

    *i*: :class:`str`
        Name of the input triangulation, usually automatically determined by
        the appropriate module; for example this is ``Components.tri`` for
        pyCart jobs
        
    *o*: :class:`str`
        Name of the output file, usually automatically determined by the
        appropriate module; for example this is ``Components.o.tri`` for pyCart
        jobs.  This file needs to have the original component IDs mapped back
        onto it
        
    *ascii*: {``true``} | ``false``
        Create ASCII output triangulation
        
    *cutout*: ``true`` | {``false``}
        Perform subtraction operations instead of unions
        
    *intersect_rm*: ``true`` | {``false``}
        Option to remove small triangles
        
    *intersect_smalltri*: {``1e-4``} | :class:`float` > 0
        Cutoff size if using *rm* option


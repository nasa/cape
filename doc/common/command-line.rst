
.. _cli:

Command-Line Interface
======================

The command-line interface is accessed through three scripts (one for each
CFD solver), ``pycart``, ``pyfun``, and ``pyover``.  These three scripts are
simple Python scripts located in ``$PYCART/bin``.

Command-line options are interpreted by :func:`cape.argread.readkeys`. Multiple
letters following a single hyphen, such as ``pycart -cf``, is interpreted as
two options, equivalent to ``pycart -c -f``. Multiple characters following two
hyphens, such as ``pycart --help`` are interpreted as a single option with a
longer name. 
        
The master input file for one of these scripts is a JSON file that is usually
identified with the ``-f`` flag.  For example

    .. code-block:: console
    
        $ pycart -f run/poweroff.json
        
will use the file :file:`run/poweroff.json` for its settings.  Each call to one
of these scripts interprets the current directory, i.e. ``$PWD``, as the home
directory for the analysis.  In the example above, the home directory is
``$PWD``, not ``$PWD/run``.


.. _cli-commands:

Basic Commands
--------------
Options to ``pycart`` and the other scripts can be grouped into commands,
modifiers, and subsetting options.  The subsetting options are used to identify
a subset of the full run matrix, and the command options identify an action for
``pycart`` to take.  Modifiers provide a small set of options for the the major
commands.  The following section describes commands using a :ref:`simple
example <pycart-ex-bullet>` from pyCart.

Furthermore, these commands are listed in order of precedence.  If a user
specifies more than one of these options, only one action will actually be
taken, and that order of precedence is demonstrated in the table below.

    +--------------------------+------------------------------------------+
    | Command                  | Description                              |
    +==========================+==========================================+
    | ``pycart -h``            | Display help message and quit            |
    +--------------------------+------------------------------------------+
    | ``pycart --help``        | Display help message and quit            |
    +--------------------------+------------------------------------------+
    | ``pycart -c``            | Check and display status                 |
    +--------------------------+------------------------------------------+
    | ``pycart  --batch``      | Run command as a PBS job; this submits a |
    |                          | PBS job that runs the same command with  |
    |                          | the ``--batch`` flag removed             |
    +--------------------------+------------------------------------------+
    | ``pycart --aero``        | Create/update force & moment databook    |
    +--------------------------+------------------------------------------+
    | ``pycart --pt $PT``      | Update point sensor data book for point  |
    |                          | sensor group *PT*; if *PT* is not        |
    |                          | given, update all point sensor groups    |
    +--------------------------+------------------------------------------+
    | ``pycart --ll $COMP``    | Update line load data book for component |
    |                          | *COMP*; if *COMP* is not given, update   |
    |                          | all line load data books                 |
    +--------------------------+------------------------------------------+
    | ``pycart --triqfm $COMP``| Update patch load database for component |
    |                          | *COMP*; if *COMP* is not given, update   |
    |                          | all TriqFM data books                    |
    +--------------------------+------------------------------------------+
    | ``pycart --archive``     | Create archive of solution in an outside |
    |                          | folder and clean up the working folder   |
    +--------------------------+------------------------------------------+
    | ``pycart --clean``       | Delete files according to the settings   |
    |                          | from *ProgressDeleteFiles* even if case  |
    |                          | is not completed                         |
    +--------------------------+------------------------------------------+
    | ``pycart --unarchive``   | Copy files from archive to working       |
    |                          | directory                                |
    +--------------------------+------------------------------------------+
    | ``pycart --skeleton``    | Clean even more files after archiving    |
    +--------------------------+------------------------------------------+
    | ``pycart --report $REP`` | Create/update a report called *REP*; if  |
    |                          | *REP* is omitted, create the first       |
    |                          | report available                         |
    +--------------------------+------------------------------------------+
    | ``pycart --qdel``        | Kill PBS jobs                            |
    +--------------------------+------------------------------------------+
    | ``pycart --kill``        | Kill PBS jobs                            |
    +--------------------------+------------------------------------------+
    | ``pycart -e $CMD``       | Run command *CMD* in each case folder    |
    +--------------------------+------------------------------------------+
    | ``pycart --exec $CMD``   | Run command *CMD* in each case folder    |
    +--------------------------+------------------------------------------+
    | ``pycart -n $N``         | Submit or run up to *N* jobs             |
    +--------------------------+------------------------------------------+
    | ``pycart --extend``      | Extend the last phase by the number of   |
    |                          | steps that phase would run for           |
    +--------------------------+------------------------------------------+
    | ``pycart --apply``       | Rewrite input files and PBS files using  |
    |                          | appropriate JSON settings; can also be   |
    |                          | used to add one or more phases           |
    +--------------------------+------------------------------------------+
    | ``pycart``               | Running with no command is equivalent to |
    |                          | ``pycart -n 10``                         |
    +--------------------------+------------------------------------------+
    
.. _cli-h:

Help Message
************
To see of the options from the command line, just run ``pycart
-h`` or ``pycart --help``.

    .. code-block:: console
    
        $ pycart -h
        
        Python interface for Cart3D: pycart

        This function provides a master interface for pyCart. The functionality from
        this script is also accessible from the 'pyCart' module using relatively
        simple commands.
        
        USAGE
        
            $ pycart [options]
            
        EXAMPLES
        
            The basic call submits all jobs prescribed in the file 'pyCart.json'
                    
                $ pycart
                
            This command uses the inputs from 'poweron.json' and only displays
            statuses.  No jobs are submitted.
                
                $ pycart -f poweron.json -c
                
         ...

        
The help messages are printed as raw text on the command line in such a format
that is interpreted as 
`reStructuredText <http://docutils.sourceforge.net/rst.html>`_  at
:mod:`pc_doc` (for ``pycart``), :mod:`pf_doc` (for ``pyfun``), and
:mod:`po_doc` (for ``pyover``).
        

.. _cli-c:

Status Check
************
The first command is the help command, which was discussed previously.  The
second command is to check the status, which may show something like the
following.

    .. code-block:: console
    
        $ pycart -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 DONE    200/200     .   5.0
        1    poweroff/m2.0a0.0b0.0 INCOMP  100/200     .   2.2
        2    poweroff/m2.0a2.0b0.0 RUN     157/200     Q   3.1   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
                
        ---=1, INCOMP=1, RUN=1, DONE=1,
        
The status check will show one of eight statuses for each case in the run
matrix or :ref:`subset <cli-subset>`, and these are explained in the table
below.

    +------------+------------------------------------------------------+
    | Status     | Description                                          |
    +============+======================================================+
    | ``---``    | Folder does not yet exist or does not have all files |
    +------------+------------------------------------------------------+
    | ``INCOMP`` | Case is not running, not submitted in the queue, and |
    |            | has not reached the requested number of iterations   |
    +------------+------------------------------------------------------+
    | ``QUEUE``  | Case is not running but is present in the PBS queue  |
    +------------+------------------------------------------------------+
    | ``RUN``    | Case is currently running, locally or as PBS job     |
    +------------+------------------------------------------------------+
    | ``ZOMBIE`` | Case appears to be running, but no files have        |
    |            | changed in the last 30 minutes                       |
    +------------+------------------------------------------------------+
    | ``ERROR``  | Case has been marked as erroneous for some reason    |
    +------------+------------------------------------------------------+
    | ``DONE``   | Case is not running and has reached requested number |
    |            | of iterations                                        |
    +------------+------------------------------------------------------+
    | ``PASS``   | Case meets criteria for ``DONE`` and has been marked |
    |            | as acceptable with a ``p`` in the run matrix file    |
    +------------+------------------------------------------------------+
    | ``PASS*``  | Case was marked ``PASS`` by user but does not meet   |
    |            | criteria for ``DONE``                                |
    +------------+------------------------------------------------------+

This command also has a modifier ``-j`` that will also show the PBS job number
as another column in the status message.  This can be written shorthand as
``pycart -cj``.  Also, the ``-j`` modifier extends to any other command that
shows the status message above as part of its process (e.g. submitting jobs).

    
.. _cli-batch:

Batch Commands
**************
The ``--batch`` option is a special command that creates a PBS job that runs
what is otherwise the same command.  This creates a simple PBS job in the
folder ``batch-pbs``.  As an exception, the ``-h`` and ``-c`` options do not
require much work, and so they can not be submitted as batch jobs.
    

.. _cli-aero:

Data Book Updates
*****************
The ``--aero`` flag creates or updates the force and moment database.  The list
of components included in the database, along with other defining options, are
specified in the ``"DataBook"`` section of the input JSON file.  When a user 
runs this command, each case (:ref:`subsetting commands <cli-subset>` still
apply) is shown along with a brief message of the status of the database and
action of this update.

    .. code-block:: console
    
        $ pycart --aero
        poweroff/m0.84a0.0b0.0
          Adding new databook entry at iteration 300.
        poweroff/m0.84a2.0b0.0
          Not enough iterations (100) for analysis.
        poweroff/m0.88a0.0b0.0
        poweroff/m0.88a2.0b0.0
          Updating from iteration 200 to 300.
        poweroff/m0.88a4.0b0.0
          Databook up to date.

The point sensor data book (``--pt``) and line load data book (``--ll``) work
in the same way.  The exception is that these two extended data book examples
can be commanded to update only one component of the data book.  Suppose there
is a setup with a center, left, and right component.  Then

    .. code-block:: console
    
        $ pycart --ll LL_C
        
only updates the data book for line loads on the center component (assuming
there is a line load component called ``LL_C`` for the center body defined in
the ``"DataBook"`` section of the JSON file).  Meanwhile, the command ``pycart
--ll`` will update the data book for all three components.

Also, these commands will both run the necessary post-processing and collect
the results into the data book.  For instance, it will run ``triloadCmd`` to
compute line loads unless they already exist.


.. _cli-archive:

Archiving Solutions
*******************
Once a case has been completed and marked as ``PASS``, the ``pycart --archive``
command can be issued to perform several tasks.

    #. Delete any files that are not deemed necessary to be saved
    #. Archive folders or other groups into tar balls to reduce file count
    #. Copy reduced set of files to external location for backup
    #. Clean up the working folder
    
All of these steps can be heavily customized using the options in the
``"Archive"`` subsection of the ``"RunControl"`` section of the JSON file.
There are two main modes of archive generation.  The first mode is to archive
the entire case as an entire folder, which leads to a file such as
``m2.50a2.0.tar`` in the archive.  The second mode is to copy several files
within the folder.  Using the second option, files can either be saved
individually or in groups as tar balls.  Which mode is selected is based on the
*ArchiveType* option; the second mode is selected unless *ArchiveType* is
``"full"``.

In both modes, there are three different opportunities to delete files:

    #. Delete files as the cases is running, using *ProgressDeleteFiles*
    #. Delete files after completion but before archiving, *PreDeleteFiles*
    #. Delete files after creating archive, *PostDeleteFiles*
    
These options can be set to keep around the *n* most recent files meeting a
particular file glob; for example

    .. code-block:: javascript
    
        "ProgressDeleteFiles": [
            {"q.[0-9]*": 2},
            {"x.[0-9]*": 2}
        ]
        
This tells pyOver to keep the 2 most recent ``q.$N`` and the two most recent
``x.$N`` files; this can be very useful in keeping down hard drive usage for
large runs.


.. _cli-clean:

Trimming Excess Files While Running
***********************************
The ``--clean`` command can be applied even if the case is not marked ``PASS``
or even it is currently running.  It applies the *ProgressDeleteFiles* and
*ProgressDeleteFolders* options from the *RunControl>Archive* section of the
JSON file.

Note that inappropriate settings for these two options may cause pyCart to
delete files needed for running!


.. _cli-report:

Creating Reports
****************
Creating :ref:`automated reports <report>` is done using the command ``pycart
--report``.  In the ``"Report"`` section of the JSON file, a number of reports
can be defined.  A sample section of the JSON file looks like the following:

    .. code-block:: javascript
    
        "Report": {
            // List of reports
            "Reports": ["case", "sweep", "flow"],
            // Definitions for each report
            "cases": {
                ...
            },
            "sweep": {
                ...
            },
            "flow": {
                ...
            },
            // Definitions for each figure
            "Figures": {
                ...
            },
            // Definitions for each subfigure
            "Subfigures": {
                ...
            }
        }

In this case, the following commands are possible.

    .. code-block:: console
    
        $ pycart --report
        $ pycart --report case
        $ pycart --report sweep
        $ pycart --report flow
        
The first two of these commands are equivalent since not naming the report
explicitly in the command defaults to the first report in the ``"Reports"``
list.

Each command creates a file ``report/report-$REP.pdf``.  Further, subsequent
calls to ``pycart --report`` only updates each figure of each case if needed.
This means that figures are only updated when more iterations have been run
since the last time the figure was created or the settings for that figure have
changed.


.. _cli-qdel:

Killing Jobs
************
Occasionally it is necessary to stop a number of PBS jobs that have either been
submitted to the queue or are currently running.  Rather than manually killing
each job (using the ``qdel`` command, for instance), pyCart provides a command
to delete PBS jobs for an arbitrarily-sized subset of the run matrix.  The
following two commands are equivalent.

    .. code-block:: console
    
        $ pycart --qdel
        $ pycart --kill


.. _cli-e:

Execute a Script in Each Case Folder
************************************
The command ``pycart -e $CMD`` can be used to execute a command in each folder
in a run matrix.  This can be something simple, such as ``pycart -e ls`` to
list the files in each folder.

    .. code-block:: console
    
        $ pycart -e ls
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 RUN     130/200     .        0.0 
          Executing system command:
            ls
        body.dat	    Components.i.tri  forces.dat      input.cntl     preSpec.c3d.cntl
        bullet_no_base.dat  conditions.json   functional.dat  Mesh.c3d.Info  run_cart3d.pbs
        bullet_total.dat    Config.xml	      history.dat     Mesh.mg.c3d    RUNNING
        cap.dat		    entire.dat	      input.00.cntl   Mesh.R.c3d
        case.json	    flowCart.out      input.c3d       moments.dat
            exit(0)
        1    poweroff/m2.0a0.0b0.0 DONE    200/200     .            
          Executing system command:
            ls
        body.dat	    clic.dat		cutPlanes.dat	input.cntl	  run.00.200
        bullet_no_base.dat  clic.i.dat		entire.dat	loadsCC.dat	  run_cart3d.pbs
        bullet_total.dat    clic.w.dat		export.png	macro.xml	  test.pdf
        cap.dat		    Components.i.tri	export.py	marco.py	  test.png
        case.json	    Components.i.triq	forces.dat	Mesh.c3d.Info	  test.pvcc
        check.00200	    Components.w2.triq	functional.dat	Mesh.mg.c3d	  test.pvd
        checkDT.00200	    Components.w.triq	history.dat	Mesh.R.c3d	  test.pvsm
        clic		    conditions.json	input.00.cntl	moments.dat
        clic.cntl	    Config.xml		input.c3d	preSpec.c3d.cntl
            exit(0)
        2    poweroff/m2.0a2.0b0.0 ---     /           .            
        3    poweroff/m2.0a2.0b2.0 ---     /           .            
        
        ---=2, RUN=1, DONE=1,
    
It can also be used to execute a specific script, for example

    .. code-block:: console
    
        $ pycart -e tools/fixProblem.py
        
This command copies the script ``tools/fixProblem.py`` to each case folder and
then executes ``./fixProblem.py`` in each folder.  The commands ``pycart -e``
and ``pycart --exec`` are equivalent.


.. _cli-extend:

Extending a Case to Repeat the Last Phase
*****************************************
Often a case does not look like it has fully converged after inspecting the
report.  The command

    .. code-block:: console
    
        $ pycart --extend [OPTIONS]
        
goes into each folder, determines the last phase number and how many iterations
it normally runs for, and extends the case by that many iterations.  The
primary effect of this command is to edit the last entry of *PhaseIters* in the
:file:`case.json` file.

pyCart is fairly diligent at determining the appropriate number of iterations
for which to extend the case.  For example, for OVERFLOW runs it will determine
the last namelist and determine the extension amount from the *GLOBAL>NSTEPS*
option in that namelist.

This command comes with two additional options: ``--restart`` and ``--imax
$N``.  The ``--imax $N`` option prevents pyCart from extending the maximum
iteration count beyond *N*.

Furthermore, the default behavior of the ``--extend`` command is modify the
iteration count and then leave the case in an ``INCOMP`` status, which can feel
a little disappointing.  (The reason is that the user may want to change the
status of many cases, and this prevents restarting a surprising load of cases.)
Adding either ``--qsub`` or ``--restart`` tells pyCart to submit or restart the
case after extending it.


.. _cli-apply:

Apply New Settings and/or Add Phases
************************************
The command ``pycart --apply`` will try to apply all the settings from the
current JSON file to each case that meets the subsetting criteria associated
with the command.  There are usually three steps to this process:

    * Rewrite all input files, such as :file:`input.cntl` for Cart3D or
      namelist files for OVERFLOW and FUN3D. 
    * Rewrite the PBS scripts
    * Rewrite ``case.json`` with new *RunControl* settings
    
Each of these steps may be affected by the presence of additional phases in the
JSON file.  For example, if the case in its present setup has three phases, but
:file:`pyOver.json` has five phases, the ``pyover --extend`` command will write
two additional input namelists called :file:`run.03.inp` and
:file:`run.04.inp`.

The ``--qsub`` option also works with this command.

.. _cli-n:

Submitting or Running Jobs
***************************
Submitting jobs is usually done with the ``-n`` flag.  If *no* other commands
are issued, it is the same as using ``-n 10``.  Only jobs with a status of
``---`` or ``INCOMP`` will be run or submitted.

If a JSON file has the setting ``"qsub": true`` in the ``"RunControl"`` section
of the JSON file, it is submitted as a PBS job. Otherwise, it is run within the
same shell that issued the ``pycart`` command.  An exception to this is when
the ``-n`` command is issued in combination with ``--batch``.  Running as a
``--batch`` PBS job overrides ``"qsub"`` and runs one or more jobs within the
batch job.

This command accepts several modifiers:

    +-------------------+---------------------------------------------+
    | Modifier          | Description                                 |
    +===================+=============================================+
    | ``--no-restart``  | Only setup and/or run and submit new cases; |
    |                   | do not submit ``INCOMP`` jobs that have     |
    |                   | iterations remaining                        |
    +-------------------+---------------------------------------------+
    | ``--no-start``    | Set up new cases but do not actually start  |
    |                   | simulation or submit job                    |
    +-------------------+---------------------------------------------+
    | ``--no-qsub``     | Keep all tasks local and do not submt PBS   |
    |                   | job(s)                                      |
    +-------------------+---------------------------------------------+


.. _cli-modification:

Command Modifiers
-----------------
Several options are used to specify additional settings to some or all
commands.  The most important of these is the ``-f`` flag, which tells
``pycart``, ``pyfun``, or ``pyover`` to use a specific JSON input file.

    .. code-block:: console
    
        $ pycart -f run/poweroff.json -c
        
If the ``-f`` flag is not used, a default file name of either
:file:`pyCart.json`, :file:`pyFun.json`, or :file:`pyOver.json` is assumed.
Pointing to specific JSON files using the ``-f`` flag is useful for run
folders that have two or more related CFD configurations.

Another modifier option is ``-j``, which adds a column showing the job number
to the stats message printout.  Another modifier option that affects the status
message is the ``-u $USER`` flag, which instructs pyCart to check the queue for
jobs submitted by a different user.  The default is to use the username of the
user issuing the command, but the ``-u`` flag can be used to check the status
of another user's jobs.  Somewhat related is the ``-q $QUE`` flag, which allows
the user to override the queue setting in the JSON file and submit new jobs to
a specific queue.

The ``--qsub`` option modifies some post-processing commands to be submitted as
PBS jobs in each folder (instead of being run in the current terminal). 
Arguably a command in its own right, the ``--batch`` modifier creates a PBS job
that runs the same command (except with the ``--batch`` flag removed).

The following table lists all modifiers and the commands that are affected by
them.

    +---------------+---------------------------------+-------------------+
    | Modifier      | Description                     | Affected commands |
    +===============+=================================+===================+
    | ``-f $FNAME`` | Use JSON file called *FNAME*    | All except ``-h`` |
    +---------------+---------------------------------+-------------------+
    | ``-j``        | Print PBS job numbers if run    | ``-c``, ``-e``,   |
    |               | job is present in the queue     | ``-n``            |
    +---------------+---------------------------------+-------------------+
    | ``-q $QUE``   | Submit new jobs to PBS queue    | ``-n``            |
    |               | called *QUE* and ignore ``"q"`` |                   |
    |               | option in ``"PBS"`` section     |                   |
    +---------------+---------------------------------+-------------------+
    | ``-u $USER``  | Check PBS queue for jobs        | ``-c``, ``-e``,   |
    |               | submitted by user *USER*        | ``-n``            |
    +---------------+---------------------------------+-------------------+
    | ``--delete``  | Remove subset of cases from     | ``--aero``,       |
    |               | a data book                     | ``--ll``,         |
    |               |                                 | ``--triqfm``      |
    +---------------+---------------------------------+-------------------+
    | ``--imax $N`` | Do not exceed iteration *N*     | ``--extend``      |
    +---------------+---------------------------------+-------------------+
    | ``--restart`` | Restart/resubmit case after     | ``--extend``,     |
    |               | modifying settings              | ``--apply``       |
    +---------------+---------------------------------+-------------------+
    | ``--batch``   | Rerun command as a PBS job      | All except ``-h`` |
    +---------------+---------------------------------+-------------------+
          
.. _cli-subset:

Run Matrix Subsetting
---------------------
Identifying a subset of the run matrix using the command line is an important
tool.  This system of scripts and modules has been used for run matrix of more
than 5000 cases, which could result in, for example, a very unwieldy status
message.  Although it is possible to comment out lines of the run matrix file,
pyCart provides much more powerful tools to identify specific subsets of the
run matrix.

There are five separate subsetting options, and it is possible to use more than
one of them in the same command.  When doing so, they are treated as logical
**AND**, so that only cases meeting all criteria are shown, run, or processed.
Finally, these subset commands can be applied to all commands that interact
with individual cases.  In other words, it applies to all commands except
``pycart -h``.  The following dedicates a subsection to each of the five
subsetting options.  The subsections will use a :ref:`simple example
<pycart-ex-bullet>` from ``$PYCART/examples/pycart/bullet``.

    .. code-block:: console
    
        $ pycart -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 DONE    200/200     .   5.0
        1    poweroff/m2.0a0.0b0.0 INCOMP  100/200     .   2.2
        2    poweroff/m2.0a2.0b0.0 RUN     157/200     Q   3.1   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
                
        ---=1, INCOMP=1, RUN=1, DONE=1,

    
.. _cli-subset-I:

Specific Indices
****************
Limiting to a specific index or list of indices is simple.  Consider the
following examples using the ``pycart -I`` option.  The first version is to
identify an individual case.

    .. code-block:: console
    
        $ pycart -c -I 1
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m2.0a0.0b0.0 INCOMP  100/200     .   2.2
                
        INCOMP=1,

It is also possible to get a specific list of cases.

    .. code-block:: console
    
        $ pycart -c -I 0,2,3
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 DONE    200/200     .   5.0
        2    poweroff/m2.0a2.0b0.0 RUN     157/200     Q   3.1   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
                
        ---=1, RUN=1, DONE=1,
        
Finally, a range of cases can be identified using a ``:`` character.  Note that
this relies on Python's zero-based indexing, which is something of an acquired
taste.

    .. code-block:: console
    
        $ pycart -cI 1:3
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m2.0a0.0b0.0 INCOMP  100/200     .   2.2
        2    poweroff/m2.0a2.0b0.0 RUN     157/200     Q   3.1   
                
        INCOMP=1, RUN=1,


.. _cli-subset-cons:

Using Constraints
*****************
Perhaps the most useful subsetting command is to give explicit constraints.

    .. code-block:: console
    
        $ pycart -c --cons "alpha==2"
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        2    poweroff/m2.0a2.0b0.0 RUN     157/200     Q   3.1   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
                
        ---=1, RUN=1,

Multiple constraints can be separated with commas.

    .. code-block:: console
    
        $ pycart -c --cons "alpha==2,beta==2"
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        2    poweroff/m2.0a2.0b0.0 RUN     157/200     Q   3.1   
                
        RUN=1,
        
The variables included in the constraint list must be run matrix variables
listed in the ``"Keys"`` option of the ``"Trajectory"`` section of the JSON
file.  The *key* must be the first thing in each constraint, so for example
``pycart --cons "2==alpha"`` will not work.

Finally, the Python relational operators ``==``, ``<``, ``<=``, ``>``, and
``>=`` can be combined with other operations.  For example, the following
command isolates cases with Mach number ending in 0.5.

    .. code-block:: console
    
        $ pycart -c --cons "Mach%1==0.5"
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 DONE    200/200     .   5.0
                
        DONE=1,


.. _cli-subset-filter:

Filtering by Folder Name
************************
The ``--filter`` option allows a user to restrict the command to cases that
include raw text in their full folder name.

    .. code-block:: console
    
        $ pycart -c --filter "a2"
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        2    poweroff/m2.0a2.0b0.0 RUN     157/200     Q   3.1   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
                
        ---=1, RUN=1,
        

.. _cli-subset-glob:

Filtering by File Glob
**********************
The ``--glob`` option is similar to ``--filter`` except that it uses standard
file globs.  This is useful for identifying cases with a range of text instead
of a very specific case.  However, the entire name of the case must match the
glob, so the user may need to add ``'*'`` to the beginning and end of the
command.

    .. code-block:: console
    
        $ pycart -c --glob "*a[1-3]*"
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        2    poweroff/m2.0a2.0b0.0 RUN     157/200     Q   3.1   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
                
        ---=1, RUN=1,
        

.. _cli-subset-regex:

Using Regular Expressions
*************************
The ``--re`` option is basically a better version of ``--glob``.  It uses
Python's standard :mod:`re` module and only reports cases that contain at least
one match for the regular expression.

    .. code-block:: console
    
        $ pycart -c --re "a[1-3]"
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        2    poweroff/m2.0a2.0b0.0 RUN     157/200     Q   3.1   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
                
        ---=1, RUN=1,
    

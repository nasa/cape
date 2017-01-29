
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

    .. code-block:: bash
    
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
    | ``pycart --archive``     | Create archive of solution in an outside |
    |                          | folder and clean up the working folder   |
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
    | ``pycart``               | Running with no command is equivalent to |
    |                          | ``pycart -n 10``                         |
    +--------------------------+------------------------------------------+
    
.. _cli-h:

Help Message
************
To see of the options from the command line, just run ``pycart
-h`` or ``pycart --help``.

    .. code-block:: bash
    
        $ pycart -h
        
        Python interface for Cart3D: :file:`pycart`
        ===========================================
        
        This function provides a master interface for pyCart.  All of the functionality
        from this script is also accessible from the :mod:`pyCart` module using
        relatively simple commands.
        
        :Usage:
            .. code-block:: bash
            
                $ pycart [options]
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

    .. code-block:: none
    
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

    .. code-block:: bash
    
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

    .. code-block:: bash
    
        $ pycart --ll LL_C
        
only updates the data book for line loads on the center component (assuming
there is a line load component called ``LL_C`` for the center body defined in
the ``"DataBook"`` section of the JSON file).  Meanwhile, the command ``pycart
--ll`` will update the data book for all three components.

Also, these commands will both run the necessary post-processing and collect
the results into the data book.  For instance, it will run ``triloadCmd`` to
compute line loads unless they already exist.


          
.. _cli-subset:

Run Matrix Subsetting
---------------------

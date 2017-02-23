
.. _cape-json-uncategorized:

--------------------------
Uncategorized JSON Options
--------------------------

Several inputs in the JSON control files are outside of each category.  A few of
those are common to each module, and they are described here.  The JSON syntax
for these options (shown with some non-default values) is 

    .. code-block:: javascript
    
        "nSubmit": 10,
        "umask": 0027,
        "ShellCmds": [
            "module load cape",
            "export PATH=$PATH:~/usr/cart3d"
        ],
        "BatchShellCmds": [
            "module load tecplot/360EX_2016r1",
            "module load texlive/2008"
        ],
        "PythonPath": ["tools"],
        "Modules": ["mymod", "thatmod"]
        
Links to additional options for each specific solver are found below.

    * :ref:`Cart3D <pycart-json-uncategorized>`
    * :ref:`FUN3D <pyfun-json-uncategorized>`
    * :ref:`OVERFLOW <pyover-json-uncategorized>`


Maximum Number of Jobs to Submit
================================
This parameter sets the maximum number of jobs that CAPE or any of the 
derivative modules will submit with a single command-line call.

    .. code-block:: javascript
    
        "nSubmit": 10
        
However, this value can be overridden from the command line using the ``-n``
option.

    .. code-block:: bash
    
        $ cape -n 20
        
File Permissions Mask
=====================
The ``umask`` parameter is important to set on systems with shared files. There
are a variety of reasons that PBS jobs may start using a file permissions mask
different from the user's preferred value. Typically, the most appropriate
``umask`` is ``0027``, which causes new files to have permissions ``640`` and
new directories or executable files to have permissions ``750``. This enables
users within the same group to read files but not edit them. Many systems will
use the default mask ``700``, which means that even users in the same group
cannot see or read the files.

Startup Shell Commands
======================

An important miscellaneous option, especially for cases submitted as PBS jobs,
lists commands to run within the shell before running any Cart3D/FUN3D/whatever
commands.  This is a list of strings that will be placed at the top of the run
script in each case directory.  There is also ``"BatchShellCmds"``, which
specifies additional commands that are run in ``pycart --batch`` commands. By
default, this is an empty list, which means no startup commands are issued at
the beginning of PBS jobs. 

    .. code-block:: javascript
    
        "ShellCmds": [],
        "BatchShellCmds": []
        
When Cape sets up a case, it creates a run script :file:`run_cape.pbs` in each
folder (or, if there is a nontrivial run sequence, :file:`run_cape.00.pbs`,
:file:`run_cape.01.pbs`, etc.). The run script can use BASH, ``csh``, or any
other shell, and this is set in the "PBS" section of the JSON file. The default
is BASH (that is, ``"/bin/bash"``), but some users prefer ``csh``.

If your ``rc`` file for your selected shell contains the necessary commands to
run Cart3D, a possible option is to use the following.

    .. code-block:: javascript
    
        "ShellCmds": ["source ~/.bashrc"]
        
(or ``". ~/.bashrc"``, as appropriate) This is discouraged unless Cart3D,
OVERFLOW, or FUN3D is basically the only software you ever use. A better option
is to put the commands that are needed in the JSON file, which makes that file
portable and less subject to later errors or system changes. Here is an example
that I use to run Cart3D on NASA's *Pleiades* supercomputer.

    .. code-block:: javascript
    
        "ShellCmds": [
            ". $MODULESHOME/init/bash",
            "module use -a /u/ddalle/share/modulefiles",
            "module load cart3d",
            "module load pycart",
            "module load mpt"
        ]
        
The first command is necessary because PBS jobs are started with very few
environment variables set.  For running cases in parallel, this command (or
sourcing a premade :file:`.*shrc` file) is necessary.  Another thing to note
here is that you also need to tell the interpreter where the pyCart commands
are---hence the ``"module load pycart"`` line.

A common use for ``--batch`` commands is to create reports that are
time-consuming and inappropriate for a supercomputer's login nodes.  For such
cases it is typical to add Tecplot and LaTeX modules for batch jobs.

    .. code-block:: javascript
    
        "BatchShellCmds": [
            "module load tecplot/360EX_2016r1",
            "module load texlive/2008"
        ]


.. _cape-json-uncategorized-mod:

Special Python Modules
======================
Some advanced runs require features that simply do not fit into the main
pyCart, pyOver, or whichever appropriate set of options. This would be true no
matter much work is put into the code. Indeed, one of the purposes of Cape is
to still be helpful when this situation inevitably occurs.

The following two lines allow the user to define a custom Python module (or
list of modules, if the user deems that appropriate).

    .. code-block:: javascript
    
        "PythonPath": ["tools/"],
        "Modules": ["mymod", "thatmod"]
        
The first variable causes pyCart to append the folder ``tools/`` to the
environment variable ``$PYTHONPATH``.  This means that files in ``tools/`` can
be imported as Python modules.

The second variable lists Python modules that will be imported every time the
the JSON file is loaded.  The way this is implemented is by just
running the standard ``import mymod`` and ``import thatmod`` syntax, and then
the functions and other information in those modules will be available to
Cape for that run.

.. _cape-json-misc-dict:

Miscellaneous Options Dictionary
================================

The full list of available options and their possible values is described below.
Alternate options are separated by ``|`` charcters, and default values are
between ``{}`` characters.
        
    *BatchShellCmds*: {``[]``} | :class:`list` (:class:`str`)
        List of shell commands to run after *ShellCmds* for batch PBS jobs

    *ShellCmds*: {``[]``} | :class:`list` (:class:`str`)
        List of shell commands to run at start of each script
    
    *PythonPath*: {``""``} | :class:`str` | :class:`list` (:class:`str`)
        Path or list of paths to add to ``$PYTHONPATH`` environment variable
        
    *Modules*: {``[]``} | :class:`str` | :class:`list` (:class:`str`)
        Python module or list of Python modules to import with this file
        
    *nSubmit*: {``10``} | :class:`int`
        Maximum number of jobs to submit/run
        
    *umask*: {``0027``} | :class:`int` | :class:`str`
        File permissions mask interpreted as octal number


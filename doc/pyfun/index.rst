
pyFun Documentation
===================

This is the documentation for pyFun/pyfun, which provides a Python module and 
scripts for interacting with FUN3D.

The basic usage for this module is to create a file called :file:`pyFun.json`
and use the script `pyfun`.  In addition to this control file for pyFun,
several other FUN3D input files (or, more accurately, input template files)
must be provided.

    * FUN3D namelist, :file:`fun3d.nml`
    * Mesh file(s), for example :file:`pyfun.b8.ugrid`
    * Boundary condition map, for example :file:`pyfun.mapbc`
    * There must also be a run matrix, which can either be specified within
      :file:`pyFun.json` or within a separate file

All of these files can have different names for cases when that helps in the
organization of runs, but these are the default file names.  If all these files
exist in the same folder, the following is the basic command that will start up
to 10 cases.

    .. code-block:: bash
    
        $ pyfun
        
The following will start up to *N* cases.

    .. code-block:: bash
    
        $ pyfun -n $N
        
This will start no cases but show you the status of all the runs.

    .. code-block:: bash
    
        $ pyfun -c
        
Changing the flag to ``-cj`` will also show PBS job numbers for cases where a
current job is either running or in the queue.  Supposing that the file is not
called :file:`pyFun.json` but :file:`run.json`, the ``-f`` flag can be used.
If at least some cases have partially run, the command to update or create the
data book (i.e. analyze force and moment coefficients and statistics as
specified in the control file), the ``--aero`` flag will lead to that action.

    .. code-block:: bash
    
        $ pyfun -f run.json --aero
        
The last option to be highlighted on the front page is the automated report
generation capability.  This command generates a multipage PDF based on settings
in the "Report" section of :file:`run.json` that usually gives a summary of each
case in the run directory on its own page.

    .. code-block:: bash
    
        $ pyfun -f run.json --report
        
There are many more flags to `pyfun`, including options to compress or archive
folders, plot iterative histories, and constrain to only a subset of the run
matrix.

Controlling runs with pyFun really focuses on the :file:`pyFun.json` file.
This file is divided into sections, some of which are mandatory.


.. toctree::
    :maxdepth: 2

    examples/index
    json/index

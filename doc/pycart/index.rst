
pyCart Documentation
====================

Welcome to the documentation for pyCart, which provides a Python module for
interacting with Cart3D.

The basic usage for this module is to create a file called :file:`pyCart.json`
and use the script `pycart`.  In addition to this control file for pyCart,
several other Cart3D input files (or, more accurately, input template files)
must be provided.

    * Cart3D input control file, :file:`input.cntl`
    * Surface triangulation file, :file:`Components.i.tri`
    * Optionally, :file:`Config.xml` to name and group surface components
    * If an adaptive run, an adaptive run script, :file:`aero.csh`
    * There must also be a run matrix, which can either be specified within
      :file:`pyCart.json` or within a separate file

All of these files can have different names for cases when that helps in the
organization of runs, but these are the default file names.  If all these files
exist in the same folder, the following is the basic command that will start up
to 10 cases.

    .. code-block:: bash
    
        $ pycart
        
The following will start up to *N* cases.

    .. code-block:: bash
    
        $ pycart -n $N
        
This will start no cases but show you the status of all the runs.

    .. code-block:: bash
    
        $ pycart -c
        
Changing the flag to ``-cj`` will also show PBS job numbers for cases where a
current job is either running or in the queue.  Supposing that the file is not
called :file:`pyCart.json` but :file:`run.json`, the ``-f`` flag can be used.
If at least some cases have partially run, the command to update or create the
data book (i.e. analyze force and moment coefficients and statistics as
specified in the control file), the ``--aero`` flag will lead to that action.

    .. code-block:: bash
    
        $ pycart -f run.json --aero
        
The last option to be highlighted on the front page is the automated report
generation capability.  This command generates a multipage PDF based on settings
in the "Report" section of :file:`run.json` that usually gives a summary of each
case in the run directory on its own page.

    .. code-block:: bash
    
        $ pycart -f run.json --report
        
There are many more flags to `pycart`, including options to compress or archive
folders, plot iterative histories, and constrain to only a subset of the run
matrix.

Controlling runs with pyCart really focuses on the :file:`pyCart.json` file.
This file is divided into sections, some of which are mandatory.  The three
most important sections describe the inputs to `flowCart`, the meshing options,
and the nature of the run matrix including variable definitions.


.. toctree::
    :maxdepth: 2

    json/index
    pyCart/index

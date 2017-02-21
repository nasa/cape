.. Documentation for the various Python modules

****************************************************
Introduction to pyCart Application Program Interface
****************************************************

The various methods used by pyCart, pyFun, and pyOver are based on a set of
Python modules with a few capabilities written in C (with a full Python/C
interface).  In order to reduce the number of lines, most of the base
functionality is provided by a common module :mod:`cape`.  Then each of the
modules for individual solvers (:mod:`pyCart`, :mod:`pyFun`, :mod:`pyOver`) are
based off of this module.  Furthermore, pyCart is a heavily object-oriented
implementation in which the vast majority of capabilities come from customized
classes.

The most critical class is :class:`cape.cntl.Cntl`, which acts as a template
for the three classes that interact with actually submitting jobs and checking
status.  An instance of one of these classes comes from reading all the
settings from a master :ref:`JSON file <cape-json>`.  When a user utilizes the
primary command-line interface via a command such as

    .. code-block:: bash
    
        $ pycart -f run/poweroff.json -c
        
the program works by loading an instance of :class:`pyCart.cart3d.Cart3d` by
reading :file:`run/poweroff.json` and using methods of that instance.  In this
case, the Python version of the above is shown below.

    .. code-block:: python
        
        # Load the pyCart API module
        import pyCart
        
        # Read the settings from the JSON file
        cart3d = pyCart.Cart3d('run/powroff.json')
        
        # Do something (in this case, the "-c" command)
        cart3d.DisplayStatus()

The basic modules based on this model are listed below:

    * :mod:`cape.cntl.Cntl`
    * :mod:`pyCart.cart3d.Cart3d`
    * :mod:`pyFun.fun3d.Fun3d`
    * :mod:`pyOver.overflow.Overflow`
    
Reading the JSON files depends heavily on the :mod:`cape.options` module.  The
majority of :ref:`JSON settings <cape-json>` are defined in :mod:`cape.json`,
with some of the additional settings particular to each individual solver
defined in :mod:`pyCart.options`, etc.

There are also a collection of helper modules, such as :mod:`pyCart.report`.
These typically provide one or more classes (such as
:class:`pyCart.report.Report`) which add a few methods to the :mod:`cape`
version.  This leads to a definition for the :mod:`pyCart` version of the
module that starts something like the following.

    .. code-block:: python
    
        # Import CAPE version
        import cape.report
        
        # Definition for pyCart.report.Report based on cape.report.Report
        class Report(cape.report.Report):
            
            ...
            
Then the code in :file:`pyCart/report.py` contains either methods that are
particular to Cart3D or methods that need to be modified from the definitions
in :file:`cape/report.py`.

There are a few modules that provide tools that are not primarily based on
classes.  There is a set of so-called "case" modules, which are the interface
to running the individual programs for each solver.  For example,
:mod:`pyCart.case` contains the function :func:`pyCart.case.run_flowCart`,
which runs from within a case folder.  The "case" modules are also based on
:mod:`cape.case` but in a different way.  These modules begin with the
following line, and then additional commands that are particular to each solver
are created in subsequent lines.

    .. code-block:: python
    
        # Import all methods from the CAPE version
        from cape.case import *
        
        # Load local modules
        from . import cmd
        from . import bin
        ...
        
Here is a list of modules that are not primarily based on classes.  Modules
that are particular to a solver are listed as children of the :mod:`cape`
module.

    * :mod:`cape.case`
        - :mod:`pyCart.case`
        - :mod:`pyFun.case`
        - :mod:`pyOver.case`
    * :mod:`cape.util`
    * :mod:`cape.geom`
    * :mod:`cape.convert`
    * :mod:`cape.color`
    * :mod:`cape.bin`
        - :mod:`pyCart.bin`
        - :mod:`pyFun.bin`
        - :mod:`pyOver.bin`
    * :mod:`cape.cmd`
        - :mod:`pyCart.cmd`
        - :mod:`pyFun.cmd`
        - :mod:`pyOver.cmd`
        
Finally, in addition to the :class:`cape.cntl.Cntl` class, there are several
key classes that form the basis for the key pyCart functionality.

    +---------------------------------+-------------------------------------+
    | Class                           | Description and Discussion          |
    +=================================+=====================================+
    | :class:`cape.tri.Tri`           | Interface to Cart3D-style surface   |
    |                                 | triangulations, can read several    |
    |                                 | formats, and ``triq`` files also    |
    |                                 | can be read                         |
    +---------------------------------+-------------------------------------+
    | :class:`cape.dataBook.DataBook` | pyCart-style database class         |
    +---------------------------------+-------------------------------------+
    | :class:`cape.dataBook.DBBase`   | Template class for reading and      |
    |                                 | interacting with databooks for a    |
    |                                 | single databook product             |
    +---------------------------------+-------------------------------------+
    | :class:`cape.dataBook.CaseData` | Template class for reading and      |
    |                                 | interacting with data from a single |
    |                                 | case                                |
    +---------------------------------+-------------------------------------+
    | :class:`cape.report.Report`     | Interface to automated reports      |
    +---------------------------------+-------------------------------------+
    | :class:`cape.fileCntl.FileCntl` | Template for interacting with all   |
    |                                 | settings files                      |
    +---------------------------------+-------------------------------------+
    | :class:`cape.namelist.Namelist` | Class for reading Fortran namelists |
    +---------------------------------+-------------------------------------+
    

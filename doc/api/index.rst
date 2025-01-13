.. Documentation for the various Python modules

****************************************************
Introduction to CAPE Application Program Interface
****************************************************

The various methods used by pyCart, pyFun, and pyOver are based on a set of
Python modules with a few capabilities written in C (with a full Python/C
interface). In order to reduce the number of lines, most of the base
functionality is provided by a common module :mod:`cape`. Then each of the
modules for individual solvers (:mod:`cape.pycart`, :mod:`cape.pyfun`,
:mod:`cape.pyover`) are based off of this module. Furthermore, pyCart is a
heavily object-oriented implementation in which the vast majority of
capabilities come from customized classes.

The most critical class is :class:`cape.cntl.Cntl`, which acts as a template
for the three classes that interact with actually submitting jobs and checking
status.  An instance of one of these classes comes from reading all the
settings from a master :ref:`JSON file <cape-json>`.  When a user utilizes the
primary command-line interface via a command such as

    .. code-block:: bash
    
        $ pycart -f run/poweroff.json -c
        
the program works by loading an instance of :class:`cape.pycart.cntl.Cntl` by
reading :file:`run/poweroff.json` and using methods of that instance.  In this
case, the Python version of the above is shown below.

    .. code-block:: python
        
        # Load the pyCart API module
        import cape.pycart
        
        # Read the settings from the JSON file
        cntl = cape.pycart.Cntl('run/powroff.json')
        
        # Do something (in this case, the "-c" command)
        cntl.DisplayStatus()

The basic modules based on this model are listed below:

    * :mod:`cape.cntl.Cntl`
    * :mod:`cape.pycart.cntl.Cntl`
    * :mod:`cape.pyfun.cntl.Cntl`
    * :mod:`cape.pykes.cntl.Cntl`
    * :mod:`cape.pyover.cntl.Cntl`
    
Reading the JSON files depends heavily on the :mod:`cape.options` module.  The
majority of :ref:`JSON settings <cape-json>` are defined in :mod:`cape.json`,
with some of the additional settings particular to each individual solver
defined in :mod:`cape.pycart.options`, etc.

There are also a collection of helper modules, such as :mod:`cape.pycart.report`.
These typically provide one or more classes (such as
:class:`cape.pycart.report.Report`) which add a few methods to the :mod:`cape`
version.  This leads to a definition for the :mod:`cape.pycart` version of the
module that starts something like the following.

    .. code-block:: python
    
        # Import CAPE version
        import cape.cfdx.report
        
        # Definition for pyCart.report.Report based on cape.cfdx.report.Report
        class Report(cape.cfdx.report.Report):
            
            ...
            
Then the code in :file:`cape/pycart/report.py` contains either methods that are
particular to Cart3D or methods that need to be modified from the definitions
in ``cape/cfdx/report.py``.
        
Here is a list of modules that are not primarily based on classes.  Modules
that are particular to a solver are listed as children of the :mod:`cape`
module.

    * :mod:`cape.cfdx.case`
        - :mod:`cape.pycart.case`
        - :mod:`cape.pyfun.case`
        - :mod:`cape.pyover.case`
    * :mod:`cape.argread`
    * :mod:`cape.util`
    * :mod:`cape.geom`
    * :mod:`cape.convert`
    * :mod:`cape.color`
    * :mod:`cape.cfdx.cmdgen`
        - :mod:`cape.pycart.cmdgen`
        - :mod:`cape.pyfun.cmdgen`
        - :mod:`cape.pyover.cmdgen`
    * :mod:`cape.cfdx.cmdrun`
        - :mod:`cape.pycart.cmdrun`
        - :mod:`cape.pyfun.cmdrun`
        - :mod:`cape.pyover.cmdrun`
        
Finally, in addition to the :class:`cape.cntl.Cntl` class, there are several
classes that form the basis for the key CAPE functionality.

.. list-table::
    :header-rows: 1
    
    * - Class
      - Description and discussion
    * - :class:`cape.tri.Tri`
      - Interface to Cart3D-style surface triangulations, can read several
        formats, and ``triq`` files also can be read
    * - :class:`cape.cfdx.dataBook.DataBook`
      - CFD raw data database class
    * - :class:`cape.cfdx.dataBook.CaseData`
      - Template class for reading and interacting with iterative data from a
        single case
    * - :class:`cape.cfdx.report.Report`
      - Interface to automated reports
    * - :class:`cape.filecntl.filecntl.FileCntl`
      - Template for interacting with many CFD solver settings file types
    * - :class:`cape.nmlfile.NmlFile`
      - Class for reading Fortran namelists
    * - :class:`cape.filecntl.namelist2.Namelist2`
      - Alternative interface to Fortran namelists


.. toctree::
    :maxdepth: 2

    cape/index
    cfdx/index
    pycart/index
    pyfun/index
    pylava/index
    pylch/index
    pykes/index
    pyover/index
    dkit/index
    tnakit/index

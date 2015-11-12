.. pyCart documentation master file, created by
   sphinx-quickstart on Wed Jun 11 08:36:23 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
CAPE Documentation
==================

Welcome to Cape, a computational aerodynamics processing environment for
efficient interaction with several high-fidelity aerodynamics codes.  This
includes tools for pre-processing, executing the codes, and performing
post-processing tasks.  The approach of Cape is to provide tools that make users
more efficient at some or all of the modeling and analysis process.  It may be
useful to use Cape for just one step of a particular user's project, or it may
be useful to use it for the entire process.

Currently, Cape has interfaces for 
`Cart3D <http://people.nas.nasa.gov/~aftosmis/cart3d/>`_ and 
`FUN3D <http://fun3d.larc.nasa.gov/>`_.  The Cart3D interface, :mod:`pyCart` has
been used for several NASA projects.  One example was the creation of an
aerodynamic database for booster separation for the Space Launch System, which
included over 10,000 different adaptive Cart3D runs in a 12-dimensional run
matrix.

The FUN3D interface, :mod:`pyFun` is relatively new, but reuses most of the
code used to build :mod:`pyCart`.  Both modules are built off of common tools in
the :mod:`cape` module, and so much of the usage is common between the two
interfaces.

Each interface is a portmanteau of "Python" and the name of the solver.  The
command-line interface is invoked with the commands ``pycart`` and ``pyfun``.
In addition, there are several scripts matching the glob ``pc_*.py`` for
isolated tasks such as converting grid formats.

Cape Inputs and JSON Files
--------------------------
Inputs to Cape can be given as either command-line arguments, input files
associated with the CFD solver, or JSON files.  `JSON <http:www.json.org>`_ is a
simple but extensible format similar to XML.  There are interpreters for many
languages.


pyCart Capabilities
-------------------
The Cart3D interface, pyCart, has quite a few capabilities.  Some highlights
and critical aspects are listed below.

    * Automatically create parametrically-named folders for cases in a matrix
    
    * Copy Cart3D input files to run files and edit them according to run matrix
      variables and global input settings
      
    * Create PBS scripts and submit them automatically
    
    * Simplified interface for refining initial volume mesh using component
      names
    
    * Built-in variables such as Mach number, angle of attack, and sideslip
    
    * Rotate components (such as fins or separating bodies) using Euler angles
      or a hinge axis
    
    * Automated averaging of iterative histories and statistics for databases
    
    * Automated reports of iterative histories and Tecplot layouts
    
    * Interface for custom functions for more advanced run matrix variables such
      as thrust settings

The basic usage for this module is to create a file called :file:`pyCart.json`
and use the script ``pycart``.  In addition to this control file for pyCart,
several other Cart3D input files (or, more accurately, input template files)
can be provided or interfaced.  The names below are canonical names; it is
possible to use other file names when setting up configurations.

    * Cart3D input control file, :file:`input.cntl`
    * Surface triangulation file, :file:`Components.i.tri`
    * Surface triangulation component names, :file:`Config.xml`
    * If an adaptive run, an adaptive run script, :file:`aero.csh`
    * Run matrix, either specified within :file:`pyCart.json` or a file


pyFun Capabilities
------------------
The FUN3D interface, pyFun, is newer and has somewhat fewer capabilities,
although most of the database and reporting capabilities are inherited from
pyCart.  Some capabilities are highlighted below.

    * Automatically create parametrically-named folders for cases in a matrix
    
    * Copy FUN3D input files to run files and edit them according to run matrix
      variables and global input settings
      
    * Create PBS scripts and submit them automatically
    
    * Built-in variables such as Mach number, angle of attack, sideslip,
      Reynolds number, and temperature
    
    * Automated averaging of iterative histories and statistics for databases
    
    * Automated reports of iterative histories and Tecplot layouts
    
The basic usage for this module is to create a file called :file:`pyFun.json`
and use the script ``pyfun``.  The required FUN3D input files are described
below.

    * FUN3D run settings template, :file:`fun3d.nml`
    * Mesh file; a recommended format is AFLR3, :file:`*.ugrid`
    * Boundary condition file, :file:`*.mapbc`
    * Run matrix, either specified within :file:`pyFun.json` or a file


.. toctree::
    :maxdepth: 2
    :numbered:
    
    install
    pycart/index
    script/index
    template/index
    
..    pyCart/cart3d



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`


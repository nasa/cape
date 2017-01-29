.. pyCart documentation master file, created by
   sphinx-quickstart on Wed Jun 11 08:36:23 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
pyCart Documentation
====================

Welcome to pyCart, a computational aerodynamics processing environment for
efficient interaction with several high-fidelity aerodynamics codes. This
includes tools for pre-processing, executing the codes, and performing
post-processing tasks. The approach of Cape is to provide tools that make users
more efficient at some or all of the modeling and analysis process. It may be
useful to use Cape for just one step of a particular user's project, or it may
be useful to use it for the entire process.

Most users really would rather evaluate new tools by seeing some examples, so
here are the links to the example pages for the main solvers:

    * :ref:`Cart3D Examples Using pyCart <pycart-examples>`
    * :ref:`FUN3D Examples Using pyFun <pyfun-examples>`
    * :ref:`OVERFLOW Examples Using pyOver <pyover-examples>`

Currently, pyCart has interfaces for `Cart3D
<http://people.nas.nasa.gov/~aftosmis/cart3d/>`_, `OVERFLOW
<http://overflow.arc.nasa.gov>`_, and `FUN3D <http://fun3d.larc.nasa.gov/>`_.
The Cart3D interface, :mod:`pyCart` has been used for several NASA projects.
One example was the creation of an aerodynamic database for booster separation
for the Space Launch System, which included over 10,000 different adaptive
Cart3D runs in a 12-dimensional run matrix.

The FUN3D interface, :mod:`pyFun`, is relatively new but reuses most of the
code used to build :mod:`pyCart`. Both modules are built off of common tools in
the :mod:`cape` module, and so much of the usage is common between the two
interfaces.

The OVERFLOW interface (also written "Overflow") interface, :mod:`pyOver`, also
uses much of the underlying code of :mod:`cape`.  It can be used to run
OVERFLOW, interact with databases, and there is a Python interface to solution
files including easy dimensionalization of state variables.

The central Python module that contains most of the code for each module is
called :mod:`cape`, which may be mentioned from time to time.

Each interface is a portmanteau of "Python" and the name of the solver. The
command-line interface is invoked with the commands ``pycart``, ``pyfun``, and
``pyover``. In addition, there are several scripts matching the glob
``pc_*.py`` for isolated tasks such as converting grid formats.

**Cape Inputs and JSON Files**
Inputs to Cape can be given as either command-line arguments, input files
associated with the CFD solver, or JSON files. `JSON <http:www.json.org>`_ is a
simple but extensible format similar to XML. There are interpreters for many
languages.


**Common CAPE Capabilities**
The following bullet list of capabilities is common to all interfaces including
any that will be added in the future.

    * Automatically create parametrically-named folders for cases in a matrix
    
    * Copy input files to run folders and edit them according to run matrix
      variables and multiple phases of global input settings
      
    * Create PBS scripts and submit them automatically
    
    * Built-in variables such as Mach number, angle of attack, sideslip,
      Reynolds number, and temperature
    
    * Automated averaging of iterative histories and statistics for databases
    
    * Automated reports using LaTeX (usually supports Tecplot and ParaView
      images)

**pyCart Capabilities**
The Cart3D interface, pyCart, has quite a few capabilities.  Some highlights
and critical aspects are listed below.

    * Copy Cart3D input files to run files and edit them according to run matrix
      variables and global input settings
      
    * Simplified interface for refining initial volume mesh using component
      names
    
    * Rotate components (such as fins or separating bodies) using Euler angles
      or a hinge axis
    
    * Automated reports of iterative histories and Tecplot layouts
    
    * Interface for custom functions for more advanced run matrix variables
      such as thrust settings

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


**pyFun Capabilities**
The FUN3D interface, pyFun, is newer and has somewhat fewer capabilities,
although most of the database and reporting capabilities are inherited from
pyCart.  Some capabilities are highlighted below.

    * Copy FUN3D input files to run files and edit them according to run matrix
      variables and global input settings
      
    * Interface with AFLR3 for volume mesh generation, which can be used to
      have multiple bodies in different relative positions for each case in a
      run matrix
      
    * Interface to Chimera Grid Tools ``triload`` to calculate sectional loads
    
The basic usage for this module is to create a file called :file:`pyFun.json`
and use the script ``pyfun``.  The required FUN3D input files are described
below.

    * FUN3D run settings template, :file:`fun3d.nml`
    * Mesh file; a recommended format is AFLR3, :file:`*.ugrid`
    * Boundary condition file, :file:`*.mapbc`
    * Run matrix, either specified within :file:`pyFun.json` or a file
    
**pyOver Capabilities**
The OVERFLOW interface, pyOver, is also new but will have a rapidly expanding
set of capabilities.  Much of the database and report capabilities are
inherited from pyCart.  Some highlights are below.

    * Copy OVERFLOW input namelists and edit them according to run matrix
      variables and multiple phases of input settings
      
    * Highly generalized interface to OVERFLOW namelists that doesn't need
      great amounts of user preparation
    
    * Built-in interface to OVERFLOW *q* files with simple access to
      coefficients and dimensional states
      
The basic usage for this module involves a file :file:`pyOver.json`, and the
primary script is ``pyover``.  Required input files depend on whether the case
is being run in DCF or Pegasus5 mode.  In this module, the user is responsible
for informing pyOver which mesh files are required, although a default list is
defined.  In general the required files are the following.

    * OVERFLOW namelist template, :file:`overflow.inp`
    * Mesh files, usually in the ``common/`` folder
    * Run Matrix, either within :file:`pyOver.json` or in a CSV-like file

.. toctree::
    :maxdepth: 2
    :numbered:
    
    install
    moreinfo/index
    common/index
    cape/index
    pycart/index
    pyfun/index
    pyover/index
    script/index
    

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`


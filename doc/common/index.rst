
.. _common:

Usage and Common Settings
=========================

This section covers basic usage of the pyCart modules that work for each
solvers.  That is, there is no need for separate descriptions for Cart3D,
OVERFLOW, and FUN3D.

The most import pyCart tools are the command-line scripts ``pycart``,
``pyfun``, and ``pyover``.  These are used to submit jobs, check status, and
perform post-processing.  Each of the three has an extensive set of
command-line options, and most of these are available to all three scripts.
The three scripts are all Python scripts in the ``$CAPE/bin/`` folder, and
they must have at a minimum a :ref:`JSON file <json-syntax>` as an input.

In addition, all the functionality of the ``pycart``, ``pyfun``, and
``pyover`` scripts can be accessed via a Python API.  Each of the three
scripts has its own module: :mod:`pyCart.cart3d`, :mod:`pyFun.fun3d`, or
:mod:`pyOver.overflow`.  These are typically more useful for accessing
post-processing data, and it can also be useful in creating advanced scenarios
that cannot be set up fully within the JSON input file.

Finally, there a number of additional scripts in the ``$CAPE/bin/`` folder
that serve more specific purposes such as converting file formats. 


.. toctree::
    :maxdepth: 2
    
    setup
    command-line
    file/json
    file/matrix
    freestream
    report/index
    json/index
    scripting
    


.. _common:

Usage and Common Settings
=========================

This section covers basic usage of the CAPE modules that work for each solvers.
That is, there is no need for separate descriptions for Cart3D, OVERFLOW, and
FUN3D.

The most important :mod:`cape` tools are the command-line scripts ``pycart``,
``pyfun``, and ``pyover``. These are used to submit jobs, check status, and
perform post-processing. Each of the three has an extensive set of command-line
options, and most of these are available to all three scripts. Each executable
script has at a minimum a :ref:`JSON file <json-syntax>` as an input.

    .. note::

        For CAPE developers, the three scripts (and other CAPE executables) are
        all located in the ``$CAPE/bin/`` folder. For regular users who install
        :mod:`cape` using :mod:`pip` as recommended in these docs, the
        executables will live alongside the other local executables installed
        by :mod:`pip`.

        On a Linux system, this will often be ``$HOME/.local/bin``.

In addition, all the functionality of the ``pycart``, ``pyfun``, and ``pyover``
scripts can be accessed via a Python API. Each of the three scripts has its own
module: :mod:`cape.pycart.cntl`, :mod:`cape.pyfun.cntl`, or
:mod:`cape.pyover.cntl`. These are typically more useful for accessing
post-processing data, and it can also be useful in creating advanced scenarios
that cannot be set up fully within the JSON input file.

Finally, there a number of additional executables that serve more specific
purposes such as converting file formats.


.. toctree::
    :maxdepth: 3
    
    setup
    command-line
    file/json
    file/matrix
    freestream
    report/index
    json/index
    scripting
    

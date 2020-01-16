
.. _pyover-example-powered-nacelle:

--------------------------------
OVERFLOW Powered Nacelle Example
--------------------------------

This pyOver example shows how to use pyover to run one of the simple test cases
that come with the OVERFLOW source code. This example starts with the grids and
inputs files that are created within the OVERFLOW examples, and documents how
to create the pyOver setup, run matrix, how to run several OVERFLOW cases,
covers some post-processing.  This example is located in 

    * ``$CAPE/examples/pyover/02_powered_nacelle/``

This example shows how to use pyOver for a test case with two related
configurations, a flow-through axisymmetric nacelle, and a powered axisymmetric
nacelle.  The example comes with the grids and input files ready to run
OVERFLOW. However, if one desires to generate these files locally, here are the
commands that were used to create the grid and input files in the OVERFLOW
source.  This assumes that the OVERFLOW source bundle has been installed in a
directory whose absolute path is given by the environment variable
``$OVERHOME``.  The following generates double-precision, little-endian
unformatted versions of the grid files.

  .. code-block:: bash

    export GFORTRAN_CONVERT_UNIT=little_endian
    export FC=gfortran
    export FFLAGS=-fdefault-real-8
    cd $OVERHOME/test/powered_nacelle/grids_ft
    ./makegrids
    cd ../run_ft
    rsync -av Config.xml xrays.in grid.in \
       $CAPE/test/pyover/02_powered_nacelle/common_flowthrough/.
    rsync -av mixsur.{inp,fmp} grid.{ibi,ib,map,i.tri,nsf} \
       $CAPE/test/pyover/02_powered_nacelle/common_flowthrough/fomo/.
    rsync -av m0.80.1.inp
       $CAPE/test/pyover/02_powered_nacelle/common_flowthrough/overflow.inp
    cd ../grids
    ./makegrids
    cd ../run
    rsync -av Config.xml xrays.in grid.in \
       $CAPE/test/pyover/02_powered_nacelle/common_powered/.
    rsync -av pr136.1.inp \
       $CAPE/test/pyover/02_powered_nacelle/common_powered/overflow.inp
    rsync -av mixsur.{inp,fmp} grid.{ibi,ib,map,i.tri,nsf} \
       $CAPE/test/pyover/02_powered_nacelle/common_powered


Input Run Matrix
----------------

pyOver Configuration File
-------------------------


Execution
---------


Report Generation
-----------------




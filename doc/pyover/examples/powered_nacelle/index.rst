
.. _pyover-example-powered-nacelle:

--------------------------------
OVERFLOW Powered Nacelle Example
--------------------------------

This pyOver example shows how to use pyover to run one of the simple
test cases that come with the OVERFLOW source code. 
post-processing for a simple bullet geometry.  This example is located in 

    * ``$CAPE/examples/pyover/02_powered_nacelle/``

This example shows how to use pyOver for a test case with two related
configurations, a flow-through axisymmetric nacelle, and a powered axisymmetric
nacelle.  The example comes with the grids and input files ready to run
OVERFLOW. However, if one desires to generate these files locally, here are
the commands used to create the grid and input files in the OVERFLOW source.
This assumes that the OVERFLOW source bundle has been installed in a directory
whose absolute path is given by the environment variable ``$OVERHOME``.

  .. code-block:: bash

    export GFORTRAN_CONVERT_UNIT=little_endian
    export FC=gfortran
    export FFLAGS=-fdefault-real-8
    cd $OVERHOME/test/powered_nacelle/grids_ft
    ./makegrids
    cd ../run_ft
    rsync -av xrays.in mixsur.{inp,fmp} grid.{ibi,ib,in,map,i.tri,nsf} \
        $CAPE/test/pyover/02_powered_nacelle/common_ft
    rsync -av Config.xml xrays.in mixsur.{inp,fmp} grid.{ibi,ib,in,map,i.tri,nsf} \
        $CAPE/test/pyover/02_powered_nacelle/common_ft
    cd ../grids
    ./makegrids
    cd ../run
    rsync -av xrays.in mixsur.{inp,fmp} grid.{ibi,ib,in,map,i.tri,nsf} \
        $CAPE/test/pyover/02_powered_nacelle/common_powered
    rsync -av pr136.1.inp \
        $CAPE/test/pyover/02_powered_nacelle/common_powered/overflow.inp




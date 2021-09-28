.. Documentation for the overall pyCart module.

.. _Cart3D:

*******************************
Cart3D Documentation for pyCart
*******************************

Some independent documentation for Cart3D is provided here; it is written from
the perspective of pyCart and Cape and closely follows the :mod:`cape.pycart` module.

Note the two following locations for stand-alone Cart3D documentation.  The
second link requires entering some basic information but is free.  These were
most recently accessed on April 19, 2016.

    * `<http://people.nas.nasa.gov/~aftosmis/cart3d/>`_

This documentation is really about the Cart3D input and output files and the
Cart3D binaries.  All or most of the content is discussed in the context of
pyCart and how the two software suites work together.  Many of the details are
made theoretically invisible to users of pyCart, but having a reference is
useful when problems or difficult scenarios arise.

The following table describes input files to Cart3D.

    +----------------------+--------------------------------------------------+
    | File                 | Description and Type                             |
    +======================+==================================================+
    | ``input.cntl``       | General flow solver input file, sets Mach        |
    |                      | number, angle of attack, solver inputs, line     |
    |                      | sensors,etc.                                     |
    +----------------------+--------------------------------------------------+
    | ``Config.xml``       | Surface component naming file                    |
    +----------------------+--------------------------------------------------+
    | ``Components.i.tri`` | Surface triangulation                            |
    +----------------------+--------------------------------------------------+
    | ``preSpec.c3d.cntl`` | Customizes volume mesh (input to ``cubes``)      |
    +----------------------+--------------------------------------------------+
    | ``aero.csh``         | (Only for adaptive runs) adaptive inputs and run |
    |                      | script; executable                               |
    +----------------------+--------------------------------------------------+

There are some other files that are inputs to the flow solver ``flowCart`` but
are usually created by initialization files.

    +----------------------+--------------------------------------------------+
    | File                 | Description and Type                             |
    +======================+==================================================+
    | ``input.c3d``        | Contains mesh domain boundary                    |
    +----------------------+--------------------------------------------------+
    | ``Mesh.c3d.Info``    | History of modifications to volume mesh          |
    +----------------------+--------------------------------------------------+
    | ``Mesh.mg.c3d``      | Mesh file with multigrid levels                  |
    +----------------------+--------------------------------------------------+

Typical output files are given below.  Some of these file names are slightly
different 

    +-----------------------+-------------------------------------------------+
    | File                  | Description and Type                            |
    +=======================+=================================================+
    | ``Components.i.triq`` | Surface triangulation with solution state       |
    +-----------------------+-------------------------------------------------+
    | ``Components.i.plt``  | Surface solution Tecplot file                   |
    +-----------------------+-------------------------------------------------+
    | ``Components.i.dat``  | ASCII surface solution Tecplot file             |
    +-----------------------+-------------------------------------------------+
    | ``cutPlanes.plt``     | Tecplot file for cut planes of solution         |
    +-----------------------+-------------------------------------------------+
    | ``loadsCC.dat``       | Summary of forces and moments on each           |
    |                       | component; not used by pyCart                   |
    +-----------------------+-------------------------------------------------+
    | ``$COMP.dat``         | Iterative force/moment history for component    |
    |                       | named *COMP*                                    |
    +-----------------------+-------------------------------------------------+
    | ``check.%05i``        | Cart3D restart file                             |
    +-----------------------+-------------------------------------------------+
    
    
.. toctree::
    :maxdepth: 3
    
    


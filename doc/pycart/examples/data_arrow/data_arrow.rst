
.. _pycart-ex-data-arrow:

Demo 7: Data Book Plots and Reports
===================================

Using the geometry from :ref:`Example 2 <pycart-ex-arrow>` and 
:ref:`Example 7 <pycart-ex-lineload-arrow>`, this example computes forces and
moments on a larger number of cases in order to

continues the analysis and adds computation of sectional loads.  The example is
located in ``$PYCART/examples/pycart/07_data_arrow``.

The geometry used for this shape is a capped cylinder with four fins and 9216
faces and seven components.  This example is set to run 30 cases with a square
matrix of Mach number and angle of attack.

    .. code-block:: console
    
        $ pycart -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m0.50a00.0   ---     /           .            
        1    poweroff/m0.50a01.0   ---     /           .            
        2    poweroff/m0.50a02.0   ---     /           .            
        3    poweroff/m0.50a05.0   ---     /           .            
        4    poweroff/m0.50a10.0   ---     /           .            
        5    poweroff/m0.80a00.0   ---     /           .            
        6    poweroff/m0.80a01.0   ---     /           .            
        7    poweroff/m0.80a02.0   ---     /           .            
        8    poweroff/m0.80a05.0   ---     /           .            
        9    poweroff/m0.80a10.0   ---     /           .            
        10   poweroff/m0.95a00.0   ---     /           .            
        11   poweroff/m0.95a01.0   ---     /           .            
        12   poweroff/m0.95a02.0   ---     /           .            
        13   poweroff/m0.95a05.0   ---     /           .            
        14   poweroff/m0.95a10.0   ---     /           .            
        15   poweroff/m1.10a00.0   ---     /           .            
        16   poweroff/m1.10a01.0   ---     /           .            
        17   poweroff/m1.10a02.0   ---     /           .            
        18   poweroff/m1.10a05.0   ---     /           .            
        19   poweroff/m1.10a10.0   ---     /           .            
        20   poweroff/m1.40a00.0   ---     /           .            
        21   poweroff/m1.40a01.0   ---     /           .            
        22   poweroff/m1.40a02.0   ---     /           .            
        23   poweroff/m1.40a05.0   ---     /           .            
        24   poweroff/m1.40a10.0   ---     /           .            
        25   poweroff/m2.20a00.0   ---     /           .            
        26   poweroff/m2.20a01.0   ---     /           .            
        27   poweroff/m2.20a02.0   ---     /           .            
        28   poweroff/m2.20a05.0   ---     /           .            
        29   poweroff/m2.20a10.0   ---     /           .            
        
        ---=30,
        
Setup
-----
The interesting part of the JSON setup to get the data we need for this example
is in the ``"Config"`` and ``"DataBook"`` sections.  The ``"Config"`` section
tells Cart3D which things for which to track iterative histories.  In addition,
it defines the reference area and any reference points (such as the Moment
reference point).

    .. code-block:: javascript
    
        "Config": {
            // File to name each integer CompID and group them into families 
            "File": "arrow.xml",
            // Declare forces and moments
            "Force": [
                "cap",
                "body",
                "fins",
                "arrow_no_base",
                "arrow_total",
                "fin1",
                "fin2",
                "fin3",
                "fin4"
            ],
            "RefPoint": {
                "arrow_no_base": "MRP",
                "arrow_total": "MRP", 
                "fins": "MRP",
                "fin1": "MRP",
                "fin2": "MRP",
                "fin3": "MRP",
                "fin4": "MRP"
            },
            // Define some points for easier reference
            "Points": {
                "MRP": [3.5, 0.0, 0.0]
            },
            // Reference quantities
            "RefArea": 3.14159,
            "RefLength": 2.0
        }
    
The *Config>Force* instructs Cart3D to report the force on the named components
at each iteration (in addition to any components in the template ``input.cntl``
file).  In particular, it adds a line such as ``Force cap`` for each listed
component.  The *Config>RefPoint* performs a similar function to report the
moment at each iteration as well.  In Cart3D, it is possible to have a force
only, a moment only, or both.  Either way, forces and/or moments will be put
into the file ``cap.dat``.

Each component requesting a moment needs a moment reference point.  Instead of
typing out the moment reference point for each requested moment, we define a
reference point called ``"MRP"`` in *Config>Points*.  This makes it easier to
change the reference point, but the *Config>Points* parameter has some other
advantages.  It can automatically be translated by a run matrix variable (i.e.
trajectory key); for example, it could be used to keep track of a point on the
leading edge of a deflected fin.

The ``"DataBook"`` section defines which quantities are of interest for
recording into a database.  A ``"DataBook"`` has data that is stored outside of
the run folders and has a more permanent feeling.  The portion of the JSON file
is shown below.

    .. code-block:: javascript
    
        "DataBook": {
            // List of data book components
            "Components": [
                "cap",
                "body",
                "fins",
                "arrow_no_base",
                "arrow_total",
                "fuselage",
                "fin1",
                "fin2",
                "fin3",
                "fin4"
            ],
            // Location of data book
            "Folder": "data/",
            // Parameters for collecting data
            "nFirst": 0,
            "nStats": 100,
            "nMin": 100,
            // Basic component
            "arrow_no_base": {"Type": "FM"},
            "arrow_total":  {"Type": "FM"},
            "fins": {"Type": "FM"},
            "fin1": {"Type": "FM"},
            "fin2": {"Type": "FM"},
            "fin3": {"Type": "FM"},
            "fin4": {"Type": "FM"},
            "fuselage": {
                "Type": "FM",
                "CompID": ["arrow_no_base", "-fins"]
            }
        }
    
The parameter *DataBook>Components* lists the components that go into the
databook.  All of these except for ``"fuselage"`` were defined in
*Config>Forces*, and some were also in *Config>Moments*.  The default databook
component type for pyCart is ``"Force"``; here we have changed the type to
``"FM"`` (short for "force & moment") for components where the moment is
available.

The ``"fuselage"`` key shows how we can in some cases get iterative histories
for components we forgot to track.  We define the ``"fuselage"`` component to
be the force and moment on ``"arrow_no_base"`` minus the force and moment onf
``"fins"``.  To add components, just omit the ``"-"`` prefix.

DataBook Interface
-------------------
This example is set up so that the user can run the 30 cases using typical
commands introduced in previous examples.  However, the databook is already
provided in the ``data/`` folder.  It contains files such as ``aero_cap.csv``,
``aero_body.csv``, and so on for each component in *DataBook>Components*.  An
example file is partially shown below.
        
    :download:`aero_arrow_no_base.csv`:
    
    .. code-block:: none
    
        # Database statistics for 'arrow_no_base' extracted on 2017-03-21 21:20:36 
        #
        #mach,alpha,config,Label,CA,CY,CN, ... CA_min,CA_max,CA_std, ... nOrders,nIter,nStats
        0.5,0,poweroff,,0.3478,-0.0002,0.02083, ... 5.02,200,100
        ...
        2.2,10,poweroff,,0.8580,0.0002,0.9261, ... 1.51,200,100

One can interact with the data book from any Python interactive shell (IPython
is highly recommended).  This example shows how to interface with the databook,
which can be a useful skill to investigate trends, etc.

    .. code-block:: pycon
    
        >>> import pyCart
        >>> cart3d = pyCart.Cart3d()
        >>> cart3d.ReadDataBook()
        >>> cart3d.DataBook.Components
        [u'cap',
         u'body',
         u'fins',
         u'arrow_no_base',
         u'arrow_total',
         u'fuselage',
         u'fin1',
         u'fin2',
         u'fin3',
         u'fin4']
        >>> DBfins = cart3d.DataBook['fins']
        >>> I = cart3d.x.Filter(['alpha==2'])
        >>> DBfins.PlotCoeff('CN', I)
        
This quick example opens up a :mod:`matplotlib` figure which leads to the
result below.  However, it is usually easier to use the ``pycart --report``
command.
        
    .. figure:: fig1.*
        :width: 3.8in
        
        Example plot of *CN* created from pyCart DataBook API
        
Reports
-------

.. _pycart-ex-data-arrow:

Demo 7: Data Book Plots and Reports
===================================

Using the geometry from :ref:`Example 2 <pycart-ex-arrow>` and :ref:`Example 7
<pycart-ex-lineload-arrow>`, this example computes forces and moments on a
larger number of cases in order to continues the analysis and adds computation
of sectional loads. To get started, clone this repoand run the following easy
commands:

    .. code-block:: console

        $ git clone https://github.com/nasa-ddalle/pycart07-data_arrow.git
        $ cd pycart07-data_arrow
        $ ./copy-files.py
        $ cd work/

This will copy all of the files into a newly created ``work/`` folder. Follow
the instructions below by entering that ``work/`` folder; the purpose is that
you can easily delete the ``work/`` folder and restart the tutorial at any
time.

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
        
**Note:** the data book is not created or updated automatically once the cases
are completed.  The data book is only created or updated using the command
``pycart --aero``.  For this example, the data book already exists, but for
practical usage this is an important step.

One can interact with the data book from any Python interactive shell (IPython
is highly recommended).  This example shows how to interface with the databook,
which can be a useful skill to investigate trends, etc.

    .. code-block:: pycon
    
        >>> import cape.pycart
        >>> cntl = pyCart.Cntl()
        >>> cntl.ReadDataBook()
        >>> cntl.DataBook.Components
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
        >>> DBfins = cntl.DataBook['fins']
        >>> I = cntl.x.Filter(['alpha==2'])
        >>> DBfins.PlotCoeff('CN', I)
        
This quick example opens up a :mod:`matplotlib` figure which leads to the
result in :numref:`fig-pycart-ex07-raw-CN`.  However, it is usually easier to
use the ``pycart --report`` command.
        
    .. _fig-pycart-ex07-raw-CN:
    .. figure:: fig1.*
        :width: 3.8in
        
        Example plot of *CN* created from pyCart DataBook API
        
Reports
-------
Options for automated reports are set in the ``"Reports"`` section of the JSON
file.  This example defines four reports, and all of them are so-called "Sweep"
reports.  Instead of plotting iterative histories for each case, plots are made
for the forces and moments for a collection of cases.  This results in, for
example, plots of normal force as a function of Mach number.  The header
section of the ``"Reports"`` section is shown below.

    .. code-block:: javascript
    
        "Report": {
            // List of reports
            "Reports": ["mach", "mach-carpet", "alpha", "alpha-carpet"],
            // Define the report
            "mach": {
                "Title": "Cart3D Force \\& Moment Mach Sweep",
                "Subtitle": "Example \\texttt{07\\_data\\_arrow}",
                "Restriction": "pyCart Example - Distribution Unlimited",
                "Sweeps": "mach"
            },
            "mach-carpet": {
                "Title": "Cart3D Force \\& Moment Mach Sweep",
                "Subtitle": "Example \\texttt{07\\_data\\_arrow}",
                "Restriction": "pyCart Example - Distribution Unlimited",
                "Sweeps": "mach-carpet"
            },
            "alpha": {
                "Title": "Cart3D Force \\& Moment Mach Sweep",
                "Subtitle": "Example \\texttt{07\\_data\\_arrow}",
                "Restriction": "pyCart Example - Distribution Unlimited",
                "Sweeps": "alpha"
            },
            "alpha-carpet": {
                "Title": "Cart3D Force \\& Moment Mach Sweep",
                "Subtitle": "Example \\texttt{07\\_data\\_arrow}",
                "Restriction": "pyCart Example - Distribution Unlimited",
                "Sweeps": "alpha-carpet"
            }
        }

Mach Sweeps
^^^^^^^^^^^
One can see that these are "sweep" reports because the key *Report>Sweeps* key
is defined and *Report>Figures* is not.  It is possible to put both into the
same report, but that's not done here because the example is set up to be
possible without actually running the cases.  Anyway, try creating the first
report using the following command.

    .. code-block:: console
    
        $ pycart --report mach
        
This creates five pages with nine Mach sweep plots per page.  Each page is a
single page, and there are five pages because we have a square run matrix with
five different angles of attack.  Rather than specifying too much detail, an
example plot is provided in :numref:`fig-pycart-ex07-a2-fuselage-CLM` and
:numref:`fig-pycart-ex07-a2-fins-CN`.

    .. _fig-pycart-ex07-a2-fuselage-CLM:
    .. figure:: alpha02/mach_fuse_CLM.*
        :width: 3.8 in
        
        Mach sweep of ``fuselage``/*CLM* at 2 degrees angle of attack

    .. _fig-pycart-ex07-a2-fins-CN:
    .. figure:: alpha02/mach_fins_CN.*
        :width: 3.8 in
        
        Mach sweep of *CN* on each fin at 2 degrees angle of attack

The inputs that led to these two figures (*mach_fuse_CLM* for
:numref:`fig-pycart-ex07-a2-fuselage-CLM`; *mach_fins_CN* for
:numref:`fig-pycart-ex07-a2-fins-CN`) are shown below.  This is an excerpt from
the *Report>Subfigures* section of ``pyCart.json``.

    .. code-block:: javascript
    
        // Mach sweep
        "mach_arrow": {
            "Type": "SweepCoeff",
            "Width": 0.33,
            "FigureWidth": 5.5,
            "FigureHeight": 4.2,
            "LineOptions": {
                "marker": "o",
                "color": ["b", "g", "m", "darkorange", "purple"],
                "ls": "-"
            },
            "Component": "arrow_no_base",
            "XLabel": "Mach number"
        },
        "mach_fuse_CLM": {
            "Type": "mach_arrow",
            "Component": "fuselage",
            "Coefficient": "CLM"
        },
        "mach_fins_CN": {
            "Type": "mach_arrow",
            "Component": ["fin1", "fin2", "fin3", "fin4"],
            "Coefficient": "CN"
        }
        
The *Type* parameter is set to ``"SweepCoeff"`` here for each plot.  The full
path to this setting is *Report>Subfigures>mach_arrow>Type*, and this setting
is inherited by all the other ``mach_*`` subfigures.  In
*mach_arrow>LineOptions*, we set formatting options to be used by the Mach
sweep plots.  A list of values, such as shown here in *color*, causes pyCart to
cycle through the different plot styles.  In this example, the first line is
blue, the second line is green, etc.  See :mod:`matplotlib` for a full set of
available plot options.


The main settings are *Component* and *Coefficient*.  Once the main template
for the subfigures is set (here in *mach_arrow*), the other plots can usually
be created by just changing the *Component* and *Coefficient*.

The *mach_fins_CN* subfigure also demonstrates how users can plot multiple
lines on the same plot by having a list of components.
:numref:`fig-pycart-ex07-a2-fins-CN` shows this example.  Because the sideslip
is zero, the two fins on the side, fin 2 and fin 4 are right on top of each
other.  The top fin (fin 1) and bottom fin (fin 3) are not as symmetric.

Users are encouraged to create the report and explore the other aspects of the
example in the resulting PDF and the JSON file.

Carpet Plots
^^^^^^^^^^^^
In order to get into the plots quicker, the previous subsection skipped the
definition of the actual sweeps.  The *Report>Sweeps* definition from
``pyCart.json`` is shown below.

    .. code-block:: javascript

        "Sweeps": {
            // Mach sweep
            "mach": {
                "Figures": ["SweepTables", "MachSweep"],
                "EqCons": ["alpha"],
                "XAxis": "mach"
            },
            // Mach sweep with alpha carpet
            "mach-carpet": {
                "Figures": ["SweepTables", "MachSweep"],
                "EqCons": [],
                "CarpetEqCons": ["alpha"],
                "XAxis": "mach"
            },
            // Alpha sweep
            "alpha": {
                "Figures": ["SweepTables", "AlphaSweep"],
                "EqCons": ["mach"],
                "XAxis": "alpha"
            },
            // Alpha sweep with Mach carpet
            "alpha-carpet": {
                "Figures": ["SweepTables", "AlphaSweep"],
                "EqCons": [],
                "CarpetEqCons": ["mach"],
                "XAxis": "alpha"
            }
        }
        
Notice in the excerpt from the top level of the ``"Report"`` section at the
beginning of this example, each named "report" has a *Sweeps* key.  That
selects one or more "sweep" from *Report>Sweeps*.  Inspecting the JSON file
probably makes more sense than this attempt to explain it in words.

Anyway, the ``"mach"`` sweep lists two figures, ``"SweepTables"`` and
``"MachSweep"``, and more importantly an "equality constraint" in the form of
setting *EqCons* to ``["alpha"]``.  This means that each case that goes into
one Mach sweep must have the same value of *alpha*.  It is also possible to use
*TolCons* which allows the user to specify that all cases must have an angle of
attack within a certain tolerance.  The *TolCons* key is especially useful for
comparing results to wind tunnel data, which may have some slight variations in
test conditions.

In addition to *EqCons* and *TolCons*, there is also *GlobalCons*, which limits
which cases are eligible to be included in any sweep.  For example, we could
set ``"GlobalCons": ["mach > 1.0"]`` to limit the results to only supersonic
cases. 

Also, the ``"Figures"`` key works in the same way within ``"Sweeps"`` as it
does in regular reports.  See the previous examples and the example
``pyCart.json`` for more information on how to define figures.  Finally,
the *XAxis* key simply designates a run matrix variable (trajectory key) to use
as the independent variable in the plots.

The focus of this subsection is the ``"mach-carpet"`` sweep and its use of
*CarpetEqCons*.  Both *CarpetEqCons* and *CarpetTolCons* work in a similar way
to *EqCons* and *TolCons*.  However, "carpet" constraints allow the user to
plot multiple sweeps on the same figure.  Here the report ``"mach-report"`` has
no *EqCons*, so the entire run matrix goes into the same result, and there is
only one page of plots in the automated report.  

Create the carpet plot by running the following command:

    .. code-block:: console
    
        $ pycart --report mach-carpet

A pair of selected plots from this report are shown in
:numref:`fig-pycart-ex07-fuselage-mach-carpet-CLM` and
:numref:`fig-pycart-ex07-arrow-mach-carpet-CN`.  There are five curves in each
of the two figures, each with a different color.  Each individual curve is a
Mach sweep at a constant angle of attack.
        
    .. _fig-pycart-ex07-fuselage-mach-carpet-CLM:
    .. figure:: mach-carpet/mach_fuse_CLM.*
        :width: 3.8 in
        
        Mach sweeps of ``fuselage`` pitching moment

    .. _fig-pycart-ex07-arrow-mach-carpet-CN:
    .. figure:: mach-carpet/mach_arrow_CN.*
        :width: 3.8 in
        
        Mach sweeps of ``fuselage`` normal force coefficient
    
This is probably the most informative type of plot for a CFD configuration if
the main product is a force & moment database.  For example
:numref:`fig-pycart-ex07-fuselage-mach-carpet-CLM` shows that the fuselage on
its own transitions from stable to unstable at Mach 1 (although the fins more
than make up for the static instability with the moment reference point).
:numref:`fig-pycart-ex07-arrow-mach-carpet-CN` shows that the overall normal
force coefficient is mainly a function of angle of attack but with a spike
around Mach 1.

Angle of Attack Sweeps
^^^^^^^^^^^^^^^^^^^^^^
Reconfiguring these plots to be angle of attack sweeps is straightforward.
:numref:`fig-pycart-ex07-fuselage-alpha-carpet-CLM` is the counterpart to
:numref:`fig-pycart-ex07-fuselage-mach-carpet-CLM`, and
:numref:`fig-pycart-ex07-arrow-alpha-carpet-CN` is the counterpart to
:numref:`fig-pycart-ex07-arrow-mach-carpet-CN`.  These plots are created by
running ``pycart --report alpha-carpet``.

        
    .. _fig-pycart-ex07-fuselage-alpha-carpet-CLM:
    .. figure:: alpha-carpet/aoa_fuse_CLM.*
        :width: 3.8 in
        
        Alpha sweeps of ``fuselage`` pitching moment

    .. _fig-pycart-ex07-arrow-alpha-carpet-CN:
    .. figure:: alpha-carpet/aoa_arrow_CN.*
        :width: 3.8 in
        
        Alpha sweeps of ``fuselage`` normal force coefficient

The trends with angle of attack are relatively straightforward.  In this narrow
range of angle of attack, it anticipated that the normal force would be linear
with *alpha*.  Interestingly, the fuselage *CLM* vs *alpha* curve has a stable
slope only at Mach 0.5 (and kind of 0.8).


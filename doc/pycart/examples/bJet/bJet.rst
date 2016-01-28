
Demo 4: Business Jet, Data Book, and Automated Reports
======================================================

The following example uses a more complex geometry to demonstrate Cart3D's
capabilities and the ease with which complex geometries can be analyzed.  Let's
run the first case in the ``$PYCART/examples/pycart/bJet`` folder.

    .. code-block:: none
    
        $ pycart -n 1
        Case Config/Run Directory   Status  Iterations  Que CPU Time
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m0.84a0.0b0.0 ---     /           .   
          Case name: 'poweroff/m0.84a0.0b0.0' (index 0)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
             Writing triangulation: 'Components.i.tri'
         > autoInputs -r 8 -t Components.i.tri -maxR 11 -nDiv 4
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'autoInputs.out')
         > cubes -pre preSpec.c3d.cntl -maxR 11 -reorder -a 10 -b 2
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'cubes.out')
         > mgPrep -n 3
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'mgPrep.out')
             Starting case 'poweroff/m0.84a0.0b0.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -tm 0
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'flowCart.out')
         > flowCart -his -clic -restart -N 300 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -tm 1
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'flowCart.out')
        
        Submitted or ran 1 job(s).
        
        ---=1, 

A sample graphic of the surface pressure made with ParaView is shown below.

    .. figure:: bJet01.png
        :width: 5in
    
        Business jet triangulation colored by pressure coefficient from
        ``poweroff/m0.84a0.0b0.0`` solution
        
Phase Control
-------------
This is an example with more complex geometry, obviously, but it also is an
example of a two-phase solution procedure. This can be seen in the sample output
above because there are two ``flowCart`` commands. The following is a snippet
from the :file:`pyCart.json` file.

    .. code-block:: javascript
    
        "RunControl": {
            // Run sequence
            "PhaseSequence": [0, 1],
            "PhaseIters": [0, 300],
            // Number of threads
            "nProc": 4,
            
            // flowCart settings
            "flowCart": {
                "it_fc": [200, 100],
                "first_order": [1, 0],
                "cfl": 1.1,
                "mg_fc": 3,
                "limiter": 2,
                "tm": [0, 1],
                "y_is_spanwise": false,
                "binaryIO": false,
                "tecO": true
            },
            
            // Parameters for autoInputs
            "autoInputs": {
                "r": 8,
                "nDiv": 4
            },
            
            // Parameters for cubes
            "cubes": {
                "maxR": 11,
                "pre": "preSpec.c3d.cntl",
                "cubes_a": 10,
                "cubes_b": 2,
                "reorder": true
            }
        },
        
In this ``"RunControl"`` section is the option ``"PhaseSequence": [0, 1]``, which
tells pyCart to run phase 0 followed by phase 1.  Phase 0 is run exactly once
because *PhaseIters[0]* is ``0``, and phase 1 is repeated until at least
*PhaseIters[1]* total (i.e., including previous phases) iterations have been
completed.

The *it_fc* option inside the ``"flowCart"`` section specifies how many
iterations in each call to ``flowCart``.  In this case, *it_fc[0]* is ``200``,
so phase 0 runs for 200 iterations, and hence the ``flowCart -N 200`` command
above.  Since *it_fc[1]* is 100, phase 1 runs ``flowCart -restart -N 300``,
where *N* is the **total** number of iterations at which ``flowCart`` exits.
The dual nature of the *first_order* option means that phase 0 is run in
first-order mode while subsequent phases will all be second-order.  All the
other options in the ``"flowCart"`` section that are not specified as a list use
the same option for all phases.

Configuration
-------------
Let's also look at the ``"Config"`` section of :file:`pyCart.json`.

    .. code-block:: javascript
    
        // Describe the reference values and config.
        "Config": {
            // Defer to a file for most things.
            "File": "Config.xml",
            // Which forces should be reported
            "Force": ["fuselage", "wing", "htail", "vtail", "engines"],
            // Reference values
            "RefArea": 1005.3,
            "RefLength": 66.3,
            // The moment point can be specified as a dictionary of components.
            "Points": {"MRP": [0.0, 0.0, 0.0]},
            "RefPoint": {
                "fuselage": "MRP",
                "wing":     "MRP",
                "htail":    "MRP",
                "vtail":    "MRP",
                "engines":  "MRP"
            }
        },

The *Force* section lists out the components for which iterative force histories
are reported while running ``flowCart``.  Similarly, the *RefPoint* section
specifies which components will also have aerodynamic moments reported.  An
interesting feature demonstrated in this example is how the moment reference
point is not defined directly for each component.  Instead, a common reference
point is defined in the *Points* variable, and pyCart automatically refers to
this point when creating Cart3D's standard :file:`input.cntl` input file.  This
saves a little bit of effort if a reference point happens to move a little bit,
but it is also useful in cases where reference points may shift from case to
case---for example when studying a separation problem or moving fins.

Database Management
-------------------
Let's also look at some of pyCart's database management capabilities.  In
particular, we'll look at automated calculation of mean values and standard
deviations of aerodynamic forces and moments.

Much like the ``"Config"``, section, the data book, which is controlled by the
``"DataBook"`` section of :file:`pyCart.json`, needs a list of components to
keep track of.  In the JSON file snippet below taken from the
:file:`pyCart.json` file from the business jet example, we're tracking five
components, and we are recording both the forces and moments for each.

    .. code-block:: javascript
    
        "DataBook": {
            // List of components to place in data book
            "Components": ["fuselage", "wing", "htail", "vtail", "engines"],
            // Number of iterations to use for statistics.
            "nStats": 50,
            "nMin": 200,
            // Place to put the data book
            "Folder": "data",
            // Information about each component.
            "fuselage": {"Type": "FM"},
            "wing":     {"Type": "FM"},
            "htail":    {"Type": "FM"},
            "vtail":    {"Type": "FM"},
            "engines":  {"Type": "FM"}
        },
        
The ``{"Type": "FM"}`` specifier just means that its a default force & moment
component.  Another common value of *Type* is ``"Force"``, which just ignores
any moment histories.  These are pretty vanilla data book component definitions;
it is also possible to specify a transformation if you want to resolve the
forces and/or moments in a different coordinate system or scale some of the
results.

Two other important parameters are *nStats* and *nMin*.  The *nMin* parameter in
this case means that only iterations after iteration 200 can be used to compute
the mean value and standard deviation in the database.  Using this *nMin*
parameter is a good error-prevention technique because it automatically leaves
holes in the database for cases that have not run sufficiently far.  The
*nStats* parameter means that pyCart will use the last 50 iterations available
to compute the mean.

To create or update the data book, run the following command.

    .. code-block:: none
    
        $ pycart --aero
        poweroff/m0.84a0.0b0.0
          Adding new databook entry at iteration 300.
        poweroff/m0.84a2.0b0.0
        poweroff/m0.88a0.0b0.0
        poweroff/m0.88a2.0b0.0
        
In this case, ``pycart`` runs through the run matrix (it is possible to restrict
this command to a subset of cases just like any ``pycart`` command) and checks
if any case meets the criteria to be entered into the databook.  Every case must
be run at least *nMin* + *nStats* iterations.  This creates a few files in the
``data/`` folder.  Specifically, there is a ``aero_$COMP.csv`` file for each
*COMP* in the ``"Components"`` field.  As an example, the contents of
:file:`aero_fuselage.csv` are the following.

    .. code-block:: none
    
        # aero data for 'fuselage' extracted on 2016-01-27 16:38:05 
        #
        # Reference Area = 1.005300E+03
        # Reference Length = 6.630000E+01
        # Nominal moment reference point:
        # XMRP = 0.000000E+00
        # YMRP = 0.000000E+00
        # ZMRP = 0.000000E+00
        #
        # Mach,alpha,beta,config,Label,CA,CY,CN,CLL,CLM,CLN,CA_min,CA_max,
            CA_std,CA_err,CY_min,CY_max,CY_std,CY_err,CN_min,CN_max,CN_std,
            CN_err,CLL_min,CLL_max,CLL_std,CLL_err,CLM_min,CLM_max,CLM_std,
            CLM_err,CLN_min,CLN_max,CLN_std,CLN_err,nOrders,nIter,nStats
        0.84,0.0,0.0,poweroff,,8.93902000E-03,-4.91405000E-03,9.77648294E-06,
            1.94313000E-06,-4.68098922E-06,-1.51877000E-03,8.93902000E-03,
            8.93902000E-03,6.93889390E-18,0.00000000E+00,-4.91405000E-03,
            -4.91405000E-03,2.60208521E-18,0.00000000E+00,9.77640000E-06,
            9.77653000E-06,3.80553837E-11,1.36840069E-11,1.94313000E-06,
            1.94313000E-06,0.00000000E+00,0.00000000E+00,-4.68100000E-06,
            -4.68098000E-06,6.52089771E-12,2.16772288E-12,-1.51877000E-03,
            -1.51877000E-03,1.30104261E-18,8.03348895E-20,6.7302,300,50

This is a fairly self-explanatory file in which lines starting with ``#`` are
comments. The indentations shown in the sample are line continuations; the
actual contents of the file contains two very long lines. 

Automated Reports
-----------------
This business jet also contains a demo of pyCart's automated report capability.
Calling ``pyCart --report`` results in a multi-page PDF created using LaTeX.
There are two modes for these reports: one creates various figures for each case
in the run matrix, and the other creates various plots for groups of cases.  The
example below shows the set of plots for the one case we've run in this example.

    .. figure:: report-case.*
        :width: 5.5in
        
    Example report page for case ``poweroff/m0.84a0.0b0.0``

This is the second page of the report generated from the command

    .. code-block:: none
    
        $ pycart -I 0 --report

It contains two tables; one of these summarizes the run conditions (i.e., the
values of the run matrix input variables), and the other presents selected force
and moment results.  Then there is a set of nine plots that show selected
quantities at each iteration.  A higher-resolution view of the residual history
plot is below.

    .. figure:: L1.*
        :width: 3.5in
    
    L1 density residual history for ``poweroff/m0.84a0.0b0.0``
    
The settings for this automated report are specified in the ``"Report"`` section
of :file:`pyCart.json`.



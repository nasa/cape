
.. _cape-json-DataBook:

---------------------
Data Book Definitions
---------------------

The data book defines quantities that Cape tracks as results.  It also allows
for the definition of "Target" data for comparison to any CFD results.  Data
book components can be integrated forces and moments, point sensors, line loads,
or potentially other subjects.  The following JSON example shows some of the
capabilities.

    .. code-block:: javascript
    
        "DataBook": {
            // List of components or data book items
            "Components": ["fuselage", "wing", "tail", "engines", "P1"],
            // Number of iterations to use for statistics
            "nStats": 50,
            "nMin": 200,
            // Place to put the data book
            "Folder": "data"
            // Sorting order
            "Sort": ["mach", "alpha", "beta"],
            // Information for each component
            "fuselage": {
                "Type": "FM",
                "Targets": {"CA": "WT/CAFC", "CY": "WT/CY", "CN": "WT/CN"}
            },
            "wing":     {"Type": "FM"},
            "tail":     {"Type": "FM"},
            "engines":  {"Type": "Force"},
            // Information for a point sensor
            "P1": {
                "Type": "PointSensor",
                "Points": ["P101", "P102"],
                "nStats": 20
            },
            // Target data from a wind tunnel
            "Targets": {
                "WT": {
                    "File": "data/wt.csv",
                    "Trajectory": {"mach": "Mach"},
                    "Tolerances": {
                        "mach": 0.02, "alpha": 0.25, "beta": 0.1
                    }
                }
            }
        }
        
Links to default options for each specific solver are found below.

    * :ref:`Cart3D <pycart-json-DataBook>`
    * :ref:`FUN3D <pyfun-json-DataBook>`
    * :ref:`OVERFLOW <pyover-json-DataBook>`
        
General Data Book Definitions
=============================

The following dictionary of options describes the general definitions of the
data book.  These are applicable to any data book for any of the solvers.

    *Components*: :class:`list` (:class:`str`)
        List of components to analyze and create data book entries for
        
    *nStats*: [ {``1``} | :class:`int` ]
        Number of iterations to use for computing statistics (such as mean,
        iterative history min and max, and standard deviation)
    
    *nMin*: [ {``0``} | :class:`int` ]
        Minimum iteration number allowed for inclusion in results and
        statistics; a case must have at least ``nMin+nStats`` iterations for
        inclusion in the data book
        
    *nMaxStats*: [ {``None``} | :class:`int` ]
        Optional parameter for maximum number of iterations for inclusion in
        statistics; Cape will use ``nStats<=n<=nMaxStats`` iterations based on
        its estimate of resulting iterative uncertainty
    
    *Folder*: :class:`str`
        Location in which to store data book (relative to JSON root)
        
    *Sort*: :class:`str` | :class:`list` (:class:`str`)
        Trajectory key(s) on which to sort data book (in reverse order if a
        :class:`list`); ignored if not the name of a trajectory variable
        
Each component in *Components* can be either a force, moment, force & moment,
point sensor, or alternative data type defined for specific solvers.  For each
force and/or moment component, a file such as :file:`aero_fuselage.csv`,
:file:`aero_wing.csv`, etc. is created in the location specified by *Folder*.
The *Folder* is created if necessary when reading/writing the data book.

Point sensor components create files such as :file:`pt_P101.csv` and
:file:`pt_P102.csv` in the same directory.

The *Sort* option, if specified, is used by Cape to sort he lines of the data
book files before writing to file.  Without a *Sort* key, lines in the data book
files are written in the order in which they were processed.

.. _cape-json-DBComp:

Data Book Component Definitions
===============================

Each data book component has at least the following possible options.
Additional components may have more options that are not defined in this
universal context.

The values for *nStats*, *nMin*, and *nMaxStats* are optional and will default
to those from the general data book.  This capability enables users to specify
different values of these parameters for different components.

    *comp*: :class:`dict`
        Definition for individual component *comp*

        *Type*: {``"FM"``} | :class:`str`
            Type of component being tracked
            
        *Coefficients*: :class:`list` (:class:`str`)
            List of coefficients for this component
            
        *Transformations*: {``[]``} | :class:`list` (:class:`dict`)
            List of transformation dictionaries
        
        *Targets*: {``{}``} | ``T`` | :class:`dict` (:class:`str`)
            Dictionary of column names for target values
        
        *nStats*: [ {``1``} | :class:`int` ]
            Number of iterations to use for computing statistics 
        
        *nMin*: [ {``0``} | :class:`int` ]
            Minimum iteration number allowed for inclusion in results
            
        *nMaxStats*: [ {``None``} | :class:`int` ]
            Maximum number of iterations for inclusion in statistics
            
        *CA*: {``["mu","min","max","std","err"]``} | :class:`list` (:class:`str`)
            List of statistical properties for coefficient *CA*
        
        *Cp*: {``["mu","std","min","max"]``} | :class:`list` (:class:`str`)
            List of statistical properties for coefficient *Cp*
            
Each coefficient can have a list of statistical properties defined for it.  The
example above only lists *CA* and *Cp*, but this may be defined for any
coefficient defined in the *Coefficients* section.
    
.. _cape-json-DBTransformation:

Data Book Transformations
-------------------------
Each data book may have a list of transformations.  This may be useful, for
example, if a component has been rotated with respect to the CFD axes, and the
forces and moments, which have been recorded in the CFD axes, need to be
transformed to reflect the correct values in the body frame of the rotated
component.  Another example is that moment coefficients are often recorded about
the opposite-direction *x* and *z* axes, and so it is necessary to multiply the
values of *CLL* and *CLN* by ``-1``.

    *Transformations*: ``[E]`` | ``[S]`` | :class:`list` (:class:`dict`)
        List of data transformations
        
        *E*: :class:`dict`
            Definition for an Euler 3-2-1 transformation
            
            *Type*: ``"Euler321"``
                Specify the transformation type
                
            *phi*: {``"phi"``} | ``"-phi"`` | :class:`str`
                Name of the trajectory variable to use for the roll angle
                transformation value
                
            *theta*: {``"theta"``} | ``"-theta"`` | :class:`str`
                Name of the trajectory variable to use for the pitch angle
                transformation value
                
            *psi*: {``"psi"``} | ``"-psi"`` | :class:`str`
                Name of the trajectory variable to use for the yaw angle
                transformation value
                
        *S*: :class:`dict`
            Definition for coefficient scaling
            
            *Type*: ``"ScaleCoeffs"``
                Specify the transformation type
                
            *CA*: {``1.0``} | :class:`float`
                Scale factor by which to multiply *CA* values
                
            *CY*: {``1.0``} | ``-1.0`` | :class:`float`
                Scale factor by which to multiply *CY* values
                
            *CLL*: ``1.0`` | {``-1.0``} | :class:`float`
                Scale factor by which to multiply *CLL* values
                
            *CLN*: ``1.0`` | {``-1.0``} | :class:`float`
                Scale factor by which to multiply *CLN* values
                
.. _cape-json-DBCompTarget:

Data Book Component Target Definitions
--------------------------------------
Defining target values for a data book component is relatively straightforward.
The target data book, defined below, may use column names differing from *CA*,
*CY*, etc., and this section is used to specify what columns to use.  By
default, Cape will try to compare to a target value from the first target
defined; to be more explicit, one can use a forward slash to specify both which
target to use and which column within that target.  For example, to use a column
named *CAFC* in target *WT* use ``"WT/CAFC"`` in the JSON file.

    *Targets*: {``{}``} | ``T`` | :class:`dict` (:class:`str`)
        Dictionary of column names for target values
        
        *T*: :class:`dict`
            Dictionary of target values
            
            *CA*: {``"CA"``} | ``targ/col`` | :class:`str`
                Name of column for target *CA* value; column *col* from
                target *targ* where *targ* defaults to first column if this
                field does not contain a forward slash
            
            *Cp*: {``"Cp"``} | ``targ/col`` | :class:`str`
                Name of column for target *Cp* value; column *col* from
                target *targ* where *targ* defaults to first column if this
                field does not contain a forward slash
            
        
.. _cape-json-DBCompFM:

Force and Moment Data Book Components
-------------------------------------
The following dictionary fills in some of the values applicable to force and
moment objects.  It is a more specific version of the :ref:`more general
dictionary <cape-json-DBComp>`. 

    *comp*: :class:`dict`
        Definition for individual component *comp*

        *Type*: {``"FM"``} | ``"Force"`` | ``"Moment"``
            Type of component being tracked
            
        *Coefficients*: :class:`list` (:class:`str`)
            List of coefficients for this component
            
        *Transformations*: {``[S]``} | :class:`list` (:class:`dict`)
            List of transformation dictionaries
            
            *S*: :class:`dict`
                Transformation dictionary
                
                *Type*: {``"ScaleCoeffs"``} | ``"Euler321"``
                    Transformation type
        
        *Targets*: {``{}``} | ``T`` | :class:`dict` (:class:`str`)
            Dictionary of column names for target values
        
        *CA*: {``["mu","min","max","std","err"]``} | :class:`list` (:class:`str`)
            List of statistical properties for axial force coefficient
            
        *CY*: {``["mu","min","max","std","err"]``} | :class:`list` (:class:`str`)
            List of statistical properties for lateral force coefficient
            
        *CN*: {``["mu","min","max","std","err"]``} | :class:`list` (:class:`str`)
            List of statistical properties for normal force coefficient
            
        *CLL*: {``["mu","min","max","std","err"]``} | :class:`list` (:class:`str`)
            List of statistical properties for rolling moment coefficient
            
        *CLM*: {``["mu","min","max","std","err"]``} | :class:`list` (:class:`str`)
            List of statistical properties for pitching moment coefficient
            
        *CLN*: {``["mu","min","max","std","err"]``} | :class:`list` (:class:`str`)
            List of statistical properties for yawing moment coefficient
            
        *CL*: {``["mu","min","max","std","err"]``} | :class:`list` (:class:`str`)
            List of statistical properties for lift coefficient
            
        *CD*: {``["mu","min","max","std","err"]``} | :class:`list` (:class:`str`)
            List of statistical properties for drag coefficient
            
.. _cape-json-DBCompPointSensor:
        
Point Sensor Data Book Components
---------------------------------
The following dictionary fills in some of the values applicable to point sensor
objects.  It is a more specific version of the :ref:`more general
dictionary <cape-json-DBComp>`.  Furthermore, the list of available coefficients
and their particular interpretation varies from solver to solver.

    *comp*: :class:`dict`
        Definition for individual component *comp*

        *Type*: {``"FM"``} | ``"Force"`` | ``"Moment"``
            Type of component being tracked
            
        *Coefficients*: :class:`list` (:class:`str`)
            List of coefficients for this component
        
        *Targets*: {``{}``} | ``T`` | :class:`dict` (:class:`str`)
            Dictionary of column names for target values
        
        *Cp*: {``["mu","std","min","max"]``} | :class:`list` (:class:`str`)
            List of statistical properties for pressure coefficient
            
        *M*: {``["mu","std","min","max"]``} | :class:`list` (:class:`str`)
            List of statistical properties for Mach number
            
        *rho*: {``["mu","std","min","max"]``} | :class:`list` (:class:`str`)
            Statistical properties for density or nondimensional density
            
        *p*: {``["mu","std","min","max"]``} | :class:`list` (:class:`str`)
            Statistical properties for pressure or nondimensional pressure
            
        *T*: {``["mu","std","min","max"]``} | :class:`list` (:class:`str`)
            Statistical properties for temperature or nondimensional temperature
            
.. _cape-json-DBTarget:

Target or Comparison Data Sources
=================================

The *Targets* key is an optional parameter that points to another data source
(or multiple other data sources) for use as a reference value both in the data
book files and plots. Each "Target" is read from a single file that contains
columns used to map points in that file to run matrix conditions and one or more
force/moment coefficients for one or more components in the data book. The list
of *Targets* parameters is given below.

    *Targets*: {``{}``} | ``{targ: T}`` | :class:`dict` (:class:`dict`)
        List of target dict descriptions
        
        *targ*: :class:`str`
            Name of target
        
        *T*: :class:`dict`
            Individual target description
            
            *Label*: :class:`str`
                Label to be used for this data source, e.g. in plot legends;
                defaults to value of *Name* option
                
            *File*: :class:`str`
                File name of the data source
            
            *Delimiter*: {``", "``} | ``","`` | ``" "`` | :class:`str`
                Delimiter to be used when reading/writing data book files
            
            *Comment*: {``"#"``} | :class:`str`
                Character used to denote comment line in source file
                
            *Components*: :class:`list` (:class:`str`)
                List of components to which this target file applies; default is
                all components in the data book
            
            *Trajectory*: :class:`dict` (:class:`str`)
                Dictionary of column names for trajectory variables to be used
                for comparing trajectory cases to target data points.  Any case
                that has matching values for all keys listed in this
                :class:`dict` will be considered to be at matching conditions
            
            *Tolerances*: :class:`dict` (:class:`float`)
                Dictionary of tolerances for matching rows of the target to data
                book points.  For example, including ``"mach": 0.02"`` means the
                target data point can differ in Mach number from the Cape data
                book point by up to 0.02 (inclusive)



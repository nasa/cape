
---------------------
Data Book Definitions
---------------------

The section in :file:`pyCart.json` labeled "DataBook" is an optional section
used to allow pyCart to create a data book of relevant coefficients and
statistics.  The section can also be used to define a series of plots that make
exploring the data book easier for the user and customer.

This section has very few defaults, so the following is an example that shows
many of the features of the pyCart data book.  The example is taken from the SLS
booster separation work that was done to compare to a wind tunnel test (UPWT
1891 at Langley Research Center's Unitary Plan Wind Tunnel), but with some of
the repetitive aspects removed for brevity.

    .. code-block:: javascript
    
        "DataBook": {
            "Components": ["CORE_No_Base", "RSRB_No_Base", "LSRB_No_Base"],
            "nStats": 5000,
            "Folder": "data/poweron",
            "Sort": "n",
            "Targets": [{
                "Name": "UPWT 1891",
                "File": "data/WT.poweron.dat",
                "Trajectory": {
                    "alpha": "ALPHAC",
                    "beta": "BETAC",
                    "dxR": "DXBR", "dyR": "DYBR", "dzR": "DZBR",
                    "dphiR": "DPHIBR",
                    "dthetaR": "DTHETABR",
                    "dpsiR": "DPSIBR"
                }
            }],
            "CORE_No_Base": {
                "Type": "FM",
                "Targets": {
                    "CA": "CAFC", "CLL": "CLLFC",
                    "CY": "CYFC", "CLM": "CLMFC",
                    "CN": "CNFC", "CLN": "CLNFC"
                }
            },
            "LSRB_No_Base": {
                "Type": "FM",
                "Targets": {
                    "CY": "CYFBL", "CLM": "CLMBL",
                    "CN": "CNFBL", "CLN": "CLNBL"
                },
                "Transformations": [
                    {"Type": "Euler321",
                        "psi": "dpsiL", "theta": "dthetaL", "phi": "dphiL"},
                    {"Type": "ScaleCoeffs", "CLL": -1.0, "CLN": -1.0}
                ]
            },
            "RSRB_No_Base": {
                "Type": "FM",
                "Targets": {
                    "CY": "CYFBR", "CLM": "CLMBR",
                    "CN": "CNFBR", "CLN": "CLNBR"
                },
                "Transformations": [
                    {"Type": "Euler321",
                        "psi": "dpsiR", "theta": "dthetaR", "phi": "dphiR"},
                    {"Type": "ScaleCoeffs", "CLL": -1.0, "CLN": -1.0}
                ]
            },
            "Plot": [
                {
                    "XAxis": "dxR",
                    "XLabel": "Axial SRB Displacement",
                    "YAxis": "CA",
                    "YLabel": "Axial force coefficient (CA)",
                    "Lable": "CORE",
                    "Components": ["CORE_No_Base"],
                    "Restriction": "SBU - ITAR",
                    "Sweep": {"alpha": 0.02, "beta": 0.05, "dyR": 0.05,
                        "dzR": 0.05, "dthetaR": 0.15, "dpsiR": 0.15},
                    "PlotOptions": {"lw": 2, "color": "k",
                        "marker": ["o", "s", "^", "*", "p"]},
                    "TargetOptions": {"lw": 1.5, "color": "r",
                        "marker": ["o", "s", "^", "*", "p"]},
                    "StandardDeviation": 1.0,
                    "StDevOptions": {"lw": 0.05, "color": "#001080",
                        "facecolor": "#9090EF", "alpha": 0.6},
                    "MinMax": false,
                    "MinMaxOptions": {"color": "g", "alpha": 0.4} 
                },
                {
                    "YAxis": "CY",
                    "YLabel": "Side force coefficient (CY)"
                },
                {
                    "YAxis": "CN",
                    "YLabel": "Normal force coefficient (CN)"
                },
                {
                    "Components": ["LSRB_No_Base", "RSRB_No_Base"],
                    "Label": "SRB",
                    "YAxis": "CY",
                    "YLabel": "Side force coefficient (CY)"
                },
                {
                    "YAxis": "CN",
                    "YLabel": "Normal force coefficient (CN)"
                }
            ]
        }

Clearly there are a lot of pieces to the data book definition (many of which are
optional).  These can be roughly divided into the following sections: general
definitions and setup, target definitions for comparing results, component
definitions, and plotting directives.

General Data Book Definitions
=============================

The following dictionary of options describes the general options.

    *Components*: :class:`list` (:class:`str`)
        List of components to analyze and create data book entries for
        
    *nStats*: :class:`int`
        Number of iterations to use for computing statistics (such as mean,
        iterative history min and max, and standard deviation)
    
    *Folder*: :class:`str`
        Location in which to store data book (relative to pyCart root)
        
    *Sort*: :class:`str`
        Trajectory key on which to sort data book; ignored if empty or not the
        name of a trajectory variable
        
These options are relatively straightforward.  The result of creating or
updating the data book will be a file such as :file:`aero_CORE_No_Base.dat`,
:file:`aero_LSRB_No_Base.dat`, etc. for each component in the *Components* list.
These files will be placed in the location *Folder*, which is created if
necessary.

When pyCart updates the data book, it only updates cases that from the active
trajectory that have new iterations.  Meanwhile, the data book can contain
results that are not in the current trajectory (for example if the user has
commented out some lines of the current trajectory file).

The *Sort* key, if specified causes pyCart to sort the lines of those data book
files before writing them.  The data is basically considered to be unsorted by
pyCart (search routines are used before collecting plot data, for example), but
having a small amount of organization in the files helps maintain sanity for a
user that inspects them manually.

Target or Comparison Data Sources
=================================

The *Targets* key is an optional parameter that points to another data source
for use as a reference value both in the data book files and any plots.
Currently these data sources must be a single file, and the user specifies in
this section the location for that file, a label for the data source, and a list
of columns that correspond to the trajectory variables.  The list of *Targets*
parameters is given below.

    *Targets*: {``[]``} | ``[T]`` | :class:`list` (:class:`dict`)
        List of target dict descriptions
        
        *T*: :class:`dict`
            Individual target description
            
            *Name*: :class:`str`
                Label to be used for this data source
                
            *File*: :class:`str`
                File name of the data source
            
            *Delimiter*: {``", "``} | ``","`` | ``" "`` | :class:`str`
                Delimiter to be used when reading/writing data book files
            
            *Trajectory*: :class:`dict` (:class:`str`)
                Dictionary of column names for trajectory variables to be used
                for comparing trajectory cases to target data points.  Any case
                that has matching values for all keys listed in this
                :class:`dict` will be considered to be at matching conditions
                
Data Book Component Definitions
===============================

Each component listed in *DataBook["Components"]* must have its own definition
section.  In some cases these will be relatively trivial, but there are also
several customization options available for more complex scenarios.  The
following list gives a description of available parameters; they are all
optional except *Type*, and that has a default value.

    *comp*: :class:`dict`
        An individual component description
        
        *Type*: {``"FM"``} | ``"Force"`` | ``"Moment"``
            Specifies which coefficients to analyze
            
        *Targets*: :class:`dict`
            Dictionary of coefficients and target column names.  This takes the
            form of key names that are force or moment coefficients with key
            values of target column names.  For example ``{"CA": "CAFC"}`` tells
            pyCart to compare *CA* from the current component to the column from
            the target called *CAFC*.  If there is more than one target data
            source, use the *Name* of the target followed by a forward slash,
            e.g. ``{"CA": "UPWT1891/CAFC"}``.
            
        *Transformations*: ``[E]`` | ``[S]`` | :class:`list` (:class:`dict`)
            List of transformation dictionaries.  This can be useful if the
            component is rotated with respect to the Cart3D axes, for example.
            It can also be used to transform the coefficients to the stability
            axes.
            
            *E*: :class:`dict`
                Definition for an Euler 3-2-1 transformation
                
                *Type*: ``"Euler321"``
                    Specify the transformation type
                    
                *phi*: {``"phi"``} | :class:`str`
                    Name of the trajectory variable to use for the roll angle
                    transformation value
                    
                *theta*: {``"theta"``} | :class:`str`
                    Name of the trajectory variable to use for the pitch angle
                    transformation value
                    
                *psi*: {``"psi"``} | :class:`str`
                    Name of the trajectory variable to use for the yaw angle
                    transformation value
                    
            *T*: :class:`dict`
                Definition for coefficient scaling
                
                *Type*: ``"ScaleCoeffs"``
                    Specify the transformation type
                    
                *CA*: {``1.0``} | :class:`float`
                    Scale factor by which to multiply *CA* values
                    
                *CY*: {``1.0``} | ``-1.0`` | :class:`float`
                    Scale factor by which to multiply *CY* values
                    
                *CLN*: {``1.0``} | ``-1.0`` | :class:`float`
                    Scale factor by which to multiply *CLN* values
                    
Data Book Plotting Options
==========================

This is the section in which the user can specify optional plots to sweep
through the data book in various ways in order to gain a better understanding of
the results.

The section is essentially a list of :class:`dict`\ s that each specify a single
type of plot.  However, pyCart has the space-saving feature that any option that
is not specified in a plot defaults to whatever value it had in the previous
plot.  Suppose that you have decided that you want all the Cart3D data to be
plotted with black lines, and all the target data should be plotted with red
lines.  Then you would set this in the *PlotOptions* and *TargetOptions* keys,
respectively, for the first plot and never enter those options again for the
remaining plots.  In the example at the top of this page, that is indeed the
method that was used.

The list below is a dictionary for the possible options for an individual plot. 
The required options are *XAxis*, *YAxis*, *Sweep*, and *Components*, but keep
in mind that these don't necessarily need to be defined for each plot if the
values from the previous plot are appropriate.

    *P*: :class:`dict`
        Specifications for an individual plot; each parameter defaults to
        previous plot's value
        
        *XAxis*: :class:`str` [required]
            Name of trajectory variable to use as x-axis for data book sweeps
            
        *XLabel*: {*XAxis*} | :class:`str`
            Text to use for label on plot's x-axis.
            
        *YAxis*: :class:`str` [required]
            Name of trajectory variable to use as y-axis
            
        *YLabel*: {*YAxis*} | :class:`str`
            Text to use for label on plot's y-axis
            
        *Components*: :class:`list` (:class:`str`) [required]
            List of components to include in plot
            
        *Label*: {``"-".join(Components)``} | :class:`str`
            Label to use for this plot in the output file name
            
        *Sweep*: ``{"x0": v0}`` | :class:`dict` (:class:`float`) [required]
            Dictionary of trajectory variable tolerances used to define
            parameter sweeps.  For example, if the trajectory keys are
            ``"Mach"``, ``"alpha"``, ``"beta"``, and you want to plot sweeps
            versus Mach number for fixed *alpha* and *beta*, use ``"Sweep":
            {"alpha": 0.01, "beta": 0.01}``
            
            *x0*: :class:`str`
                Name of a trajectory key used to define sweeps
                
            *v0*: :class:`float` >=0
                Tolerance to allow in variation during the sweep
                
        *Restriction*: {``""``} | ``"SBU - ITAR"`` | :class:`str`
            Distribution limitation to print at bottom center of each plot
            
        *PlotOptions*: {``{"color": "k", "marker": "^"}``} | :class:`dict`
            Dictionary of plot options for Cart3D data.  See
            `matplotlib.pyplot.plot
            <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_
            for details.  If a parameter is a list, e.g. ``{"marker": ["^", "s",
            "o"]}``, pyCart will loop through the values for each line in the
            series.
            
        *TargetOptions*: {``{"color": "r", "marker": "o"}``} | :class:`dict`
            Dictionary of plot options for reference (target) data.  See above
            for further details.
            
        *StandardDeviation*: {``0``} | :class:`float`
            If this parameter is nonzero, a filled region is plotted above and
            below the mean  of *YAxis* from the iterative history is plotted.
            The height of the filled region is 2 times *StandardDeviation* times
            the iterative history standard deviation at each point.
            
        *StDevOptions*: ``{"alpha": 0.5, "lw": 0.2}`` | :class:`dict`
            Dictionary of plot options for standard deviation window. See
            `matplotlib.pyplot.fill_between
            <http://matplotlib.org/
            api/pyplot_api.html#matplotlib.pyplot.fill_between>`_ for full
            description of possible options.
            
        *MinMax*: ``true`` | {``false``}
            If ``true``, plot the minimum and maximum value of *YAxis* from the
            last *nStats* iterations at each point.
            
        *Carpet*: {``{}``} | :class:`dict`
            This variable has the same structure as *Sweep* from above but has a
            roughly perpendicular function.  Suppose that the trajectory keys
            are ``"Mach"``, ``"alpha"``, and ``"beta"``.  To get sweeps versus
            Mach number for different values of *alpha*, but with all lines on
            the same plot having the same value of *beta*, set *Sweep* equal to
            ``{"beta": 0.01}``, and set *Carpet* equal to ``{"alpha": 0.01}``.
            

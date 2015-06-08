
---------------------
Data Book Definitions
---------------------

The section in :file:`pyCart.json` labeled "DataBook" is an optional section
used to allow pyCart to create a data book of relevant coefficients and
statistics.  It enables collecting key results and statistics for entire run
matrices into a small number of files.

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
            }
        }

Clearly there are a lot of pieces to the data book definition (many of which are
optional).  These can be roughly divided into the following sections: general
definitions and setup, target definitions for comparing results,  and component
definitions.

General Data Book Definitions
=============================

The following dictionary of options describes the general options.

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
        statistics; pyCart will use ``nStats<=n<=nMaxStats`` iterations based on
        its estimate of resulting iterative uncertainty
    
    *Folder*: :class:`str`
        Location in which to store data book (relative to pyCart root)
        
    *Sort*: :class:`str` | :class:`list` (:class:`str`)
        Trajectory key(s) on which to sort data book (in reverse order if a
        :class:`list`); ignored if not the name of a trajectory variable
        
These options are relatively straightforward.  The result of creating or
updating the data book will be a file such as :file:`aero_CORE_No_Base.csb`,
:file:`aero_LSRB_No_Base.csv`, etc. for each component in the *Components* list.
These files will be placed in the location *Folder*, which is created if
necessary.

When pyCart updates the data book, it only updates cases that from the active
trajectory that have new iterations.  Meanwhile, the data book can contain
results that are not in the current trajectory (for example if the user has
commented out some lines of the current trajectory file or multiple run matrices
are combined into a common data book).

The *Sort* key, if specified causes pyCart to sort the lines of those data book
files before writing them.  The data is basically considered to be unsorted by
pyCart (search routines are used before collecting plot data, for example), but
having a small amount of organization in the files helps maintain sanity for a
user that inspects them manually or other tool that uses the data book.

Target or Comparison Data Sources
=================================

The *Targets* key is an optional parameter that points to another data source
(or multiple other data sources) for use as a reference value both in the data
book files and plots. Each "Target" is read from a single file that contains
columns used to map points in that file to run matrix conditions and one or more
force/moment coefficients for one or more components in the data book. The list
of *Targets* parameters is given below.

    *Targets*: {``[]``} | ``[T]`` | :class:`list` (:class:`dict`)
        List of target dict descriptions
        
        *T*: :class:`dict`
            Individual target description
            
            *Name*: :class:`str`
                Identifier to be used for this label
            
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
                    
                *phi*: {``"phi"``} | ``"-phi"`` | :class:`str`
                    Name of the trajectory variable to use for the roll angle
                    transformation value
                    
                *theta*: {``"theta"``} | ``"-theta"`` | :class:`str`
                    Name of the trajectory variable to use for the pitch angle
                    transformation value
                    
                *psi*: {``"psi"``} | ``"-psi"`` | :class:`str`
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

            

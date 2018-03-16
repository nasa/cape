
.. _cape-json-Report:

--------------------------------------
Automated Report Generation with LaTeX
--------------------------------------

The section in :file:`pyCart.json` labeled "Report" is for generating automated
reports of results.  It requires a fairly complete installation of `pdfLaTeX`.
Further, an installation of Tecplot 360 or ParaView enhances the capability of
the report generation.

    .. code-block:: javascript
    
        "Report": {
            "Archive": false,
            "Reports": ["case", "mach"],
            "case": {
                "Title": "Automated Cart3D Report",
                "Subtitle": "Forces, Moments, \\& Residuals",
                "Author": "Cape Developers",
                "Affiliation": "NASA Ames",
                "Logo": "NASA_logo.pdf",
                "Frontispiece": "NASA_logo.pdf",
                "Restriction": "Distribution Unlimited",
                "Figures": ["Summary", "Forces"],
                "FailFigures": ["Summary", "Surface"],
                "ZeroFigures": ["Summary", "Surface"]
            },
            "mach": {
                "Title": "Results for Mach Sweeps",
                "Sweeps": "mach"
            },
            "Sweeps": {
                "mach": {
                    "Figures": ["SweepCond", "SweepCoeff"],
                    "EqCons": ["alpha", "beta"],
                    "XAxis": "mach"
                }
            },
            "Figures": {
                "Summary": {
                    "Alignment": "left",
                    "Subfigures": ["Conditions", "Summary"]
                },
                "Forces": {
                    "Alignment": "center",
                    "Header": "Force, moment, \\& residual histories",
                    "Subfigures": ["CA", "CY", "CN", "L1"]
                },
                "SweepCond": {
                    "Subfigures": ["SweepConds", "SweepCases"],
                },
                "SweepCoeff": {
                    "Subfigures": ["mach_CA", "mach_CN"],
                },
            },
            "Subfigures": {
                "Conditions": {
                    "Type": "Conditions",
                    "Alignment": "left",
                    "Width": 0.35,
                    "SkipVars": []
                },
                "Summary": {
                    "Type": "Summary"
                },
                "CA": {
                    "Type": "PlotCoeff",
                    "Component": "wing",
                    "Coefficient": "CA",
                    "Width": 0.5,
                    "StandardDeviation": 1.0, 
                    "nStats": 200
                },
                "CY": {"Type": "CA" "Coefficient": "CY"},
                "CN": {"Type": "CA" "Coefficient": "CN"},
                "L1": {"Type": "PlotL1"}
                "mach_CA": {
                    "Type": "SweepCoeff",
                    "Width": 0.5,
                    "Component": "wing",
                    "Coefficient": "CA"
                },
                "mach_CN": {"Type": "mach_CA", "Coefficient": "CN"}
            }
        }
        
Links to additional options for each specific solver are found below.

    * :ref:`Cart3D <pycart-json-Report>`
    * :ref:`FUN3D <pyfun-json-Report>`
    * :ref:`OVERFLOW <pyover-json-Report>`

These sections are put into action by calls of ``cape --report``, where
``cape`` can be replaced by ``pycart``, ``pyfun``, or ``pyover``, as
appropriate. This section enables some powerful capabilities, and it is often
the longest section of the JSON file.

There are three primary fields: "Sweeps", "Figures", and "Subfigures", along
with two minor settings of "Archive" and "Reports". The example above has a
report named "case" that produces one page for each solution and a report
called "mach" that creates sweeps of results from the data book. Users may
build a specific report with a command such as ``cape --report case`` (assuming
there is a report called ``"case"``). With no value (i.e. ``cape --report``),
the first report in the *Reports* field is created.

Because this section often becomes very long, a useful tool is to separate the
definitions into multiple JSON files.  Using the example above may allow the
user to replace that section with the following syntax.

    .. code-block:: javascript
    
        "Report": {
            
            "Archive": false,
            "Reports": ["case", "mach"],
            "case": JSONFile("Report-case.json")
            "mach": {
                "Title": "Results for Mach Sweeps",
                "Sweeps": "mach"
            },
            "Sweeps": JSONFile("Report-Sweeps.json")
            "Figures": JSONFile("Report-Figures.json")
            "Subfigures": JSONFile("Report-Subfigures.json")
        }

The base level option names for this parameter are described in dictionary
format below.

The description of the available options is shown below.  If *Reports* is not
defined, the list of reports is 

    *Reports*: :class:`list` (:class:`str`) | ``["R1", "R2"]``
        List of reports defined in this JSON file
        
    *Archive*: {``true``} | ``false``
        Whether or not to tar folders in the report folder in order to reduce
        file count
        
    *Sweeps*: ``{}`` | ``{[S]}`` | :class:`dict` (:class:`dict`)
        Dictionary of sweep definitions (combined plots of subsets of cases)
        
    *Figures*: ``{}`` | ``{[F]}`` | :class:`dict` (:class:`dict`)
        Dictionary if figure definitions
        
    *Subfigures*: ``{}`` | ``{[U]}`` | :class:`dict` (:class:`dict`)
        Dictionary of subfigure definitions to be used by the figures 
    
    *R1*: :class:`dict`
        Definition of report named ``"R1"``
        
    *R2*: :class:`dict`
        Definition of report named ``"R2"``

.. toctree::
    :maxdepth: 3
    
    main
    Sweeps
    Figures
    Subfigures/index
    

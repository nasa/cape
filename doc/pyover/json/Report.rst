
.. _pyover-json-Report:

--------------------------------------
Automated Report Generation with LaTeX
--------------------------------------

The section in :file:`pyOver.json` labeled "Report" is for generating automated
reports of results.  It requires a fairly complete installation of `pdfLaTeX`.
Further, an installation of Tecplot 360 or ParaView enhances the capability of
the report generation.

    .. code-block:: javascript
    
        "Report": {
            "Archive": false,
            "Reports": ["case", "mach"],
            "case": {
                "Title": "Automated OVERFLOW Report",
                "Author": "Cape Developers",
                "Restriction": "Distribution Unlimited",
                "Figures": ["Summary", "Forces"],
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
                    "Subfigures": ["CA", "L1"]
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
                    "Width": 0.5
                },
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
        
All settings, their meanings, and their possible values are described the
corresponding :ref:`Cape Report section <cape-json-Report>`.


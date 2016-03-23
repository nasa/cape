
.. _pyfun-json-DataBook:

---------------------
Data Book Definitions
---------------------

The section in :file:`pyFun.json` labeled "DataBook" is an optional section
used to allow pyFun to create a data book of relevant coefficients and
statistics.  It enables collecting key results and statistics for entire run
matrices into a small number of files.

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
        
All settings, their meanings, and their possible values are described the
corresponding :ref:`Cape DataBook section <cape-json-DataBook>`.


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
            "fuselage": {"Type": "FM"},
            "wing":     {"Type": "FM"},
            "tail":     {"Type": "FM"},
            "engines":  {"Type": "Force"},
            // Target data from a wind tunnel
            "Targets": {
                "WT": {
                    "File": "data/wt.csv",
                    "Trajectory": {"mach": "Mach"},
                    "Tolerances": {
                        "mach": 0.02, "alpha": 0.25, "beta": 0.1
                    }
                }
            },
            // Information for a point sensor
            "P1": {
                "Type": "PointSensor",
                "Points": ["P101", "P102"],
                "nStats": 20
            }
        }
        

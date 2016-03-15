
.. _cape-json-Config:

----------------------------------------
Solution Configuration & Component Names
----------------------------------------

The ``"Config"`` section of a Cape JSON file is used to give definitions of
components, parts of vehicles, points, and similar things.  This is generally
pretty customized for each solver, but some very common options are defined here
in :mod:`cape` to reduce duplicated code.

The JSON syntax with some nontrivial values is below.

    .. code-block:: javascript
    
        "Config": {
            "Components": ["total", "wing"],
            "ConfigFile": "Config.xml",
            "RefArea": 1.57,
            "RefLength": {
                "total": 0.5,
                "wing": 1.0
            },
            "RefPoint": "MRP",
            "Point": {
                "MRP": [0.0, 0.0, 0.0],
                "CG": [2.5. 0.0, 0.2]
            }
        }


.. _json-syntax:

JSON Files for pyCart
=====================

The various pyCart utilities and modules extensively utilize the 
`JSON <http://www.json.org>`_ format, which is a simple file type that is
easily read into Python and many other languages (a 
`list of interpreters <http://www.json.org>`_ can be found on the official
website).  In CAPE, these files are used to store many types of information.
In fact, using pyCart mostly boils down to three tasks:

    * preparation of input files for the CFD solver, including mesh generation,
    * interacting with JSON files, and
    * issuing the appropriate command-line commands
    
The JSON format originated with Javascript in mind, but we can think of it as
lightly modified Python syntax without too much trouble.  In Python terms, a
JSON file contains the definition for a single dictionary (that is,
:class:`dict`).  The keys of the :class:`dict` must all be strings using double
quotes, and the value of each key can be :class:`str`, :class:`int`,
:class:`float`, :class:`bool`, :class:`NoneType`, :class:`list`, or
:class:`dict`.

The following example shows all of the valid types that can be used in a JSON
file.

    .. code-block:: javascript
    
        {
            "RunControl": {
                "PhaseSequence": [0, 1],
                "PhaseIters": [200, 400],
                "nProc": 8,
                "MPI": true,
            },
            "Config": {
                "File": "Config.xml",
                "RefArea": 3.141593,
                "RefLength": 2.0,
                "RefPoint": null
            }
        }
        
This would be interpreted in Python as the following :class:`dict`.

    .. code-block:: python
    
        {
            "RunControl": {
                "PhaseSequence": [0, 1],
                "PhaseIters": [200, 400],
                "nProc": 8,
                "MPI": True,
            },
            "Config": {
                "File": "Config.xml",
                "RefArea": 3.141593,
                "RefLength": 2.0,
                "RefPoint": None
            }
        }

There are a few conversions between Python and JSON syntax:

    * Boolean parameters are not capitalized; ``true`` -> ``True``, ``false`` ->
      ``False``
    * None-type variable has a different name; ``null`` -> ``None``
    * Strings must use double quotes
    * The key names, i.e. the things to the left of the ``:``, must be strings
    
In CAPE, the most common reason for using ``null`` as a value for a setting is
to force the program to ignore any defaults.  This could be the case if you
have set some parameter in an input file and don't want CAPE to touch it.

Finally, there are two major additions to CAPE's implementation of JSON:

    * Lines beginning with ``//`` or ``#`` will be ignored as comments
    * It is possible to include the contents of other JSON files
    
Going back to the example contents used above, suppose we have two JSON files.

    :file:`cape.json`:
    
        .. code-block:: javascript
        
            {
                // Specific run control settings
                "RunControl": {
                    "PhaseSequence": [0, 1],
                    "PhaseIters": [200, 400],
                    "nProc": 8,
                    "MPI": true,
                },
                // Common problem configuration settings
                "Config": JSONFile("Config.json")
            }
            
    :file:`Config.json`:
    
        .. code-block:: javascript
        
            {
                "File": "Config.xml",
                "RefArea": 3.141593,
                "RefLength": 2.0,
                "RefPoint": null
            }
            
The various CAPE modules will then automatically replace
``JSONFile("Config.json")`` with the contents of :file:`Config.json`.  This can
be very useful when conducting sensitivity studies in which most of the
contents of the input file remain constant.

A downside of this approach is that most CAPE input files end up being invalid
JSON files.  A script is provided that replaces comments with empty lines and
expands any ``JSONFile()`` commands (which can be recursive).

    .. code-block:: console
    
        $ pc_ExpandJSON.py -i cape.json -o expand.json
        
The file resulting from this command, :file:`expand.json`, is shown below.

    .. code-block:: javascript
    
        {
        
            "RunControl": {
                "PhaseSequence": [0, 1],
                "PhaseIters": [200, 400],
                "nProc": 8,
                "MPI": true,
            },
            
            "Config": {
                "File": "Config.xml",
                "RefArea": 3.141593,
                "RefLength": 2.0,
                "RefPoint": null
            }
        }

Finally, CAPE provides helpful error messages when typos are present in the
JSON file. This is usually a missing ``:``, extra ``,``, or something similar.
They can be very difficult to track down, but such syntax errors are
accompanied with the line containing the problem and the line above and below.


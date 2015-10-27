
--------------------------------
Surface Configuration and Naming
--------------------------------

Cart3D has an optional (but very useful and thus highly recommended) input file
called :file:`Config.xml` that allows a user to define specific names for
components and also group them together.  The "Config" section of the
:file:`pyCart.json` master settings file points to this file (which can have a
different name but will always be saved as :file:`Config.xml` in the run
directories) and also defines reference length and area.  Finally, this section
also tells `flowCart` what points to report force and moment histories for and
defines moment reference points where necessary.

The following :file:`pyCart.json` snippet shows all the options in action and
lists the defaults.

    .. code-block:: javascript
    
        "Config": {
            "File": "Config.xml",
            "RefArea": 1.0,
            "RefLength": 1.0,
            "RefPoint": [0.0, 0.0, 0.0],
            "Force": ["entire"],
            "Xslices": [],
            "Yslices": [],
            "Zslices": []
        }
        
These are indeed the defaults, but this example is misleadingly simple.  The
capabilities of the "Config" section are much better demonstrated by the
following example.

    .. code-block:: javascript
    
        "Config": {
            "RefArea": {"default": 18.0, "Rudder": 1.2},
            "RefLength": {"default": 2.6, "Rudder": 0.6},
            "RefPoint": {
                "Wings": [7.0, 0.0, -0.2],
                "Rudder": [12.5, 0.0, 2.1],
                "Body": [0.0, 0.0, 0.0]
            },
            "Force": ["Wings", "Body", "Rudder", "Tail"]
        }
        
In this example, a default value is specified for both the reference area and
reference length, but a different value is defined for the ``"Rudder"``
component.  Moment histories are requested for three components, and all of them
use a different moment reference point.  Finally, the force histories are
requested for four components.

The full dictionary of "Config" options is shown below.

    *File*: {``Config.xml``} | :class:`str`
        Name of XML file containing component names and groupings
        
    *RefArea*: {``1.0``} | :class:`float` | :class:`dict` (:class:`float`)
        Reference area or :class:`dict` of reference areas for different
        components
        
    *RefLength*: {``1.0``} | :class:`float` | :class:`dict` (:class:`float`)
        Reference length or :class:`dict` of reference lengths for different
        components
        
    *RefPoint*: {``[0.0, 0.0, 0.0]``} | :class:`dict` (:class:`list`)
        Three-dimensional float specifying global reference point or
        :class:`dict` of components and their moment reference points
        
    *Force*: {``["entire"]``} | :class:`list` (:class:`str`)
        List of components on which to request force history
        
    *Xslices*: {``[]``} | ``[0.0]`` | :class:`list` (:class:`float`)
        List of x-coordinates at which to extract cut planes
        
    *Yslices*: {``[]``} | ``[0.0]`` | :class:`list` (:class:`float`)
        List of y-coordinates at which to extract cut planes
        
    *Zslices*: {``[]``} | ``[0.0]`` | :class:`list` (:class:`float`)
        List of z-coordinates at which to extract cut planes

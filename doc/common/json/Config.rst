
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
            "File": "Config.xml",
            "RefArea": 1.57,
            "RefLength": {
                "total": 0.5,
                "wing": 1.0
            },
            "RefPoint": "MRP",
            "Points": {
                "MRP": [0.0, 0.0, 0.0],
                "CG": [2.5. 0.0, 0.2]
            }
        }
        
Links to additional options for each specific solver are found below.

    * :ref:`Cart3D <pycart-json-Config>`
    * :ref:`FUN3D <pyfun-json-Config>`
        
The *Components* parameter defines a list of components that the user wants Cape
to know about.  Quite often, the meaning of the named component is actually
defined in the *ConfigFile*, for example the way it is done in Cart3D with
:file:`Config.xml`.  In most cases, these *Components* refer to subsets of the
surface mesh(es), and they can be used to define components that pyCart, pyOver,
or pyFun can move around.

The *RefArea* and *RefLength* parameters are universal enough to be included in
the base :mod:`cape` module because almost any CFD solver or code that uses CFD
results will utilize reference length and reference area.  The *RefPoint*
parameter is used to define moment reference points and, for some solvers, also
the list of components for which moments should be reported.  In addition, it is
possible to define different reference areas/lengths for different components
using a :class:`dict` as in the example *RefLength* above.

Finally, the *Point* parameter is a unique capability of Cape that allows points
to be conveniently defined in one spot.  This can be convenient for a list of
components that have the same moment reference point.  Furthermore, it is
essential for configurations that have multiple translations and/or rotations.
Since the rotation point and rotation axis may move, Cape automatically updates
*Points* according to the definition of those translations and/or rotations.

The dictionary of options is given below.

    *Components*: {``[]``} | :class:`list` (:class:`str`)
        List of components for Cape to know about, usually surface subsets
        
    *File*: {``"Config.xml"``} | :class:`str`
        Name of file defining surface and/or volume grid subsets
        
    *Points*: {``{}``} | :class:`dict` (:class:`list`)
        Dictionary of named points and their coordinates
        
    *RefArea*: {``3.14159``} | :class:`float` | :class:`dict` (:class:`float`)
        Reference area or dictionary of reference areas for each component
        
    *RefLength*: {``1.0``} | :class:`float` | :class:`dict` (:class:`float`)
        Reference length or dictionary of reference lengths for each component
        
    *RefPoint*: {``null``} | :class:`dict` (:class:`list`) | :class:`list`
        Moment reference point or dictionary of moment reference points
        
    *RefSpan*: {``null``} | :class:`float` | :class:`dict` (:class:`float`)
        Reference span; falls back to *RefLength* if not specified


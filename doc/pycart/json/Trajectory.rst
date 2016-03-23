
.. _pycart-json-Trajectory:

------------------------
Trajectory or Run Matrix
------------------------

A required section of :file:`pyCart.json` describes the variables that define
the run conditions and the values of those variables.  In many cases, these will
be familiar input variables like Mach number, angle of attack, and sideslip
angle; but in other cases they will require additional description within the
:file:`pyCart.json` file.  The generic run matrix description is found in the
:ref:`Cape Trajectory section <cape-json-Trajectory>`.

The following trajectory keys are in addition to :ref:`the Cape list
<cape-json-TrajectoryKeys>` available to all solvers.

[*translation*]:
Translate one or more components in a specified direction

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "translation",
            "Value": "float",
            "CompID": [],
            "CompIDSymmetric": [],
            "Vector": [],
            "Points": [],
            "PointsSymmetric": []
        }

[*rotation*]:
Rotate one or more components about a specified vector

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "rotation",
            "Value": "float",
            "CompID": [],
            "CompIDSymmetric": [],
            "VectorSymmetry": [1.0, -1.0, 1.0],
            "AngleSymmetry": -1.0,
            "Vector": [[0,0,0], [1,0,0]],
            "Points": [],
            "PointsSymmetric": []
        }
        
These two variable types are somewhat complex but make it relatively simple to
move components around and do all the necessary bookkeeping.  The *Vector*
option describes the direction in which a component is translated or the vector
about which a component is rotated.  The distance by which a component is
translated (in the direction of *Vector*) or the angle by which a component is
rotated is then the value of that variable in the run matrix.  The options
relating to *Symmetry* define components that are moved in an equal but opposite
transformation.  Finally, the *Points* option allows pyCart to automatically
transform points along with the component.  The most obvious reason for this is
to keep a moment reference point in the same position in the body frame of the
rotated component, but it is also useful when there are multiple rotations, the
user wants to sample a point or cut plane at a fixed body position, or many
other potential applications.

A long-form dictionary of the relevant translation and rotation options is given
below.

    *CompID*: {``[]``} | :class:`list` (:class:`int` | :class:`str`)
        List of components by name or index that are translated/rotated
        
    *CompIDSymmetric*: {``[]``} | :class:`list` (:class:`int` | :class:`str`)
        List of components to translate/rotate in symmetric direction
        
    *AngleSymmetry*: {``-1.0``} | :class:`float`
        Multiplier on angle for components rotated in symmetric direction.  This
        is complementary to modifying the rotation vector using *VectorSymmetry*
    
    *Points*: {``[]``} | :class:`list` (:class:`str`)
        List of points by name that are translated/rotated
        
    *PointsSymmetric*: {``[]``} | :class:`list` (:class:`str`)
        List of points that are translated/rotated in symmetric direction
    
    *Vector*: {``[[0,0,0],[0,1,0]]``} | :class:`list` (:class:`list` | :class:`str`)
        Two points defining the vector about which to rotate or the direction in
        which to translate.  The points can be defined either as a list of three
        floats or a named point; named points will be moved by prior
        translations and rotations
        
    *VectorSymmetry*: {``[1.0,1.0,1.0]``} | :class:`list` (:class:`float`), shape=(3,)
        Multipliers applied to each component of the rotation vector


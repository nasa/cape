-------------------------------------------------------
``Config``: surface configuration and reference options
-------------------------------------------------------

**Option aliases:**

* *File* → *ConfigFile*

**Recognized options:**

*Components*: {``[]``} | :class:`list`\ [:class:`str`]
    list of components to request from solver
*ConfigFile*: {``'Config.xml'``} | :class:`str`
    configuration file name
*Points*: {``{}``} | :class:`dict`
    dictionary of reference point locations
*RefArea*: {``1.0``} | :class:`float` | :class:`dict`
    reference area [for a component]
*RefLength*: {``1.0``} | :class:`float` | :class:`dict`
    reference length ro moment or pitching moment
*RefPoint*: {``0.0``} | :class:`float` | :class:`dict` | :class:`str`
    reference point for moment calculations
*RefSpan*: {``None``} | :class:`float` | :class:`dict`
    reference length for yaw and rolling moments


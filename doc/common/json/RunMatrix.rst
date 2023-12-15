---------------------------------
Options for ``RunMatrix`` section
---------------------------------

**Option aliases:**

* *Cols* -> *Keys*
* *Defns* -> *Definitions*
* *cols* -> *Keys*
* *defns* -> *Definitions*
* *file* -> *File*
* *gas* -> *Freestream*
* *keys* -> *Keys*
* *prefix* -> *Prefix*

**Recognized options:**

*File*: {``None``} | :class:`str`
    run matrix data file name
*Freestream*: {``{}``} | :class:`dict`
    properties of freestream gas model
*GroupMesh*: {``False``} | ``True``
    value of option "GroupMesh"
*GroupPrefix*: {``'Grid'``} | :class:`str`
    default prefix for group folders
*Keys*: {``['mach', 'alpha', 'beta']``} | :class:`list`\ [:class:`str`]
    list of run matrix variables
*Prefix*: {``''``} | :class:`str`
    default prefix for case folders
*Values*: {``{}``} | :class:`dict`
    value of option "Values"

**Subsections:**

.. toctree::
    :maxdepth: 1

    RunMatrix-Definitions

--------
``Mesh``
--------

**Option aliases:**

* *ConfigFile* ? *ConfigMapBCFile*
* *BCFile* ? *MapBCFile*
* *MapBC* ? *MapBCFile*

**Recognized options:**

*ConfigMapBCFile*: {``None``} | :class:`str`
    seperate ``.mapbc`` file only for naming surface components
*CopyAsFiles*: {``None``} | :class:`dict`
    file(s) to copy and rename; source file is left-hand side and target file name is right-hand side
*CopyFiles*: {``None``} | :class:`list`\ [:class:`str`]
    file(s) to copy to run folder w/o changing file name
*LinkAsFiles*: {``None``} | :class:`dict`
    file(s) to link and rename; source file is left-hand side and target file name is right-hand side
*LinkFiles*: {``None``} | :class:`list`\ [:class:`str`]
    file(s) to link into run folder w/o changing file name
*LinkMesh*: {``False``} | ``True``
    option to link mesh file(s) instead of copying
*MapBCFile*: {``None``} | :class:`str`
    name of the boundary condition map file
*MeshFile*: {``None``} | :class:`str`
    original mesh file name(s)
*TriFile*: {``None``} | :class:`str`
    original surface triangulation file(s)
*WriteTri*: {``True``} | ``False``
    whether to write surface triangulation file


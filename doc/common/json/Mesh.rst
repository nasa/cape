---------------------------------
``Mesh``: options for mesh inputs
---------------------------------

**Recognized options:**

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
*MeshFile*: {``None``} | :class:`str`
    original mesh file name(s)
*TriFile*: {``None``} | :class:`str`
    original surface triangulation file(s)


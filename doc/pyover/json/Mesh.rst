
.. _pyover-json-mesh:

********************
Mesh Section Options
********************
The options below are the available options in the Mesh Section of the ``pyover.json`` control file

..
    start-Mesh-copyfiles

*CopyFiles*: {``[]``} | :class:`list`\ [:class:`str`]
    list of files to copy into case folder

..
    end-Mesh-copyfiles

..
    start-Mesh-type

*Type*: ``'dcf'`` | ``'peg5'``
    overall meshing stragety

..
    end-Mesh-type

..
    start-Mesh-configdir

*ConfigDir*: {``None``} | :class:`str`
    folder from which to copy/link mesh files

..
    end-Mesh-configdir

..
    start-Mesh-linkfiles

*LinkFiles*: {``['grid.in', 'xrays.in', 'fomo/grid.ibi', 'fomo/grid.nsf', 'fomo/grid.ptv', 'fomo/mixsur.fmp']``} | :class:`list`\ [:class:`str`]
    list of files to link into case folder

..
    end-Mesh-linkfiles


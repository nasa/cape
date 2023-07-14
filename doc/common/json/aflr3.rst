---------------
"aflr3" section
---------------

*o*: {``None``} | :class:`str`
    output file for AFLR3
*run*: {``None``} | ``True`` | ``False``
    whether or not to run AFLR3
*keys*: {``{}``} | :class:`dict`
    AFLR3 options using ``key=val`` format
*cdfr*: {``None``} | :class:`float`
    AFLR3 max geometric growth rate
*i*: {``None``} | :class:`str`
    input file for AFLR3
*blr*: {``None``} | :class:`float`
    AFLR3 boundary layer stretching ratio
*nqual*: {``0``} | :class:`int`
    number of AFLR3 mesh quality passes
*blds*: {``None``} | :class:`float`
    AFLR3 initial boundary-layer spacing
*blc*: {``None``} | ``True`` | ``False``
    AFLR3 prism layer option
*bli*: {``None``} | :class:`int`
    number of AFLR3 prism layers
*flags*: {``{}``} | :class:`dict`
    AFLR3 options using ``-flag val`` format
*mdf*: ``1`` | {``2``}
    AFLR3 volume grid distribution flag
*angqbf*: {``None``} | :class:`float`
    AFLR3 max angle on surface triangles
*angblisimx*: {``None``} | :class:`float`
    AFLR3 max angle b/w BL intersecting faces
*grow*: {``None``} | :class:`float`
    AFLR3 off-body growth rate
*mdsblf*: ``0`` | {``1``} | ``2``
    AFLR3 BL spacing thickness factor option
*cdfs*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64`
    AFLR3 geometric growth exclusion zone size
*BCFile*: {``None``} | :class:`str`
    AFLR3 boundary condition file


---------
AFLR3Opts
---------

*mdsblf*: ``0`` | {``1``} | ``2``
    AFLR3 BL spacing thickness factor option
*i*: {``None``} | :class:`str`
    input file for AFLR3
*nqual*: {``0``} | :class:`int`
    number of AFLR3 mesh quality passes
*run*: {``None``} | ``True`` | ``False``
    whether or not to run AFLR3
*cdfs*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64`
    AFLR3 geometric growth exclusion zone size
*BCFile*: {``None``} | :class:`str`
    AFLR3 boundary condition file
*cdfr*: {``None``} | :class:`float`
    AFLR3 max geometric growth rate
*angqbf*: {``None``} | :class:`float`
    AFLR3 max angle on surface triangles
*blds*: {``None``} | :class:`float`
    AFLR3 initial boundary-layer spacing
*mdf*: ``1`` | {``2``}
    AFLR3 volume grid distribution flag
*o*: {``None``} | :class:`str`
    output file for AFLR3
*blc*: {``None``} | ``True`` | ``False``
    AFLR3 prism layer option
*flags*: {``{}``} | :class:`dict`
    AFLR3 options using ``-flag val`` format
*angblisimx*: {``None``} | :class:`float`
    AFLR3 max angle b/w BL intersecting faces
*blr*: {``None``} | :class:`float`
    AFLR3 boundary layer stretching ratio
*grow*: {``None``} | :class:`float`
    AFLR3 off-body growth rate
*keys*: {``{}``} | :class:`dict`
    AFLR3 options using ``key=val`` format
*bli*: {``None``} | :class:`int`
    number of AFLR3 prism layers


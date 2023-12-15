-----------------------------
Options for ``aflr3`` section
-----------------------------

**Recognized options:**

*BCFile*: {``None``} | :class:`str`
    AFLR3 boundary condition file
*angblisimx*: {``None``} | :class:`float`
    AFLR3 max angle b/w BL intersecting faces
*angqbf*: {``None``} | :class:`float`
    AFLR3 max angle on surface triangles
*blc*: {``None``} | ``True`` | ``False``
    AFLR3 prism layer option
*blds*: {``None``} | :class:`float`
    AFLR3 initial boundary-layer spacing
*bli*: {``None``} | :class:`int`
    number of AFLR3 prism layers
*blr*: {``None``} | :class:`float`
    AFLR3 boundary layer stretching ratio
*cdfr*: {``None``} | :class:`float`
    AFLR3 max geometric growth rate
*cdfs*: {``None``} | :class:`int` | :class:`float`
    AFLR3 geometric growth exclusion zone size
*flags*: {``{}``} | :class:`dict`
    AFLR3 options using ``-flag val`` format
*grow*: {``None``} | :class:`float`
    AFLR3 off-body growth rate
*i*: {``None``} | :class:`str`
    input file for AFLR3
*keys*: {``{}``} | :class:`dict`
    AFLR3 options using ``key=val`` format
*mdf*: ``1`` | {``2``}
    AFLR3 volume grid distribution flag
*mdsblf*: ``0`` | {``1``} | ``2``
    AFLR3 BL spacing thickness factor option
*nqual*: {``0``} | :class:`int`
    number of AFLR3 mesh quality passes
*o*: {``None``} | :class:`str`
    output file for AFLR3
*run*: {``None``} | ``True`` | ``False``
    whether or not to run AFLR3


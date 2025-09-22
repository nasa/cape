----------------------------
CLI options for ``comp2tri``
----------------------------

**Option aliases:**

* *gmptagoffset* -> *gmptagoffset*
* *ifile* -> *i*
* *ifiles* -> *i*
* *keepcomps* -> *keepComps*
* *maketags* -> *makeGMPtags*
* *ofile* -> *o*
* *output* -> *o*
* *tagoffset* -> *gmpTagOffset*

**Recognized options:**

*ascii*: {``None``} | ``True`` | ``False``
    output file will be ASCII (if not trix)
*config*: {``None``} | ``True`` | ``False``
    write ``Config.xml`` using component tags
*dp*: {``False``} | ``True``
    use double-precision vert-coordinates
*gmp2comp*: {``None``} | :class:`object`
    copy GMPtags to IntersectComponents
*gmpTagOffset*: {``None``} | :class:`int`
    renumber GMPtags by NxOffset for file *N*
*i*: {``None``} | :class:`list`\ [:class:`str`]
    input file or list thereof
*inflate*: {``None``} | ``True`` | ``False``
    inflate geometry to break degeneracies
*keepComps*: {``None``} | ``True`` | ``False``
    preserve 'intersect' component tags
*makeGMPtags*: {``None``} | ``True`` | ``False``
    create GMPtags from volume indexes
*o*: {``'Components.tri'``} | :class:`str`
    output filename
*run*: {``None``} | ``True`` | ``False``
    whether to execute program
*trix*: {``None``} | ``True`` | ``False``
    output file will be eXtended-TRI (trix) format
*v*: {``False``} | ``True``
    verbose mode


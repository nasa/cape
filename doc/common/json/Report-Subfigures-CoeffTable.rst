------------------------------------
Options for ``CoeffTable`` subfigure
------------------------------------

**Option aliases:**

* *nStats* -> *NStats*
* *nMinStats* -> *NMinStats*
* *nMaxStats* -> *NMaxStats*
* *Parent* -> *Type*
* *parent* -> *Type*
* *pos* -> *Position*
* *type* -> *Type*
* *width* -> *Width*

**Recognized options:**

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"
*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"
*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"
*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"
*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"
*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*EpsFormat*: {``'%.2e'``} | :class:`dict` | :class:`str`
    printf-style text format for sampling error
*Header*: {``''``} | :class:`str`
    subfigure header
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*MuFormat*: {``'%.4f'``} | :class:`dict` | :class:`str`
    printf-style text format for mean value
*NMaxStats*: {``None``} | :class:`int`
    max number of iterations to allow in averaging window
*NMinStats*: {``None``} | :class:`int`
    min iteration number to include in averaging window
*NStats*: {``None``} | :class:`int`
    nominal (minimum) number of iterations to average over
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*SigmaFormat*: {``'%.4f'``} | :class:`dict` | :class:`str`
    printf-sylte text format for standard deviation
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``0.33``} | :class:`float`
    value of option "Width"


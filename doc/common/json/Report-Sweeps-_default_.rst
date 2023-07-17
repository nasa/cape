-------------------
Options for default
-------------------

**Option aliases:**

* *EqConstraints* -> *EqCons*
* *EqualityCons* -> *EqCons*
* *EqualityConstraints* -> *EqCons*
* *GlobalConstraints* -> *GlobalCons*
* *TolConstraints* -> *TolCons*
* *ToleranceConstraints* -> *TolCons*
* *XAxis* -> *XCol*
* *YAxis* -> *YCol*
* *cols* -> *EqCons*
* *cons* -> *GlobalCons*
* *figs* -> *Figures*
* *itol* -> *IndexTol*
* *mask* -> *Indices*
* *nmin* -> *MinCases*
* *tols* -> *TolCons*
* *xcol* -> *XCol*
* *xk* -> *XCol*
* *xkey* -> *XCol*
* *yk* -> *YCol*
* *ykey* -> *YCol*

**Recognized options:**

*CarpetEqCons*: {``None``} | :class:`list`\ [:class:`str`]
    run matrix keys that are constant on carpet subsweep
*EqCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys that must be constant on a sweep
*Figures*: {``None``} | :class:`list`\ [:class:`str`]
    list of figures in sweep report
*GlobalCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of global constraints for sweep
*IndexTol*: {``None``} | :class:`int`
    max delta of run matrix/databook index for single sweep
*Indices*: {``None``} | :class:`list`\ [:class:`int`]
    explicit list of run matrix/databook indices to include
*MinCases*: {``3``} | :class:`int`
    minimum number of data points in a sweep to include plot
*RunMatrixOnly*: {``False``} | ``True``
    option to restrict databook to current run matrix
*TolCons*: {``None``} | :class:`dict`
    tolerances for run matrix keys to be in same sweep
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis of sweep plots
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis of sweep contour plots


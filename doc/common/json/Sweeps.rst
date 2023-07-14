----------------
"Sweeps" section
----------------


Options for default
===================

*CarpetEqCons*: {``None``} | :class:`list`\ [:class:`str`]
    run matrix keys that are constant on carpet subsweep
*Indices*: {``None``} | :class:`list`\ [:class:`int`]
    explicit list of run matrix/databook indices to include
*GlobalCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of global constraints for sweep
*MinCases*: {``3``} | :class:`int`
    minimum number of data points in a sweep to include plot
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis of sweep plots
*TolCons*: {``None``} | :class:`dict`
    tolerances for run matrix keys to be in same sweep
*IndexTol*: {``None``} | :class:`int`
    max delta of run matrix/databook index for single sweep
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis of sweep contour plots
*EqCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys that must be constant on a sweep
*Figures*: {``None``} | :class:`list`\ [:class:`str`]
    list of figures in sweep report
*RunMatrixOnly*: ``True`` | {``False``}
    option to restrict databook to current run matrix



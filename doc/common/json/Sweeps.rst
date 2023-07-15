-------------------
SweepCollectionOpts
-------------------


Options for default
===================

*GlobalCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of global constraints for sweep
*EqCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys that must be constant on a sweep
*Figures*: {``None``} | :class:`list`\ [:class:`str`]
    list of figures in sweep report
*IndexTol*: {``None``} | :class:`int`
    max delta of run matrix/databook index for single sweep
*CarpetEqCons*: {``None``} | :class:`list`\ [:class:`str`]
    run matrix keys that are constant on carpet subsweep
*MinCases*: {``3``} | :class:`int`
    minimum number of data points in a sweep to include plot
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis of sweep contour plots
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis of sweep plots
*RunMatrixOnly*: ``True`` | {``False``}
    option to restrict databook to current run matrix
*TolCons*: {``None``} | :class:`dict`
    tolerances for run matrix keys to be in same sweep
*Indices*: {``None``} | :class:`list`\ [:class:`int`]
    explicit list of run matrix/databook indices to include




            
.. _cape-json-ReportSweep:

Sweep Definitions
=================

Each sweep has a definition that is similar to a report but with additional
options to divide the run matrix into subsets.  For example, if the run matrix
has three independent variables (which pyCart calls trajectory keys) of
``"Mach"``, ``"alpha"``, and ``"beta"``, then a common sweep would be to plot
results as a function of Mach number for constant *alpha* and *beta*.  To do
that, one would put ``"EqCons": ["alpha", "beta"]`` in the sweep definition.

The full list of available options is below.

    *Sweeps*: ``{}`` | ``{[S]}`` | :class:`dict` (:class:`dict`)
        Dictionary of sweep definitions (combined plots of subsets of cases)
        
        *S*: :class:`dict`
            Dictionary of sweep definitions for sweep named ``"S"``
            
            *Figures*: {``[]``} | :class:`list` (:class:`str`)
                List of figures to include for each sweep subset
                
            *EqCons*: {``[]``} | :class:`list` (:class:`str`)
                List of trajectory keys to hold constant for each subset
                
            *TolCons*: {``{}``} | :class:`dict` (:class:`float`)
                Dictionary of trajectory keys to hold within a certain tolerance
                from the value of that key for the first case in the subset
                
            *IndexTol*: {``None``} | :class:`int`
                If used, only allow the index of the first and last cases in a
                subset to differ by this value
                
            *XAxis*: {``None``} | :class:`str`
                Name of trajectory key used to sort subset; if ``None``, sort by
                data book index
                
            *TrajectoryOnly*: ``true`` | {``false``}
                By default, the data book is the source for sweep plots; this
                option can restrict the plots to points in the current run
                matrix
                
            *GlobalCons*: {``[]``} | :class:`list` (:class:`str`)
                List of global constraints to only divide part of the run matrix
                into subsets
                
            *Indices*: {``None``} | :class:`list` (:class:`int`)
                If used, list of indices to divide into subsets
                
            *MinCases*: {``1``} | :class:`int`
                Minimum number of cases for a sweep to be reported
                
            *CarpetEqCons*: ``[]`` | :class:`list` (:class:`str`)
                Some sweep subfigures allow a sweep to be subdivided into
                subsweeps; this could be used to create plots of *CN* versus
                *Mach* with several lines each having constant *alpha*
                
            *CarpetTolCons*: ``{}`` | :class:`dict` (:class:`float`)
                Tolerance constraints for subdividing sweeps

The subsets are defined so that each case meeting the *GlobalCons* is placed
into exactly one subset.  For each subset, pyCart begins with the first
available case and applies the constraints using that point as a reference.
                
Constraints can be defined in more complex ways than the example given prior to
the list of options.  For relatively simple run matrices, grouping cases by
constant values of one or more trajectory keys (i.e. using *EqCons*) may be
adequate, but other run matrices may require more advanced settings.

For example, wind tunnel data often is collected at conditions that are not
exactly constant, i.e. the angle of attack may fluctuate slightly.  Instead of
using *EqCons*, a better option in this case would be to include ``"TolCons":
{"alpha": 0.02}``.  Then all cases in a subset would have an angle of attack
within ``0.02`` of the angle of attack of the first point of the subset.

Another advanced capability is to use *EqCons* such as ``["k%10"]`` or
``["k/10%10"]``.  This could be used to require each case to have the same ones
digit or the same tens digit of some trajectory variable called *k*.


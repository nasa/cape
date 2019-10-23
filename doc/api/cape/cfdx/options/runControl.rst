
.. automodule:: cape.cfdx.options.runControl

    Environment variables
    ------------------------

    .. autoclass:: cape.cfdx.options.runControl.Environ
        :members:
        
    Overall run control and system options
    --------------------------------------
        
    .. autoclass:: cape.cfdx.options.runControl.RunControl
        :members: get_Environ, get_ulimit,
            get_nIter, get_PhaseSequence, get_PhaseIters, get_nSeq,
            get_MPI, get_nProc, get_mpicmd, get_qsub,
            get_Resubmit, get_Continue

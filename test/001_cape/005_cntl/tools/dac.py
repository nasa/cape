#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Initialization functions
def InitCntl(cntl):
    """A function that gets called when *cntl* is loaded

    :Call:
        >>> InitAeroDB(cntl)
    :Inputs:
        *cntl*: :class:`cape.cntl.Cntl`
            Global JSON interface
    :Versions:
        * 2018-10-09 ``@ddalle``: First version
    """
    # Override a setting for testing purposes
    cntl.opts.set_PhaseIters(100, 1)
    cntl.opts.set_PhaseSequence([0, 1])

# Case function: apply the architecture
def ApplyTag(cntl, i, **kw):
    """Modify architecture settings for ``poweroff``

    :Call:
        >>> ApplyTag(cntl, i, **kw)
    :Inputs:
        *cntl*: :class:`cape.pyfun.cntl.Cntl`
            Overflow settings interface
        *i*: :class:`int`
            Case number
        *san*: ``True`` | {``False``}
            Whether or not to submit to Sandy Bridge architecture
        *ivy*: ``True`` | {``False``}
            Whether or not to submit to Ivy Bridge architecture
        *has*: ``True`` | {``False``}
            Whether or not to submit to Haswell architecture
        *bro*: ``True`` | {``False``}
            Whether or not to submit to Broadwell architecture
        *bro_ele*: ``True`` | {``False``}
            Whether or not to submit to Broadwell architecture (*Electra*)
    :Versions:
        * 2017-04-12 ``@ddalle``: First version
    """
    # Process the architecture to use
    if kw.get('ivy', False):
        # Ivy Bridge nodes
        cntl.opts.set_PBS_model("ivy")
        cntl.opts.set_PBS_ncpus(20)
        cntl.opts.set_PBS_mpiprocs(20)
        cntl.opts.set_PBS_select(15)
        cntl.opts.set_nProc(300)
    elif kw.get('wes', False):
        # Westmere nodes (*Merope*)
        cntl.opts.set_PBS_model("wes")
        cntl.opts.set_PBS_ncpus(12)
        cntl.opts.set_PBS_mpiprocs(12)
        cntl.opts.set_PBS_select(30)
        cntl.opts.set_nProc(360)
    elif kw.get('san', False):
        # Sandy Bridge nodes
        cntl.opts.set_PBS_model("san")
        cntl.opts.set_PBS_ncpus(16)
        cntl.opts.set_PBS_mpiprocs(16)
        cntl.opts.set_PBS_select(20)
        cntl.opts.set_nProc(320)
    elif kw.get('has', False):
        # Haswell nodes
        cntl.opts.set_PBS_model("has")
        cntl.opts.set_PBS_ncpus(24)
        cntl.opts.set_PBS_mpiprocs(24)
        cntl.opts.set_PBS_select(10)
        cntl.opts.set_nProc(240)
    elif kw.get('bro_ele', False):
        # Broadwell/Electra nodes
        cntl.opts.set_PBS_model("bro_ele")
        cntl.opts.set_PBS_ncpus(28)
        cntl.opts.set_PBS_mpiprocs(28)
        cntl.opts.set_PBS_select(10)
        cntl.opts.set_nProc(280)
    elif kw.get('sky_ele', False):
        # Skylake/Electra nodes
        cntl.opts.set_PBS_model("sky_ele")
        cntl.opts.set_PBS_ncpus(40)
        cntl.opts.set_PBS_mpiprocs(40)
        cntl.opts.set_PBS_select(10)
        cntl.opts.set_nProc(400)
    else:
        # Broadwell nodes
        cntl.opts.set_PBS_model("bro")
        cntl.opts.set_PBS_ncpus(28)
        cntl.opts.set_PBS_mpiprocs(28)
        cntl.opts.set_PBS_select(10)
        cntl.opts.set_nProc(280)

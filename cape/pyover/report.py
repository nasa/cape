r"""
:mod:`cape.pyover.report`: Automated report interface
======================================================

The ``pyover`` module for generating automated results reports using
PDFLaTeX provides a single class :class:`pyOver.report.Report`, which is
based off the CAPE version :class:`cape.cfdx.report.Report`. The
:class:`cape.cfdx.report.Report` class is a sort of dual-purpose object
that contains a file interface using :class:`cape.texfile.Tex` combined with
a capability to create figures for each case or sweep of cases mostly
based on :mod:`cape.cfdx.databook`.

An automated report is a multi-page PDF generated using PDFLaTeX.
Usually, each CFD case has one or more pages dedicated to results for
that casecntl. The user defines a list of figures, each with its own list of
subfigures, and these are generated for each case in the run matrix
(subject to any command-line constraints the user may specify). Types of
subfigures include

    * Table of the values of the input variables for this case
    * Table of force and moment values and statistics
    * Iterative histories of forces or moments (one or more coefficients)
    * Iterative histories of residuals
    * Images using a Tecplot layout
    * Many more

In addition, the user may also define "sweeps," which analyze groups of
cases defined by user-specified constraints. For example, a sweep may be
used to plot results as a function of Mach number for sets of cases
having matching angle of attack and sideslip angle. When using a sweep,
the report contains one or more pages for each sweep (rather than one or
more pages for each case).

Reports are created using system commands of the following format.

    .. code-block: console

        $ pyover --report

The class has an immense number of methods, which can be somewhat
grouped into bookkeeping methods and plotting methods.  The main
user-facing methods are :func:`cape.cfdx.report.Report.UpdateCases` and
:func:`cape.cfdx.report.Report.UpdateSweep`.  Each type of subfigure has
its own method, for example
:func:`cape.cfdx.report.Report.SubfigPlotCoeff` for ``"PlotCoeff"``  or
:func:`cape.cfdx.report.Report.SubfigPlotL2` for ``"PlotL2"``.

:See also:
    * :mod:`cape.cfdx.report`
    * :mod:`cape.pyover.options.Report`
    * :mod:`cape.options.Report`
    * :class:`cape.cfdx.databook.DBComp`
    * :class:`cape.cfdx.databook.CaseFM`
    * :class:`cape.cfdx.lineload.LineLoadDataBook`

"""

# Standard library
import os
import shutil

# Third-party modules

# CAPE submodules
from . import casecntl
from ..cfdx import report as capereport
from ..filecntl.tecfile import Tecscript


# Class to interface with report generation and updating.
class Report(capereport.Report):
    r"""Interface for automated report generation

    :Call:
        >>> R = pyOver.report.Report(oflow, rep)
    :Inputs:
        *oflow*: :class:`cape.pyover.cntl.Cntl`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyOver.report.Report`
            Automated report interface
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """

    # String conversion
    def __repr__(self):
        """String/representation method

        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        return '<pyOver.Report("%s")>' % self.rep
    # Copy the function
    __str__ = __repr__

    # Read a Tecplot script
    def ReadTecscript(self, fsrc):
        r"""Read a Tecplot script interface

        :Call:
            >>> R.ReadTecscript(fsrc)
        :Inputs:
            *R*: :class:`pyOver.report.Report`
                Automated report interface
            *fscr*: :class:`str`
                Name of file to read
        :Versions:
            * 2016-10-25 ``@ddalle``: First version
        """
        return Tecscript(fsrc)

    # Function to link appropriate visualization files
    def LinkVizFiles(self, sfig=None, i=None):
        r"""Create links to appropriate visualization files

        Specifically, ``Components.i.plt`` and ``cutPlanes.plt`` or
        ``Components.i.dat`` and ``cutPlanes.dat`` are created.

        :Call:
            >>> R.LinkVizFiles()
        :Inputs:
            *R*: :class:`pyOver.report.Report`
                Automated report interface
        :See Also:
            :func:`pyCart.casecntl.LinkPLT`
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        # Defer to function from :func:`pyOver.case`
        casecntl.LinkX()
        casecntl.LinkQ()
        # Extract Options
        opts = self.cntl.opts
        # Check for splitmq options
        fsplitmq = opts.get_SubfigOpt(sfig, "SplitmqI")
        # Exit if None
        if fsplitmq is None:
            # Try another name
            fsplitmq = opts.get_SubfigOpt(sfig, "splitmq")
        # Exit if none
        if fsplitmq is None:
            return
        # Assemble absolute path
        if os.path.isabs(fsplitmq):
            # Already absolute path
            fabs = fsplitmq
        else:
            # Append root directory
            fabs = os.path.join(self.cntl.RootDir, fsplitmq)
        # Get the file name alone
        fname = os.path.split(fabs)[-1]
        # Check if the file exists
        if not os.path.isfile(fabs):
            return
        # Check which ``q`` and ``x`` files to use as input and output
        fqi = opts.get_SubfigOpt(sfig, "QIn")
        fqo = opts.get_SubfigOpt(sfig, "QOut")
        fxi = opts.get_SubfigOpt(sfig, "Xin")
        fxo = opts.get_SubfigOpt(sfig, "XOut")
        # Defaults
        if fqi is None:
            fqi = "q.pyover.p3d"
        if fqo is None:
            fqo = "q.pyover.srf"
        if fxi is None:
            fxi = "x.pyover.p3d"
        if fxo is None:
            fxo = "x.pyover.srf"
        # Check if the surf file exists
        if not os.path.isfile(fqi):
            # No input file to use
            qq = False
        elif os.path.isfile(fqo):
            # If both files exist, check iteration nubers
            nqi = casecntl.checkqt(fqi)
            try:
                nqo = casecntl.checkqt(fqo)
            except Exception:
                nqo = 0
            # Check if surface file is already up-to-date
            if nqi <= nqo:
                # Up-to-date
                qq = False
            else:
                # Old q.srf file
                os.remove(fqo)
                qq = True
        else:
            # No q.srf file
            qq = True
        # Check if the surf grid file exists
        if not os.path.isfile(fxi):
            # No input file
            qx = False
        elif os.path.isfile(fxo):
            # Update *x.srf* only if updating *q.srf*
            if qq:
                # Delete the link
                os.remove(fxo)
                qx = True
            else:
                # Up-to-date files
                qx = False
        else:
            # No x.srf file
            qx = True
        # Exit if nothing to do
        if not (qx or qq):
            return
        # Name for the file here
        fspq = 'splitmq.%s.i' % sfig
        fspx = 'splitmx.%s.i' % sfig
        fspqo = 'splitmq.%s.o' % sfig
        fspxo = 'splitmx.%s.o' % sfig
        # Copy the file to this directory
        shutil.copy(fabs, fname)
        # Update or create q.srf if necessary
        if qq:
            # Edit the splitmq files
            casecntl.EditSplitmqI(fname, fspq, fqi, fqo)
            # Split the solution
            cmd = 'splitmq < %s > %s' % (fspq, fspqo)
            print("    %s" % cmd)
            ierr = os.system(cmd)
            # Check for errors
            if ierr:
                return
            # Delete files
            os.remove(fspq)
            os.remove(fspqo)
        # Update or create x.srf if necessary
        if qx:
            # Edit the splitmx file
            casecntl.EditSplitmqI(fname, fspx, fxi, fxo)
            # Split the surface grid
            cmd = 'splitmx < %s > %s' % (fspx, fspxo)
            print("    %s" % cmd)
            ierr = os.system(cmd)
            # Check for errors
            if ierr:
                return
            # Delete files
            os.remove(fspx)
            os.remove(fspxo)
        # Delete the template
        os.remove(fname)


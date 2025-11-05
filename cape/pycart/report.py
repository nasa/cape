r"""
:mod:`cape.pycart.report`: Automated report interface
=====================================================

The pyCart module for generating automated results reports using
PDFLaTeX provides a single class :class:`pyCart.report.Report`, which is
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
    * Iterative histories of forces or moments
    * Iterative histories of residuals
    * Images using a Tecplot layout
    * Many more

In addition, the user may also define "sweeps," which analyze groups of
cases defined by user-specified constraints. For example, a sweep may be
used to plot results as a function of Mach number for sets of cases
having matching angle of attack and sideslip angle. When using a sweep,
the report contains one or more pages for each sweep (rather than one or
more pages for each case).

Reports are usually created using system commands of the following
format.

    .. code-block: console

        $ pycart --report

The class has an immense number of methods, which can be somewhat
grouped into bookkeeping methods and plotting methods.  The main
user-facing methods are :func:`cape.cfdx.report.Report.UpdateCases` and
:func:`cape.cfdx.report.Report.UpdateSweep`.  Each
:ref:`type of subfigure <cape-json-ReportSubfigure>` has its own method,
for example :func:`cape.cfdx.report.Report.SubfigPlotCoeff` for
``"PlotCoeff"``  or :func:`cape.cfdx.report.Report.SubfigPlotL2` for
``"PlotL2"``.

:See also:
    * :mod:`cape.cfdx.report`
    * :mod:`cape.pycart.options.Report`
    * :mod:`cape.options.Report`
    * :class:`cape.cfdx.databook.DBComp`
    * :class:`cape.cfdx.databook.CaseFM`
    * :class:`cape.cfdx.lineload.LineLoadDataBook`

"""

# Standard library
import os

# Third-party modules
import numpy as np

# Local imports
from .casecntl import LinkPLT
from ..cfdx import report as capereport
from ..filecntl.tecfile import Tecscript


# Dedicated function to load pointSensor only when needed.
def ImportPointSensor():
    """Import :mod:`cape.pycart.pointSensor` if not loaded

    :Call:
        >>> pyCart.report.ImportPointSensor()
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
    """
    # Make global variables
    global pointsensor
    # Check for PyPlot.
    try:
        pointsensor
    except Exception:
        # Load the modules.
        from . import pointsensor


# Dedicated function to load lineload only when needed.
def ImportLineLoad():
    """Import :mod:`cape.pycart.lineload` if not loaded

    :Call:
        >>> pyCart.report.ImportLineLoad()
    :Versions:
        * 2016-06-10 ``@ddalle``: First version
    """
    # Make global variables
    global lineload
    # Check for PyPlot.
    try:
        lineload
    except Exception:
        # Load the modules
        from . import lineload


# Class to interface with report generation and updating.
class Report(capereport.Report):
    """Interface for automated report generation

    :Call:
        >>> R = pyCart.report.Report(cart3d, rep)
    :Inputs:
        *cart3d*: :class:`cape.pycart.cntl.Cntl`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyCart.report.Report`
            Automated report interface
    :Versions:
        * 2015-03-07 ``@ddalle``: Started
        * 2015-03-10 ``@ddalle``: First version
    """

    # Read point sensor history
    def ReadPointSensor(self):
        """Read iterative history for a case

        :Call:
            >>> P = R.ReadPointSensor()
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
        :Outputs:
            *P*: :class:`pyCart.pointsensor.CasePointSensor`
                Iterative history of point sensors
        :Versions:
            * 2015-12-07 ``@ddalle``: First version
        """
        # Make sure the modules are present.
        ImportPointSensor()
        # Read point sensors history
        return pointsensor.CasePointSensor()

    # Read line loads
    def ReadLineLoad(self, comp, i, targ=None, update=False):
        """Read line load for a case

        :Call:
            >>> LL = R.ReadLineLoad(comp, i, targ=None, update=False)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of line load component
            *i*: :class:`int`
                Case number
            *targ*: {``None``} | :class:`str`
                Name of target data book to read, if not ``None``
            *update*: ``True`` | {``False``}
                Whether or not to attempt an update if case not in data book
        :Outputs:
            *LL*: :class:`pyCart.lineload.CaseLL`
                Individual case line load interface
        :Versions:
            * 2016-06-10 ``@ddalle``: First version
        """
        # Make sure the modules are present.
        ImportLineLoad()
        # Ensure configuration is present
        self.cntl.ReadConfig()
        # Read the data book and line load data book
        self.cntl.ReadDataBook()
        # Get data book handle.
        if targ is None:
            # General data book
            DB = self.cntl.DataBook
            # Read the line load data book
            DB.ReadLineLoad(comp, conf=self.cntl.config)
            DBL = DB.LineLoads[comp]
            # Use the index directly
            j = i
        else:
            # Pointer to the data book
            DB = self.cntl.DataBook
            # Read the target as appropriate
            DB.ReadTarget(targ)
            # Target data book
            DBT = DB.Targets[targ]
            # Read Line load
            DBT.ReadLineLoad(comp, conf=self.cntl.config)
            DBL = DBT.LineLoads[comp]
            # Update the trajectory
            DBL.UpdateRunMatrix()
            # Target options
            topts = self.cntl.opts.get_TargetDataBookByName(targ)
            # Find a match
            J = DBL.FindTargetMatch(DB, i, topts, keylist='tol')
            # Check for a match
            if len(J) == 0: return None
            # Get the first match
            j = J[0]
            # Move the handle.
            DB = DBT
        # Read the case
        DBL.ReadCase(j)
        # Check auto-update flag
        if update and j not in DBL:
            # Update the case
            DBL.UpdateCase(j)
            # Read the case
            DBL.ReadCase(j)
        # Output the case line load
        return DBL.get(j)

    # Function to plot point sensor history
    def SubfigPlotPoint(self, sfig, i, q):
        """Plot iterative history of a point sensor state

        :Call:
            >>> lines = R.SubfigPlotPoint(sfig, i, q)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to redraw images
        :Versions:
            * 2015-12-07 ``@ddalle``: First version
        """
        # Save current folder
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
        # Extract the point
        pt = opts.get_SubfigOpt(sfig, "Point")
        # Get the group
        grp = opts.get_SubfigOpt(sfig, "Group")
        # Get the state
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # List of coefficients
        if type(coeff).__name__ in ['list', 'ndarray']:
            # List of coefficients
            nCoeff = len(coeff)
        else:
            # One entry
            nCoeff = 1
        # Check for list of points
        if type(pt).__name__ in ['list', 'ndarray']:
            # List of components
            nCoeff = max(nCoeff, len(pt))
        # Current status
        nIter  = self.cntl.CheckCase(i)
        # Get caption
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption
        if fcpt is None:
            # Check for a list
            if type(pt).__name__ in ['list']:
                # Join them, e.g. "[P1,P2]/Cp"
                fcpt = "[" + ",".join(pt) + "]"
            else:
                # Use the point name
                fcpt = pt
            # Add the coefficient title
            fcpt = "%s/%s" % (fcpt, coeff)
            # Ensure there are not underscores
            fcpt = fcpt.replace('_', r'\_')
        # Get the vertical alignment.
        hv = opts.get_SubfigOpt(sfig, "Position")
        # Get subfigure width
        wsfig = opts.get_SubfigOpt(sfig, "Width")
        # First line
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header
        fhdr = opts.get_SubfigOpt(sfig, "Header")
        # Alginment
        algn = opts.get_SubfigOpt(sfig, "Alignment")
        # Set alignment
        if algn.lower() == "center":
            lines.append('\\centering\n')
        # Write the header
        if fhdr:
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
            lines.append('\\vskip-6pt\n')
        # Check for image update
        if not q:
            # File name to check for
            fpdf = '%s.pdf' % sfig
            # Check for the file
            if os.path.isfile(fpdf):
                # Include the graphics.
                lines.append('\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fpdf))
            # Set the caption.
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
        # Go to the run directory.
        os.chdir(self.cntl.RootDir)
        os.chdir(frun)
        # Read the Aero history.
        P = self.ReadPointSensor()
        # Loop through plots
        for k in range(nCoeff):
            # Get the point and coefficient
            pt    = opts.get_SubfigOpt(sfig, "Point", k)
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Numbers of iterations
            nStats = opts.get_SubfigOpt(sfig, "nStats",     k)
            nMin   = opts.get_SubfigOpt(sfig, "nMinStats",  k)
            nMax   = opts.get_SubfigOpt(sfig, "nMaxStats",  k)
            nLast  = opts.get_SubfigOpt(sfig, "nLastStats", k)
            # Default to databook options
            if nStats is None: nStats = opts.get_DataBookNStats(grp)
            if nMin   is None: nMin   = opts.get_DataBookNMin(grp)
            if nMax   is None: nMax   = opts.get_DataBookNMaxStats(grp)
            if nLast  is None: nLast  = opts.get_nLastStats(grp)
            # Numbers of iterations for plots
            nPlotIter  = opts.get_SubfigOpt(sfig, "nPlot",      k)
            nPlotFirst = opts.get_SubfigOpt(sfig, "nPlotFirst", k)
            nPlotLast  = opts.get_SubfigOpt(sfig, "nPlotLast",  k)

            # Check if there are iterations.
            if nIter < 2: continue
            # Don't use iterations before *nMin*
            nMax = min(nMax, nIter-nMin)
            # Get the manual range to show
            dc = opts.get_SubfigOpt(sfig, "Delta", k)
            # Get the multiple of standard deviation to show
            ksig = opts.get_SubfigOpt(sfig, "StandardDeviation", k)
            # Get the multiple of iterative error to show
            uerr = opts.get_SubfigOpt(sfig, "IterativeError", k)
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigureWidth", k)
            figh = opts.get_SubfigOpt(sfig, "FigureHeight", k)
            # Plot options
            kw_p = opts.get_SubfigOpt(sfig, "LineOptions",   k)
            kw_m = opts.get_SubfigOpt(sfig, "MeanOptions",   k)
            kw_s = opts.get_SubfigOpt(sfig, "StDevOptions",  k)
            kw_u = opts.get_SubfigOpt(sfig, "ErrPltOptions", k)
            kw_d = opts.get_SubfigOpt(sfig, "DeltaOptions",  k)
            # Label options
            sh_m = opts.get_SubfigOpt(sfig, "ShowMu", k)
            sh_s = opts.get_SubfigOpt(sfig, "ShowSigma", k)
            sh_d = opts.get_SubfigOpt(sfig, "ShowDelta", k)
            sh_e = opts.get_SubfigOpt(sfig, "ShowEpsilon", k)
            # Label formatting
            f_s = opts.get_SubfigOpt(sfig, "SigmaFormat", k)
            f_e = opts.get_SubfigOpt(sfig, "EpsilonFormat", k)
            # Draw the plot.
            h = P.PlotState(coeff, pt, n=nPlotIter, nAvg=nStats,
                nFirst=nPlotFirst, nLast=nPlotLast,
                LineOptions=kw_p, MeanOptions=kw_m,
                d=dc, DeltaOptions=kw_d,
                k=ksig, StDevOptions=kw_s,
                u=uerr, ErrPltOptions=kw_u,
                ShowMu=sh_m, ShowDelta=sh_d,
                ShowSigma=sh_s, SigmaFormat=f_s,
                ShowEspsilon=sh_e, EpsilonFormat=f_e,
                FigWidth=figw, FigHeight=figh)
        # Change back to report folder.
        os.chdir(fpwd)
        # Check for a figure to write.
        if nIter >= 2:
            # Get the file formatting
            fmt = opts.get_SubfigOpt(sfig, "Format")
            dpi = opts.get_SubfigOpt(sfig, "DPI")
            # Figure name
            fimg = '%s.%s' % (sfig, fmt)
            fpdf = '%s.pdf' % sfig
            # Save the figure.
            if fmt in ['pdf']:
                # Save as vector-based image.
                h['fig'].savefig(fimg)
            elif fmt in ['svg']:
                # Save as PDF and SVG
                h['fig'].savefig(fimg)
                h['fig'].savefig(fpdf)
            else:
                # Save with resolution.
                h['fig'].savefig(fimg, dpi=dpi)
                h['fig'].savefig(fpdf)
            # Close the figure.
            h['fig'].clf()
            # Include the graphics.
            lines.append('\\includegraphics[width=\\textwidth]{%s/%s}\n'
                % (frun, fpdf))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

    # Function to write pressure coefficient table
    def SubfigPointSensorTable(self, sfig, i, q):
        """Create lines for a "PointSensorTable" subfigure

        :Call:
            >>> lines = R.SubfigPointTable(sfig, i, q)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to redraw images
        :Versions:
            * 2015-12-07 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Extract options
        opts = self.cntl.opts
        # Current statuts
        nIter = self.cntl.CheckCase(i)
        # Coefficients to plot
        coeffs = opts.get_SubfigOpt(sfig, "Coefficients")
        # Get the targets
        targs = opts.get_SubfigOpt(sfig, "Targets")
        # Target data and target options
        DBT = {}
        topts = {}
        # Read those targets into the data book, if necessary
        for targ in targs:
            # Read the target data book
            self.cntl.ReadDataBook()
            self.cntl.DataBook.ReadTarget(targ)
            # Append it to the list
            DBT[targ] = self.cntl.DataBook.GetTargetByName(targ)
            # Check for coefficient lists
            tcoeffs = opts.get_SubfigOpt(sfig, "%s.Coefficients"%targ)
            # Default to main list
            if tcoeffs is None: tcoeffs = coeffs
            # Initialize the options for this target
            topts[targ] = {"Coefficients": tcoeffs}
            # Get index of first matching point
            J = DBT[targ].FindMatch(self.cntl.x, i)
            # Check for matches
            if len(J) == 0:
                # No match
                topts[targ]["Index"] = None
            else:
                # Take the first match
                topts[targ]["Index"] = J
            # Loop through the coefficients
            for coeff in tcoeffs:
                # Get the list of statistical fields for DB and target
                fsd = opts.get_SubfigOpt(sfig, coeff)
                fst = opts.get_SubfigOpt(sfig, '%s.%s' % (targ,coeff))
                # Save the settings
                if fst is None:
                    # Use defaults
                    topts[targ][coeff] = fsd
                else:
                    # Use specific settings
                    topts[targ][coeff] = fst
        # Get the points.
        grp = opts.get_SubfigOpt(sfig, "Group")
        pts = opts.get_SubfigOpt(sfig, "Points")
        # Process point list if a group is specified
        if grp: pts = opts.get_DBGroupPoints(grp)
        # Numbers of iterations for statistics
        nStats = opts.get_SubfigOpt(sfig, "nStats")
        nMin   = opts.get_SubfigOpt(sfig, "nMinStats")
        # Get the status and data book options
        if nStats is None: nStats = opts.get_DataBookNStats(grp)
        if nMin   is None: nMin   = opts.get_DataBookNMin(grp)
        # Iteration at which to build table
        nOpt = opts.get_SubfigOpt(sfig, "Iteration")
        # Make sure current progress is a number
        if nIter is None: nIter = 0
        # Translate into absolute iteration number if relative.
        if nOpt == 0:
            # Use last iteration (standard)
            nCur = nIter
        elif nOpt < 0:
            # Use iteration relative to the end
            nCur = nIter + 1 + nOpt
        else:
            # Use the number
            nCur = nOpt
        # Initialize statistics
        S = {}
        # Get statistics if possible.
        if nCur >= max(1, nMin+nStats):
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(self.cntl.x.GetFullFolderNames(i))
            # Read the point sensor history
            P = self.ReadPointSensor()
            # Loop through components
            for pt in pts:
                # Process point index
                if type(pt).__name__.startswith('int'):
                    # Integer specification (dangerous)
                    kpt = pt
                else:
                    # Specified by name
                    kpt = P.GetPointSensorIndex(pt)
                # Check for adequate iterations
                if np.sum(P.i>=nMin) < nStats: continue
                # Get the statistics.
                S[pt] = P.GetStats(kpt, nStats=nStats, nLast=nCur)
        # Go back to original folder.
        os.chdir(fpwd)
        # Get the vertical alignment.
        hv = self.cntl.opts.get_SubfigOpt(sfig, 'Position')
        # Get subfigure width
        wsfig = self.cntl.opts.get_SubfigOpt(sfig, 'Width')
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = self.cntl.opts.get_SubfigOpt(sfig, 'Header')
        # Add the iteration number to header
        line = '\\textbf{\\textit{%s}} (Iteration %i' % (fhdr, nCur)
        # Add number of iterations used for statistics
        if len(S)>0:
            # Add stats count.
            line += (', {\\small\\texttt{nStats=%i}}' % nStats)
        # Close parentheses
        line += ')\\par\n'
        # Write the header.
        lines.append('\\noindent\n')
        lines.append(line)
        # Begin the table with the right amount of columns.
        line = '\\begin{tabular}{l'
        for coeff in coeffs:
            # Get the statsitical values to print
            fs = opts.get_SubfigOpt(sfig, coeff)
            # Column count
            nf = len(fs)
        # Loop through targets
        for targ in targs:
            # List of coefficients
            for coeff in topts[targ]["Coefficients"]:
                # Get list of target statistical fields for this coeff
                nf += len(topts[targ][coeff])
        # Append that many centered columns.
        line += ('|c'*nf)
        # Write the line to begin the table
        lines.append(line + '}\n')
        # Write headers.
        lines.append('\\hline \\hline\n')
        lines.append('\\textbf{\\textsf{Point}}\n')
        # Write headers
        for coeff in coeffs:
            # Loop through suffixes for this coefficient
            for fs in opts.get_SubfigOpt(sfig, coeff):
                # Get the symbol
                sym = self.GetStateSymbol(coeff, fs)
                # Append line line
                lines.append(' & ' + sym + ' \n')
        # Write headers for each target
        for targ in targs:
            # Get name of target with underscores removed
            ltarg = targ.replace('_', r'\_')
            # Check coefficients for this target
            for coeff in topts[targ]["Coefficients"]:
                # Loop through suffixes for this coefficient
                for fs in topts[targ][coeff]:
                    # Get the symbol
                    sym = self.GetStateSymbol(coeff, fs)
                    # Append line line
                    lines.append(' & ' + ltarg + '/' + sym + ' \n')
        # End header
        lines.append('\\\\\n')
        lines.append('\\hline\n')
        # Loop through points.
        for pt in pts:
            # Write point name.
            lines.append('\\texttt{%s}\n' % pt.replace('_', r'\_'))
            # Initialize line
            line = ''
            # Loop through the coefficients.
            for coeff in coeffs:
                # Loop through statistics
                for fs in opts.get_SubfigOpt(sfig, coeff):
                    # Process the statistic type
                    if nCur <= 0 or pt not in S:
                        # No iterations
                        line += '& $-$ '
                    elif fs == 'mu':
                        # Mean value
                        line += ('& $%.4f$ ' % S[pt][coeff])
                    elif fs == 'std':
                        # Standard deviation
                        line += ('& %.2e ' % S[pt][coeff+'_'+fs])
                    elif (coeff+'_'+fs) in S[pt]:
                        # Other statistic
                        line += ('& $%.4f$ ' % S[pt][coeff+'_'+fs])
                    else:
                        # Missing
                        line += '& $-$ '
            # Loop through targets
            for targ in targs:
                # Loop through the coefficients.
                for coeff in topts[targ]["Coefficients"]:
                    # Loop through statistics
                    for fs in topts[targ][coeff]:
                        # Name of field
                        c = ('%s.%s_%s' % (pt, coeff, fs)).rstrip('_mu')
                        # Column names
                        ckeys = DBT[targ].ckeys
                        # Name of column
                        if grp not in ckeys or c not in ckeys[grp]:
                            # No match for this coefficient
                            line += '& $-$ '
                            continue
                        else:
                            # Get the column
                            col = ckeys[grp][c]
                        # index
                        J = topts[targ]["Index"]
                        # Process the statistic type
                        if J is None or col not in DBT[targ]:
                            # No iterations
                            line += '& $-$ '
                        elif fs == 'mu':
                            # Mean value
                            val = np.mean(DBT[targ][col][J])
                            line += ('& $%.4f$ ' % val)
                        elif fs == 'std':
                            # Standard deviation
                            val = np.mean(DBT[targ][col][J])
                            line += ('& %.2e ' % val)
                        else:
                            # Other statistic
                            val = np.mean(DBT[targ][col][J])
                            line += ('& $%.4f$ ' % val)
            # Finish the line and append it.
            line += '\\\\\n'
            lines.append(line)
        # Finish table and subfigure
        lines.append('\\hline \\hline\n')
        lines.append('\\end{tabular}\n')
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

    # Function to plot mean coefficient for a sweep
    def SubfigSweepPointHist(self, sfig, fswp, I, q):
        r"""Plot a histogram of a point sensor coefficient over several cases

        :Call:
            >>> R.SubfigSweepCoeff(sfig, fswp, I)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indices in the sweep
            *q*: ``True`` | ``False``
                Whether or not to redraw images
        :Versions:
            * 2016-01-16 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Extract options and trajectory
        x = self.cntl.DataBook.x
        opts = self.cntl.opts
        # Case folder
        frun = x.GetFullFolderNames(I[0])
        # Get the component.
        grp = opts.get_SubfigOpt(sfig, "Group")
        # Get the point
        pt = opts.get_SubfigOpt(sfig, "Point")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # Get caption
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption
        if fcpt is None:
            # Use the point name and the coefficient
            fcpt = "%s/%s" % (pt, coeff)
        # Ensure that there are not underscores.
        fcpt = fcpt.replace("_", r"\_")
        # Initialize subfigure
        lines = self.SubfigInit(sfig)
        # Check for image update
        if not q:
            # File name to check for
            fpdf = '%s.pdf' % sfig
            # Check for the file
            if os.path.isfile(fpdf):
                # Include the graphics.
                lines.append('\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fpdf))
            # Set the caption.
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
        # Get list of targets
        targs = self.SubfigTargets(sfig)
        # Read the point sensor group data book
        self.cntl.DataBook.ReadPointSensor(grp)
        # Get the point sensor
        DBP = self.cntl.DataBook.PointSensors[grp][pt]
        # Get the targets
        targs = self.SubfigTargets(sfig)
        # Target labels
        ltarg = opts.get_SubfigOpt(sfig, "TargetLabel")
        # Ensure list
        if type(ltarg).__name__ != 'list':
            ltarg = [ltarg]
        # Initialize target values
        vtarg = []
        # Number of targets
        ntarg = len(targs)
        # Loop through targets.
        for i in range(ntarg):
            # Select the target
            targ = targs[i]
            # Read the target if not present.
            self.cntl.DataBook.ReadTarget(targ)
            # Get the data book target
            DBT = self.cntl.DataBook.GetTargetByName(targ)
            # Get the target co-sweep
            jt = self.GetTargetSweepIndices(fswp, I[0], targ)
            # Get the proper coefficient
            ckey = '%s.%s' % (pt, coeff)
            # Check for a match in the target
            if grp in DBT.ckeys and ckey in DBT.ckeys[grp]:
                # Get the name of the column in the target
                col = DBT.ckeys[grp][ckey]
                # Save the value.
                if len(jt) > 0:
                    vtarg.append(np.mean(DBT[col][jt]))
                else:
                    # No value!
                    vtarg.append(None)
            else:
                # No value!
                vtarg.append(None)
            # Default label
            if len(ltarg) < i or ltarg[i] in [None,False]:
                ltarg[i] = targ
        # Distribution variable
        xk = opts.get_SweepOpt(fswp, "XAxis")
        # Form options for point sensor histogram
        kw_h = {
            # Reference values
            "StDev":          opts.get_SubfigOpt(sfig, "StandardDeviation"),
            "Delta":          opts.get_SubfigOpt(sfig, "Delta"),
            "PlotMu":         opts.get_SubfigOpt(sfig, "PlotMu"),
            "OutlierSigma":   opts.get_SubfigOpt(sfig, "OutlierSigma"),
            # Target information
            "TargetValue":    vtarg,
            "TargetLabel":    ltarg,
            # Axis labels
            "XLabel":         opts.get_SubfigOpt(sfig, "XLabel"),
            "YLabel":         opts.get_SubfigOpt(sfig, "YLabel"),
            # Figure dimensions
            "FigureWidth":    opts.get_SubfigOpt(sfig, "FigWidth"),
            "FigureHeight":   opts.get_SubfigOpt(sfig, "FigHeight"),
            # Text labels of reference values
            "ShowMu":         opts.get_SubfigOpt(sfig, "ShowMu"),
            "ShowSigma":      opts.get_SubfigOpt(sfig, "ShowSigma"),
            "ShowDelta":      opts.get_SubfigOpt(sfig, "ShowDelta"),
            "ShowTarget":     opts.get_SubfigOpt(sfig, "ShowTarget"),
            # Format flags
            "MuFormat":       opts.get_SubfigOpt(sfig, "MuFormat"),
            "SigmaFormat":    opts.get_SubfigOpt(sfig, "SigmaFormat"),
            "DeltaFormat":    opts.get_SubfigOpt(sfig, "DeltaFormat"),
            "TargetFormat":   opts.get_SubfigOpt(sfig, "TargetFormat"),
            # Plot options
            "HistOptions":    opts.get_SubfigOpt(sfig, "HistOptions"),
            "MeanOptions":    opts.get_SubfigOpt(sfig, "MeanOptions"),
            "StDevOptions":   opts.get_SubfigOpt(sfig, "StDevOptions"),
            "DeltaOptions":   opts.get_SubfigOpt(sfig, "DeltaOptions"),
            "TargetOptions":  opts.get_SubfigOpt(sfig, "TargetOptions")
        }
        # Plot the histogram with labels
        h = DBP.PlotValueHist(coeff, I, **kw_h)
        # Change back to report folder
        os.chdir(fpwd)
        # Get the file formatting
        fmt = opts.get_SubfigOpt(sfig, "Format")
        dpi = opts.get_SubfigOpt(sfig, "DPI")
        # Figure name
        fimg = '%s.%s' % (sfig, fmt)
        fpdf = '%s.pdf' % sfig
        # Save the figure.
        if fmt.lower() in ['pdf']:
            # Save as vector-based image.
            h['fig'].savefig(fimg)
        elif fmt.lower() in ['svg']:
            # Save as PDF and SVG
            h['fig'].savefig(fimg)
            h['fig'].savefig(fpdf)
        else:
            # Save with resolution
            h['fig'].savefig(fimg, dpi=dpi)
            h['fig'].savefig(fpdf)
        # Close the figure
        h['fig'].clf()
        # Include the graphics
        lines.append('\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
            % (fswp, frun, fpdf))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

    # Process state symbol
    def GetStateSymbol(self, coeff, fs):
        """Get a TeX symbol for a coefficient and statistical field

        :Call:
            >>> sym = R.GetStateSymbol(coeff, fs)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report instance
            *coeff*: Cp | dp | rho | M | T | p
                Name of coefficient
            *fs*: mu | std | err | min | max | u
                Statistical quantity
        :Outputs:
            *sym*: :class:`str`
                TeX symbol (including ``$`` chars) for this field
        :Versions:
            * 2015-12-18 ``@ddalle``: First version
        """
        # Process label
        if coeff == 'Cp':
            # Pressure coefficient
            lbl = "C_p"
        elif coeff == 'dp':
            # Delta pressure
            lbl = r"(p-p_\infty)/p_\infty"
        elif coeff == 'rho':
            # Static density
            lbl = r'\rho/\rho_\infty'
        elif coeff == 'U':
            # x-velocity
            lbl = r'u/a_\infty'
        elif coeff == 'V':
            # y-velocity
            lbl = r'v/a_\infty'
        elif coeff == 'W':
            # z-velocity
            lbl = r'w/a_\infty'
        elif coeff == 'P':
            # weird pressure
            lbl = r'p/\gamma p_\infty'
        else:
            # Something else?
            lbl = coeff
        # Check suffix type
        if fs in ['std', 'sigma']:
            # Standard deviation
            sym = r'$\sigma(%s)$' % lbl
        elif fs in ['err', 'eps', 'epsilon']:
            # Sampling error
            sym = r'$\varepsilon(%s)$' % lbl
        elif fs == 'max':
            # Maximum value
            sym = 'max$(%s)$' % lbl
        elif fs == 'min':
            # Minimum value
            sym = 'min$(%s)$' % lbl
        elif fs == 'u':
            # Uncertainty
            sym = '$U(%s)$' % lbl
        else:
            # Mean
            sym = '$%s$' % lbl
        # Output
        return sym

    # Read a Tecplot script
    def ReadTecscript(self, fsrc):
        """Read a Tecplot script interface

        :Call:
            >>> R.ReadTecscript(fsrc)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *fscr*: :class:`str`
                Name of file to read
        :Versions:
            * 2016-10-25 ``@ddalle``: First version
        """
        return Tecscript(fsrc)

    # Function to link appropriate visualization files
    def LinkVizFiles(self, sfig=None, i=None):
        """Create links to appropriate visualization files

        Specifically, ``Components.i.plt`` and ``cutPlanes.plt`` or
        ``Components.i.dat`` and ``cutPlanes.dat`` are created.

        :Call:
            >>> R.LinkVizFiles(sfig, i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of the subfigure
            *i*: :class:`int`
                Case index
        :See Also:
            :func:`pyCart.casecntl.LinkPLT`
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        # Defer to function from :func:`pyCart.case`
        LinkPLT()



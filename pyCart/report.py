"""Function for Automated Report Interface"""

# File system interface
import os

# Local modules needed
from . import tex, tar
# Data book and plotting
from .dataBook import Aero, CaseFM, CaseResid

#<!--
# ---------------------------------
# I consider this portion temporary

# Get the umask value.
umask = 0027
# Get the folder permissions.
fmask = 0777 - umask
dmask = 0777 - umask

# Change the umask to a reasonable value.
os.umask(umask)

# ---------------------------------
#-->

# Class to interface with report generation and updating.
class Report(object):
    """Interface for automated report generation
    
    :Call:
        >>> R = pyCart.report.Report(cart3d, rep)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyCart.report.Report
            Automated report interface
    :Versions:
        * 2015-03-07 ``@dalle``: Started
    """
    # Initialization method
    def __init__(self, cart3d, rep):
        """Initialization method"""
        # Save the interface
        self.cart3d = cart3d
        # Check for this report.
        if rep not in cart3d.opts.get_ReportList():
            # Raise an exception
            raise KeyError("No report named '%s'" % rep)
        # Go to home folder.
        fpwd = os.getcwd()
        os.chdir(cart3d.RootDir)
        # Create the report folder if necessary.
        if not os.path.isdir('report'): os.mkdir(report, dmask)
        # Go into the report folder.
        os.chdir('report')
        # Get the options and save them.
        self.rep = rep
        self.opts = cart3d.opts['Report'][rep]
        # Initialize a dictionary of handles to case LaTeX files
        self.cases = {}
        # Read the file if applicable
        self.OpenMain()
        
        
        # Return
        os.chdir(fpwd)
        
        
    # Function to open the master latex file for this report.
    def OpenMain(self):
        """Open the primary LaTeX file or write skeleton if necessary
        
        :Call:
            >>> R.OpenMain()
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get and save the report file name.
        self.fname = 'report-' + self.rep + '.tex'
        # Go to the report folder.
        fpwd = os.getcwd()
        os.chdir(self.cart3d.RootDir)
        os.chdir('report')
        # Check for the file and create if necessary.
        if not os.path.isfile(self.fname): self.WriteSkeleton()
        # Open the interface to the master LaTeX file.
        self.tex = tex.Tex(self.fname)
        # Check quality.
        if len(self.tex.SectionNames) < 5:
            raise IOError("Bad LaTeX file '%s'" % self.fname)
        # Return
        os.chdir(fpwd)
        
    # Function to create the file for a case
    def UpdateCase(self, i):
        """Open, create if necessary, and update LaTeX file for a case
        
        :Call:
            >>> R.OpenCase(i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the case name.
        fgrp = self.cart3d.x.GetGroupFolderNames(i)
        fdir = self.cart3d.x.GetFolderNames(i)
        # Go to the report directory if necessary.
        fpwd = os.getcwd()
        os.chdir(self.cart3d.RootDir)
        os.chdir('report')
        # Create the folder if necessary.
        if not os.path.isdir(fgrp): os.mkdir(fgrp, dmask)
        # Go to the group folder.
        os.chdir(fgrp)
        # Create the case folder if necessary.
        if not (os.path.isfile(fdir+'.tar') or os.path.isdir(fgrp)):
            os.mkdir(fdir, dmask)
        # Go into the folder.
        self.cd(fdir)
        # Read the status file.
        nr, stsr = self.ReadCaseIter()
        # Get the actual iteration number.
        n = self.CheckCase(i)
        sts = self.CheckStatus(i)
        # Check if there's anything to do.
        if not (nr is None) or (nr < n) or (stsr != sts):
            # Go home and quit.
            os.chdir(fpwd)
            return
        # Check for the file.
        if not os.path.isfile(self.fname): self.WriteCaseSkeleton(i)
        # Open it.
        self.cases[i] = tex.Tex(self.fname)
        # Set the iteration number and status header.
        self.SetHeaderStatus(i)
        
        
        
    # Read the iteration to which the figures for this report have been updated
    def ReadCaseJSONIter():
        """Read JSON file to determine what the current iteration for the report is
        
        The status is read from the present working directory
        
        :Call:
            >>> n, sts = R.ReadCaseJSONIter()
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
        :Outputs:
            *n*: :class:`int` or :class:`float`
                Iteration at which the figure has been created
            *sts*: :class:`str`
                Status for the case
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        # Check for the file.
        if not os.path.isfile('report.json'): return None
        # Open the file
        f = open('report.json')
        # Read the settings.
        opts = json.load(f)
        # Close the file.
        f.close()
        # Read the status for this iteration
        return tuple(opts.get(self.rep, [None, None]))
        
    # Set the iteration of the JSON file.
    def SetCaseJSONIter(n, sts):
        """Mark the JSON file to say that the figure is up to date
        
        :Call:
            >>> R.SetCaseJSONIter(n, sts)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *n*: :class:`int` or :class:`float`
                Iteration at which the figure has been created
            *sts*: :class:`str`
                Status for the case
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Check for the file.
        if os.path.isfile('report.json'):
            # Open the file.
            f = open('report.json')
            # Read the settings.
            opts = json.load(f)
            # Close the file.
            f.close()
        else:
            # Create empty settings.
            opts = {}
        # Set the iteration number.
        opts[self.rep] = [n, sts]
        # Create the updated JSON file
        f = open('report.json', 'w')
        # Write the contents.
        json.dump(opts)
        # Close file.
        f.close()
        
    # Function to create the skeleton for a master LaTeX file
    def WriteSkeleton(self):
        """Create and write preamble for master LaTeX file for report
        
        :Call:
            >>> R.WriteSkeleton()
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Go to the report folder.
        fpwd = os.getcwd()
        os.chdir(self.cart3d.RootDir)
        os.chdir('report')
        # Create the file (delete if necessary)
        f = open(self.fname, 'w')
        # Write the universal header.
        f.write('%$__Class\n')
        f.write('\\documentclass[letter,10pt]{article}\n\n')
        # Write the preamble.
        f.write('%$__Preamble\n')
        # Margins
        f.write('\\usepackage[margin=0.6in,top=0.7in,headsep=0.1in,\n')
        f.write('    footskip=0.15in]{geometry}\n')
        # Other packages
        f.write('\\usepackage{graphicx}\n')
        f.write('\\usepackage{subcaption}\n')
        f.write('\\usepackage{hyperref}\n')
        f.write('\\usepackage{fancyhdr}\n')
        f.write('\\usepackage{amsmath}\n')
        f.write('\\usepackage{amssymb}\n')
        f.write('\\usepackage{times}\n')
        f.write('\\usepackage{DejaVuSansMono}\n')
        f.write('\\usepackage[T1]{fontenc}\n\n')
        
        # Set the title and author.
        f.write('\\title{%s}\n' % self.cart3d.opts.get_ReportTitle(self.rep))
        f.write('\\author{%s}\n' % self.cart3d.opts.get_ReportAuthor(self.rep))
        
        # Format the header and footer
        f.write('\n\\fancypagestyle{pycart}{%\n')
        f.write(' \\renewcommand{\\headrulewidth}{0.4pt}%\n')
        f.write(' \\renewcommand{\\footrulewidth}{0.4pt}%\n')
        f.write(' \\fancyhf{}%\n')
        f.write(' \\fancyfoot[C]{\\textbf{\\textsf{%s}}}%%\n'
            % self.cart3d.opts.get_ReportRestriction(self.rep))
        f.write(' \\fancyfoot[R]{\\thepage}%\n')
        # Check for a logo.
        if len(self.opts.get('Logo','')) > 0:
            f.write(' \\fancyfoot[L]{\\raisebox{-0.32in}{%\n')
            f.write('  \\includegraphics[height=0.45in]{%s}}}%%\n'
                % self.opts['Logo'])
        # Finish this primary header/footer format
        f.write('}\n\n')
        # Empty header/footer format for first page
        f.write('\\fancypagestyle{plain}{%\n')
        f.write(' \\renewcommand{\\headrulewidth}{0pt}%\n')
        f.write(' \\renewcommand(\\footrulewidth}{0pt}%\n')
        f.write(' \\fancyhf{}%\n')
        f.write('}\n\n')
        
        # Small captions if needed
        f.write('\\captionsetup[subfigure]')
        f.write('{textfont=sf,font+=footnotesize}\n\n')
        
        # Macros for setting cases.
        f.write('\\newcommand{\\thecase}{}\n')
        f.write('\\newcommand{\\setcase}[1]{\\renewcommand{\\thecase}{#1}}\n')
        
        # Actual document
        f.write('\n%$__Begin\n')
        f.write('\\begin{document}\n')
        f.write('\\pagestyle{plain}\n')
        f.write('\\maketitle\n\n')
        f.write('\\pagestyle{pycart}\n\n')
        
        # Skeleton for the main part of the report.
        f.write('%$__Cases\n')
        
        # Termination of the report
        f.write('\n%$__End\n')
        f.write('\end{document}\n')
        
        # Close the file.
        f.close()
        
        # Return
        os.chdir(fpwd)
        
    # Function to write skeleton for a case.
    def WriteCaseSkeleton(i):
        """Initialize LaTeX file for case *i*
        
        :Call:
            >>> R.WriteCaseSkeleton(i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
        """
        # Get the name of the case
        frun = self.cart3d.x.GetFullFolderNames(i)
        
        # Create the file (delete if necessary)
        f = open(self.fname, 'w')
        
        # Write the header.
        f.write('\n\\newpage\n')
        f.write('\\setcase{%s}\n' % frun.replace('_', '\\_'))
        f.write('\\phantomsection\n')
        f.write('\\addcontentsline{toc}{section}{\\texttt{\\thecase}}\n')
        f.write('\\fancyhead[L]{\\texttt{\\thecase}}\n\n')
        
        # Empty section for the figures
        f.write('%$__Figures\n')
        
        # Close the file.
        f.close()
        
    # Function to set the upper-right header
    def SetHeaderStatus(i):
        """Set header to state iteration progress and summary status
        
        :Call:
            >>> R.WriteCaseSkeleton(i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
        """
        # Get case current iteration
        n = cart3d.CheckStatus(i)
        # Get case number of required iterations
        nMax = cart3d.GetLastIter(i)
        # Get status
        sts = cart3d.CheckCaseStatus(i)
        # Form iteration string
        if n is None:
            # Unknown.
            fitr = '-/%s' % self.cart3d.opts.get_IterSeq(-1)
        else:
            # Use the values.
            fitr = '%s/%s' % (n, nMax)
        # Form string for the status type
        if sts == "PASS":
            # Set green
            fsts = '\\color{green}\\textbf{PASS}'
        elif sts == "ERROR":
            # Set red
            fsts = '\\color{red}\\textbf{ERROR}'
        else:
            # No color (black)
            fsts = '\\textbf{%s}' % sts
        # Form the line.
        line = '\\fancyhead[R]{\\textsf{%s, \large%s}}\n' % (fitr, fsts)
        # Put the line into the text.
        self.cases[i].ReplaceOrAddLineToSectionStartsWith(
            '_header', '\\fancyhead[R]', line, -1)
        # Update sections.
        self.UpdateLines()
            
        
    
        
    # Function to go into a folder, respecting archive option
    def cd(self, fdir):
        """Interface to :func:`os.chdir`, respecting "Archive" option
        
        This function can only change one directory at a time.
        
        :Call:
            >>> R.cd(fdir)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *fdir*: :class:`str`
                Name of directory to change to
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get archive option.
        q = self.cart3d.opts.get_ReportArchive(rep)
        # Check direction.
        if fdir.startswith('..'):
            # Check archive option.
            if q:
                # Tar and clean up if necessary.
                tar.chdir_up()
            else:
                # Go up a folder.
                os.chdir('..')
        else:
            # Check archive option.
            if q:
                # Untar if necessary
                tar.chdir_in(fdir)
            else:
                # Go into the folder
                os.chdir(fdir)
        
        

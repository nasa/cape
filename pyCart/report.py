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
        # Got to the report folder.
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
        
        
        
        
        

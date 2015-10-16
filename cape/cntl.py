"""
CAPE base module for CFD control: :mod:`cape.cape`
==================================================

This module provides tools and templates for tools to interact with various CFD
codes and their input files.  The base class is :class:`cape.cntl.Cntl`, and the
derivative classes include :class:`pyCart.cart3d.Cart3d`.

The derivative classes are used to read input files, set up cases, submit and/or
run cases, and be an interface for the various CAPE options

:Versions:
    * 2015-09-20 ``@ddalle``: Started
"""

# Numerics
import numpy as np
# Configuration file processor
import json
# File system
import os

# Local modules
from . import options
from . import queue
from . import case

# Functions and classes from other modules
from trajectory import Trajectory
from config     import Config

# Import triangulation
from tri import Tri, RotatePoints



# Class to read input files
class Cntl(object):
    """
    Class for handling global options, setup, and execution of CFD codes
    
    :Call:
        >>> cntl = cape.Cntl(fname="cape.json")
    :Inputs:
        *fname*: :class:`str`
            Name of JSON settings file from which to read options
    :Outputs:
        *cntl*: :class:`cape.cntl.Cntl`
            Instance of CAPE control interface
    :Versions:
        * 2015-09-20 ``@ddalle``: Started
    """
    # Initialization method
    def __init__(self, fname="cape.json"):
        """Initialization method for :mod:`cape.cntl.Cntl`"""
        
        # Read settings
        self.opts = options.Options(fname=fname)
        
        #Save the current directory as the root
        self.RootDir = os.getcwd()
        
        # Import modules
        self.ImportModules()
        
        # Process the trajectory.
        self.x = Trajectory(**self.opts['Trajectory'])

        # Job list
        self.jobs = {}
        
        # Set umask
        os.umask(self.opts.get_umask())
        
        
    # Output representation
    def __repr__(self):
        """Output representation method for Cntl class
        
        :Versions:
            * 2015-09-20 ``@ddalle``: First version
        """
        # Display basic information
        return "<cape.Cntl(nCase=%i)>" % self.x.nCase
        
    # Function to import user-specified modules
    def ImportModules(self):
        """Import user-defined modules, if any
        
        :Call:
            >>> cntl.ImportModules()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of CAPE control interface
        :Versions:
            * 2014-10-08 ``@ddalle``: First version (pyCart)
            * 2015-09-20 ``@ddalle``: Moved to parent class
        """
        # Get Modules.
        lmod = self.opts.get("Modules", [])
        # Ensure list.
        if not lmod:
            # Empty --> empty list
            lmod = []
        elif type(lmod).__name__ != "list":
            # Single string
            lmod = [lmod]
        # Loop through modules.
        for imod in lmod:
            # Status update
            print("Importing module '%s'." % imod)
            # Load the module by its name
            exec('self.%s = __import__("%s")' % (imod, imod))
        
    # Function to prepare the triangulation for each grid folder
    def ReadTri(self):
        """Read initial triangulation file(s)
        
        :Call:
            >>> cntl.ReadTri()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
        :Versions:
            * 2014-08-30 ``@ddalle``: First version
        """
        # Only read triangulation if not already present.
        try:
            self.tri
            return
        except Exception:
            pass
        # Get the list of tri files.
        ftri = self.opts.get_TriFile()
        # Status update.
        print("  Reading tri file(s) from root directory.")
        # Go to root folder safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read them.
        if type(ftri).__name__ == 'list':
            # Read the initial triangulation.
            tri = Tri(ftri[0])
            # Save the number of nodes to this point.
            tri.iTri = [tri.nTri]
            # Loop through the remaining tri files.
            for f in ftri[1:]:
                # Append the file.
                tri.Add(Tri(f))
                # Save the node number.
                tri.iTri.append(tri.nTri)
        else:
            # Just read the triangulation file.
            tri = Tri(ftri)
            # Save the one break point.
            tri.iTri = [tri.nTri]
        # Save it.
        self.tri = tri
        # Check for a config file.
        os.chdir(self.RootDir)
        self.tri.config = Config(self.opts.get_ConfigFile())
        # Make a copy of the original to revert to after rotations, etc.
        self.tri0 = self.tri.Copy()
        # Return to original location.
        os.chdir(fpwd)
            
    # Make a directory
    def mkdir(self, fdir):
        """Make a directory with the correct permissions
        
        :Call:
            >>> cntl.mkdir(fdir)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *fdir*: :class:`str`
                Directory to create
        :Versions:
            * 2015-09-27 ``@ddalle``: First version
        """
        # Get umask
        umask = self.opts.get_umask()
        # Apply mask
        dmask = 0777 - umask
        # Make the directory.
        os.mkdir(fdir, dmask)
        
    # Function to display current status
    def DisplayStatus(self, **kw):
        """Display current status for all cases
        
        This prints case names, current iteration numbers, and so on.
        
        :Call:
            >>> cntl.DisplayStatus(j=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *j*: :class:`bool`
                Whether or not to display job ID numbers
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-10-04 ``@ddalle``: First version
            * 2014-12-09 ``@ddalle``: Added constraints
        """
        # Force the "check" option to true.
        kw['c'] = True
        # Call the job submitter but don't allow submitting.
        self.SubmitJobs(**kw)
        
    # Master interface function
    def SubmitJobs(self, **kw):
        """Check jobs and prepare or submit jobs if necessary
        
        :Call:
            >>> cart3d.SubmitJobs(**kw)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *c*: :class:`bool`
                If ``True``, only display status; do not submit new jobs
            *j*: :class:`bool`
                Whether or not to display job ID numbers
            *n*: :class:`int`
                Maximum number of jobs to submit
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-10-05 ``@ddalle``: First version
            * 2014-12-09 ``@ddalle``: Added constraints
        """
        # Get flag that tells pycart only to check jobs.
        qCheck = kw.get('c', False)
        # Get flag to show job IDs
        qJobID = kw.get('j', False)
        # Check whether or not to kill PBS jobs
        qKill = kw.get('qdel', False)
        # No submissions if we're just deleting.
        if qKill: qCheck = True
        # Maximum number of jobs
        nSubMax = int(kw.get('n', 10))
        # Get list of indices.
        I = self.x.GetIndices(**kw)
        # Get the case names.
        fruns = self.x.GetFullFolderNames(I)
        
        # Get the qstat info (safely; do not raise an exception).
        jobs = queue.qstat(u=kw.get('u'))
        # Save the jobs.
        self.jobs = jobs
        # Initialize number of submitted jobs
        nSub = 0
        # Initialize number of jobs in queue.
        nQue = 0
        # Maximum length of one of the names
        if len(fruns) > 0:
            # Check the cases
            lrun = max([len(frun) for frun in fruns])
        else:
            # Just use a default value.
            lrun = 0
        # Make sure it's as long as the header
        lrun = max(lrun, 21)
        # Print the right number of '-' chars
        f = '-'; s = ' '
        # Create the string stencil.
        if qJobID:
            # Print status with job numbers.
            stncl = ('%%-%is ' * 6) % (4, lrun, 7, 11, 3, 7)
            # Print header row.
            print(stncl % ("Case", "Config/Run Directory", "Status", 
                "Iterations", "Que", "Job ID"))
            # Print "---- --------" etc.
            print(f*4 + s + f*lrun + s + f*7 + s + f*11 + s + f*3 + s + f*7)
        else:
            # Print status without job numbers.
            stncl = ('%%-%is ' * 5) % (4, lrun, 7, 11, 3)
            # Print header row.
            print(stncl % ("Case", "Config/Run Directory", "Status", 
                "Iterations", "Que"))
            # Print "---- --------" etc.
            print(f*4 + s + f*lrun + s + f*7 + s + f*11 + s + f*3)
        # Initialize dictionary of statuses.
        total = {'PASS':0, 'PASS*':0, '---':0, 'INCOMP':0,
            'RUN':0, 'DONE':0, 'QUEUE':0, 'ERROR':0}
        # Loop through the runs.
        for j in range(len(I)):
            # Case index.
            i = I[j]
            # Extract case
            frun = fruns[j]
            # Check status.
            sts = self.CheckCaseStatus(i, jobs, u=kw.get('u'))
            # Get active job number.
            jobID = self.GetPBSJobID(i)
            # Append.
            total[sts] += 1
            # Get the current number of iterations
            n = self.CheckCase(i)
            # Switch on whether or not case is set up.
            if n is None:
                # Case is not prepared.
                itr = "/"
                que = "."
            else:
                # Case is prepared and might be running.
                # Get last iteration.
                nMax = self.GetLastIter(i)
                # Iteration string
                itr = "%i/%i" % (n, nMax)
                # Check the queue.
                if jobID in jobs:
                    # Get whatever the qstat command said.
                    que = jobs[jobID]["R"]
                else:
                    # Not found by qstat (or not a jobID at all)
                    que = "."
                # Check for queue killing
                if qKill and (jobID in jobs):
                    # Delete it.
                    self.StopCase(i)
            # Print info
            if qJobID and jobID in jobs:
                # Print job number.
                print(stncl % (j, frun, sts, itr, que, jobID))
            elif qJobID:
                # Print blank job number.
                print(stncl % (j, frun, sts, itr, que, ""))
            else:
                # No job number.
                print(stncl % (j, frun, sts, itr, que))
            # Check status.
            if qCheck: continue
            # If submitting is allowed, check the job status.
            if sts in ['---', 'INCOMP']:
                # Prepare the job.
                self.PrepareCase(i)
                # Start (submit or run) case
                self.StartCase(i)
                # Increase job number
                nSub += 1
            # Don't continue checking if maximum submissions reached.
            if nSub >= nSubMax: break
        # Extra line.
        print("")
        # State how many jobs submitted.
        if nSub:
            print("Submitted or ran %i job(s).\n" % nSub)
        # Status summary
        fline = ""
        for key in total:
            # Check for any cases with the status.
            if total[key]:
                # At least one with this status.
                fline += ("%s=%i, " % (key,total[key]))
        # Print the line.
        if fline: print(fline)
        
    # Function to start a case: submit or run
    def StartCase(self, i):
        """Start a case by either submitting it 
        
        This function checks whether or not a case is submittable.  If so, the
        case is submitted via :func:`cape.queue.pqsub`, and otherwise the
        case is started using a system call.
        
        It is assumed that the case has been prepared.
        
        :Call:
            >>> pbs = cntl.StartCase(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
        """
        # Check status.
        if self.CheckCase(i) is None:
            # Case not ready
            return
        elif self.CheckRunning(i):
            # Case already running!
            return
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get case name and go to the folder.
        frun = self.x.GetFullFolderNames(i)
        os.chdir(frun)
        # Print status.
        print("     Starting case '%s'." % frun)
        # Start the case by either submitting or calling it.
        pbs = self.CaseStartCase()
        # Display the PBS job ID if that's appropriate.
        if pbs:
            print("     Submitted job: %i" % pbs)
        # Go back.
        os.chdir(fpwd)
        # Output
        return pbs
    
    def CaseStartCase(self):
        """Start a case by either submitting it or running it
        
        This function relies on :mod:`cape.case`, and so it is customized for
        the correct solver only in that it calls the correct *case* module.
        
        :Call:
            >>> pbs = cart3d.CaseStartCase()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        return case.StartCase()
        
    # Function to terminate a case: qdel and remove RUNNING file
    def StopCase(self, i):
        """
        Stop a case by deleting its PBS job and removing the :file:`RUNNING`
        file.
        
        :Call:
            >>> cart3d.StopCase(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Check status.
        if self.CheckCase(i) is None:
            # Case not ready
            return
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name and go there.
        frun = self.x.GetFullFolderNames(i)
        os.chdir(frun)
        # Stop the job if possible.
        case.StopCase()
        # Go back.
        os.chdir(fpwd)
        
    
        
    # Function to determine if case is PASS, ---, INCOMP, etc.
    def CheckCaseStatus(self, i, jobs=None, auto=False, u=None):
        """Determine the current status of a case
        
        :Call:
            >>> sts = cart3d.CheckCaseStatus(i, jobs=None, auto=False, u=None)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
            *jobs*: :class:`dict`
                Information on each job, ``jobs[jobID]`` for each submitted job
            *u*: :class:`str`
                User name (defaults to ``os.environ['USER']``)
        :Versions:
            * 2014-10-04 ``@ddalle``: First version
            * 2014-10-06 ``@ddalle``: Checking queue status
        """
        # Current iteration count
        n = self.CheckCase(i)
        # Try to get a job ID.
        jobID = self.GetPBSJobID(i)
        # Default jobs.
        if jobs is None:
            # Use current status.
            jobs = self.jobs
        # Check for auto-status
        if (jobs=={}) and auto:
            # Call qstat.
            self.jobs = queue.qstat(u=u)
            jobs = self.jobs
        # Check if the case is prepared.
        if self.CheckError(i):
            # Case contains :file:`FAIL`
            sts = "ERROR"
        elif n is None:
            # Nothing prepared.
            sts = "---"
        else:
            # Check if the case is running.
            if self.CheckRunning(i):
                # Case currently marked as running.
                sts = "RUN"
            elif self.CheckError(i):
                # Case has some sort of error.
                sts = "ERROR"
            else:
                # Get maximum iteration count.
                nMax = self.GetLastIter(i)
                # Check current count.
                if jobID in jobs:
                    # It's in the queue, but apparently not running.
                    if jobs[jobID]['R'] == "R":
                        # Job running according to the queue
                        sts = "RUN"
                    else:
                        # It's in the queue.
                        sts = "QUEUE"
                elif n >= nMax:
                    # Not running and sufficient iterations completed.
                    sts = "DONE"
                else:
                    # Not running and iterations remaining.
                    sts = "INCOMP"
        # Check if the case is marked as PASS
        if self.x.PASS[i]:
            # Check for cases marked but that can't be done.
            if sts == "DONE":
                # Passed!
                sts = "PASS"
            else:
                # Funky
                sts = "PASS*"
        # Output
        return sts
        
    # Check a case.
    def CheckCase(self, i):
        """Check current status of run *i*
        
        Because the file structure is different for each solver, some of this
        method may need customization.  This customization, however, can be kept
        to the functions :func:`cape.case.GetCurrentIter` and
        :func:`cape.cntl.Cntl.CheckNone`.
        
        :Call:
            >>> n = cntl.CheckCase(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *n*: :class:`int` or ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2014-09-27 ``@ddalle``: First version
            * 2015-09-27 ``@ddalle``: Generic version
            * 2015-10-14 ``@ddalle``: Removed dependence on :mod:`case`
        """
         # Check input.
        if type(i).__name__ not in ["int", "int64", "int32"]:
            raise TypeError(
                "Input to :func:`Cntl.CheckCase()` must be :class:`int`.")
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Initialize iteration number.
        n = 0
        # Check if the folder exists.
        if (not os.path.isdir(frun)): n = None
        # Check that test.
        if n is not None:
            # Go to the group folder.
            os.chdir(frun)
            # Check the history iteration
            n = self.CaseGetCurrentIter()
        # If zero, check if the required files are set up.
        if (n == 0) and self.CheckNone(): n = None
        # Return to original folder.
        os.chdir(fpwd)
        # Output.
        return n
        
    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentIter(self):
        """Get the current iteration number from the appropriate module
        
        This function utilizes the :mod:`cape.case` module, and so it must be
        copied to the definition for each solver's control class
        
        :Call:
            >>> n = cntl.CaseGetCurrentIter()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *n*: :class:`int` or ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        return case.GetCurrentIter()
        
        
    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self):
        """Check if case *i* has the necessary files to run
        
        :Versions:
            * 2015-09-27 ``@ddalle``: First version
        """
        return False
    
    
    # Get PBS job ID if possible
    def GetPBSJobID(self, i):
        """Get PBS job number if one exists
        
        :Call:
            >>> pbs = cntl.GetPBSJobID(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *pbs*: :class:`int` or ``None``
                Most recently reported job number for case *i*
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
        """
        # Check the case.
        if self.CheckCase(i) is None: return None
        # Go to the root folder
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Go there.
        os.chdir(frun)
        # Check for a "jobID.dat" file.
        if os.path.isfile('jobID.dat'):
            # Read the file.
            try:
                # Open the file and read the first line.
                line = open('jobID.dat').readline()
                # Get the job ID.
                pbs = int(line.split()[0])
            except Exception:
                # Unsuccessful reading for some reason.
                pbs = None
        else:
            # No file.
            pbs = None
        # Return to original directory.
        os.chdir(fpwd)
        # Output
        return pbs
        
    # Write a PBS header
    def WritePBSHeader(self, f, i, j):
        """Write common part of PBS script
        
        :Call:
            >>> cntl.WritePBSHeader(f, i, j)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *f*: :class:`file`
                Open file handle
            *i*: :class:`int`
                Case index
            *j*: :class:`int`
                Run index
        :Versions:
            * 2015-09-30 ``@ddalle``: Separated from WritePBS
        """
        # Get the shell path (must be bash)
        sh = self.opts.get_PBS_S(j)
        # Write to script both ways.
        f.write('#!%s\n' % sh)
        f.write('#PBS -S %s\n' % sh)
        # Get the shell name.
        lbl = self.GetPBSName(i)
        # Write it to the script
        f.write('#PBS -N %s\n' % lbl)
        # Get the rerun status.
        PBS_r = self.opts.get_PBS_r(j)
        # Write if specified.
        if PBS_r: f.write('#PBS -r %s\n' % PBS_r)
        # Get the option for combining STDIO/STDOUT
        PBS_j = self.opts.get_PBS_j(j)
        # Write if specified.
        if PBS_j: f.write('#PBS -j %s\n' % PBS_j)
        # Get the number of nodes, etc.
        nnode = self.opts.get_PBS_select(j)
        ncpus = self.opts.get_PBS_ncpus(j)
        nmpis = self.opts.get_PBS_mpiprocs(j)
        smodl = self.opts.get_PBS_model(j)
        # Form the -l line.
        line = '#PBS -l select=%i:ncpus=%i' % (nnode, ncpus)
        # Add other settings
        if nmpis: line += (':mpiprocs=%i' % nmpis)
        if smodl: line += (':model=%s' % smodl)
        # Write the line.
        f.write(line + '\n')
        # Get the walltime.
        t = self.opts.get_PBS_walltime(j)
        # Write it.
        f.write('#PBS -l walltime=%s\n' % t)
        # Check for a group list.
        PBS_W = self.opts.get_PBS_W(j)
        # Write if specified.
        if PBS_W: f.write('#PBS -W %s\n' % PBS_W)
        # Get the queue.
        PBS_q = self.opts.get_PBS_q(j)
        # Write it.
        if PBS_q: f.write('#PBS -q %s\n\n' % PBS_q)
        
        # Go to the working directory.
        f.write('# Go to the working directory.\n')
        f.write('cd %s\n\n' % os.getcwd())
        
        # Write a header for the shell commands.
        f.write('# Additional shell commands\n')
        # Loop through the shell commands.
        for line in self.opts.get_ShellCmds():
            # Write it.
            f.write('%s\n' % line)
        
    # Check if a case is running.
    def CheckRunning(self, i):
        """Check if a case is currently running
        
        :Call:
            >>> q = cart3d.CheckRunning(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`RUNNING` file in it
        :Versions:
            * 2014-10-03 ``@ddalle``: First version
        """
        # Safely go to root.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the RUNNING file.
        q = os.path.isfile(os.path.join(frun, 'RUNNING'))
        # Go home.
        os.chdir(fpwd)
        # Output
        return q
            
    # Check for a failure.
    def CheckError(self, i):
        """Check if a case has a failure
        
        :Call:
            >>> q = cntl.CheckError(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`FAIL` file in it
        :Versions:
            * 2015-01-02 ``@ddalle``: First version
        """
        # Safely go to root.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the RUNNING file.
        q = os.path.isfile(os.path.join(frun, 'FAIL'))
        # Go home.
        os.chdir(fpwd)
        # Output
        return q
        
    # Get last iter
    def GetLastIter(self, i):
        """Get minimum required iteration for a given run to be completed
        
        :Call:
            >>> nIter = cart3d.GetLastIter(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *nIter*: :class:`int`
                Number of iterations required for case *i*
        :Versions:
            * 2014-10-03 ``@ddalle``: First version
        """
        # Check the case
        if self.CheckCase(i) is None:
            return None
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Go there.
        os.chdir(frun)
        # Read the local case.json file.
        fc = case.ReadCaseJSON()
        # Option for desired iterations
        N = fc.get('IterSeq', 0)
        # Return to original location.
        os.chdir(fpwd)
        # Output the last entry (if list)
        return options.getel(N, -1)
        
    # Get PBS name
    def GetPBSName(self, i):
        """Get PBS name for a given case
        
        :Call:
            >>> lbl = cntl.GetPBSName(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl` or derivative
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *lbl*: :class:`str`
                Short name for the PBS job, visible via `qstat`
        :Versions:
            * 2014-09-30 ``@ddalle``: First version
        """
        # Extract the trajectory.
        x = self.x
        # Initialize label.
        lbl = ''
        # Loop through keys.
        for k in x.keys[0:]:
            # Skip it if not part of the label.
            if not x.defns[k].get('Label', True):
                continue
            # Default print flag
            if x.defns[k]['Value'] == 'float':
                # Float: get two decimals if nonzero
                sfmt = '%.2f'
            else:
                # Simply use string
                sfmt = '%s'
            # Non-default strings
            slbl = x.defns[k].get('PBSLabel', x.abbrv[k])
            sfmt = x.defns[k].get('PBSFormat', sfmt)
            # Apply values
            slbl = slbl + (sfmt % getattr(x,k)[i])
            # Strip underscores
            slbl = slbl.replace('_', '')
            # Strop trailing zeros and decimals if float
            if x.defns[k]['Value'] == 'float':
                slbl = slbl.rstrip('0').rstrip('.')
            # Append to the label.
            lbl += slbl
        # Check length.
        if len(lbl) > 15:
            # 16-char limit (or is it 15?)
            lbl = lbl[:15]
        # Output
        return lbl
        
    # Prepare a case.
    def PrepareCase(self, i):
        """Prepare case for running if necessary
        
        :Call:
            >>> n = cntl.PrepareCase(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to analyze
        :Versions:
            * 2014-09-30 ``@ddalle``: First version
            * 2015-09-27 ``@ddalle``: Template version
        """
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if prepared.
        if n is not None: return None
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Save current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Make the directory if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Go there.
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        
        # Write a JSON files with flowCart and plot settings.
        self.WriteCaseJSON(i)
        
        # Return to original location.
        os.chdir(fpwd)
    
    # Write flowCart options to JSON file
    def WriteCaseJSON(self, i):
        """Write JSON file with `flowCart` and related settings for case *i*
        
        :Call:
            >>> cart3d.WriteCaseJSON(i)
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-12-08 ``@ddalle``: First version
        """
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Check if it exists.
        if not os.path.isdir(frun):
            # Go back and quit.
            os.chdir(fpwd)
            return
        # Go to the folder.
        os.chdir(frun)
        # Write folder.
        f = open('case.json', 'w')
        # Dump the flowCart settings.
        json.dump(self.opts['flowCart'], f, indent=1)
        # Close the file.
        f.close()
        # Return to original location
        os.chdir(fpwd)
        
    

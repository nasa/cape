"""
:mod:`cape.cfdx.queue`: PBS queue interface module
==================================================

This module contains direct interface for functions like `qsub` and `qstat`.
These methods provide an easy interface to command-line PBS utilities and also
provide some access to the PBS information.  For example, the method
:func:`cape.queue.pqsub` writes a file ``"jobID.dat"`` with the PBS job number
of the submitted job.

"""
# OS interface
import subprocess as sp
import os

# For processing qstat lines
import re


# Function to call `qsub` and get the PBS number
def qsub(fname):
    """Submit a PBS script and return the job number

    :Call:
        >>> pbs = cape.queue.qsub(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of PBS script to submit
    :Outputs:
        *pbs*: :class:`int` or ``None``
            PBS job ID number if submission was successful
    :Versions:
        * 2014-10-05 ``@ddalle``: Version 1.0
        * 2021-08-09 ``@ddalle``: Version 2.0
            - Support Python 3
            - Use full job ID (not just :class:`int`) as a backup
    """
    # Call the command with safety
    try:
        # Call `qsub` with output
        stdout, _ = sp.Popen(['qsub', fname], stdout=sp.PIPE).communicate()
        # Get the job ID
        try:
            # Get the integer job number
            return int(stdout.decode("utf-8").split('.')[0])
        except Exception:
            # Use the full name
            return stdout.decode("utf-8").strip()
    except Exception:
        # Print a message, but don't fail.
        print("Submitting PBS script failed:\n  '%s/%s'"
            % (os.getcwd(), fname))
        print("  Recevied STDOUT: %si (%s)" % (stdout, type(stdout)))
        # Failed; return None
        return None


# Function to call `qsub` and get the PBS number
def sbatch(fname):
    """Submit a Slurm script and return the job number

    :Call:
        >>> pbs = cape.queue.sbatch(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of Slurm script to submit
    :Outputs:
        *pbs*: :class:`int` or ``None``
            Slurm job ID number if submission was successful
    :Versions:
        * 2014-10-05 ``@ddalle``: First version
    """
    # Call the command with safety
    try:
        # Call `qsub` with output
        stdout, _ = sp.Popen(['sbatch', fname], stdout=sp.PIPE).communicate()
        # Decode
        txt = stdout.decode("utf-8")
        # Get the integer job number.
        return int(txt.split()[-1].strip())
    except Exception:
        # Print a message, but don't fail.
        print("Submitting Slurm script failed:\n  '%s/%s'"
            % (os.getcwd(), fname))
        # Failed; return None
        return None


# Function to delete jobs from the queue.
def qdel(jobID):
    """Delete a PBS job by number

    :Call:
        >>> cape.queue.qdel(jobID)
    :Inputs:
        *pbs*: :class:`int` or :class:`list` (:class:`int`)
            PBS job ID number if submission was successful
    :Versions:
        * 2014-12-26 ``@ddalle``: First version
    """
    # Convert to list if necessary.
    if type(jobID).__name__ not in ['list', 'ndarray']:
        # Convert to list.
        jobID = [jobID]
    # Call the command with safety.
    for jobI in jobID:
        try:
            # Call `qdel`
            sp.Popen(['qdel', str(jobI)], stdout=sp.PIPE).communicate()
            # Status update.
            print("     Deleted PBS job %i" % jobI)
        except Exception:
            print("     Failed to delete PBS job %s" % jobI)


# Function to delete jobs from the Slurm queue.
def scancel(jobID):
    """Delete a Slurm job by number

    :Call:
        >>> cape.queue.scancel(jobID)
    :Inputs:
        *pbs*: :class:`int` or :class:`list` (:class:`int`)
            PBS job ID number if submission was successful
    :Versions:
        * 2018-10-10 ``@ddalle``: First version
    """
    # Convert to list if necessary.
    if type(jobID).__name__ not in ['list', 'ndarray']:
        # Convert to list.
        jobID = [jobID]
    # Call the command with safety.
    for jobI in jobID:
        try:
            # Call `qdel`
            sp.Popen(['scancel', str(jobI)], stdout=sp.PIPE).communicate()
            # Status update.
            print("     Deleted Slurm job %i" % jobI)
        except Exception:
            print("     Failed to delete Slurm job %s" % jobI)


# Function to call `qsub` and save the job ID
def pqsub(fname, fout="jobID.dat"):
    """Submit a PBS script and save the job number in an *fout* file

    :Call:
        >>> pbs = cape.queue.pqsub(fname, fout="jobID.dat")
    :Inputs:
        *fname*: :class:`str`
            Name of PBS script to submit
        *fout*: :class:`str`
            Name of output file to contain PBS job number
    :Outputs:
        *pbs*: :class:`int` or ``None``
            PBS job ID number if submission was successful
    :Versions:
        * 2014-10-06 ``@ddalle``: Version 1.0
        * 2021-08-09 ``@ddalle``: Version 1.1; allow non-int PBS IDs
    """
    # Submit the job.
    pbs = qsub(fname)
    # Create the file if the submission was successful.
    if pbs:
        # Create the output file.
        f = open(fout, 'w')
        # Write the job id.
        f.write('%s\n' % pbs)
        # Close the file.
        f.close()
    # Return the number.
    return pbs


# Function to call `abatch` and save the job ID
def psbatch(fname, fout="jobID.dat"):
    """Submit a PBS script and save the job number in an *fout* file

    :Call:
        >>> pbs = cape.queue.psbatch(fname, fout="jobID.dat")
    :Inputs:
        *fname*: :class:`str`
            Name of PBS script to submit
        *fout*: :class:`str`
            Name of output file to contain PBS job number
    :Outputs:
        *pbs*: :class:`int` or ``None``
            PBS job ID number if submission was successful
    :Versions:
        * 2018-10-10 ``@ddalle``: Version 1.0
        * 2021-08-09 ``@ddalle``: Version 1.1; allow non-int job IDs
    """
    # Submit the job.
    pbs = sbatch(fname)
    # Create the file if the submission was successful.
    if pbs:
        # Create the output file.
        f = open(fout, 'w')
        # Write the job id.
        f.write('%s\n' % pbs)
        # Close the file.
        f.close()
    # Return the number.
    return pbs


# Function to get the job ID
def pqjob(fname="jobID.dat"):
    """Read the PBS job number from file

    :Call:
        >>> pbs = cape.queue.pqjob()
        >>> pbs = cape.queue.pqjob(fname="jobID.dat")
    :Inputs:
        *fname*: :class:`str`
            Name of file to read containing PBS job number
    :Outputs:
        *pbs*: :class:`int`
            PBS job ID number if possible or ``0`` if file could not be read
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
    """
    # Be safe.
    try:
        # Read the file.
        line = open(fname).readline().strip()
        # Get the first value.
        pbs = int(line.split()[0])
        # Output
        return pbs
    except Exception:
        return None


# Function to get `qstat` information
def qstat(u=None, J=None):
    """Call `qstat` and process information

    :Call:
        >>> jobs = cape.queue.qstat(u=None)
        >>> jobs = cape.queue.qstat(J=None)
    :Inputs:
        *u*: :class:`str`
            User name, defaults to ``os.environ[USER]``
        *J*: :class:`int`
            Specific job ID for which to check
    :Outputs:
        *jobs*: :class:`dict`
            Information on each job, ``jobs[jobID]`` for each submitted job
    :Versions:
        * 2014-10-06 ``@ddalle``: First version
        * 2015-06-19 ``@ddalle``: Added ``qstat -J`` option
    """
    # Process username
    if u is None:
        u = os.environ['USER']
    # Form the command
    if J is not None:
        # Call for a specific job.
        cmd = ['qstat', '-J', str(J)]
    else:
        # Call for a user.
        cmd = ['qstat', '-u', u]
    # Call the command with safety.
    try:
        # Call `qstat` with output.
        txt = sp.Popen(cmd, stdout=sp.PIPE).communicate()[0].decode("utf-8")
        # Split into lines.
        lines = txt.split('\n')
        # Initialize jobs.
        jobs = {}
        # Loop through lines.
        for line in lines:
            # Check for lines that don't start with PBS ID number.
            if not re.match('[0-9]', line): continue
            # Split into values.
            v = line.split()
            # Get the job ID
            jobID = int(v[0].split('.')[0])
            # Save the job info.
            jobs[jobID] = {'u':v[1], 'q':v[2], 'N':v[3], 'R':v[7]}
        # Output.
        return jobs
    except Exception:
        # Failed or no qstat command
        return {}


# Function to get `qstat` information
def squeue(u=None, J=None):
    """Call `qstat` and process information

    :Call:
        >>> jobs = cape.queue.squeue(u=None)
        >>> jobs = cape.queue.squeue(J=None)
    :Inputs:
        *u*: :class:`str`
            User name, defaults to ``os.environ[USER]``
        *J*: :class:`int`
            Specific job ID for which to check
    :Outputs:
        *jobs*: :class:`dict`
            Information on each job, ``jobs[jobID]`` for each submitted job
    :Versions:
        * 2014-10-06 ``@ddalle``: First version
        * 2015-06-19 ``@ddalle``: Added ``qstat -J`` option
    """
    # Process username
    if u is None:
        u = os.environ['USER']
    # Form the command
    if J is not None:
        # Call for a specific job.
        cmd = ['squeue', '-dd', str(J)]
    else:
        # Call for a user.
        cmd = ['squeue', '-u', u]
    # Call the command with safety.
    try:
        # Call `qstat` with output.
        txt = sp.Popen(cmd, stdout=sp.PIPE).communicate()[0].decode("utf-8")
        # Split into lines.
        lines = txt.split('\n')
        # Initialize jobs.
        jobs = {}
        # Loop through lines.
        for line in lines:
            # Check for lines that don't start with PBS ID number.
            if not re.match('[0-9]', line): continue
            # Split into values.
            v = line.split()
            # Get the job ID
            jobID = int(v[0].split('.')[0])
            # Save the job info.
            jobs[jobID] = {'u':v[3], 'q':v[1], 'N':v[6], 'R':v[4]}
        # Output.
        return jobs
    except Exception:
        # Failed or no qstat command
        return {}


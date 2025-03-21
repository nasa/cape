r"""
:mod:`cape.cfdx.queue`: Manage PBS and Slurm interfaces
=============================================================

This module contains direct interface for functions like ``qsub`` and
``qstat``. These methods provide an easy interface to command-line PBS
utilities and also provide some access to the PBS information. For
example, the method :func:`pqsub` writes a file ``jobID.dat`` with the
PBS job number of the submitted job.

"""

# Standard library
import getpass
import os
import re
from io import IOBase
from subprocess import Popen, PIPE
from typing import Optional, Union

# Local imports
from ..errors import assert_which


# PBS/Slurm job ID file
JOB_ID_FILE = "jobID.dat"
JOB_ID_FILES = (
    os.path.join("cape", "jobID.dat"),
    "jobID.dat",
)


# Function to call `qsub` and get the PBS number
def qsub(fname: str) -> str:
    r"""Submit a PBS script and return the job number

    :Call:
        >>> jobid = qsub(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of PBS script to submit
    :Outputs:
        *jobid*: :class:`str` | ``None``
            PBS job ID if submission was successful
    :Versions:
        * 2014-10-05 ``@ddalle``: v1.0
        * 2021-08-09 ``@ddalle``: v2.0
            - Support Python 3
            - Use full job ID (not just :class:`int`) as a backup

        * 2023-11-07 ``@ddalle``: v2.1; test for ``qsub``
        * 2024-06-18 ``@ddalle``: v3.0; string output
    """
    # Check for missing ``qsub`` function
    assert_which("qsub")
    # Call ``qsub`` with captured output
    proc = Popen(['qsub', fname], stdout=PIPE)
    # Get STDOUT
    stdout, _ = proc.communicate()
    # Check return code
    if proc.returncode:
        # Print a message, but don't fail
        print(f"Submitting PBS script failed:\n  > {os.path.abspath(fname)}")
    # Decode output and split by '.'
    jobname = stdout.strip().decode()
    # Return everything up to second '.'
    return _job(jobname)


# Function to call `qsub` and get the PBS number
def sbatch(fname: str) -> Optional[str]:
    r"""Submit a Slurm script and return the job number

    :Call:
        >>> jobid = sbatch(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of Slurm script to submit
    :Outputs:
        *jobid*: :class:`str` | ``None``
            Slurm job ID if submission was successful
    :Versions:
        * 2022-10-05 ``@ddalle``: v1.0
        * 2024-06-18 ``@ddalle``: v2.0; string output
    """
    # Check for missing ``sbatch`` function
    assert_which("sbatch")
    # Call ``qsub`` with captured output
    proc = Popen(['sbatch', fname], stdout=PIPE)
    # Get STDOUT
    stdout, _ = proc.communicate()
    # Check return code
    if proc.returncode:
        # Print a message, but don't fail
        print(f"Submitting Slurm script failed:\n  > {os.path.abspath(fname)}")
    # Decode output and split by '.'
    jobname = stdout.strip().decode()
    # Take everything up to second '.'
    return _job(jobname)


# Function to delete jobs from the queue.
def qdel(jobID: Union[str, int], force: bool = False):
    r"""Delete a PBS job by ID

    :Call:
        >>> qdel(jobID, force=False)
    :Inputs:
        *jobID*: :class:`str` | :class:`int`
            PBS job ID number if submission was successful
        *force*: ``True`` | {``False``}
            Option to force-kill troublesome job
    :Versions:
        * 2014-12-26 ``@ddalle``: v1.0
        * 2024-06-12 ``@ddalle``: v2.0; eliminate loop
        * 2024-06-16 ``@ddalle``: v2.1; add *force*
    """
    # Convert to str if int
    jobID = _job(jobID)
    # ``qdel`` command
    cmdlist = ["qdel", jobID]
    # Add -W force option?
    if force:
        cmdlist += ["-W", "force"]
    # Call ``qdel``
    proc = Popen(['qdel', jobID], stdout=PIPE, stderr=PIPE)
    # Wait for command
    proc.communicate()
    # Status update
    if proc.returncode == 0:
        print(f"     Deleted PBS job {jobID}")
    else:
        print(f"     Failed to delete PBS job {jobID}")


# Function to delete jobs from the Slurm queue.
def scancel(jobID: Union[str, int]):
    r"""Delete a Slurm job by ID

    :Call:
        >>> scancel(jobID)
    :Inputs:
        *pbs*: :class:`int` or :class:`list`\ [:class:`int`]
            PBS job ID number if submission was successful
    :Versions:
        * 2022-10-05 ``@ddalle``: v1.0
    """
    # Convert to str if int
    jobID = _job(jobID)
    # Call ``scancel``
    proc = Popen(['scancel', str(jobID)], stdout=PIPE)
    proc.communicate()
    # Status update
    if proc.returncode == 0:
        print(f"     Deleted Slurm job {jobID}")
    else:
        print(f"     Failed to delete Slurm job {jobID}")


# Function to call `qsub` and save the job ID
def pqsub(fname: str, fout: str = JOB_ID_FILE) -> str:
    r"""Submit a PBS script and save the job number in an *fout* file

    :Call:
        >>> pbs = pqsub(fname, fout="jobID.dat")
    :Inputs:
        *fname*: :class:`str`
            Name of PBS script to submit
        *fout*: :class:`str`
            Name of output file to contain PBS job number
    :Outputs:
        *pbs*: :class:`int` | ``None``
            PBS job ID number if submission was successful
    :Versions:
        * 2014-10-06 ``@ddalle``: v1.0
        * 2021-08-09 ``@ddalle``: v1.1; allow non-int PBS IDs
    """
    # Submit the job
    jobname = qsub(fname)
    # Keep up to second '.'
    jobID = _job(jobname)
    # Create the file if the submission was successful
    if jobID:
        # Create file
        with openfile_w(fout, 'w') as fp:
            # Write the job id
            fp.write(f"{jobID}\n")
    # Return the job
    return jobID


# Function to call `abatch` and save the job ID
def psbatch(fname: str, fout: str = JOB_ID_FILE) -> str:
    r"""Submit a PBS script and save the job number in an *fout* file

    :Call:
        >>> pbs = psbatch(fname, fout="jobID.dat")
    :Inputs:
        *fname*: :class:`str`
            Name of PBS script to submit
        *fout*: :class:`str`
            Name of output file to contain PBS job number
    :Outputs:
        *pbs*: :class:`int` | ``None``
            Slurm job ID number if submission was successful
    :Versions:
        * 2022-10-05 ``@ddalle``: v1.0
    """
    # Submit the job
    jobID = sbatch(fname)
    # Create the file if the submission was successful.
    if jobID:
        # Open file
        with openfile_w(fout, 'w') as fp:
            # Write the job id
            fp.write(f'{jobID}\n')
    # Return the number.
    return jobID


# Read all current job IDs
def read_job_ids() -> list:
    r"""Read IDs of all jobs listed in ``jobID.dat`` or similar file

    :Call:
        >>> job_ids = read_job_ids()
    :Outputs:
        *job_ids*: :class:`list`\ [:class:`str`]
            List of job identifiers
    :Versions:
        * 2024-06-13 ``@ddalle``: v1.0
    """
    # Initialize list
    job_ids = []
    # Open the jobID.dat or similar file
    with open_jobfile_r() as fp:
        # Check if no file was found
        if fp is None:
            return job_ids
        # Read lines of file
        for raw_line in fp.readlines():
            # Strip line
            line = raw_line.strip()
            # Remove comments
            jcom = line.find('#')
            job_id = jcom[:jcom].strip()
            # Check for comment, empty line, or duplicate
            if job_id and job_id not in job_ids:
                # Save it
                job_ids.append(job_id)
    # Output
    return job_ids


# Function to get `qstat` information
def qstat(
        u: Optional[str] = None,
        J: Optional[Union[str, int]] = None) -> dict:
    r"""Call ``qstat`` and process information

    :Call:
        >>> jobs = qstat(u=None, J=None)
    :Inputs:
        *u*: {``None``} | :class:`str`
            User name, defaults to current user's name
        *J*: {``None``} | :class:`str` | :class:`int`
            Specific job ID for which to check
    :Outputs:
        *jobs*: :class:`dict`
            Information on each job, ``jobs[jobID]`` for each submitted job
    :Versions:
        * 2014-10-06 ``@ddalle``: v1.0
        * 2015-06-19 ``@ddalle``: v1.1; add ``qstat -J`` option
    """
    # Process username
    if u is None:
        u = getpass.getuser()
    # Form the command
    if J is not None:
        # Call for a specific job
        cmd = ['qstat', '-J', str(J)]
    else:
        # Call for a user
        cmd = ['qstat', '-u', u]
    # Initialize jobs
    jobs = {}
    # Call the command with safety
    try:
        # Call `qstat` with output.
        txt = Popen(cmd, stdout=PIPE).communicate()[0].decode("utf-8")
        # Split into lines
        lines = txt.split('\n')
        # Loop through lines.
        for line in lines:
            # Check for lines that don't start with PBS ID number
            if not re.match('[0-9]', line.strip()):
                continue
            # Split into parts
            v = line.split()
            # Get the job ID
            jobID = v[0]
            # Save the job info
            jobs[jobID] = dict(u=v[1], q=v[2], N=v[3], R=v[7])
        # Output
        return jobs
    except Exception:
        # Failed or no qstat command
        return jobs


# Function to get `qstat` information
def squeue(
        u: Optional[str] = None,
        J: Optional[Union[str, int]] = None) -> dict:
    r"""Call ``squeue`` and process information

    :Call:
        >>> jobs = squeue(u=None)
        >>> jobs = squeue(J=None)
    :Inputs:
        *u*: :class:`str`
            User name, defaults to ``os.environ[USER]``
        *J*: :class:`int`
            Specific job ID for which to check
    :Outputs:
        *jobs*: :class:`dict`
            Info on each job, ``jobs[jobID]`` for each submitted job
    :Versions:
        * 2023-06-16 ``@ddalle``: v1.0
        * 2024-06-18 ``@ddalle``: v1.1; add
    """
    # Process username
    if u is None:
        u = os.environ['USER']
    # Form the command
    if J is not None:
        # Call for a specific job
        cmd = ['squeue', '-dd', str(J)]
    else:
        # Call for a user
        cmd = ['squeue', '-u', u]
    # Initialize jobs
    jobs = {}
    # Call the command with safety
    try:
        # Call ``squeue`` with output
        txt = Popen(cmd, stdout=PIPE).communicate()[0].decode("utf-8")
        # Split into lines.
        lines = txt.split('\n')
        # Loop through lines.
        for line in lines:
            # Check for lines that don't start with PBS ID number.
            if not re.match('[0-9]', line.strip()):
                continue
            # Split into values.
            v = line.split()
            # Get the job ID
            jobID = v[0]
            # Save the job info
            jobs[jobID] = dict(u=v[3], q=v[1], N=v[6], R=v[4])
        # Output.
        return jobs
    except Exception:
        # Failed or no qstat command
        return jobs


# Get job ID
def get_job_id() -> Optional[str]:
    r"""Get job ID of currently running job (if any)

    :Call:
        >>> job_id = get_job_id()
    :Outputs:
        *job_id*: :class:`str` | ``None``
            Current PBS/Slurm job ID, if found
    :Versions:
        * 2024-06-12 ``@ddalle``: v1.0
    """
    # Look for PBS job ID; then Slurm
    pbs_id = os.environ.get("PBS_JOBID")
    slurm_id = os.environ.get("SLURM_JOB_ID")
    # Return either that is not None (or return None)
    job_id = slurm_id if pbs_id is None else pbs_id
    return job_id


# Open file for writing, calling mkdir() if needed
def openfile_w(fname: str, mode: str = 'w') -> IOBase:
    r"""Open a file for writing, creating folder if needed

    :Call:
        >>> fp = openfile_w(fname, mode='w')
    :Inputs:
        *fname*: :class:`str`
            Name of file to write
        *mode*: {``'w'``} | ``'wb'``
            Mode for resulting file
    :Outputs:
        *fp*: :class:`IOBase`
            File handle, open for writing
    :Versions:
        * 2024-06-12 ``@ddalle``: v1.0
    """
    # Check for folders to create
    parts = fname.split(os.sep)
    # Cumulative folder
    fdir = ''
    # Loop through parts (none if just file name)
    for part in parts[:-1]:
        # Accumulate path
        fdir = os.path.join(fdir, part)
        # Create folder if needed
        if not os.path.isdir(fdir):
            os.mkdir(fdir)
    # Open file
    return open(fname, mode)


def open_jobfile_r() -> Optional[IOBase]:
    r"""Open the best available PBS/Slurm job ID file for reading

    :Call:
        >>> fp = open_jobfile_r()
    :Outputs:
        *fp*: :class:`IOBase` | ``None``
            File handle, open for reading, if available
    :Versions:
        * 2024-06-13 ``@ddalle``: v1.0
    """
    # Loop through candidate job ID files
    for fname in JOB_ID_FILES:
        # Check if file exists
        if os.path.isfile(fname):
            return open(fname, 'r')


def _job(jobname: str) -> str:
    r"""Standardize PBS/Slurm job IDs

    For example this removes ``.nasa.nasa.gov`` from the output of NAS
    ``qsub`` commands (in order to match output of NAS ``qstat``).

    :Call:
        >>> jobID = _job(jobname)
    :Inputs:
        *jobname*: :class:`str`
            Raw job identifier
    :Outputs:
        *jobID*: :class:`str`
            Standardized job identifier w/ at most one ``'.'``
    :Versions:
        * 2024-06-20 ``@ddalle``: v1.0
    """
    return ".".join(str(jobname).split('.')[:2])

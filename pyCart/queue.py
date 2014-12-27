"""
Cart3D queue interface module: :mod:`pyCart.queue`
==================================================

Contains direct interface for functions like `qsub` and `qstat`.

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
        >>> pbs = pyCart.queue.qsub(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of PBS script to submit
    :Outputs:
        *pbs*: :class:`int` or ``None``
            PBS job ID number if submission was successful
    :Versions:
        * 2014-10-05 ``@ddalle``: First version
    """
    # Call the command with safety
    try:
        # Call `qsub` with output
        txt = sp.Popen(['qsub', fname], stdout=sp.PIPE).communicate()[0]
        # Get the integer job number.
        return int(txt.split('.')[0])
    except Exception:
        # Print a message, but don't fail.
        print("Submitting PBS script failed:\n  '%s/%s'"
            % (os.getcwd(), fname)) 
        # Failed; return None
        return None
        
# Function to delete jobs from the queue.
def qdel(jobID):
    """Delete a PBS job by number
    
    :Call:
        >>> pyCart.queue.qdel(jobID)
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
            sp.Popen(['qdel', str(jobI)], stdout=sp.PIPE).communicate()[0]
            # Status update.
            print("  Deleted PBS job %i" % jobI)
        except Exception:
            print("  Failed to delete PBS job %s" % jobI)
        

# Function to call `qsub` and save the job ID
def pqsub(fname, fout="jobID.dat"):
    """Submit a PBS script and save the job number in an *fout* file
    
    :Call:
        >>> pbs = pyCart.queue.pqsub(fname, fout="jobID.dat")
    :Inputs:
        *fname*: :class:`str`
            Name of PBS script to submit
        *fout*: :class:`str`
            Name of output file to contain PBS job number
    :Outputs:
        *pbs*: :class:`int` or ``None``
            PBS job ID number if submission was successful
    :Versions:
        * 2014-10-06 ``@ddalle``: First version
    """
    # Submit the job.
    pbs = qsub(fname)
    # Create the file if the submission was successful.
    if pbs:
        # Create the output file.
        f = open(fout, 'w')
        # Write the job id.
        f.write('%i\n' % pbs)
        # Close the file.
        f.close()
    # Return the number.
    return pbs
        
# Function to get `qstat` information
def qstat(u=None):
    """Call `qstat` and process information
    
    :Call:
        >>> jobs = pyCart.queue.qstat(u=None)
    :Inputs:
        *u*: :class:`str`
            User name, defaults to ``os.environ[USER]``
    :Outputs:
        *jobs*: :class:`dict`
            Information on each job, ``jobs[jobID]`` for each submitted job
    :Versions:
        * 2014-10-06 ``@ddalle``: First versoin
    """
    # Process username
    if u is None:
        u = os.environ['USER']
    # Call the command with safety.
    try:
        # Call `qstat` with output.
        txt = sp.Popen(['qstat', '-u', u], stdout=sp.PIPE).communicate()[0]
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
        
    

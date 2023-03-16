r"""
:mod:`cape.cfdx.options`: CAPE options interface module
========================================================

The :mod:`cape.cfdx.options` provides tools to read, access, modify, and
write settings for :mod:`cape`. The class is based off of the built-in
:class:`dict` class, so its default behaviors, such as
``opts['RunControl']`` or ``opts.get('RunControl')`` are also present.
In addition, many convenience methods, such as
``opts.set_PhaseIters(n)``, which sets the number of iterations to run,
are provided.

In addition, this module controls default values of each CAPE parameter
in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    * Values directly specified in the input file, e.g. ``cape.json``

    * Values specified in the default control file,
      ``$CAPE/settings/cape.default.json``

    * Hard-coded defaults from this module

:See Also:
    * :mod:`cape.optdict`
"""

# Standard library
import os

# Local imports
from . import util
from .pbsopts import PBSOpts
from .slurmopts import SlurmOpts
from .databookopts import DataBookOpts
from .Report import Report
from .meshopts import MeshOpts
from .configopts import ConfigOpts
from .runctlopts import RunControlOpts
from ...optdict import OptionsDict, INT_TYPES


# Other imports
odict = util.odict
rc = util.rc


# Class definition
class Options(OptionsDict):
    r"""Options structure, subclass of :class:`dict`

    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary to be transformed into :class:`cape.options.Options`
    :Outputs:
        *opts*: :class:`Options`
            Options interface
    :Versions:
        * 2014-07-28 ``@ddalle``: Version 1.0
    """
   # ================
   # Class attributes
   # ================
   # <
    # Accepted options/sections
    _optlist = {
        "BatchPBS",
        "BatchSlurm",
        "CaseFunction",
        "Config",
        "DataBook",
        "InitFunction",
        "Mesh",
        "Modules",
        "PBS",
        "PostPBS",
        "PostSlurm",
        "PythonExec",
        "PythonPath",
        "Report",
        "RunControl",
        "RunMatrix",
        "Slurm",
        "ZombieFiles",
        "nSubmit",
        "umask",
    }

    # Aliases
    _optmap = {
        "Trajectory": "RunMatrix",
    }

    # Known option types
    _opttypes = {
        "BatchShellCmds": str,
        "CaseFunction": str,
        "InitFunction": str,
        "Modules": str,
        "PythonExec": str,
        "PythonPath": str,
        "RunMatrix": dict,
        "ZombieFiles": str,
        "nSubmit": INT_TYPES,
        "umask": INT_TYPES + (str,),
    }

    # Option default list depth
    _optlistdepth = {
        "BatchShellCmds": 1,
        "CaseFunction": 1,
        "InitFunction": 1,
        "PythonPath": 1,
        "Modules": 1,
        "ShellCmds": 1,
        "ZombieFiles": 1,
    }

    # Defaults
    _rc = {
        "ZombieFiles": [
            "*.out"
        ],
        "nSubmit": 10,
    }

    # Descriptions for methods
    _rst_descriptions = {
        "CaseFunction": "function(s) to execute in case right before starting",
        "BatchShellCmds": "additional shell commands for batch jobs",
        "InitFunction": "function(s) to run immediately after parsing JSON",
        "Modules": "list of Python modules to import",
        "PythonExec": "specific Python executable to use for jobs",
        "PythonPath": "folder(s) to add to Python path for custom modules",
        "ZombieFiles": "file name flobs to check mod time for zombie status",
        "nSubmit": "maximum number of jobs to submit at one time",
    }

    # Section classes
    _sec_cls = {
        "BatchPBS": PBSOpts,
        "BatchSlurm": SlurmOpts,
        "Config": ConfigOpts,
        "DataBook": DataBookOpts,
        "Mesh": MeshOpts,
        "PBS": PBSOpts,
        "PostPBS": PBSOpts,
        "PostSlurm": SlurmOpts,
        "Report": Report,
        "RunControl": RunControlOpts,
        "Slurm": SlurmOpts,
    }

    # Prefixes for section opts
    _sec_prefix = {
        "BatchPBS": "PBS_",
        "BatchSlurm": "Slurm_",
        "PBS": "PBS_",
        "PostPBS": "PBS_",
        "PostSlurm": "Slurm_",
        "Slurm": "Slurm_",
    }

    # Parents
    _sec_parent = {
        "BatchPBS": "PBS",
        "BatchSlurm": "Slurm",
        "PostPBS": "PBS",
        "PostSlurm": "Slurm",
    }
   # >

   # =============
   # Configuration
   # =============
   # <
    # Initialization hook
    def init_post(self):
        r"""Initialization hook for :class:`Options`

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`Options`
                Options interface
        :Versions:
            * 2022-10-23 ``@ddalle``: Version 1.0
        """
        # Read the defaults
        defs = util.getCapeDefaults()
        # Apply the defaults.
        self = util.applyDefaults(self, defs)
        # Add extra folders to path.
        self.AddPythonPath()

    # Function to add to the path.
    def AddPythonPath(self):
        r"""Add requested locations to the Python path

        :Call:
            >>> opts.AddPythonPath()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Versions:
            * 2014-10-08 ``@ddalle``: Version 1.0
        """
        # Get the "PythonPath" option
        lpath = self.get("PythonPath", [])
        # Quit if empty.
        if (not lpath): return
        # Ensure list.
        if type(lpath).__name__ != "list":
            lpath = [lpath]
        # Loop through elements.
        for fdir in lpath:
            # Add absolute path, not relative.
            os.sys.path.append(os.path.abspath(fdir))

    # Make a directory
    def mkdir(self, fdir, sys=False):
        r"""Make a directory with the correct permissions

        :Call:
            >>> opts.mkdir(fdir, sys=False)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fdir*: :class:`str`
                Directory to create
            *sys*: ``True`` | {``False``}
                Whether or not to replace ``None`` with system setting
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
        """
        # Get umask
        umask = self.get_umask(sys=sys)
        # Test for NULL umask
        if umask is None:
            # Make directory with default permissions
            try:
                # Make the directory
                os.mkdir(fdir)
            except Exception as e:
                # Check for making directory that exists
                if e.errno == 17:
                    # No problem; go on
                    pass
                else:
                    # Other error; valid
                    raise e
        else:
            # Apply umask
            dmask = 0o777 - umask
            # Make the directory.
            try:
                # Attempt the command
                os.mkdir(fdir, dmask)
            except Exception as e:
                # Check for making directory that exists
                if e.errno == 17:
                    # No problem; go on
                    pass
                else:
                    # Other error; valid
                    raise e
   # >

   # =====
   # Tools
   # =====
   # <
    # Write a PBS header
    def WritePBSHeader(self, f, lbl, j=0, typ=None, wd=None):
        r"""Write common part of PBS script

        :Call:
            >>> opts.WritePBSHeader(f, i=None, j=0, typ=None, wd=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *f*: :class:`file`
                Open file handle
            *lbl*: :class:`str`
                Name of the PBS job
            *j*: :class:`int`
                Phase number
            *typ*: {``None``} | ``"batch"`` | ``"post"``
                Group of PBS options to use
            *wd*: {``None``} | :class:`str`
                Folder to enter when starting the job
        :Versions:
            * 2015-09-30 ``@ddalle``: Separated from WritePBS
            * 2016-09-25 ``@ddalle``: Supporting "BatchPBS" and "PostPBS"
            * 2016-12-20 ``@ddalle``: Created version in options interface
        """
        # Get the shell path (must be bash)
        sh = self.get_PBS_S(j, typ=typ)
        # Write to script both ways.
        f.write('#!%s\n' % sh)
        f.write('#PBS -S %s\n' % sh)
        # Write it to the script
        f.write('#PBS -N %s\n' % lbl)
        # Get the rerun status.
        PBS_r = self.get_PBS_r(j, typ=typ)
        # Write if specified.
        if PBS_r:
            f.write('#PBS -r %s\n' % PBS_r)
        # Get the option for combining STDIO/STDOUT
        PBS_j = self.get_PBS_j(j, typ=typ)
        # Write if specified.
        if PBS_j:
            f.write('#PBS -j %s\n' % PBS_j)
        # Get the number of nodes, etc.
        nnode = self.get_PBS_select(j, typ=typ)
        ncpus = self.get_PBS_ncpus(j, typ=typ)
        nmpis = self.get_PBS_mpiprocs(j, typ=typ)
        nomp  = self.get_PBS_ompthreads(j, typ=typ)
        smodl = self.get_PBS_model(j, typ=typ)
        saoe  = self.get_PBS_aoe(j, typ=typ)
        # Form the -l line.
        line = '#PBS -l select=%i:ncpus=%i' % (nnode, ncpus)
        # Add other settings
        if nmpis: line += (':mpiprocs=%i' % nmpis)
        if smodl: line += (':model=%s' % smodl)
        if nomp:  line += (':ompthreads=%s' % nomp)
        if saoe:  line += (':aoe=%s' % saoe)
        # Write the line.
        f.write(line + '\n')
        # Get the walltime.
        t = self.get_PBS_walltime(j, typ=typ)
        # Write it.
        f.write('#PBS -l walltime=%s\n' % t)
        # Get the priority
        PBS_p = self.get_PBS_p(j, typ=typ)
        # Write it.
        if PBS_p is not None:
            f.write('#PBS -p %s\n' % PBS_p)
        # Check for a group list.
        PBS_W = self.get_PBS_W(j, typ=typ)
        # Write if specified.
        if PBS_W: f.write('#PBS -W %s\n' % PBS_W)
        # Get the queue.
        PBS_q = self.get_PBS_q(j, typ=typ)
        # Write it.
        if PBS_q: f.write('#PBS -q %s\n\n' % PBS_q)
        # Process working directory
        if wd is None:
            # Default to current directory
            pbsdir = os.getcwd()
        else:
            # Use the input
            pbsdir = wd
        # Go to the working directory.
        f.write('# Go to the working directory.\n')
        f.write('cd %s\n\n' % pbsdir)

        # Get umask option
        umask = self.get_umask()
        # Write the umask
        if umask > 0:
            f.write('# Set umask.\n')
            f.write('umask %04o\n\n' % umask)

        # Write a header for the shell commands.
        f.write('# Additional shell commands\n')
        # Loop through the shell commands.
        for line in self.get_ShellCmds(typ=typ):
            # Write it.
            f.write('%s\n' % line)

    # Write a Slurm header
    def WriteSlurmHeader(self, f, lbl, j=0, typ=None, wd=None):
        r"""Write common part of Slurm script

        :Call:
            >>> opts.WriteSlurmHeader(f, i=None, j=0, typ=None, wd=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *f*: :class:`file`
                Open file handle
            *lbl*: :class:`str`
                Name of the Slurm job
            *j*: :class:`int`
                Phase number
            *typ*: {``None``} | ``"batch"`` | ``"post"``
                Group of PBS options to use
            *wd*: {``None``} | :class:`str`
                Folder to enter when starting the job
        :Versions:
            * 2018-10-10 ``@ddalle``: Forked from :func:`WritePBSHeader`
        """
        # Get the shell path (must be bash)
        sh = self.get_Slurm_shell(j, typ=typ)
        # Write to script both ways.
        f.write('#!%s\n' % sh)
        # Write it to the script
        f.write('#SBATCH --job-name %s\n' % lbl)
        # Get the number of nodes, etc.
        acct  = self.get_Slurm_A(j, typ=typ)
        nnode = self.get_Slurm_N(j, typ=typ)
        ncpus = self.get_Slurm_n(j, typ=typ)
        que   = self.get_Slurm_p(j, typ=typ)
        gid   = self.get_Slurm_gid(j, typ=typ)
        # Write commands
        if acct:  f.write("#SBATCH -A %s\n" % acct)
        if nnode: f.write("#SBATCH -N %s\n" % nnode)
        if ncpus: f.write("#SBATCH -n %s\n" % ncpus)
        if que:   f.write("#SBATCH -p %s\n" % que)
        if gid:   f.write("#SBATCH --gid %s\n" % gid)
        # Get the walltime.
        t = self.get_Slurm_time(j, typ=typ)
        # Write it.
        f.write('#SBATCH --time=%s\n' % t)
        # Process working directory
        if wd is None:
            # Default to current directory
            pbsdir = os.getcwd()
        else:
            # Use the input
            pbsdir = wd
        # Go to the working directory.
        f.write('# Go to the working directory.\n')
        f.write('cd %s\n\n' % pbsdir)

        # Get umask option
        umask = self.get_umask()
        # Write the umask
        if umask > 0:
            f.write('# Set umask.\n')
            f.write('umask %04o\n\n' % umask)

        # Write a header for the shell commands.
        f.write('# Additional shell commands\n')
        # Loop through the shell commands.
        for line in self.get_ShellCmds(typ=typ):
            # Write it.
            f.write('%s\n' % line)
   # >

   # ==============
   # Global Options
   # ==============
   # <
    # Function to get the shell commands
    def get_ShellCmds(self, typ=None):
        r"""Get shell commands, if any

        :Call:
            >>> cmds = opts.get_ShellCmds(typ=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *typ*: {``None``} | ``"batch"`` | ``"post"``
                Add additional commands for batch or post-processing jobs
        :Outputs:
            *cmds*: :class:`list`\ [:class:`str`]
                List of initialization commands
        :Versions:
            * 2015-11-08 ``@ddalle``: Moved to "RunControl"
        """
        # Get the commands.
        cmds = self.get('ShellCmds', [])
        # Turn to a list if not.
        if type(cmds).__name__ != 'list':
            cmds = cmds.split(';')
        # Check type
        if typ in ["batch"]:
            # Get commands for batch jobs
            cmds_a = self.get('BatchShellCmds', [])
        elif typ in ["post"]:
            # Get commands for post-processing
            cmds_a = self.get('PostShellCmds', [])
        else:
            # No additional commands
            cmds_a = []
        # Turn to a list if necessary
        if type(cmds_a).__name__ != 'list':
            cmds_a = cmds_a.split(';')
        # Output
        return cmds + cmds_a

    # Function to set the shell commands
    def set_ShellCmds(self, cmds):
        r"""Set shell commands

        :Call:
            >>> opts.set_ChellCmds(cmds=[])
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *cmds*: :class:`list`\ [:class:`str`]
                List of initialization commands
        :Versions:
            * 2015-11-08 ``@ddalle``: Version 1.0
        """
        # Set them.
        self['ShellCmds'] = cmds

    # Method to determine if groups have common meshes.
    def get_GroupMesh(self):
        r"""Determine whether or not groups have common meshes

        :Call:
            >>> qGM = opts.get_GroupMesh()
        :Inputs:
            *opts* :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *qGM*: :class:`bool`
                Whether cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: Version 1.0
            * 2022-10-23 ``@ddalle``: Version 1.1; hard-code default
        """
        # Safely get the trajectory.
        x = self.get('RunMatrix', {})
        return x.get('GroupMesh', False)

    # Method to specify that meshes do or do not use the same mesh
    def set_GroupMesh(self, qGM=False):
        r"""Specify that groups do or do not use common meshes

        :Call:
            >>> opts.get_GroupMesh(qGM)
        :Inputs:
            *opts* :class:`pyCart.options.Options`
                Options interface
            *qGM*: ``True`` | {``False``}
                Whether cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: Version 1.0
            * 2022-10-23 ``@ddalle``: Version 1.1; hard-code default
        """
        self['RunMatrix']['GroupMesh'] = qGM

    # Get the umask
    def get_umask(self, sys=True):
        r"""Get the current file permissions mask

        The default value is the read from the system

        :Call:
            >>> umask = opts.get_umask(sys=True)
        :Inputs:
            *opts* :class:`pyCart.options.Options`
                Options interface
            *sys*: {``True``} | ``False``
                Whether or not to use system setting as default
        :Outputs:
            *umask*: ``None`` | :class:`oct`
                File permissions mask (``None`` only if *sys* is ``False``)
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
        """
        # Read the option.
        umask = self.get('umask')
        # Check if we need to use the default.
        if umask is None:
            # Check for system defaults
            if sys and os.name != "nt":
                # Get the value.
                umask = os.popen('umask', 'r').read()
                # Convert to value.
                umask = eval('0o' + umask.strip())
            else:
                # No setting
                return None
        elif type(umask).__name__ in ['str', 'unicode']:
            # Convert to octal
            umask = eval('0o' + str(umask).strip().lstrip('0o'))
        # Output
        return umask

    # Get the directory permissions to use
    def get_dmask(self, sys=True):
        r"""Get the permissions to assign to new folders

        :Call:
            >>> dmask = opts.get_dmask(sys=True)
        :Inputs:
            *opts* :class:`pyCart.options.Options`
                Options interface
            *sys*: {``True``} | ``False``
                Whether or not to use system setting as default
        :Outputs:
            *dmask*: :class:`int` | ``None``
                New folder permissions mask
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
        """
        # Get the umask
        umask = self.get_umask()
        # Check for null umask
        if umask is not None:
            # Subtract UMASK from full open permissions
            return 0o0777 - umask

    # Apply the umask
    def apply_umask(self, sys=True):
        r"""Apply the permissions filter

        :Call:
            >>> opts.apply_umask(sys=True)
        :Inputs:
            *opts* :class:`pyCart.options.Options`
                Options interface
            *sys*: {``True``} | ``False``
                Whether or not to use system setting as default
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
            * 2017-09-05 ``@ddalle``: Version 1.1; add *sys* kwarg
        """
        # Get umask
        umask = self.get_umask()
        # Apply if possible
        if umask is not None:
            os.umask(umask)
   # >


# Add global properties
Options.add_properties(("BatchShellCmds", "PythonExec", "nSubmit"))
# Add methods from subsections
Options.promote_sections()

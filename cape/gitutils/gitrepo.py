# -*- coding: utf-8 -*-
r"""
:mod:`gitutils.gitrepo`: Interact with git repos using system interface
========================================================================

This module provides the :class:`GitRepo` class, which provides a basic
interface to a git repository (whether working or bare). Users may
perform basic git operations such as ``git-push`` and ``git-pull`` using
commands like :func:`GitRepo.push`, :func:`GitRepo.pull`.

"""

# Standard library
import datetime
import functools
import os
import re
import shutil
import sys
from base64 import b32encode
from subprocess import Popen, PIPE

# Local imports
from ._vendor import shellutils
from .giterror import (
    GitutilsFileNotFoundError,
    GitutilsSystemError,
    GitutilsValueError,
    assert_isinstance,
    trunc8_fname)


# Regular expression for deciding if a path is local
REGEX_HOST = re.compile(r"((?P<host>[A-Za-z][A-Za-z0-9-.]+):)?(?P<path>.+)$")

# Local time zone
_now_utc = datetime.datetime.now(datetime.timezone.utc)
TIME_ZONE = _now_utc.astimezone().tzinfo


# Decorator for moving directories
def run_gitdir(func):
    r"""Decorator to run a function within the parent folder

    :Call:
        >>> func = run_rootdir(func)
    :Wrapper Signature:
        >>> v = repo.func(*a, **kw)
    :Inputs:
        *func*: :class:`func`
            Name of function
        *cntl*: :class:`Cntl`
            Control instance from which to use *cntl.RootDir*
        *a*: :class:`tuple`
            Positional args to :func:`cntl.func`
        *kw*: :class:`dict`
            Keyword args to :func:`cntl.func`
    :Versions:
        * 2018-11-20 ``@ddalle``: v1.0
        * 2020-02-25 ``@ddalle``: v1.1: better exceptions
    """
    # Declare wrapper function to change directory
    @functools.wraps(func)
    def wrapper_func(self, *args, **kwargs):
        # Recall current directory
        fpwd = os.getcwd()
        # Go to specified directory
        os.chdir(self.gitdir)
        # Run the function with exception handling
        try:
            # Attempt to run the function
            v = func(self, *args, **kwargs)
        except (Exception, KeyboardInterrupt):
            # Go back to original folder
            os.chdir(fpwd)
            # Raise the error
            raise
        # Go back to original folder
        os.chdir(fpwd)
        # Return function values
        return v
    # Apply the wrapper
    return wrapper_func


# Class to interface one repo
class GitRepo(object):
    r"""Git repository interface class

    :Call:
        >>> repo = GitRepo(where=None)
    :Inputs:
        *where*: {``None``} | :class:`str`
            Path from which to look for git repo (default is CWD)
    :Outputs:
        *repo*: :class:`GitRepo`
            Interface to git repository
    :Attributes:
        * :attr:`bare`
        * :attr:`gitdir`
        * :attr:`host`
        * :attr:`shell`
    """
   # --- Class attributes ---
    # Class attributes
    __slots__ = (
        "bare",
        "gitdir",
        "host",
        "shell",
        "_tmpbase",
        "_tmpdir",
    )

   # --- __dunder__ ---
    def __init__(self, where=None):
        r"""Initialization method"""
        # Initialize slots
        #: ``None`` | :class:`gitutils._vendor.shellutils.SSH` --
        #: Optional persistent subprocess to run commands with; usually
        #: not used
        self.shell = None
        # Identify host and path
        host, path = identify_host(where)
        #: ``None`` | :class:`str` --
        #: Name of remote host, if any; usually not used
        self.host = host
        #: ``True`` | ``False`` --
        #: Whether this instance is in a bare repository
        self.bare = is_bare(where)
        #: :class:`str` -- Absolute path to root directory
        self.gitdir = self.get_gitdir(path)
        # No temporary working directory
        self._tmpbase = None
        self._tmpdir = None

   # --- Identification ---
    def get_gitdir(self, path: str):
        r"""Get absolute path to git repo root, even on bare repos

        :Call:
            >>> repo.get_gitdir(where=None, bare=None)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *path*: :class:`str`
                Working directory; can be local path or SSH pathlly)
        :Outputs:
            *gitdir*: :class:`str`
                Full path to top-level of working repo or git-dir of bare
        :Versions:
            * 2022-12-22 ``@ddalle``: v1.1; support older git vers
            * 2023-08-26 ``@ddalle``: v2.0; move to instance method
        """
        # Get host
        host = self.host
        # Get the "git-dir" for bare repos and "toplevel" for working repos
        if self.bare:
            # Get relative git dir
            gitdir, _ = shellutils.check_o(
                ["git", "rev-parse", "--git-dir"], cwd=path, host=host)
            # Absolute *gitdir* (--absolute-git-dir not avail on older git)
            gitdir = os.path.realpath(os.path.join(path, gitdir.strip()))
        else:
            gitdir, _ = shellutils.check_o(
                ["git", "rev-parse", "--show-toplevel"], cwd=path, host=host)
        # Output
        return gitdir.strip().replace("/", os.sep)

    def get_configdir(self):
        r"""Get the path to the ``*.git/`` folder, where ``config`` is

        :Call:
            >>> fdir = repo.get_configdir()
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
        :Outputs:
            *fdir*: :class:`str`
                Path to ``{project}.git/`` or ``{project}/.git/``
        """
        # Check if a bare repo
        if self.bare:
            # Bare repo is the {project}.git parent
            return self.gitdir
        else:
            # Working repo uses {project}/.git
            return os.path.join(self.gitdir, ".git")

   # --- Subprocess/shellutils ---
    def connect(self):
        r"""Connect a persistent subprocess at base of repo

        :Call:
            >>> shell = repo.connect()
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
        :Outputs:
            *shell*: :class:`shellutils.SSH`
                Subprocess running on either
        :Versions:
            * 2023-10-25 ``@ddalle``: v1.0
        """
        # Check if already connected
        if isinstance(self.shell, shellutils.SSH):
            # Already connected
            return self.shell
        # Otherwise connect
        if self.host in (None, "localhost"):
            # Connect local shell (special case)
            self.shell = shellutils.Shell()
        else:
            # Connect remote SSH shell
            self.shell = shellutils.SSH(self.host)
        # Output
        return self.shell

   # --- Config/remote/branch ---
    def get_remotes(self) -> dict:
        r"""Get names and URLs of remotes

        :Call:
            >>> remotes = repo.get_remotes()
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
        :Outputs:
            *remotes*: :class:`dict`\ [:class:`str`]
                Dictionary of remotes and URLs for each
        """
        # Run command to get list of remotes
        stdout = self.check_o(["git", "remote", "-v"], cwd=True).strip()
        lines = stdout.split('\n')
        # Initialize output
        remotes = {}
        # Loop through lines
        for line in lines:
            # Split by tab character
            name, part = line.split('\t')
            # Get address
            url = part.split(' (')[0]
            # Save remote
            remotes[name] = url
        # Output
        return remotes

    def get_branch_list(self):
        r"""Get list of branches

        :Call:
            >>> branches = repo.get_branch_list()
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
        :Outputs:
            *branches*: :class:`list`\ [:class:`str`]
                List of branches, ``branches[0]`` is HEAD branch
        """
        # Run command to list branches
        lines = self.check_o(["git", "branch"], cwd=True).strip().split('\n')
        # Initialize list
        branches = []
        # Loop through lines of STDOUT
        for line in lines:
            # Check if it starts with '*'
            if line.startswith("*"):
                # This is the HEAD branch
                branches.insert(0, line[1:].strip())
            else:
                # Another branch
                branches.append(line.strip())
        # Output
        return branches

    def validate_branch(self, branch=None):
        r"""Make sure branch exists before using it

        :Call:
            >>> repo.validate_branch(branch=None)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *branch*: {``None``} | :class:`str`
                Name of branch to checkout if non-empty
        """
        # If *branch* is ``None``, just use HEAD
        if branch is None:
            return
        # Check type
        assert_isinstance(branch, str, "branch name")
        # Get branch list
        branch_list = self.get_branch_list()
        # Make sure *branch* exists
        if branch not in branch_list:
            raise GitutilsValueError(
                f"No branch '{branch}' " +
                f"in repo {os.path.basename(self.gitdir)}")

   # --- Status Operations ---
    def check_ignore(self, fname: str) -> bool:
        r"""Check if *fname* is (or would be) ignored by git

        :Call:
            >>> q = repo.check_ignore(fname)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *fname*: :class:`str`
                Name of file
        :Outputs:
            *q*: ``True`` | ``False``
                Whether file is ignored (even if file doesn't exist)
        :Versions:
            * 2022-12-20 ``@ddalle``: v1.0
        """
        # Structure a command for git
        _, _, ierr = shellutils.call_oe(["git", "check-ignore", fname])
        # If ignored, return code is 0
        return ierr == 0

    def check_track(self, fname: str) -> bool:
        r"""Check if a file is tracked by git

        :Call:
            >>> q = repo.check_track(fname)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *fname*: :class:`str`
                Name of file
        :Outputs:
            *q*: ``True`` | ``False``
                Whether file is tracked
        :Versions:
            * 2022-12-20 ``@ddalle``: v1.0
        """
        # Structure a command for git
        stdout = self.check_o(["git", "ls-files", fname])
        # If tracked, it will be listed in stdout
        return stdout.strip() != ""

    def status(self, *fnames) -> dict:
        r"""Check status of file(s) or whole repo

        :Call:
            >>> statusdict = repo.status(*fnames)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *fnames*: :class:`tuple`\ [:class:`str`]
                File name(s); if empty, show status of entire repo
        :Outputs:
            *statusdict*: :class:`dict`\ [:class:`str`]
                Two-letter status code for each modified file
        :Versions:
            * 2023-10-30 ``@ddalle``: v1.0
        """
        # Only on working repo
        self.assert_working()
        # Structure a command
        cmd = ["git", "status", "-s"]
        cmd.extend(fnames)
        # Run status command
        stdout = self.check_o(cmd).rstrip()
        # Get status for each line
        statusdict = {}
        # Loop through lines
        for line in stdout.split("\n"):
            # Split status and file name
            status = line[:2]
            fname = line.rstrip()[3:]
            # Save it
            statusdict[fname] = status
        # Output
        return statusdict

   # --- Log operations ---
    def get_ref(self, ref="HEAD"):
        r"""Get the SHA-1 hash of commit pointed to by *ref*

        :Call:
            >>> commit = repo.get_ref(ref="HEAD")
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *ref*: {``"HEAD"``} | :class:`str`
                Git reference, can be branch name, tag, or commit hash
        :Outputs:
            *commit*: :class:`str`
                Full (SHA-1) hash of commit pointed to by *ref*
        """
        # Check input types
        assert_isinstance(ref, str, "git ref name")
        # Get commit that *ref* currently points at
        return self.check_o(["git", "rev-parse", ref]).strip()

   # --- Repo properties ---
    def assert_working(self, cmd=None):
        r"""Assert that current repo is working (non-bare)

        :Call:
            >>> repo.assert_working(cmd=None)
        :Inputs:
            *repo*: :class:`GitRepo`
                Inteface to git repository
            *cmd*: {``None``} | :class:`str`
                Command name for error message
        :Versions:
            * 2022-12-20 ``@ddalle``: v1.0
        """
        # Check if a bare repo
        if self.bare:
            # Form message
            msg = "Cannot run command in bare repo"
            # Check for a command
            if cmd:
                msg += "\n> %s" % " ".join(cmd)
            # Exception
            raise GitutilsSystemError(msg)

    def assert_bare(self, cmd=None):
        r"""Assert that current repo is working (non-bare)

        :Call:
            >>> repo.assert_working(cmd=None)
        :Inputs:
            *repo*: :class:`GitRepo`
                Inteface to git repository
            *cmd*: {``None``} | :class:`str`
                Command name for error message
        :Versions:
            * 2023-09-16 ``@ddalle``: v1.0
        """
        # Check if a bare repo
        if not self.bare:
            # Form message
            msg = "Cannot run command in working repo"
            # Check for a command
            if cmd:
                msg += "\n> %s" % " ".join(cmd)
            # Exception
            raise GitutilsSystemError(msg)

   # --- Commit ---
    def commit(self, m=None, **kw):
        r"""Commit current changes

        :Call:
            >>> repo.commit(m=None, **kw)
        :Inputs:
            *repo*: :class:`GitRepo`
                Inteface to git repository
            *m*, *message*: {``None``} | :class:`str`
                Commit message
            *a*: ``True`` | ``False``
                Option to commit all modifications to tracked files
        :Versions:
            * 2023-10-24 ``@ddalle``: v1.0
        """
        # Check for message
        msg = kw.get("message", m)
        # Basic command
        cmdi = ["git", "commit"]
        # Check for -a command
        if kw.get("a", False):
            cmdi.append("-a")
        # Process commit message
        if msg is not None:
            # Check type
            assert_isinstance(msg, str, "commit message")
            # Add message
            cmdi.extend(["-m", msg])
        # Run command
        self.check_call(cmdi)

   # --- Push/pull ---
    def pull(self, remote: str, ref="HEAD"):
        r"""Pull from a remote repo

        This function does not reproduce the full functionality of the
        original ``git-pull`` command

        :Call:
            >>> repo.pull(remote, ref="HEAD")
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *ref*: {``"HEAD"``} | :class:`str`
                Git reference, usually branch name
        """
        # Perform pull
        self.check_call(["git", "pull", remote, ref])

    def push(self, remote: str, ref="HEAD"):
        r"""Push to a remote repo

        This function does not reproduce the full functionality of the
        original ``git-push`` command

        :Call:
            >>> repo.push(remote, ref="HEAD")
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *ref*: {``"HEAD"``} | :class:`str`
                Git reference, usually branch name
        """
        # Perform pull
        self.check_call(["git", "push", remote, ref])

   # --- Add ---
    def add(self, *fnames):
        r"""Add a file, either new or modified

        Has no effect if *name* is unchanged since last commit

        :Call:
            >>> repo.add(*fnames)
        :Inputs:
            *repo*: :class:`GitRepo`
                Inteface to git repository
            *fnames*: :class:`tuple`\ [:class:`str`]
                One or more file name or file name patterns
        :Versions:
            * 2022-12-02 ``@ddalle``: v1.0
        """
        # Only perform on working repo
        self.assert_working()
        # Perform 'git add' command for each file or pattern
        for fname in fnames:
            self._add(fname)

    def _add(self, fname: str):
        self.check_call(["git", "add", fname])

   # --- Move ---
    def mv(self, fold: str, fnew: str):
        r"""Move a file or folder and inform git of change

        :Call:
            >>> repo.mv(fold, fnew)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *fold*: :class:`str`
                Name of existing file
            *fnew*: :class:`str`
                Name of file after move
        :Versions:
            * 2022-12-28 ``@ddalle``: v1.0
        """
        # Only perform on working repo
        self.assert_working()
        # Move the file and check exit status
        self.check_call(["git", "mv", fold, fnew])

   # --- Remove ---
    def rm(self, fname: str, *fnames, r=False):
        r"""Remove files or folders and stage deletions for git

        :Call:
            >>> repo.rm(fname, *fnames, r=False)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *fname*: :class:`str`
                Name of first file/folder to remove
            *fnames*: :class:`tuple`\ [:class:`str`]
                Additional file/folder names or patterns to remove
            *r*: ``True`` | {``False``}
                Recursive option needed to delete folders
        :Versions:
            * 2022-12-29 ``@ddalle``: v1.0
        """
        # Only perform on working repo
        self.assert_working()
        # Form command
        if r:
            # Recursive (remove folders)
            cmd_list = ["git", "rm", "-r", fname]
        else:
            # Only individual files
            cmd_list = ["git", "rm", fname]
        # Add additional files/folders
        cmd_list.extend(fnames)
        # Attempt to remove the files and inform git
        self.check_call(cmd_list)

   # --- Checkout ---
    def checkout_branch(self, branch=None):
        r"""Check out a branch on a working repo

        :Call:
            >>> repo.check_branch(branch=None)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *branch*: {``None``} | :class:`str`
                Name of branch to checkout if non-empty
        """
        # Check for trivial action
        if branch is None:
            return
        # Only run on working repo
        self.assert_working()
        # Check out the branch
        self.check_call(["git", "checkout", branch])

   # --- List files ---
    def ls_tree(self, *fnames, r=True, ref="HEAD"):
        r"""List files tracked by git, even in a bare repo

        Calling this function with no arguments will show all files
        tracked in *repo*.

        :Call:
            >>> filelist = repo.ls_tree(*fnames, r=True, ref="HEAD")
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *fnames*: :class:`tuple`\ [:class:`str`]
                Name of 0 or more files to search for
            *r*: {``True``} | ``False``
                Whether or not to search recursively
            *ref*: {``"HEAD"``} | :class:`str`
                Git reference, can be branch name, tag, or commit hash
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of file names meeting above criteria
        :Versions:
            * 2022-12-21 ``@ddalle``: v1.0
        """
        # Handle ref=None
        ref = _safe_ref(ref)
        # Basic command
        cmdlist = ["git", "ls-tree", "--name-only"]
        # Append -r (recursive) option if appropriate
        if r:
            cmdlist.append("-r")
        # Add ref name (branch, commit, etc.)
        cmdlist.append(ref)
        # Add any specific files or folders
        cmdlist.extend(fnames)
        # List all files
        stdout = self.check_o(cmdlist).rstrip("\n")
        # Check if empty
        if len(stdout) == 0:
            # No files
            return []
        else:
            # Split into lines
            return stdout.split("\n")

   # --- Show ---
    def show(self, fname, ref="HEAD"):
        r"""Show contents of a file, even on a bare repo

        :Call:
            >>> fbytes = repo.show(fname, ref="HEAD")
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *fname*: :class:`str`
                Name of file to read
            *ref*: {``"HEAD"``} | :class:`str`
                Git reference, can be branch name, tag, or commit hash
        :Outputs:
            *fbytes*: :class:`bytes`
                Contents of *fname* in repository, in raw bytes
        :Versions:
            * 2022-12-22 ``@ddalle``: v1.0
        """
        # Handle ref=None
        ref = _safe_ref(ref)
        # Create command
        cmdlist = ["git", "show", "%s:%s" % (ref, fname)]
        # Run command using subprocess
        proc = Popen(cmdlist, stdout=PIPE, stderr=PIPE)
        # Wait for command
        stdout, stderr = proc.communicate()
        # Check status
        if proc.returncode or stderr:
            # Fixed portion of message
            msg = (
                ("Cannot show file '%s' from ref '%s'\n" % (fname, ref)) +
                ("Return code: %i" % proc.returncode))
            # Check for STDERR
            if stderr:
                msg += ("\nSTDERR: %s" % (stderr.decode("ascii").strip()))
            # Exception
            raise GitutilsSystemError(msg)
        # Otherwise, return the result, but w/o decoding
        return stdout

   # --- Patch ---
    def patch_file(
            self,
            fname: str,
            contents,
            m=None,
            branch=None,
            clean=True):
        r"""Commit contents to a file, whether bare or working repo

        :Call:
            >>> repo.patch_file(
                fname, contents, m=None, branch=None, clean=True)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *fname*: :class:`str`
                Name of file (relative to root of git repo) to edit
            *contents*: :class:`str` | :class:`bytes`
                Contents to save for *fname*
            *m*: {``None``} | :class:`str`
                Commit message for patch; default is "Patch '{fname}'"
            *branch*: {``None``} | :class:`str`
                Name of branch to work in; default is HEAD
            *clean*: {``True``} | ``False``
                Option to delete tmp working repo if *repo* is bare
        """
        # Default commit message
        if m is None:
            m = f"Patch '{fname}'"
        # Check input types
        assert_isinstance(fname, str, "file name to patch")
        assert_isinstance(contents, (str, bytes), "contents after patch")
        assert_isinstance(m, str)
        # Check validity of *branch*
        self.validate_branch(branch)
        # Get working repo
        repo = self.get_working_repo(branch)
        # Get write mode for *contents*
        mode = 'wb' if isinstance(contents, bytes) else 'w'
        # Allow *fname* to be POSIX
        fname = fname.replace('/', os.sep)
        # Absolute path to file
        fabs = os.path.join(repo.gitdir, fname)
        fdir = os.path.dirname(fabs)
        # Make sure *fdir* exists
        if not os.path.isdir(fdir):
            f1 = trunc8_fname(fname, 24)
            raise GitutilsFileNotFoundError(
                f"Could not find folder for file '{f1}'")
        # Write the file
        with open(fabs, mode) as fp:
            fp.write(contents)
        # Add the file
        repo.check_call(["git", "add", fname], cwd=True)
        # Check if the file changed
        if repo.check_o(["git", "status", "-s", fname], cwd=True):
            # Commit
            repo.check_call(["git", "commit", "-m", m], cwd=True)
        # Done if *self* is already a working repo
        if not self.bare:
            return
        # Push back to origin if appropriate
        repo.check_call(["git", "push", "origin", "HEAD"], cwd=True)
        # Delete the temporary working repo
        if clean:
            self.rm_working_repo()

    def create_patch(self, fname: str, contents, m=None, ref="HEAD"):
        r"""Create a patch file that changes contents of a file

        :Call:
            >>> repo.patch_file(
                fname, contents, m=None, branch=None, clean=True)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *fname*: :class:`str`
                Name of file (relative to root of git repo) to edit
            *contents*: :class:`str` | :class:`bytes`
                Contents to save for *fname*
            *m*: {``None``} | :class:`str`
                Commit message for patch; default is "Patch '{fname}'"
            *ref*: {``"HEAD"``} | :class:`str`
                Git reference from which to base patch
        """
        # Check input types
        assert_isinstance(fname, str, "file name to patch")
        assert_isinstance(contents, (str, bytes), "contents after patch")
        # Default commit message
        if m is None:
            m = f"Patch '{fname}'"
        # Check if file is present
        newfile = len(self.ls_tree(fname, ref=ref)) == 0
        # Get contents of "existing" file
        if newfile:
            # File not present in repo (in *ref*)
            a = b""
        else:
            # Read file from existing *ref*
            a = self.show(fname, ref=ref)
        # Get .git/ folder to to some temporary operaions
        gitdir = self.get_configdir()
        # Folders a/ and b/ to store temporary contents w/i
        adir = os.path.join(gitdir, "a")
        bdir = os.path.join(gitdir, "b")
        # Create those folders if necessary
        for fdir in (adir, bdir):
            if not os.path.isdir(fdir):
                os.mkdir(fdir)
        # Names of files before and after
        afile = os.path.join(adir, fname)
        bfile = os.path.join(bdir, fname)
        # Write the "before" contents
        open(afile, 'wb').write(a)
        # Get write mode for output based on type of "contents"
        bmode = 'w' if isinstance(contents, str) else 'wb'
        # Write the "after" contents
        open(bfile, bmode).write(contents)
        # Command to calculate diff
        cmd1 = ["diff", "-u", f"a/{fname}", f"b/{fname}"]
        # Calculate the diff
        stdout, _, ierr = self.call_oe(cmd1, cwd=gitdir)
        # If *ierr* is ``1``, there's no diff
        if ierr == 0:
            return
        # Split it into lines
        lines = stdout.split("\n")
        # Simplify output file spec by removing date
        lines[1] = f"+++ b/{fname}"
        # Simplify input file name; customize if new
        if newfile:
            # Using /dev/null for empty file instead of a/{fname}
            lines[0] = "--- /dev/null"
            # Use the default file mode
            lines.insert(0, "new file mode 100644")
        else:
            # Simplify input file spec by removing date
            lines[0] = f"--- a/{fname}"
        # Prepend lines in reverse order for extra git commit info
        # Start with diff --git summary
        lines.insert(0, f"diff --git a/{fname} b/{fname}")
        # Prepend status summary ... can leave that empty
        lines.insert(0, "")
        lines.insert(0, "---")
        lines.insert(0, "")
        # Next is subject line (don't worry if *m* has \n's in it)
        lines.insert(0, f"Subject: [PATCH] {m}")
        # Current time
        now = datetime.datetime.now(TIME_ZONE)
        # Format it
        nowfmt = now.strftime("%a, %d %b %Y %H:%M:%S %z")
        # Now prepend the time
        lines.insert(0, f"Date: {nowfmt}")
        # Prepend user name and email
        name = self.get_user_name()
        addr = self.get_user_email()
        lines.insert(0, f"From: {name} <{addr}>")
        # Prepend the commit from which we are patching
        sha1 = self.get_ref(ref)
        lines.insert(0, f"From {sha1}")
        # Write the patch file
        with open(os.path.join(gitdir, "ab.patch"), 'w') as fp:
            fp.write('\n'.join(lines))

   # --- Shell utilities ---
    def check_o(self, cmd, codes=None, cwd=None) -> str:
        r"""Run a command, capturing STDOUT and checking return code

        :Call:
            >>> stdout = repo.check_o(cmd, codes=None, cwd=None)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *cmd*: :class:`list`\ [:class:`str`]
                Command to run in list form
            *codes*: {``None``} | :class:`list`\ [:class:`int`]
                Collection of allowed return codes (default only ``0``)
            *cwd*: {``None``} | ``True`` | :class:`str`
                Location in which to run subprocess; ``None`` is current
                working directory, and ``True`` is *repo.gitdir*
        :outputs:
            *stdout*: :class:`str`
                Captured STDOUT from command, if any
        :Versions:
            * 2023-01-08 ``@ddalle``: v1.0
        """
        # Parse special cases of *cwd*
        cwd = self._parse_cwd(cwd)
        # Run the command as requested, capturing STDOUT and STDERR
        stdout, stderr, ierr = shellutils.call_oe(cmd, cwd=cwd)
        # Check for errors, perhaps *fname* starts with --
        if codes and ierr in codes:
            # This exit code is allowed
            return stdout
        # Check for errors, perhaps mal-formed command
        if ierr:
            sys.tracebacklimit = 1
            raise GitutilsSystemError(
                ("Unexpected exit code %i from command\n" % ierr) +
                ("> %s\n\n" % " ".join(cmd)) +
                ("Original error message:\n%s" % stderr))
        # Output
        return stdout

    def check_call(self, cmd, codes=None, cwd=None, **kw) -> int:
        r"""Run a command and check return code

        :Call:
            >>> ierr = repo.check_call(cmd, codes=None, cwd=None)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *cmd*: :class:`list`\ [:class:`str`]
                Command to run in list form
            *codes*: {``None``} | :class:`list`\ [:class:`int`]
                Collection of allowed return codes (default only ``0``)
            *cwd*: {``None``} | ``True`` | :class:`str`
                Location in which to run subprocess; ``None`` is current
                working directory, and ``True`` is *repo.gitdir*
        :Outputs:
            *ierr*: :class:`int`
                Return code from subprocess
        :Versions:
            * 2023-01-08 ``@ddalle``: v1.0
        """
        # Parse special cases of *cwd*
        cwd = self._parse_cwd(cwd)
        # Run the command as requested, not capturing STDOUT and STDERR
        ierr = shellutils.call(cmd, cwd=cwd)
        # Check for errors, perhaps *fname* starts with --
        if codes and ierr in codes:
            # This exit code is allowed
            return ierr
        # Check for errors, perhaps mal-formed command
        if ierr:
            sys.tracebacklimit = 1
            raise GitutilsSystemError(
                ("Unexpected exit code %i from command\n" % ierr) +
                ("> %s\n\n" % " ".join(cmd)))
        # Output
        return ierr

    def call(self, cmd, cwd=None, **kw) -> int:
        r"""Run a command with generic options

        :Call:
            >>> ierr = repo.call(cmd, cwd=None, **kw)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *cmd*: :class:`list`\ [:class:`str`]
                Command to run in list form
            *cwd*: {``None``} | ``True`` | :class:`str`
                Location in which to run subprocess; ``None`` is current
                working directory, and ``True`` is *repo.gitdir*
            *stdout*: {``None``} | ``subprocess.PIPE`` | :class:`file`
                Option to supress or pipe STDOUT
            *stderr*: {``None``} | ``subprocess.PIPE`` | :class:`file`
                Option to supress or pipe STDERR
        :Outputs:
            *ierr*: :class:`int`
                Return code from subprocess
        :Versions:
            * 2023-10-25 ``@ddalle``: v1.0
            * 2024-04-23 ``@ddalle``: v2.0; use ``_call()``
        """
        # Run command and return only return code
        return self._call(cmd, cwd, **kw)[2]

    def _call(self, cmd, cwd=None, **kw) -> int:
        r"""Run a command with generic options

        :Call:
            >>> ierr = repo.call(cmd, cwd=None, **kw)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *cmd*: :class:`list`\ [:class:`str`]
                Command to run in list form
            *cwd*: {``None``} | ``True`` | :class:`str`
                Location in which to run subprocess; ``None`` is current
                working directory, and ``True`` is *repo.gitdir*
            *stdout*: {``None``} | ``subprocess.PIPE`` | :class:`file`
                Option to supress or pipe STDOUT
            *stderr*: {``None``} | ``subprocess.PIPE`` | :class:`file`
                Option to supress or pipe STDERR
        :Outputs:
            *ierr*: :class:`int`
                Return code from subprocess
        :Versions:
            * 2024-04-23 ``@ddalle``: v1.0
        """
        # Parse special cases of *cwd*
        cwd = self._parse_cwd(cwd)
        # Run the command as requested
        return shellutils._call(cmd, cwd=cwd, **kw)

    def call_oe(self, cmd, codes=None, cwd=None):
        r"""Run a command, capturing STDOUT and checking return code

        :Call:
            >>> stdout, stderr, ierr = repo.call_oe(cmd, cwd=None)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *cmd*: :class:`list`\ [:class:`str`]
                Command to run in list form
            *cwd*: {``None``} | ``True`` | :class:`str`
                Location in which to run subprocess; ``None`` is current
                working directory, and ``True`` is *repo.gitdir*
        :outputs:
            *stdout*: :class:`str`
                Captured STDOUT from command, if any
            *stderr*: :class:`str`
                Captured STDERR from command, if any
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2023-01-08 ``@ddalle``: v1.0
        """
        # Parse special cases of *cwd*
        cwd = self._parse_cwd(cwd)
        # Run the command as requested, capturing STDOUT and STDERR
        return shellutils.call_oe(cmd, cwd=cwd)

    def _parse_cwd(self, cwd):
        # Check for special cases
        if cwd is None:
            # Use ``os.getcwd()``
            return
        elif cwd is True:
            # Use *repo.gitdir*
            return self.gitdir
        # Make sure it's a string
        assert_isinstance(cwd, str, "current working directory")
        # Return it
        return cwd

   # --- Ignore ---
    def _ignore(self, fname):
        # Check if file already ignored by git
        if self.check_ignore(fname):
            return
        # Get path to .gitignore in same folder as *fname*
        frel, fbase = os.path.split(fname)
        fgitignore = os.path.join(frel, ".gitignore")
        # Ignore main file
        with open(fgitignore, "a") as fp:
            fp.write(fbase + "\n")
        # Add gitignore
        self._add(fgitignore)

   # --- Config ---
    def get_user_name(self) -> str:
        r"""Get current user name according to git config

        :Call:
            >>> uid = repo.get_user_name()
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
        :Outputs:
            *uid*: :class:`str`
                User name saved in ``.git/config``
        """
        return self.get_gitconfig_opt("user", "name")

    def get_user_email(self) -> str:
        r"""Get author's email address according to git config

        :Call:
            >>> addr = repo.get_user_email()
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
        :Outputs:
            *addr*: :class:`str`
                Email address
        """
        return self.get_gitconfig_opt("user", "email")

    def get_gitconfig_opt(self, sec: str, opt: str) -> str:
        r"""Get value of any git config setting

        :Call:
            >>> v = repo.get_gitconfig_opt(sec, opt)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *sec*: :class:`str`
                Name of config section to read
            *opt*: :class:`str`
                Name of option to read in config section
        :Outputs:
            *v*: :class:`object`
                Value from ``.git/config`` for *sec* > *opt*
        """
        return self.check_o(["git", "config", f"{sec}.{opt}"]).strip()

    def _from_ini(self, val):
        r"""Convert an INI-style configuration value to Python

        Converts ``"true"`` -> ``True``
        """
        # Check for special cases
        if val == "true":
            return True
        elif val == "false":
            return False
        # Otherwise return as-is
        return val

    def _to_ini(self, val) -> str:
        r"""Convert a Python value to INI-style configuration

        Uses :func:`str` except for converting to lower-case ``true``
        and ``false``.
        """
        # Check for special cases
        if val is True:
            return "true"
        elif val is False:
            return "false"
        else:
            return str(val)

   # --- Temporary Dir ---
    def get_working_repo(self, branch=None):
        r"""Check out a branch on a working repo

        :Call:
            >>> tmp_repo = repo.check_branch(branch=None)
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *branch*: {``None``} | :class:`str`
                Name of branch to checkout if non-empty
        :Outputs:
            *tmp_repo*: :class:`GitRepo`
                Interface to working repo, *repo* if it's a working repo
                or new temporary repo if *repo* is bare
        """
        # Validate branch name
        self.validate_branch(branch)
        # Just use self if a working repo
        if not self.bare:
            # Check out the branch (can fail)
            self.checkout_branch(branch)
            # Use this repo
            return self
        # Get temporary folder name
        tmpdir = self.get_tmpdir()
        # Go to parent of *gitdir*
        dirname = os.path.dirname(self.gitdir)
        # Create clone command
        cmd1 = ["git", "clone"]
        # Clone a specific branch?
        if branch:
            cmd1.extend(["-b", branch])
        # Add source and target
        cmd1.extend([self.gitdir, tmpdir])
        # Clone
        self.check_call(cmd1, cwd=dirname)
        # Return a repo
        return self.__class__(where=self._tmpdir)

    def rm_working_repo(self) -> int:
        r"""Remove the working repo, if it exists

        :Call:
            >>> repo.rm_working_repo()
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
        """
        # Check if there is a working repo
        if self._tmpdir is None:
            # Nothing to delete
            return
        # Check if the working repo still exists
        if not os.path.isdir(self._tmpdir):
            return
        # Delete the folder
        shutil.rmtree(self._tmpdir)
        # Delete attributes
        self._tmpbase = None
        self._tmpdir = None

    def get_tmpdir(self) -> str:
        r"""Create random name for a working repo if a bare repo

        :Call:
            >>> tmpdir = repo.get_tmpdir()
        :Inputs:
            *repo*: :class:`GitRepo`
                Interface to git repository
            *branch*: {``None``} | :class:`str`
                Name of branch to checkout if non-empty
        :Outputs:
            *tmpdir*: :class:`str`
                Name of folder that's a working repo
        :Attributes:
            *repo._tmpbase*: :class:`str`
                Base name of suggested working repo top-level folder
            *repo._tmpdir*: :class:`str`
                Absolute path to suggested working repo top-level folder
        """
        # No need if a working repo
        if not self.bare:
            return os.path.basename(self.gitdir)
        # Create a random string
        tmpbase = b32encode(os.urandom(15)).decode().lower()
        # Work in parent of *gitdir*
        self._tmpbase = tmpbase
        self._tmpdir = os.path.join(os.path.dirname(self.gitdir), tmpbase)
        # Output
        return self._tmpbase


# Get the top-level folder of git repository from a reference point
def get_gitdir(where=None, bare=None):
    r"""Get absolute path to git repo root, even on bare repos

    :Call:
        >>> gitdir = get_gitdir(where=None, bare=None)
    :Inputs:
        *where*: {``None``} | :class:`str`
            Working directory; can be local path or SSH path
        *bare*: {``None``} | ``True`` | ``False``
            Whether repo is bare (can be detected automatically)
    :Outputs:
        *gitdir*: :class:`str`
            Full path to top-level of working repo or git-dir of bare
    :Versions:
        * 2022-12-22 ``@ddalle``: v1.1; support older git vers
    """
    # Check for local/remote
    host, cwd = shellutils.identify_host(where)
    # Check if bare if needed
    if bare is None:
        bare = is_bare(where)
    # Get the "git-dir" for bare repos and "toplevel" for working repos
    if bare:
        # Get relative git dir
        gitdir, _ = shellutils.check_o(
            ["git", "rev-parse", "--git-dir"], cwd=cwd, host=host)
        # Absolute *gitdir* (--absolute-git-dir not avail on older git)
        gitdir = os.path.realpath(os.path.join(cwd, gitdir.strip()))
    else:
        gitdir, _ = shellutils.check_o(
            ["git", "rev-parse", "--show-toplevel"], cwd=cwd, host=host)
    # Output, but git always uses forward-slashes
    return gitdir.strip().replace("/", os.sep)


# Identify local vs remote
def identify_host(where=None):
    r"""Identify possible remote host/local path breakdown

    :Call:
        >>> host, path = identify_host(where)
    :Inputs:
        *where*: {``None``} | :class:`str`
            Local path like ``"/home/user"`` or remote path like
            ``"pfe://home"``
    :Outputs:
        *host*: ``None`` | :class:`str`
            Remote host nae preceding ``:`` if any
        *path*: :class:`str`
            Path on *host*; matches *where* if no host
    :Versions:
        * 2022-03-06 ``@ddalle``: v1.0
    """
    # Default to current location
    if where is None:
        where = os.getcwd()
    # Check against pattern (guaranteed to match)
    match = REGEX_HOST.match(where)
    # Return groups
    return match.group("host"), match.group("path")


# Check if a specified location is w/i a bare git repo
def is_bare(where=None):
    r"""Check if a location is in a bare git repo

    :Call:
        >>> q = is_bare(where=None)
    :Inputs:
        *where*: {``None``} | ;class:`str`
            Location to check
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *where* is in a bare git repo
    :Versions:
        * 2023-01-08 ``@ddalle``: v1.0
    """
    # Check for local/remote
    host, cwd = shellutils.identify_host(where)
    # Check if bare
    bare, _, ierr = shellutils.call_oe(
        ["git", "config", "core.bare"], cwd=cwd, host=host)
    # Check for issues
    if ierr:
        path = _assemble_path(host, cwd)
        raise SystemError("Path is not a git repo: %s" % path)
    # Otherwise output
    return bare.strip() == "true"


def _assemble_path(host, cwd):
    if host is None:
        return cwd
    else:
        return host + ":" + cwd


def _safe_ref(ref=None):
    # Default ref
    if ref is None:
        ref = "HEAD"
    # Output
    return ref

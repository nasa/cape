
------------------------------------------------------------------
Test results: :mod:`test.903_pyover.001_bullet.test_001_pyovercli`
------------------------------------------------------------------

This page documents from the folder

    ``test/903_pyover/001_bullet``

using the file

    ``test_001_pyovercli.py``

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_001_pyovercli.py
    :language: python

Test case: :func:`test_01_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_01_run
    :language: python
    :pyobject: test_01_run

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
        def test_01_run():
            # Instantiate
            cntl = cape.pyover.cntl.Cntl()
            # Run first case
    >       cntl.SubmitJobs(I="1")
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:29: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1155: in SubmitJobs
        self.StartCase(i)
    cape/cntl.py:96: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/cntl.py:1448: in StartCase
        pbs = self.CaseStartCase()
    cape/cntl.py:1475: in CaseStartCase
        return self._case_mod.StartCase()
    cape/pyover/case.py:198: in StartCase
        run_overflow()
    cape/pyover/case.py:126: in run_overflow
        bin.callf(cmdi, f="overrun.out", check=False)
    cape/cfdx/bin.py:133: in callf
        ierr = calli(cmdi, f, e, shell, v=v)
    cape/cfdx/bin.py:99: in calli
        ierr = sp.call(cmdi, stdout=fid, stderr=fe, shell=shell)
    /nasa/pkgsrc/toss4/2022Q1-rome/lib/python3.9/subprocess.py:349: in call
        with Popen(*popenargs, **kwargs) as p:
    /nasa/pkgsrc/toss4/2022Q1-rome/lib/python3.9/subprocess.py:951: in __init__
        self._execute_child(args, executable, preexec_fn, close_fds,
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <Popen: returncode: 255 args: ['overrunmpi', '-np', '24', 'run', '01']>
    args = ['overrunmpi', '-np', '24', 'run', '01'], executable = b'overrunmpi'
    preexec_fn = None, close_fds = True, pass_fds = (), cwd = None, env = None
    startupinfo = None, creationflags = 0, shell = False, p2cread = -1
    p2cwrite = -1, c2pread = -1, c2pwrite = 13, errread = -1, errwrite = 13
    restore_signals = True, gid = None, gids = None, uid = None, umask = -1
    start_new_session = False
    
        def _execute_child(self, args, executable, preexec_fn, close_fds,
                           pass_fds, cwd, env,
                           startupinfo, creationflags, shell,
                           p2cread, p2cwrite,
                           c2pread, c2pwrite,
                           errread, errwrite,
                           restore_signals,
                           gid, gids, uid, umask,
                           start_new_session):
            """Execute program (POSIX version)"""
        
            if isinstance(args, (str, bytes)):
                args = [args]
            elif isinstance(args, os.PathLike):
                if shell:
                    raise TypeError('path-like args is not allowed when '
                                    'shell is true')
                args = [args]
            else:
                args = list(args)
        
            if shell:
                # On Android the default shell is at '/system/bin/sh'.
                unix_shell = ('/system/bin/sh' if
                          hasattr(sys, 'getandroidapilevel') else '/bin/sh')
                args = [unix_shell, "-c"] + args
                if executable:
                    args[0] = executable
        
            if executable is None:
                executable = args[0]
        
            sys.audit("subprocess.Popen", executable, args, cwd, env)
        
            if (_USE_POSIX_SPAWN
                    and os.path.dirname(executable)
                    and preexec_fn is None
                    and not close_fds
                    and not pass_fds
                    and cwd is None
                    and (p2cread == -1 or p2cread > 2)
                    and (c2pwrite == -1 or c2pwrite > 2)
                    and (errwrite == -1 or errwrite > 2)
                    and not start_new_session
                    and gid is None
                    and gids is None
                    and uid is None
                    and umask < 0):
                self._posix_spawn(args, executable, env, restore_signals,
                                  p2cread, p2cwrite,
                                  c2pread, c2pwrite,
                                  errread, errwrite)
                return
        
            orig_executable = executable
        
            # For transferring possible exec failure from child to parent.
            # Data format: "exception name:hex errno:description"
            # Pickle is not used; it is complex and involves memory allocation.
            errpipe_read, errpipe_write = os.pipe()
            # errpipe_write must not be in the standard io 0, 1, or 2 fd range.
            low_fds_to_close = []
            while errpipe_write < 3:
                low_fds_to_close.append(errpipe_write)
                errpipe_write = os.dup(errpipe_write)
            for low_fd in low_fds_to_close:
                os.close(low_fd)
            try:
                try:
                    # We must avoid complex work that could involve
                    # malloc or free in the child process to avoid
                    # potential deadlocks, thus we do all this here.
                    # and pass it to fork_exec()
        
                    if env is not None:
                        env_list = []
                        for k, v in env.items():
                            k = os.fsencode(k)
                            if b'=' in k:
                                raise ValueError("illegal environment variable name")
                            env_list.append(k + b'=' + os.fsencode(v))
                    else:
                        env_list = None  # Use execv instead of execve.
                    executable = os.fsencode(executable)
                    if os.path.dirname(executable):
                        executable_list = (executable,)
                    else:
                        # This matches the behavior of os._execvpe().
                        executable_list = tuple(
                            os.path.join(os.fsencode(dir), executable)
                            for dir in os.get_exec_path(env))
                    fds_to_keep = set(pass_fds)
                    fds_to_keep.add(errpipe_write)
                    self.pid = _posixsubprocess.fork_exec(
                            args, executable_list,
                            close_fds, tuple(sorted(map(int, fds_to_keep))),
                            cwd, env_list,
                            p2cread, p2cwrite, c2pread, c2pwrite,
                            errread, errwrite,
                            errpipe_read, errpipe_write,
                            restore_signals, start_new_session,
                            gid, gids, uid, umask,
                            preexec_fn)
                    self._child_created = True
                finally:
                    # be sure the FD is closed no matter what
                    os.close(errpipe_write)
        
                self._close_pipe_fds(p2cread, p2cwrite,
                                     c2pread, c2pwrite,
                                     errread, errwrite)
        
                # Wait for exec to fail or succeed; possibly raising an
                # exception (limited in size)
                errpipe_data = bytearray()
                while True:
                    part = os.read(errpipe_read, 50000)
                    errpipe_data += part
                    if not part or len(errpipe_data) > 50000:
                        break
            finally:
                # be sure the FD is closed no matter what
                os.close(errpipe_read)
        
            if errpipe_data:
                try:
                    pid, sts = os.waitpid(self.pid, 0)
                    if pid == self.pid:
                        self._handle_exitstatus(sts)
                    else:
                        self.returncode = sys.maxsize
                except ChildProcessError:
                    pass
        
                try:
                    exception_name, hex_errno, err_msg = (
                            errpipe_data.split(b':', 2))
                    # The encoding here should match the encoding
                    # written in by the subprocess implementations
                    # like _posixsubprocess
                    err_msg = err_msg.decode()
                except ValueError:
                    exception_name = b'SubprocessError'
                    hex_errno = b'0'
                    err_msg = 'Bad exception data from child: {!r}'.format(
                                  bytes(errpipe_data))
                child_exception_type = getattr(
                        builtins, exception_name.decode('ascii'),
                        SubprocessError)
                if issubclass(child_exception_type, OSError) and hex_errno:
                    errno_num = int(hex_errno, 16)
                    child_exec_never_called = (err_msg == "noexec")
                    if child_exec_never_called:
                        err_msg = ""
                        # The error must be from chdir(cwd).
                        err_filename = cwd
                    else:
                        err_filename = orig_executable
                    if errno_num != 0:
                        err_msg = os.strerror(errno_num)
    >               raise child_exception_type(errno_num, err_msg, err_filename)
    E               FileNotFoundError: [Errno 2] No such file or directory: 'overrunmpi'
    
    /nasa/pkgsrc/toss4/2022Q1-rome/lib/python3.9/subprocess.py:1821: FileNotFoundError

Test case: :func:`test_02_c`
----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_02_c
    :language: python
    :pyobject: test_02_c

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_02_c():
            # Split command and add `-m` prefix
            cmdlist = [sys.executable, "-m", "cape.pyover", "-c", "-I", "1"]
            # Run the command
            stdout, _, _ = testutils.call_o(cmdlist)
            # Check outout
            result = testutils.compare_files(stdout, "test.02.out", ELLIPSIS=True)
    >       assert result.line1 == result.line2
    E       AssertionError: assert '1    powerof...           \n' == '1    powerof...0   .   ...\n'
    E         - 1    poweroff/m0.8a4.0b0.0 DONE    1500/1500   .   ...
    E         + 1    poweroff/m0.8a4.0b0.0 INCOMP  0/1500      .
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:41: AssertionError

Test case: :func:`test_03_fm`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_03_fm
    :language: python
    :pyobject: test_03_fm

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_03_fm():
            # Instantiate
            cntl = cape.pyover.cntl.Cntl()
            # Collect aero
            cntl.cli(fm=True, I="1")
            # Read databook
            cntl.ReadDataBook()
            # Get value
    >       CN = cntl.DataBook["bullet_no_base"]["CN"][0]
    E       IndexError: index 0 is out of bounds for axis 0 with size 0
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:54: IndexError


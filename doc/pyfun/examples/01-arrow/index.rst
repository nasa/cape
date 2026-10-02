
.. _pyfun-ex01-arrow:

Demo 1: Basic Viscous FUN3D Usage on Arrow with Fins
====================================================

This is a simple pyfun example that runs FUN3D on a fixed grid on an arrow
shape with four fins.

To get started, clone the repo and run a couple of commands.

    .. code-block:: console

        $ git clone https://github.com/nasa-ddalle/pyfun01-arrow.git
        $ cd pyfun01-arrow
        $ ./copy-files.py
        $ cd work/

This will copy all of the files into a newly created ``work/`` folder. Follow
the instructions below by entering that ``work/`` folder; the purpose is that
you can easily delete the ``work/`` folder and restart the tutorial at any
time.

The geometry used for this shape is a capped cylinder with four fins. The
surface has 14140 triangles on seven wall components, and the volume mesh has
13 boundaries in total including the six far-field faces. The surface
triangulation, ``arrow-far.tri``, is shown below.

    .. figure:: arrow01.png
        :width: 4in

        Arrow shape triangulation with four fins

The files in this folder are listed below with a short description.  In this
case, the run matrix is defined within the ``pyFun.json`` file.

    * ``pyFun.json``: Master input control file for pyfun
    * ``fun3d.nml``: Template namelist file
    * ``arrow-far.ugrid``: Volume grid, ASCII AFLR3 format
    * ``arrow-far.mapbc``: Boundary conditions file
    * ``arrow-far.tri``: (Not used) Cart3D surface triangulation
    * ``arrow-far.xml``: (Not used) XML file defining the component hierarchy


.. _pyfun-ex01-env:

Setting Up the Environment
--------------------------

The ``pyfun`` command is part of CAPE, which is the successor to pyCart. On
most HPC systems the easiest way to set up your environment is with a module.
For example, the following commands work on NAS Pleiades.

    .. code-block:: bash

        module use -a /swbuild/fun3d/fun3dv14_users/modulefiles
        module use -a /home5/ddalle/share/modulefiles
        module load python3/3.11.5
        module load FUN3D_INTG_Turin/14.1
        module load cape/devel

These are the same commands found in the ``"ShellCmds"`` section of
``pyFun.json``, which CAPE runs at the top of every job script. Adjust the
FUN3D and CAPE module names to match what is available on your system. The
CAPE module does not load Python by itself, so a Python module must also be
loaded; any recent Python 3 with NumPy, PyYAML, and Matplotlib in its
*site-packages* will do (at NAS this is ``python3/3.11.5``).

It is a good idea to run these commands on the login or compute node where you
will type ``pyfun``, and to put them in a startup file or your own module.
Without a working Python, the ``pyfun`` command fails with an ``ImportError``
about NumPy.  Assuming the present working directory is the ``work/`` folder of
this demo, a good first test command is the following, which checks the status
of each case in the matrix.

    .. code-block:: console

        $ pyfun -c
        Using pyfun JSON file: pyFun.json
        Case Config/Run Directory Status  Iterations Que CPU Time
        ---- -------------------- ------- ---------- --- --------
        0    arrow/m0.80a0.0b0.0  ---     /          .
        1    arrow/m0.80a2.0b0.0  ---     /          .
        2    arrow/m0.84a0.0b0.0  ---     /          .
        3    arrow/m1.20a0.0b0.0  ---     /          .

        ---=4,

Here pyfun has read the default master file name, ``pyFun.json`` and determined
that this is a setup with four cases.  The run matrix has at least three input
variables, which are Mach number, angle of attack, and sideslip, although the
user can't tell yet whether or not there are more input variables that don't
affect the parametric folder names.

As it turns out, this is a viscous example, so we should expect Reynolds number
and freestream static temperature to be inputs.  It is permitted in pyfun not
to specify these inputs and just keep a constant value inherited from the
``fun3d.nml`` namelist, but it would be unusual to have varying Mach numbers
and a constant freestream state.  If we look at the final section in
``pyFun.json``, we see that Reynolds number and temperature are indeed
variables.

    .. code-block:: javascript

        // Run matrix description
        "RunMatrix": {
            // File and variable list
            "File": "",
            "Keys": [
                "mach", "alpha", "beta", "Re", "T", "config", "Label"
            ],
            // Modify one definition
            "Definitions": {
                "mach": {"Format": "%.2f"}
            },
            // Group settings
            "GroupMesh": false,
            // Label universal
            "Label": "",
            "config": "arrow",
            // Local values
            "mach":  [0.8, 0.8, 0.84, 1.2],
            "alpha": [0.0, 2.0, 0.0,  0.0],
            "beta":  [0.0, 0.0, 0.0,  0.0],
            "Re":    [1e3, 1e3, 1e4,  1e4],
            "T":     [478, 478, 478,  478]
        }

The second line, *Keys*, lists the five input variables we have discussed above
and two additional input variables used for bookkeeping.  Each of these
variables has a standard name, so pyfun provides a default definition and
interpretation.  However, it is possible to modify any aspect of a variable's
behavior in the *Definitions* section.

Here we have modified the *mach* definition so that pyfun explicitly includes
exactly two digits after the decimal place in the folder name (otherwise we may
have difficulty with the Mach 0.84 case).  There are many more capabilities of
this *Definitions* section.  Some of them are discussed in other examples, and
the complete guide can be found in the "RunMatrix" section of the JSON
documentation.

In this case, we have decided to specify the values of the variables within the
JSON file.  We can specify a list with one value for each case, as in ``"mach":
[0.8, 0.8, 0.84, 1.2]`` or a constant value that applies to all cases as in
``"config": "arrow"``.

The values are written to the ``&reference_physical_properties`` group of each
phase namelist.  The run-matrix temperature *T* is interpreted as Rankine, which
pyfun converts to Kelvin for FUN3D, and the freestream Mach number, angle of
attack, angle of yaw, and Reynolds number are written directly.


.. _pyfun-ex01-run:

Running a Case
--------------

To actually run a case, specifically the first case, run the following command.
The ``"RunControl"`` section of ``pyFun.json`` has ``"qsub": false``, so the
case runs immediately on the current compute node, and ``"MPI": true`` with no
*NProc* given, so FUN3D uses every process available to the current allocation.
The run below uses 128 processes and takes about one minute of wall time
(roughly one CPU-hour), and it can be aborted with a ``Ctrl-C`` command if
desired.

    .. code-block:: console

        $ pyfun -I 0
        Using pyfun JSON file: pyFun.json
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    arrow/m0.80a0.0b0.0   ---     /           .
          Case name: 'arrow/m0.80a0.0b0.0' (index 0)
             Starting case 'arrow/m0.80a0.0b0.0'
         > mpiexec -np 128 nodet_mpi --animation_freq 500
             (PWD = 'arrow/m0.80a0.0b0.0/')
             (STDOUT = 'fun3d.out')
             (STDERR = 'fun3d.err')
         > mpiexec -np 128 nodet_mpi --animation_freq 500
             (PWD = 'arrow/m0.80a0.0b0.0/')
             (STDOUT = 'fun3d.out')
             (STDERR = 'fun3d.err')

        Submitted or ran 1 job(s).

        ---=1,

Two commands are printed because this case has two phases, which are set by the
*nIter* entry of the ``"RunControl"`` section along with the ``"Fun3D"``
section of the JSON file.

While this is running, we can open another window and navigate to the same
folder.  Then we can check the status using another ``pyfun -c`` call.

    .. code-block:: console

        $ pyfun -c
        Using pyfun JSON file: pyFun.json
        Case Config/Run Directory Status  Iterations Que CPU Time
        ---- -------------------- ------- ---------- --- --------
        0    arrow/m0.80a0.0b0.0  RUN     500/1000   .   0.565556
        1    arrow/m0.80a2.0b0.0  ---     /          .
        2    arrow/m0.84a0.0b0.0  ---     /          .
        3    arrow/m1.20a0.0b0.0  ---     /          .

        ---=3, RUN=1,

Once the case is complete (not fully necessary for this demo), the status will
change to the following.

    .. code-block:: console

        $ pyfun -c
        Using pyfun JSON file: pyFun.json
        Case Config/Run Directory Status  Iterations Que CPU Time
        ---- -------------------- ------- ---------- --- --------
        0    arrow/m0.80a0.0b0.0  DONE    1000/1000  .       1.06
        1    arrow/m0.80a2.0b0.0  ---     /          .
        2    arrow/m0.84a0.0b0.0  ---     /          .
        3    arrow/m1.20a0.0b0.0  ---     /          .

        ---=3, DONE=1,

To submit the case to the queue instead of running it on the current node, set
``"qsub": true`` in the ``"RunControl"`` section. The ``"PBS"`` section of
``pyFun.json`` holds the resource requests, and it should be edited to match
the queue and node model (``model=tur_ath`` here) of your allocation.

The ``arrow/m0.80a0.0b0.0`` folder itself contains the hallmark files of a
FUN3D run with a project prefix of ``arrow`` (which is set within
``pyFun.json`` or ``fun3d.nml``) with a few additional files used by pyfun to
keep track of status.

    .. code-block:: console

        $ cd arrow/m0.80a0.0b0.0
        $ ls
        arrow.flow
        arrow_fm_base.dat
        arrow_fm_body.dat
        arrow_fm_bullet_no_base.dat
        arrow_fm_bullet_total.dat
        arrow_fm_cap.dat
        arrow_fm_fin1.dat
        arrow_fm_fin2.dat
        arrow_fm_fin3.dat
        arrow_fm_fin4.dat
        arrow_fm_fins.dat
        arrow_fm_fuselage.dat
        arrow.forces
        arrow.freeze
        arrow.grid_info
        arrow_hist.dat
        arrow.mapbc
        arrow_stream_info.dat
        arrow_tec_boundary_timestep1000.szplt
        arrow_tec_boundary_timestep500.szplt
        arrow_tec_volume.szplt
        arrow.ugrid
        cape
        case.json
        conditions.json
        fun3d.00.nml
        fun3d.01.nml
        fun3d.err
        fun3d.nml
        pyfun_start.dat
        pyfun_time.dat
        run.00.500
        run.01.1000
        run_fun3d.pbs

The ``fun3d.00.nml`` and ``fun3d.01.nml`` files are the namelists for each
phase, created from the template ``fun3d.nml`` with the ``"Fun3D"`` section and
the run-matrix values applied. The file ``fun3d.nml`` is a symbolic link to the
namelist of the phase that last ran. The ``run.00.500`` and ``run.01.1000``
files hold the screen output (``fun3d.out``) of each completed phase. Each
component listed in the ``"Config"`` section has its own force-and-moment
history file, ``arrow_fm_``\ *COMP*\ ``.dat``, and the ``cape/`` subfolder
contains CAPE's logs for the case.


.. _pyfun-ex01-aero:

Extracting Data
---------------

The ``"DataBook"`` section of ``pyFun.json`` defines an aerodynamic data book
for the components of this shape. The command below extracts the time-averaged
forces and moments from each completed case into the ``data/arrow`` folder.

    .. code-block:: console

        $ pyfun -I 0 --aero
        ...
        Component bullet_no_base (type=FM)
        arrow/m0.80a0.0b0.0
          New entry at iteration 1000
        Added or updated 1 entries

The data book files, ``data/arrow/aero_$COMP.csv``, are plain CSV files with
one line per case, and they are the basis for the sweep plots and reports
described in the :ref:`pyfun-ex02-bullet` example. Creating PDF
reports with ``pyfun --report`` also requires a working LaTeX installation,
which is not available on every compute node.

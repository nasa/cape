
------------------------
Trajectory or Run Matrix
------------------------

A required section of :file:`pyCart.json` describes the variables that define
the run conditions and the values of those variables.  In many cases, these will
be familiar input variables like Mach number, angle of attack, and sideslip
angle; but in other cases they will require additional description within the
:file:`pyCart.json` file.

Example Trajectories
====================
These possibilities can be roughly categorized into several types of
trajectories, which range in complexity.  The simplest descriptions are
self-contained and contain just a few lines, and the most complex may contain
more than a hundred lines and a pointer to an external file that sets the actual
values for thousands of jobs.

Simple Self-Contained Trajectory
--------------------------------
The following example is essentially the simplest type of run matrix that can be
defined.  Fortunately it is also quite common; there are three input variables,
which are Mach number, angle of attack, and sideslip angle; and few enough runs
to just list them directly in the :file:`pyCart.json` file.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["Mach", "alpha", "beta"],
            "Mach": [2.00, 2.00, 2.00, 2.00, 2.00, 2.50],
            "alpha": [-2.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            "beta": [0.0, 0.0, 0.0, -2.0, 2.0, 0.0]
        }
        
Without going into all of the details, this will create the following folders.

    .. code-block:: none
    
        Grid/m2.0a-2.0b0.0
        Grid/m2.0a0.0b0.0
        Grid/m2.0a2.0b0.0
        Grid/m2.0a0.0b-2.0
        Grid/m2.0a0.0b2.0
        Grid/m2.5a0.0b0.0
        
By default, pyCart uses ``Grid`` as the name of the group folder, and it uses as
many digits as necessary to print the value of each of the variables (i.e.
``m2.0`` instead of ``m2.0000``).  Because ``"Mach"``, ``"alpha"``, and
``"beta"`` are recognized variables, pyCart knows to set the appropriate values
in :file:`input.cntl` in each of the run folders.

However, all of this behavior can be controlled using more advanced options in
:file:`pyCart.json`.

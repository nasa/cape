
.. _matrix-syntax:

Syntax for Run Matrix Files
===========================

Run matrix files have a customized comma- or space-separated file.  It is
customized in the sense that entries can be separated by either commas or
spaces, and there is a capability to mark rows with extra characters as a way
of giving a final disposition of the status of each case.  The values of each
column can be strings, floats, integers, or even hexadecimal integers.

These files are prototypically given a ``.csv`` file extension, but this is
mostly so that double clicking on the files in a file browser will launch a
spreadsheet program.  Below shows a typical ``"Trajectory"`` section from a
JSON file for a simple configuration with three main keys: Mach number, angle
of attack, and sideslip angle.  There are two additional keys, which are the
``"config"`` (a group folder name) and a ``"Label"``.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["mach", "alpha", "beta", "config", "Label"],
            "File": "matrix.csv"
        }
        
The following shows some typical contents of a :file:`matrix.csv` file.

    .. code-block:: none
    
        # mach, alpha, beta, config, Label
          0.80, 0.0, 0.0, poweroff,
          0.80, 2.0, 0.0, poweroff,
          0.80, 4.0, 0.0, poweroff,
          1.20, 0.0, 0.0, poweroff,
          1.20, 2.0, 0.0, poweroff,
          1.20, 2.0, 0.0, poweroff, try2
         
This is a pretty straightforward file except possibly to point out that the
first five cases have an empty ``"Label"`` value.  After running some cases,
suppose that the first four cases ran successfully and that the user determines
those cases to be successful.  These will be marked ``PASS`` by putting a ``p``
in the first column of the appropriate lines.  However, the fifth case had some
sort of problem and cannot be run using the initial configuration.  Now the
user is doing a second attempt, which has not been completed yet.  For the user
to mark these statuses, the contents of :file:`matrix.csv` should be edited to

    .. code-block:: none
    
        # mach, alpha, beta, config, Label
        p 0.80, 0.0, 0.0, poweroff,
        p 0.80, 2.0, 0.0, poweroff,
        p 0.80, 4.0, 0.0, poweroff,
        p 1.20, 0.0, 0.0, poweroff,
        e 1.20, 2.0, 0.0, poweroff,
          1.20, 2.0, 0.0, poweroff, try2
     


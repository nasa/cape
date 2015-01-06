

------------------
PBS Script Options
------------------

The "PBS" section of :file:`pyCart.json` controls settings for PBS scripts when
Cart3D runs are submitted as PBS jobs to a high-performance computing
environment.  Regardless of whether or not the user intends to submit PBS jobs,
pyCart creates scripts called :file:`run_cart3d.pbs` or
:file:`run_cart3d.01.pbs`, :file:`run_cart3d.02.pbs`, etc.  At the top of these
files is a collection of PBS directives, which will be ignored if the script is
launched locally.

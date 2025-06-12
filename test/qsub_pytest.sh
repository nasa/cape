#!/bin/bash
#PBS -S /bin/bash
#PBS -N capetest
#PBS -r y
#PBS -j oe
#PBS -l select=1:ncpus=128:mpiprocs=128:model=rom_ait
#PBS -l walltime=2:00:00
#PBS -l site=needed=/nobackupnfs1+/home3+/home5+/vast_swbuild
#PBS -W group_list=e0847
#PBS -q sls_aero1

# Go to working directory
cd /nobackupnfs1/ddalle/cape/src/cape-devel

# Additional shell commands
. $MODULESHOME/init/bash
module purge
module use -a /home3/serogers/share/modulefiles
module use -a /home5/ddalle/share/modulefiles
module use -a /swbuild/fun3d/fun3dv14_users/modulefiles
module load python3/3.11.5
module load cape/devel
module load aflr3/16.27.3
module load overflow/2.4b_dp
module load FUN3D_INTG_Rome/14.1
module load cart3d/1.6.0
module load tecplot/2023r2
module load pkgsrc

# Execute tests
python3 drive_pytest.py

# Check for failure
if [[ "$?" != "0" ]]; then
    exit 0
fi

# Switch to python 3.6
module swap python3 python3/3.6.8
python3 drive_pytest.py push

# Check for failure
if [[ "$?" != "0" ]]; then
    exit 0
fi

# Switch to python 3.9
module swap python3 python3/3.9.12
python3 drive_pytest.py push


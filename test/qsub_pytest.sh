#!/bin/bash
#PBS -S /bin/bash
#PBS -N capetest
#PBS -r y
#PBS -j oe
#PBS -l select=1:ncpus=128:mpiprocs=128:model=rom_ait
#PBS -l walltime=2:00:00
#PBS -W group_list=e0847
#PBS -q devel

# Go to working directory
cd /nobackupp16/ddalle/cape/src/cape

# Additional shell commands
. $MODULESHOME/init/bash
module use -a /home3/serogers/share/modulefiles
module use -a /home5/ddalle/share/modulefiles
module use -a /home3/fun3d/shared/n1337/toss3/modulefiles
module load python3/3.9.12
module load cape/devel-p16
module load overflow/2.4b_dp
module load FUN3D_Rome_TecIO/13.7
module load cart3d/1.5.9

# Execute tests
python3 drive_pytest.py


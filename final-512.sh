#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=10:00:00
#SBATCH --job-name final_job
#SBATCH --output=final_output_512.txt
#SBATCH --mail-type=FAIL
 
 
module load intel/2019u3
module load python/3.6.8
module load cmake/3.21.4

icc -o trmm trmm.cpp
icc -o syrk syrk.cpp
icc -o symm symm.cpp

echo 512
echo trmm.cpp
./trmm 512 512 
echo trmm.py
python trmm.py 512 512
echo syrk.cpp
./syrk 512 512 
echo syrk.py
python syrk.py 512 512
echo symm.cpp
./symm 512 512 
echo symm.py
python symm.py 512 512


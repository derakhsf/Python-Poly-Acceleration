#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=10:00:00
#SBATCH --job-name final_job
#SBATCH --output=final_output_128.txt
#SBATCH --mail-type=FAIL
 
 
module load intel/2019u3
module load python/3.6.8
module load cmake/3.21.4

icc -o trmm trmm.cpp
icc -o syrk syrk.cpp
icc -o symm symm.cpp

echo 128
echo trmm.cpp
./trmm 128 128 
echo trmm.py
python trmm.py 128 128
echo syrk.cpp
./syrk 128 128 
echo syrk.py
python syrk.py 128 128
echo symm.cpp
./symm 128 128 
echo symm.py
python symm.py 128 128


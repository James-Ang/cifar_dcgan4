#!/bin/bash -l

#SBATCH --partition=gpu-standard
#SBATCH --job-name=dcgan
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --nodes=1
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --qos=normal
#SBATCH --nodelist=gpu02

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s2003824@siswa.um.edu.my
#SBATCH --gres=gpu:titanx:1

#gres=gpu:k40c:2
#gres=gpu:gtx1080ti:1
#gres=gpu:titanxp:2
#gres=gpu:k10:8

# Loading Required module
module purge
module load miniconda/miniconda3
#module load cudnn/cudnn-8.1.1/cuda-11.2
module load cuda/cuda-11.0

#source activate ~/conda_env
#source venv/bin/activate #library not installed yet

#python proj_files/first_test/basic_main.py
python cifar_dcgan_main.py

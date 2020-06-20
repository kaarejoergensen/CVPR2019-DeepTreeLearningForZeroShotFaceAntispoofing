#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J CVPR
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 5GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s144852@student.dtu.dk
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo log/nn_training.out
#BSUB -eo log/nn_training.out


source /dtu/sw/dcc/dcc-sw.bash
nvidia-smi
# Load the cuda module
module load cuda/10.1
module load cudnn/v7.6.5.32-prod-cuda-10.1

module load python/3.7.3

cd ~/Documents/CVPR2019-DeepTreeLearningForZeroShotFaceAntispoofing
pipenv lock -r > requirements.txt
pip install -r requirements.txt --user

python3 train.py
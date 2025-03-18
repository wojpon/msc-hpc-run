#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
### -- set the job Name --
#BSUB -J mho_gat_fixed
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -u wojtek.ponikiewski@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
### #BSUB -o batch_output_gat/mho_%J.out
### #BSUB -e batch_output_gat/mho_%J.err
# -- end of LSF options --

source /dtu/3d-imaging-center/courses/conda/conda_init.sh
conda activate env-02510
pip install pyarrow
pip install optuna
pip install torch
pip install torch-geometric
python -u /dtu/3d-imaging-center/courses/02509/groups/group10/msc-hpc-run/mho_models/multi_horizon_gat_fixed_graph.py

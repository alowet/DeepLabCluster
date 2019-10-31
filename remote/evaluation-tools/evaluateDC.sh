#!/bin/bash

#SBATCH -J EvaluatePupil
#SBATCH -N 1 # 1 node
#SBATCH -p gpu # partition (queue)
#SBATCH --gres=gpu:1
#SBATCH --mem 16000 # memory pool for all cores
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o Jobs/Job.%x.%N.%j.out # STDOUT
#SBATCH -e Jobs/Job.%x.%N.%j.err # STDERR
#SBATCH --exclude=shakgpu[01-50],aagk80gpu[09,10,11,39,50,60]
#!SBATCH --test-only

#SBATCH --mail-type=END,FAIL,TIME_LIMIT # notifications
#SBATCH --mail-user=alowet@g.harvard.edu # send-to address

module load Anaconda3/5.0.1-fasrc01
module load cuda/8.0.61-fasrc01 cudnn/6.0_cuda8.0-fasrc01
module load bazel/0.13.0-fasrc01 gcc/4.9.3-fasrc01 hdf5/1.8.12-fasrc08 cmake
source activate dcut_tf13

srun -N 1 python3 Step1_EvaluateModelonDataset.py 0 #to evaluate your model [needs TensorFlow]
srun -N 1 python3 Step2_AnalysisofResults.py 1 #to compute test & train errors for your trained model
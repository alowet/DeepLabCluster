#!/usr/bin/env python
#SBATCH -J AnalysisLoop
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 2000
#SBATCH -t 10 # minutes
#SBATCH -o job-scripts/out/Job.%x.%A_%a.%N.%j.out # STDOUT
#SBATCH -e job-scripts/err/Job.%x.%A_%a.%N.%j.err # STDERR
#SBATCH --mail-type=END,FAIL,TIME_LIMIT # notifications
#SBATCH --mail-user=alowet@g.harvard.edu # send-to address

"""
Usage: sbatch analyze_all_array.sh

Written by Adam S. Lowet, Oct. 29, 2019
"""

import os, sys
from collections import Counter

subfolder = os.getcwd().split('analysis-tools')[0]
sys.path.append(subfolder)
sys.path.append(os.path.join(subfolder, "config"))
from myconfig_analysis import videofolderroot


"""
def mkdir_p(dir):
    #make a directory (dir) if it doesn't exist
    if not os.path.isdir(dir):
        os.mkdir(dir)
    if not os.path.isdir(os.path.join(dir, 'out')):
    	os.mkdir(os.path.join(dir, 'out'))
    if not os.path.isdir(os.path.join(dir, 'err')):
    	os.mkdir(os.path.join(dir, 'err'))
"""

video_folders = []
base_names = {}
for root, dirs, files in os.walk(videofolderroot):
	for file in files:
		if file.endswith('.avi'):
			path = os.path.dirname(os.path.join(root, file))
			base_name = os.path.splitext(os.path.basename(file))[0]
			# don't add if .mp4 already exists, meaning video has been analyzed already
			if not os.path.isfile(os.path.join(path, base_name + '.mp4')):
				video_folders.append(path)
				if path in base_names: base_names[path].append(base_name)
				else: base_names[path] = [base_name]

unique_folders = list(Counter(video_folders).keys())
avi_counts = list(Counter(video_folders).values())
job_directory = "%s/job-scripts" %os.getcwd()

# Make top level directories
# mkdir_p(job_directory)

# Run a separate array job for each recording session
for i in range(len(unique_folders)):
	
	folder = unique_folders[i]
	last_folder = os.path.basename(os.path.normpath(folder))

	if not os.path.isdir(os.path.join(job_directory,'out',last_folder)):
		os.makedirs(os.path.join(job_directory,'out',last_folder))
	if not os.path.isdir(os.path.join(job_directory,'err',last_folder)):
		os.makedirs(os.path.join(job_directory,'err',last_folder))

	base_name = base_names[folder][i]
	base_stem = base_name.split('_')
	base_stem = base_stem[:-1]
	base_stem = '_'.join(base_stem)

	video_file = os.path.join(folder, base_stem + '_')
	job_file = os.path.join(job_directory, last_folder + '.job')

	with open(job_file, 'w') as fh:
		fh.writelines("#!/bin/bash\n")
		fh.writelines("#SBATCH --job-name=%sTrial\n" % last_folder)
		fh.writelines("#SBATCH --output=%s.out\n" % os.path.join(job_directory,'out',last_folder,base_stem+'.%x.%A_%a.%N.%j'))
		fh.writelines("#SBATCH --error=%s.err\n" % os.path.join(job_directory,'err',last_folder,base_stem+'.%x.%A_%a.%N.%j'))
		fh.writelines("#SBATCH --time=0-00:20\n")
		fh.writelines("#SBATCH --mem=2000\n") #memory for each array task
		fh.writelines("#SBATCH -N 1\n") # one node
		fh.writelines("#SBATCH -p fas_gpu\n") # partition (queue)
		fh.writelines("#SBATCH --gres=gpu:1\n")
		fh.writelines("#SBATCH --mail-type=FAIL,TIME_LIMIT,END\n")
		fh.writelines("#SBATCH --mail-user=$alowet@g.harvard.edu\n")
		fh.writelines("#SBATCH --export=ALL\n")
		fh.writelines("#SBATCH --exclude=shakgpu[01-50],aagk80gpu[09,10,11,39,50,60]\n")

		fh.writelines("module load Anaconda3/5.0.1-fasrc01\n")
		fh.writelines("module load cuda/8.0.61-fasrc01 cudnn/6.0_cuda8.0-fasrc01\n")
		fh.writelines("module load bazel/0.13.0-fasrc01 gcc/4.9.3-fasrc01 hdf5/1.8.12-fasrc08 cmake\n")
		fh.writelines("module load ffmpeg/2.7.2-fasrc01\n")
		fh.writelines("source activate dcut_tf13\n")

		fh.writelines("python3 -u AnalyzeVideosTrial.py %s${SLURM_ARRAY_TASK_ID}.avi\n" %video_file)
		fh.writelines("python3 -u MakingLabeledVideoTrial.py %s${SLURM_ARRAY_TASK_ID}.avi\n" %video_file)

	os.system("sbatch --array=0-" + str(avi_counts[i]-1) + " " + job_file)


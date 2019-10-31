# DeepLabCluster
Quick startup guide to using the FAS cluster to train, evaluate, and analyze video with [DeepLabCut](https://github.com/AlexEMG/DeepLabCut). If you are unfamiliar with cluster computing, a good place to start is [here](https://docs.google.com/document/d/1HxHSsm9UJd7QF6eIsLBPAaDiSg4_w653RKRPY_2-6TI/edit?usp=sharing) and [here](https://www.rc.fas.harvard.edu/resources/quickstart-guide/).

# Training a new neural network
## I. Label Frames Locally
([Credit to Korleki](https://github.com/alowet/Novelty_analysis_KA/blob/master/Docs/Training_a_new_network.md))

This is the only part of the workflow that is done on your local computer. Make sure you have Fiji installed, as well as the `Generating_a_Training_Set` directory cloned from the `local` directory of this repo.

##### 0. Configuration of your project:

Edit `myconfig.py` in the `Generating_a_Training_Set` folder.

##### 1. Selecting data to label:
```
cd Generating_a_Training_Set
python3 Step1_SelectRandomFrames_fromVideos.py
```
##### 2. Label the frames:

 - open ImageJ or Fiji
 - File > Import > Image Sequence
 ![alt text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.38.43.png)
 - within pop-up window navigate to folder with images to be labeled (Generating_a_Training_Set -> data-YOUR_NETWORK)
 ![alt text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.41.15.png)
 - click first image, then click open
 - you will see window pop up named "Sequence Options"
 
   ![alt text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.41.37.png)
 - Make sure 2 boxes are checked: "Sort names numerically" and "Use virtual stack"
 - window will pop up with all your images in stack (scroll bar at bottom)
 - in tool bar (with File, Edit, etc), click Multi-point button (to right of angle button and to left of wand button)
     - you may see this botton as "point tool" (single star). if this happens, right click and change to be multi-point
  ![alt text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.42.48.png)
  ![alt_text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.43.04.png)
 - click on body features in EXACT order for every image (order specified in myconfig.py - Step 2, bodyparts variable)
 
   ![alt text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.46.30.png)
 
   (if a point can't be determined, click in the top left corner of the image, so that X and Y positions are less than 50 pixels)
   ![alt text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.48.13.png)
 - once you get through all frames, go to Analyze -> Measure
 ![alt text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.48.36.png)
 - window will pop up: "Results"
     - the points you labeled will appear in rows, with each column representing a different feature of that point
     - make sure that the number of rows corresponds to N x BP where N = number of frames and BP = number of body parts you label in each frame
 ![alt text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.49.40.png)
 - save Results window as "Results.csv" in same folder as the images you're labeling
![alt text](https://github.com/ckakiti/Novelty_analysis_KA/blob/master/Docs/Labeling_images/Screen%20Shot%202019-10-16%20at%2012.50.13.png)

##### 3. Formatting the data I:
```
python3 Step2_ConvertingLabels2DataFrame.py
```
##### 4. Checking the formatted data:
```
python3 Step3_CheckLabels.py
```
##### 5. Formatting the data II:
```
python3 Step4_GenerateTrainingFileFromLabelledData.py
```

## II. Configuring the Anaconda environment on the cluster

##### 1. Copy all folders in the `remote` directory of this repo to your directory on the cluster
This directory is probably called something like `/n/holylfs/LABS/uchida_lab/globus_$YOUR_RC_ID`. Since your folders are currently local, you'll have to do this with either `scp` or a client like FileZilla, e.g.
```
scp -r remote $YOUR_RC_ID@login.fas.rc.harvard.edu:/n/holylfs/LABS/uchida_lab/globus_$YOUR_RC_ID/
```
You'll have to make sure to overwrite the default config.py file in the `configs` folder with your own `config.py`, which is currently only local!

##### 2. Log in to the cluster and start an interactive session.
In this example, I request two hours and 4 GB of RAM on the `test` partition.
```
srun --pty -p test -t 2:00:00 --mem 4G /bin/bash
```
For help on how to use the cluster, see this page: https://www.rc.fas.harvard.edu/resources/quickstart-guide/.

##### 3. Load the Anaconda environment from a .yml file.
First, `cd` to whereever you uploaded the contents of the `remote` directory. You may want to rename it to DeepLabCut, e.g.
```
cd /n/holylfs/LABS/uchida_lab/globus_$YOUR_RC_ID/
mv remote DeepLabCut
cd DeepLabCut
```
This folder should contain `TF1_3GPUEnv_DeeperCut.yml`. Then,
```
module load Anaconda3/5.0.1-fasrc01
module load cuda/8.0.61-fasrc01 cudnn/6.0_cuda8.0-fasrc01
conda env create -f TF1_3GPUEnv_DeeperCut.yml
```
(Thanks to Gerald Pho for this step.)

## III. Training  the deep neural network

##### 1. Get the pretrained networks.

```
cd pose-tensorflow/models/pretrained
curl http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz | tar xvz
curl http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz | tar xvz
```

##### 2. Copy the two folders generated from the `Formatting the data II` step to `/pose-tensorflow/models/`.
Since your folders are currently local, you'll have to do this with either `scp` or a client like FileZilla. For example, assuming you extracted everything to a directory titled `DeepLabCut`, then from a *local* command line, you'd run:
```
scp -r YOURexperimentNameTheDate-trainset95shuffle1 $YOUR_RC_ID@login.fas.rc.harvard.edu:/n/holylfs/LABS/uchida_lab/globus_$YOUR_RC_ID/DeepLabCut/pose-tensorflow/models/
scp -r UnaugmentedDataSet_YOURexperimentNameTheDate $YOUR_RC_ID@login.fas.rc.harvard.edu:/n/holylfs/LABS/uchida_lab/globus_$YOUR_RC_ID/DeepLabCut/pose-tensorflow/models/
```

##### 3. Start training
Make sure to edit the path in `train_requeueDC.sh` first!
```
cd pose-tensorflow
sbatch train_requeueDC.sh
```
If this is working properly, it will take ~10 hours to run.

## IV. Evaluate your network
```
cd ../evaluation-tools
sbatch evaluateDC.sh
```
Evaluation metrics will be printed to STDOUT, and images will be saved in the `evaluation-tools` directory in a new folder called `LabeledImages_...`.

## V. Analyzing videos

##### 0. Configuration of your project

Edit: `myconfig_analysis.py` in the `configs` folder within `remote`. If you do this locally (recommended), don't forget to re-upload to the cluster!

##### 1. Analyzing/making labeled videos:
```
cd ../analysis-tools
sbatch analyzeDC.sh
```
This step can be easily parallelized, making it ideal to be run on the cluster! For example, let's say you wanted to run each recording session as its own job. This is more efficient when there are many short trials, which don't deserve their own job or job array because of load on the Slurm scheduler. Use:
```
cd ../analysis-tools/parallel-session
sbatch analyze_all.sh
```
If you had relatively few sessions but many trials, each named something like `path-to-file/trial_$trial-num`, and each trial was relatively long, it would be more efficient to submit each trial as a task within a job array. For example, try:
```
cd ../analysis-tools/parallel-trial
sbatch analyze_all_array.sh
```
However, note that these parallelization methods are all directory structure and naming-convention dependent. The code is provided to give you an idea of how to do this, but it should not be expected to work out of the box.

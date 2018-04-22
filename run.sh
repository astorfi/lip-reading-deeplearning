#!/bin/bash

# To run this shell script type as follows in the terminal:
#
# For training execute: ./run.sh train/test/status path/to/video/file 
#       example: ./run.sh train data/sample_video.mp4 
#
# Argument: 
#         train/test/status: one of the following: train, test, nothing!
#         path/to/video/file: relative path to the video file that we want to perform lip tracking on that.


if [ $# -eq 2 ]; then
    # assign the provided arguments to variables
    do_training=$1
    input_filename=$2
else
    # assign the default values to variables
    do_training='train'
    input_filename="data/sample_video.mp4"
fi

if [ $do_training = 'train' ]; then

    # training
    python -u ./code/training_evaluation/train.py --num_epochs=1 --batch_size=16 --train_dir=${HOME}/results/TRAIN_CNN_3D/train_logs
    # testing - Automatically restore the latest checkpoint from all saved checkpoints
    python -u ./code/training_evaluation/test.py --checkpoint_dir=${HOME}/results/

elif [ $do_training = 'test' ]; then
  
    # Just performing the test
    python -u ./code/training_evaluation/test.py --checkpoint_dir=${HOME}/results/

else

    echo "No training or testing will be performed!"
fi

# visualizing (using pretrained model)
ln -s data/ dlib
mkdir results && ln -s results/activation ./activation
python -u ./code/lip_tracking/VisualizeLip.py --input $input_filename --output results/output_video.mp4

# create gif from mouth frames
ffmpeg -i ./results/mouth/frame_%*.png results/mouth.gif

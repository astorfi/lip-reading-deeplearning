# visualizing (using pretrained model)
ln -s data/ dlib
mkdir results/action && ln -s results/activation ./activation
python -u ./code/lip_tracking/VisualizeLip.py --input $input_filename --output ../results/output_video.mp4

# create gif from mouth frames
ffmpeg -i ./mouth/frame_%*.png ../results/mouth.gif

# Sample input
input_filename="data/sample_video.mp4"

# visualizing (using pretrained model)
mkdir results
ln -s data/ dlib
python -u ./code/lip_tracking/VisualizeLip.py --input $input_filename --output results/output_video.mp4

## create gif from mouth frames
#ffmpeg -i ./results/mouth/frame_%*.png results/mouth.gif

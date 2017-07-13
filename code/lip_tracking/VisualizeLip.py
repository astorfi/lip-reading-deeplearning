import numpy as np
import cv2
import dlib
import math
import sys
import pickle
import argparse
import os
import skvideo.io


"""
PART1: Construct the argument parse and parse the arguments
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video file")
ap.add_argument("-o", "--output", required=True,
                help="path to output video file")
ap.add_argument("-f", "--fps", type=int, default=30,
                help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
                help="codec of output video")
args = vars(ap.parse_args())

"""
PART2: Calling and defining required parameters for:

       1 - Processing video for extracting each frame.
       2 - Lip extraction from frames.
"""

# Dlib requirements.
predictor_path = 'dlib/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mouth_destination_path = os.path.dirname(args["output"]) + '/' + 'mouth'
if not os.path.exists(mouth_destination_path):
    os.makedirs(mouth_destination_path)

inputparameters = {}
outputparameters = {}
reader = skvideo.io.FFmpegReader(args["input"],
                inputdict=inputparameters,
                outputdict=outputparameters)
video_shape = reader.getShape()
(num_frames, h, w, c) = video_shape
print(num_frames, h, w, c)

# The required parameters
activation = []
max_counter = 150
total_num_frames = int(video_shape[0])
num_frames = min(total_num_frames,max_counter)
counter = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the writer
writer = skvideo.io.FFmpegWriter(args["output"])


# Required parameters for mouth extraction.
width_crop_max = 0
height_crop_max = 0


'''
Processing parameters.

    activation: set to one if the full mouth can be extracted and set to zero otherwise.
    max_counter: How many frames will be processed.
    total_num_frames: Total number of frames for the video.
    num_frames: The number of frames which are subjected to be processed.
    counter: The frame counter.
'''

"""
PART3: Processing the video.

Procedure:
     1 - Extracting each frame.
     2 - Detect the mouth in the frame.
     3 - Define a boarder around the mouth.
     4 - Crop and save the mouth.

Technical considerations:
     * - For the first frame the mouth is detected and by using a boarder the mouth is extracted and cropped.
     * - After the first frame the size of the cropped windows remains fixed unless for the subsequent frames
          a bigger windows is required. In such a case the windows size will be increased and it will be held
          fixed again unless increasing the size becoming necessary again too.
"""
# Loop over all frames.
for frame in reader.nextFrame():
    print('frame_shape:', frame.shape)

    # Process the video and extract the frames up to a certain number and then stop processing.
    if counter > num_frames:
        break

    # Detection of the frame
    detections = detector(frame, 1)

    # 20 mark for mouth
    marks = np.zeros((2, 20))

    # All unnormalized face features.
    Features_Abnormal = np.zeros((190, 1))

    # If the face is detected.
    print(len(detections))
    if len(detections) > 0:
        for k, d in enumerate(detections):

            # Shape of the face.
            shape = predictor(frame, d)

            co = 0
            # Specific for the mouth.
            for ii in range(48, 68):
                """
                This for loop is going over all mouth-related features.
                X and Y coordinates are extracted and stored separately.
                """
                X = shape.part(ii)
                A = (X.x, X.y)
                marks[0, co] = X.x
                marks[1, co] = X.y
                co += 1

            # Get the extreme points(top-left & bottom-right)
            X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),
                                                int(np.amax(marks, axis=1)[0]),
                                                int(np.amax(marks, axis=1)[1])]

            # Find the center of the mouth.
            X_center = (X_left + X_right) / 2.0
            Y_center = (Y_left + Y_right) / 2.0

            # Make a boarder for cropping.
            border = 30
            X_left_new = X_left - border
            Y_left_new = Y_left - border
            X_right_new = X_right + border
            Y_right_new = Y_right + border

            # Width and height for cropping(before and after considering the border).
            width_new = X_right_new - X_left_new
            height_new = Y_right_new - Y_left_new
            width_current = X_right - X_left
            height_current = Y_right - Y_left

            # Determine the cropping rectangle dimensions(the main purpose is to have a fixed area).
            if width_crop_max == 0 and height_crop_max == 0:
                width_crop_max = width_new
                height_crop_max = height_new
            else:
                width_crop_max += 1.5 * np.maximum(width_current - width_crop_max, 0)
                height_crop_max += 1.5 * np.maximum(height_current - height_crop_max, 0)

            # # # Uncomment if the lip area is desired to be rectangular # # # #
            #########################################################
            # Find the cropping points(top-left and bottom-right).
            X_left_crop = int(X_center - width_crop_max / 2.0)
            X_right_crop = int(X_center + width_crop_max / 2.0)
            Y_left_crop = int(Y_center - height_crop_max / 2.0)
            Y_right_crop = int(Y_center + height_crop_max / 2.0)
            #########################################################

            # # # # # Uncomment if the lip area is desired to be rectangular # # # #
            # #######################################
            # # Use this part if the cropped area should look like a square.
            # crop_length_max = max(width_crop_max, height_crop_max) / 2
            #
            # # Find the cropping points(top-left and bottom-right).
            # X_left_crop = int(X_center - crop_length_max)
            # X_right_crop = int(X_center + crop_length_max)
            # Y_left_crop = int(Y_center - crop_length_max)
            # Y_right_crop = int(Y_center + crop_length_max)
            #########################################

            if X_left_crop >= 0 and Y_left_crop >= 0 and X_right_crop < w and Y_right_crop < h:
                mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]

                # Save the mouth area.
                mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(mouth_destination_path + '/' + 'frame' + '_' + str(counter) + '.png', mouth_gray)

                print("The cropped mouth is detected ...")
                activation.append(1)
            else:
                cv2.putText(frame, 'The full mouth is not detectable. ', (30, 30), font, 1, (0, 255, 255), 2)
                print("The full mouth is not detectable. ...")
                activation.append(0)

    else:
        cv2.putText(frame, 'Mouth is not detectable. ', (30, 30), font, 1, (0, 0, 255), 2)
        print("Mouth is not detectable. ...")
        activation.append(0)


    if activation[counter] == 1:
        # Demonstration of face.
        cv2.rectangle(frame, (X_left_crop, Y_left_crop), (X_right_crop, Y_right_crop), (0, 255, 0), 2)

    # cv2.imshow('frame', frame)
    print('frame number %d of %d' % (counter, num_frames))

    # write the output frame to file
    print("writing frame %d with activation %d" % (counter + 1, activation[counter]))
    writer.writeFrame(frame)
    counter += 1

writer.close()

"""
PART4: Save the activation vector as a list.

The python script for loading a list:
    with open(the_filename, 'rb') as f:
        my_list = pickle.load(f)
"""

the_filename = os.path.dirname(args["output"]) + '/' + 'activation'
my_list = activation
with open(the_filename, 'wb') as f:
    pickle.dump(my_list, f)


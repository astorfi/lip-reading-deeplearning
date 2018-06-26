import os
from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import subprocess as sp
import numpy as np
import argparse
import random
import os
import sys
from random import shuffle
import speechpy
import datetime


######################################
####### Define the dataset class #####
######################################
class AudioDataset():
    """Audio dataset."""

    def __init__(self, files_path, audio_dir, transform=None):
        """
        Args:
            files_path (string): Path to the .txt file which the address of files are saved in it.
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # self.sound_files = [x.strip() for x in content]
        self.audio_dir = audio_dir
        self.transform = transform

        # Open the .txt file and create a list from each line.
        with open(files_path, 'r') as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        list_files = []
        for x in content:
            sound_file_path = os.path.join(self.audio_dir, x.strip().split()[1])
            try:
                with open(sound_file_path, 'rb') as f:
                    riff_size, _ = wav._read_riff_chunk(f)
                    file_size = os.path.getsize(sound_file_path)

                # Assertion error.
                assert riff_size == file_size and os.path.getsize(sound_file_path) > 1000, "Bad file!"

                # Add to list if file is OK!
                list_files.append(x.strip())
            except OSError as err:
                print("OS error: {0}".format(err))
            except ValueError:
                print('file %s is corrupted!' % sound_file_path)
            # except:
            #     print("Unexpected error:", sys.exc_info()[0])
            #     raise

        # Save the correct and healthy sound files to a list.
        self.sound_files = list_files

    def __len__(self):
        return len(self.sound_files)

    def __getitem__(self, idx):
        # Get the sound file path
        sound_file_path = os.path.join(self.audio_dir, self.sound_files[idx].split()[1])

        ##############################
        ### Reading and processing ###
        ##############################

        # Reading .wav file
        fs, signal = wav.read(sound_file_path)

        # Reading .wav file
        import soundfile as sf
        signal, fs = sf.read(sound_file_path)

        ###########################
        ### Feature Extraction ####
        ###########################

        # DEFAULTS:
        num_coefficient = 40

        # Staching frames
        frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.02,
                                                  frame_stride=0.02,
                                                  zero_padding=True)

        # # Extracting power spectrum (choosing 3 seconds and elimination of DC)
        power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=2 * num_coefficient)[:, 1:]

        logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.02, frame_stride=0.02,
                                          num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                          high_frequency=None)
        

        ########################
        ### Handling sample ####
        ########################

        # Label extraction
        label = int(self.sound_files[idx].split()[0])

        sample = {'feature': logenergy, 'label': label}

        ########################
        ### Post Processing ####
        ########################
        if self.transform:
            sample = self.transform(sample)
        else:
            feature, label = sample['feature'], sample['label']
            sample = feature, label

        return sample
        # return sample


class CMVN(object):
    """Cepstral mean variance normalization.

    """

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        # Mean variance normalization of the spectrum.
        # The following line should be Uncommented if cepstral mean variance normalization is desired!
        feature = speechpy.processing.cmvn(feature, variance_normalization=True)

        return {'feature': feature, 'label': label}

class Extract_Derivative(object):
    """
    Extract derivative features.

    """

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        # Extract derivative features
        feature = speechpy.feature.extract_derivative_feature(feature)

        return {'feature': feature, 'label': label}
    

class Feature_Cube(object):
    """Return a feature cube of desired size.

    Args:
        cube_shape (tuple): The shape of the feature cube.
    """

    def __init__(self, cube_shape):
        
        self.cube_shape = cube_shape
        if self.cube_shape != None:
            self.num_frames = cube_shape[0]
            self.num_features = cube_shape[1]
            self.num_channels = cube_shape[2]


    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']         

        if self.cube_shape != None:
            feature_cube = np.zeros((self.num_frames, self.num_features, self.num_channels), dtype=np.float32)
            feature_cube = feature[0:self.num_frames, :, :]
        else:
            feature_cube = feature
                 
        
        # return {'feature': feature_cube, 'label': label}
        return {'feature': feature_cube[None, :, :, :], 'label': label}


class ToOutput(object):
    """Return the output.

    """

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        return feature, label

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> Compose([
        >>>     CMVN(),
        >>>     Feature_Cube(cube_shape=(20, 80, 40),
        >>>     augmentation=True), ToOutput(),
        >>>        ])
        If necessary, for the details of this class, please refer to Pytorch documentation.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


if __name__ == '__main__':
    # add parser
    parser = argparse.ArgumentParser(description='Input pipeline')

    # The text file in which the paths to the audio files are available.
    # The path are relative to the directory of the audio files
    # Format of each line of the txt file is "class_label subject_dir/sound_file_name.ext"
    # Example of each line: 0 subject/sound.wav
    parser.add_argument('--file_path',
                        default=os.path.expanduser(
                            '~/github/3D-convolutional-speaker-recognition/code/0-input/file_path.txt'),
                        help='The file names for development phase')

    # The directory of the audio files separated by subject
    parser.add_argument('--audio_dir',
                        default=os.path.expanduser('~/github/lip-reading-deeplearning/code/speech-input/Audio'),
                        help='Location of sound files')
    args = parser.parse_args()

    dataset = AudioDataset(files_path=args.file_path, audio_dir=args.audio_dir,
                           transform=Compose([Extract_Derivative(), Feature_Cube(cube_shape=(15,40,3)), ToOutput()]))
    idx = 0
    feature, label = dataset.__getitem__(idx)
    print(feature.shape)
    print(label)

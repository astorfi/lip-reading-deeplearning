=====================================================================
Coupled 3D Convolutional Neural Networks for Audio-Visual Recognition
=====================================================================

This repository contains the code release for our paper titled as *"Coupled 3D
Convolutional Neural Networks for Audio-Visual Recognition"*. The link to the paper_ is
provided as well.


.. _paper: https://arxiv.org/abs/1706.05739
.. _TensorFlow: https://www.tensorflow.org/

The code has been developed using TensorFlow_. The input pipeline must be prepared by the users.
This code is aimed to provide the implementation for Coupled 3D Convolutional Neural Networks for
audio-visual matching. Lip-reading can be a specifi application for this work.


--------------
General View
--------------

*Audio-visual recognition* (AVR) has been considered as
a solution for speech recognition tasks when the audio is
corrupted, as well as a visual recognition method used
for speaker verification in multi-speaker scenarios. The approach of AVR systems is to leverage the extracted informa-
tion from one modality to improve the recognition ability of
the other modality by complementing the missing information.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Problem and the Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The essential problem is to find the correspondence between the audio and visual streams, which is the goal
of this work. **We proposed the utilization of a coupled 3D Convolutional Neural Network (CNN) architecture that can map
both modalities into a representation space to evaluate the correspondence of audio-visual streams using the learned
multimodal features**.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
How to leverage 3D Convolutional Neural Networks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proposed architecture will incorporate both spatial and temporal information jointly to
effectively find the correlation between temporal information
for different modalities. By using a relatively small network architecture and much
smaller dataset, our proposed
method surpasses the performance of the existing similar
methods for audio-visual matching which use CNNs for
feature representation. We also demonstrate that effective
pair selection method can significantly increase the performance.


--------------------
Code Implementation
--------------------

The input pipeline must be provided by the user. The rest of the implementation consider the dataset
which contains the utterance-based extracted features are stored in a ``HDF5`` file. However, this
is not a necessity because by following the code, it can be seen that the experiments can be done by
any file format as long as it is adaptable with ``TensorFlow``.


~~~~~~~~~~~
Processing
~~~~~~~~~~~

In the visual section, the videos are post-processed to have equal frame rate of 30 f/s. Then, face tracking and mouth area extraction is performed on the videos using the
dlib library [dlib]_. Finally, all mouth areas are resized to have the same size, and concatenated to form the input feature
cube. The dataset does not contain any audio files. The audio files are extracted from
videos using FFmpeg frame-work [ffmpeg]_. The processing pipeline is the below figure.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input Pipeline for this work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. image:: https://github.com/astorfi/3D-convolutional-speaker-recognition/blob/master/_images/Speech_GIF.gif
..     :target: https://github.com/astorfi/3D-convolutional-speaker-recognition/blob/master/_images/Speech_GIF.gif

The proposed architecture utilizes two non-identical ConvNets which uses a pair of speech and video
streams. The network input is a pair of features that represent lip movement and
speech features extracted from 0.3 second of a video clip. The main task is to determine if a
stream of audio corresponds with a lip motion clip within a
desired stream duration. In the two next sub-sections, we are going to explain the inputs for speech and
visual streams.


**Speech Net**


On the time axis, the temporal features are non-
overlapping 20ms windows which are used for the gen-
eration of spectrum features that possess a local characteristic.
The input speech feature map, which is represented as an image cube,
corresponds to the spectrogram
as well as the first and second order derivatives of the
MFEC features. These three channels correspond to the image depth. Collectively from a 0.3 second
clip, 15 temporal feature sets (each
forms a 40 MFEC features) can be derived which form a
speech feature cube. Each input feature map for a single audio stream has the dimensionality of 15 × 40 × 3.
This representation is depicted in the following figure:

.. image:: _images/Speech_GIF.gif

The **speech features** have been extracted using [SpeechPy]_ package.

**Visual Net**

The frame rate of each video clip used in this effort is 30 f/s.
Consequently, 9 successive image frames form the 0.3 second visual stream.
The input of the visual stream of the network is a cube of size 9x60x100,
where 9 is the number of frames that represent the temporal information. Each
channel is a 60x100 gray-scale image of mouth region.




~~~~~~~~~~~~
Architecture
~~~~~~~~~~~~

The architecture is a **coupled 3D convolutional neural network** in which *two
different networks with different sets of weights must be trained*.
For the visual network, the lip motions spatial information alongside of the temporal information are
incorporated jointly, and will be fused for exploiting the temporal
correlation. For the audio network, the extracted energy features are
considered as a spatial dimension, and the stacked audio frames form the
temporal dimension. In the proposed 3D CNN architecture, the convolutional operations
are performed on successive temporal frames for both audio-visual streams.


-----------
Disclaimer
-----------

.. _link: https://github.com/tensorflow/models/tree/master/slim

The code architecture part has been heavily inspired by Slim_ and Slim image classification
library. Please refer to this link_ for further details.

---------
Citation
---------

If you used this code please cite the following paper:

.. code:: shell

  @article{}
  }

--------
License
--------

The license is as follows:


-------------
Contribution
-------------

We are looking forward to your kind feedback. Please help us to improve the code and make
our work better. For contribution, please create the pull request and we will investigate it promptly.
Once again, we appreciate your feedback and code inspections.


.. rubric:: references

.. [SpeechPy] @misc{amirsina_torfi_2017_810392,
                    author       = {Amirsina Torfi},
                    title        = {astorfi/speech_feature_extraction: SpeechPy},
                    month        = jun,
                    year         = 2017,
                    doi          = {10.5281/zenodo.810392},
                    url          = {https://doi.org/10.5281/zenodo.810392}}

.. [dlib] D. E. King. Dlib-ml: A machine learning toolkit. Journal of Machine Learning Research, 10:1755–1758, 2009.
.. [ffmpeg] F. Developers. ffmpeg tool (version be1d324) [software], 2016.

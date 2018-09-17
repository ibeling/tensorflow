# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Creates a spectrogram Numpy array from a wav file

Once you've trained a model using the `train.py` script, you can use this tool
to convert it into a binary GraphDef file that can be loaded into the Android,
iOS, or Raspberry Pi example code. Here's an example of how to run it:

bazel run tensorflow/examples/speech_commands/freeze -- \
--sample_rate=16000 --dct_coefficient_count=40 --window_size_ms=20 \
--window_stride_ms=10 --clip_duration_ms=1000 \
--model_architecture=conv \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-1300 \
--output_file=/tmp/my_frozen_graph.pb

One thing to watch out for is that you need to pass in the same arguments for
`sample_rate` and other command line variables here as you did for the training
script.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import input_data
import models
from tensorflow.python.framework import graph_util

FLAGS = None


def create_graph(sample_rate, clip_duration_ms,
                 clip_stride_ms, window_size_ms, window_stride_ms,
                 dct_coefficient_count):
  model_settings = models.prepare_model_settings(
      10, sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)
  runtime_settings = {'clip_stride_ms': clip_stride_ms}

  wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
  decoded_sample_data = contrib_audio.decode_wav(
      wav_data_placeholder,
      desired_channels=1,
      desired_samples=model_settings['desired_samples'],
      name='decoded_sample_data')
  spectrogram = contrib_audio.audio_spectrogram(
      decoded_sample_data.audio,
      window_size=model_settings['window_size_samples'],
      stride=model_settings['window_stride_samples'],
      magnitude_squared=True)
  fingerprint_input = contrib_audio.mfcc(
      spectrogram,
      decoded_sample_data.sample_rate,
      dct_coefficient_count=dct_coefficient_count)
  fingerprint_frequency_size = model_settings['dct_coefficient_count']
  fingerprint_time_size = model_settings['spectrogram_length']
  reshaped_input = tf.reshape(fingerprint_input, [
      -1, fingerprint_time_size * fingerprint_frequency_size
  ])


  return reshaped_input, wav_data_placeholder

def main(_):
  import numpy as np

  with open(FLAGS.input_file, 'rb') as wav_file:
    wav_data = wav_file.read()

  # Create the model and load its weights.
  sess = tf.InteractiveSession()
  output, wav = create_graph(FLAGS.sample_rate,
                             FLAGS.clip_duration_ms, FLAGS.clip_stride_ms,
                             FLAGS.window_size_ms, FLAGS.window_stride_ms,
                             FLAGS.dct_coefficient_count)
  spectrogram = sess.run(output, {wav:wav_data})
  np.save(FLAGS.output_file, spectrogram)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--clip_stride_ms',
      type=int,
      default=30,
      help='How often to run recognition. Useful for models with cache.',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long the stride is between spectrogram timeslices',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  # parser.add_argument(
  #     '--start_checkpoint',
  #     type=str,
  #     default='',
  #     help='If specified, restore this pretrained model before any training.')
  # parser.add_argument(
  #     '--model_architecture',
  #     type=str,
  #     default='conv',
  #     help='What model architecture to use')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--output_file', type=str, help='Where to save the spectrogram.')
  parser.add_argument(
      '--input_file', type=str, help='Wav file to load.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

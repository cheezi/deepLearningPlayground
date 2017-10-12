import tensorflow as tf
import numpy as np
import matplotlib;
import matplotlib.pyplot as plt;
import pandas as pd
from PIL import Image
from tensorflow.python import debug as tf_debug

BATCH_SIZE = 10

def _parse_function(filename, label):
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string)
	#print(image_decoded.shape)
	image_resized = tf.image.resize_images(image_decoded, [128, 128])
	#print(image_resized.shape)
	return image_resized, label


file_name_tensor = tf.train.match_filenames_once("./train/train/*.jpg")
init = (tf.global_variables_initializer(), tf.local_variables_initializer())
l = pd.read_csv('labels.csv', usecols=[1] ,header=None)

with tf.Session() as sess:
 sess.run(init)
 file_name_list = file_name_tensor.eval()
 filenames = tf.constant(file_name_list)
 labels = tf.constant(l.values)
 dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
 dataset = dataset.map(_parse_function)
 shuffle_ds = dataset.shuffle(BATCH_SIZE)
 batch = shuffle_ds.batch(BATCH_SIZE)
 iterator = batch.make_one_shot_iterator()
 next_element = iterator.get_next()
 data = sess.run(next_element)
 print(data[1].shape)
 for i in range(BATCH_SIZE):
  plt.imshow(data[0][i]/255)
  print(data[1][i])
  plt.show()
  data = sess.run(next_element)

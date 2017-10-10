import tensorflow as tf
import numpy as np
import matplotlib;
import matplotlib.pyplot as plt;
import pandas as pd
from PIL import Image
from tensorflow.python import debug as tf_debug

def _parse_function(filename, label):
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string)
	#print(image_decoded.shape)
	image_resized = tf.image.resize_images(image_decoded, [128, 128])
	#print(image_resized.shape)
	return image_resized, label


file_name_tensor = tf.train.match_filenames_once("./train/train/*.jpg")
#image_reader = tf.WholeFileReader()
#image_name, image_file = image_reader.read(filename_queue)
#image = tf.image.decode_jpeg(image_file)
init = (tf.global_variables_initializer(), tf.local_variables_initializer())
l = pd.read_csv('labels.csv', usecols=[1] ,header=None)

with tf.Session() as sess:
	# Required to get the filename matching to run.
	sess.run(init)
	file_name_list = file_name_tensor.eval()
	filenames = tf.constant(file_name_list)
	labels = tf.constant(l.values)
	dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.map(_parse_function)
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	for index in range(10):
		data = sess.run(next_element)
		plt.imshow(data[0]/255)
		print(data[1])
		plt.show()
	# Coordinate the loading of image files.
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    #image_tensor = image.eval()
    #print(image_tensor.shape)
    #print(image_name.eval())
    #Image.fromarray(np.asarray(image_tensor)).show()

    # Finish off the filename queue coordinator.
    #coord.request_stop()
    #coord.join(threads)
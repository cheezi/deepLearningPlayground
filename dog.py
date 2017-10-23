import tensorflow as tf
import numpy as np
import matplotlib;
import matplotlib.pyplot as plt;
import pandas as pd
from PIL import Image
from tensorflow.python import debug as tf_debug

BATCH_SIZE = 10

label_lookup = ['affenpinscher','afghan_hound','african_hunting_dog','airedale','american_staffordshire_terrier','appenzeller','australian_terrier','basenji','basset','beagle','bedlington_terrier','bernese_mountain_dog','black-and-tan_coonhound','blenheim_spaniel','bloodhound','bluetick','border_collie','border_terrier','borzoi','boston_bull','bouvier_des_flandres','boxer','brabancon_griffon','briard','brittany_spaniel','bull_mastiff','cairn','cardigan','chesapeake_bay_retriever','chihuahua','chow','clumber','cocker_spaniel','collie','curly-coated_retriever','dandie_dinmont','dhole','dingo','doberman','english_foxhound','english_setter','english_springer','entlebucher','eskimo_dog','flat-coated_retriever','french_bulldog','german_shepherd','german_short-haired_pointer','giant_schnauzer','golden_retriever','gordon_setter','great_dane','great_pyrenees','greater_swiss_mountain_dog','groenendael','ibizan_hound','irish_setter','irish_terrier','irish_water_spaniel','irish_wolfhound','italian_greyhound','japanese_spaniel','keeshond','kelpie','kerry_blue_terrier','komondor','kuvasz','labrador_retriever','lakeland_terrier','leonberg','lhasa','malamute','malinois','maltese_dog','mexican_hairless','miniature_pinscher','miniature_poodle','miniature_schnauzer','newfoundland','norfolk_terrier','norwegian_elkhound','norwich_terrier','old_english_sheepdog','otterhound','papillon','pekinese','pembroke','pomeranian','pug','redbone','rhodesian_ridgeback','rottweiler','saint_bernard','saluki','samoyed','schipperke','scotch_terrier','scottish_deerhound','sealyham_terrier','shetland_sheepdog','shih-tzu','siberian_husky','silky_terrier','soft-coated_wheaten_terrier','staffordshire_bullterrier','standard_poodle','standard_schnauzer','sussex_spaniel','tibetan_mastiff','tibetan_terrier','toy_poodle','toy_terrier','vizsla','walker_hound','weimaraner','welsh_springer_spaniel','west_highland_white_terrier','whippet','wire-haired_fox_terrier','yorkshire_terrier']

def _parse_function(filename, label):
 image_string = tf.read_file(filename)
 image_decoded = tf.image.decode_jpeg(image_string)
 #print(image_decoded.shape)
 image_resized = tf.image.resize_images(image_decoded, [128, 128])
 #print(image_resized.shape)
 l = tf.one_hot(label, 120)
 return image_resized, l


file_name_tensor = tf.train.match_filenames_once("./train/*.jpg")
init = (tf.global_variables_initializer(), tf.local_variables_initializer())
l = pd.read_csv('labels.csv', usecols=[1] ,header=None)
labs = []
for s in l.values:
 i = 0
 for s2 in label_lookup:
  if s == s2:
   labs.append(i)
   break
  i = i + 1
with tf.Session() as sess:
 sess.run(init)
 file_name_list = file_name_tensor.eval()
 filenames = tf.constant(file_name_list)
 labels = tf.constant(labs)
 dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
 dataset = dataset.map(_parse_function)
 shuffle_ds = dataset.shuffle(BATCH_SIZE)
 batch = dataset.batch(BATCH_SIZE)
 iterator = batch.make_one_shot_iterator()
 next_element = iterator.get_next()
 data = sess.run(next_element)
 for i in range(1):
  plt.imshow(data[0][i]/255)
  print(data[1][i])
  #plt.show()
  data = sess.run(next_element)

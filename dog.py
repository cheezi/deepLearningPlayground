import tensorflow as tf
import time
import numpy as np
import matplotlib;
import matplotlib.pyplot as plt;
import pandas as pd
from PIL import Image
from tensorflow.python import debug as tf_debug

BATCH_SIZE = 50
EPOCHS = 5
DS_SIZE = 10000
ITERATIONS = (DS_SIZE * EPOCHS) / BATCH_SIZE

label_lookup = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier',
                'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier',
                'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
                'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer',
                'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
                'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie',
                'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound',
                'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
                'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer',
                'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog',
                'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel',
                'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
                'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute',
                'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
                'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier',
                'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug',
                'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke',
                'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu',
                'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
                'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier',
                'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
                'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']




def imgs_input_fn(file_name_list, labs, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_gray = tf.image.rgb_to_grayscale(image_decoded)
        # print(image_decoded.shape)
        image_resized = tf.image.resize_images(image_gray, [28, 28])
        image_normalized = tf.image.per_image_standardization(image_resized)
        image_reshaped = tf.reshape(image_normalized, [-1])
        # print(image_resized.shape)
        l = tf.one_hot(label, 120)
        d = image_reshaped, l
        return d
    filenames = tf.constant(file_name_list)
    labels = tf.constant(labs)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return {'image' : batch_features}, batch_labels

def my_model_fn(features, labels, mode):
    input_layer = features['image']
    logits = tf.layers.dense(inputs=input_layer, units=120)
    classes = tf.argmax(input=logits, axis=1)
    predictions = {
      "classes": classes,
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "accuracy": tf.metrics.accuracy(tf.arg_max(labels, 1),classes,name="train_accuracy")
   }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(tf.arg_max(labels, 1),classes,name="train_accuracy")
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# file_name_tensor = tf.train.match_filenames_once("./train/*.jpg")
print("read csv")
l = pd.read_csv('labels.csv', header=None)
print("shuffle index")
l_shuffled = l.reindex(np.random.permutation(l.index))
labs = []
file_name_list = []
print("build labels")
for s in l_shuffled.values:
    i = 0
    file_name_list.append("./train/" + s[0] + ".jpg")
    for s2 in label_lookup:
        if s[1] == s2:
            labs.append(i)
            break
        i = i + 1
print("session")
tf.logging.set_verbosity(tf.logging.INFO)
dog_classifier = tf.estimator.Estimator(model_fn=my_model_fn, model_dir="C:\\Users\\mjanz\\PycharmProjects\\deepLearningPlayground\\model")
tensors_to_log = {"probabilities": "softmax_tensor",
                  "accuracy": "train_accuracy"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
train_input_fn = lambda: imgs_input_fn(file_name_list, labs=labs, perform_shuffle=True,repeat_count=1, batch_size=20)
dog_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])

# with tf.Session() as sess:
#     x = tf.placeholder(tf.float32, [None, 784])
#     W = tf.Variable(tf.zeros([784, 120]))
#     b = tf.Variable(tf.zeros([120]))
#     y = tf.matmul(x, W) + b
#     y_ = tf.placeholder(tf.float32, [None, 120])
#     cross_entropy = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#     train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#     init = (tf.global_variables_initializer(), tf.local_variables_initializer())
#     print("run init")
#     sess.run(init)
#     print("run init done")
#     start = time.time()
#     j = 0
#     for i in range(EPOCHS):
#         print("run iterator.initializer")
#         sess.run(iterator.initializer)
#         print("epoch " + str(i))
#         for k in range(int(DS_SIZE / BATCH_SIZE)):
#             try:
#                 j = j + 1
#                 data, labels = sess.run(next_batch)
#                 sess.run(train_step, feed_dict={x: data,
#                                                 y_: labels})
#                 end = time.time()
#                 print("Time elapsed: " + str((end-start)) + " per Batch: " + str((end-start)/(j+1)) + " remaining: " + str(((ITERATIONS - j) * (end - start))/(j+1)) )
#             except tf.errors.OutOfRangeError:
#                 break
#         data, labels = sess.run(next_batch)
#         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         print("Accuracy: " + str(sess.run(accuracy, feed_dict={x: data,
#                                                 y_: labels})))

# for i in range(1):
#  data2D = np.squeeze(data[0][i])
#  print(data2D.shape)
#  plt.imshow(data2D, cmap='gray')
#  print(data[1][i])
#  plt.show()
#  data = sess.run(next_element)

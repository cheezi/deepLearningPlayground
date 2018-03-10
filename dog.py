import tensorflow as tf
import numpy as np
import pandas as pd
BATCH_SIZE = 50
EPOCHS = 5
DS_SIZE = 10000
IMAGE_SIZE = 112
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

def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)

    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data


def rotate_images(X_imgs):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate


def flip_images(img, label):
    X_flip = []
    #tf.reset_default_graph()
    #X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(img)
    #tf_img2 = tf.image.flip_up_down(X)
    #tf_img3 = tf.image.transpose_image(X)
    #with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #for img in X_imgs:
        #    flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
        #    X_flip.extend(flipped_imgs)
        #flipped_img = sess.run([tf_img1], feed_dict = {X: img})
    #X_flip = np.array(X_flip, dtype = np.float32)
    return tf_img1 , label


def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy

def imgs_input_fn(file_name_list, labs, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_gray = tf.image.rgb_to_grayscale(image_decoded)
        # print(image_decoded.shape)
        image_resized = tf.image.resize_images(image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
        image_normalized = tf.image.per_image_standardization(image_resized)
        image_reshaped = tf.reshape(image_normalized, [-1])
        # print(image_resized.shape)
        l = tf.one_hot(label, 10)
        d = image_normalized, l
        return d
    filenames = tf.constant(file_name_list)
    labels = tf.constant(labs)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.concatenate(dataset.map(flip_images))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return {'image' : batch_features}, batch_labels

def my_model_fn(features, labels, mode):
    input_layer = features['image']
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], data_format='channels_last', padding="same", activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[4, 4], padding="same", activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.0005))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.0005))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    pool3_flat = tf.reshape(pool3, [-1, int(IMAGE_SIZE/8)*int(IMAGE_SIZE/8)*128])
    dense = tf.layers.dense(inputs=pool3_flat, units=2048, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.0005))

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout,units=10, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.0005))
    classes = tf.argmax(input=logits, axis=1)
    predictions = {
      "classes": classes,
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
   }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    l2_loss = tf.losses.get_regularization_loss()
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits) + l2_loss
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(tf.argmax(labels, 1),classes,name="train_accuracy")
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
j = 0
for s in l_shuffled.values:
    i = 0
    if j == 10:
        break

    for s2 in label_lookup:
        if s[1] == s2:
            file_name_list.append("./train/" + s[0] + ".jpg")
            labs.append(i)
            j = j + 1
            break
        i = i + 1
        #if i == 10:
        #    break;

print(np.shape(labs))
print(np.shape(file_name_list))
DS_SIZE = np.shape(file_name_list)[0]
print(DS_SIZE)
print("session")
tf.logging.set_verbosity(tf.logging.INFO)
dog_classifier = tf.estimator.Estimator(model_fn=my_model_fn, model_dir="C:\\Users\\marcus\\PycharmProjects\\deepLearningPlayground\\model")
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
train_filenames = file_name_list[:int(DS_SIZE*0.9)]
train_labels = labs[:int(DS_SIZE*0.9)]
eval_filenames = file_name_list[int(DS_SIZE*0.9):]
eval_labels = labs[int(DS_SIZE*0.9):]
train_input_fn = lambda: imgs_input_fn(train_filenames, labs=train_labels, perform_shuffle=True,repeat_count=1, batch_size=1)
eval_input_fn = lambda: imgs_input_fn(eval_filenames, labs=eval_labels, perform_shuffle=False,repeat_count=1, batch_size=1)
#for i in range(0, 100):
dog_classifier.train(input_fn=train_input_fn, steps=20, hooks=[logging_hook])
eval_results = dog_classifier.evaluate(input_fn=eval_input_fn)
print(str(i)+": "+str(eval_results))

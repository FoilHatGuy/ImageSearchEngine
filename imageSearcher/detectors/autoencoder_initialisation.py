import os

from IPython import display

import json
import time
from . import CVAE

# import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# from tensorflow.python.ops.numpy_ops import np_config
#
# np_config.enable_numpy_behavior()
# # tf.config.run_functions_eagerly(True)

with open("../config.json") as f:
    config = json.load(f)


def generate_sample(predictions):
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('./imgs/image_sample.png')
    # plt.draw()


optimizer = tf.keras.optimizers.Adam(1e-4)

epochs = 50
start_epoch = 1
# set the dimensionality of the latent space to a plane for visualization later
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
model = CVAE.CVAE()
random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, model.latent_dim])

train_size = 1000
batch_size = 8
test_size = 200

datadir = "../data/rusdata2"
dataset_params = {"directory":  datadir,
                  "shuffle":    False,
                  "image_size": (256, 256),
                  "labels":     None,
                  "color_mode": "rgb",
                  "batch_size": batch_size}  # ,
# "validation_split": 0.15,
# "seed":             60065

# test_images = tf.keras.utils.image_dataset_from_directory(datadir).unbatch()
# images_dataset = tf.data.Dataset.from_tensors(np.array(
#         [tf.keras.utils.load_img(str(datadir + "/" + path), target_size=(256, 256)) for path in
#          os.listdir(datadir)]))  # tf.data.Dataset()
images_dataset = tf.keras.utils.image_dataset_from_directory(**dataset_params)
print("WHOLE DATASET LENGTH BEFORE AUG:", len(images_dataset))
print("WHOLE DATASET BEFORE AUGMENTATION:", images_dataset)

# images_dataset = images_dataset.repeat(5)
# print(len(images_dataset))

# def preprocess_images(images):
#     image = images.numpy()
#     print(image)
#     # images = images.reshape((images.shape[0], *dataset_params["image_size"], 1)) / 255.
#     return
#
# for test_sample in images_dataset.take(1):
#     tf.keras.utils.array_to_img(test_sample[0]).show()
#     break

images_dataset = images_dataset.cache().repeat(5).map(
        lambda image: image / 255
).map(
        lambda img: tf.image.convert_image_dtype(img, tf.float32)
).map(
        lambda image: tf.image.random_flip_left_right(image)
).map(
        lambda image: tf.image.random_contrast(image, lower=0.4, upper=1)
).shuffle(500)
# #
# for test_sample in images_dataset.take(1):
#     print(test_sample)
#     (generate_sample(test_sample))
#     break

# images_dataset = images_dataset.map(lambda x: tf.image.convert_image_dtype(x, tf.float32))

# for test_sample in images_dataset.take(1):
#     print(test_sample)
#     tf.keras.utils.array_to_img(test_sample[0]).show()
#     break

print("WHOLE DATASET:", images_dataset)

print("WHOLE LEN:", len(images_dataset))
train_count = int(len(images_dataset) * .1).__floor__()
test_dataset = images_dataset.take(train_count)
train_dataset = images_dataset.skip(train_count)

print("TEST LEN:", len(test_dataset))
print("TRAIN LEN:", len(train_dataset))

print(train_dataset)


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('./imgs/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.draw()


# Pick a sample of the test set for generating output images
# assert batch_size >= num_examples_to_generate
# print( len(test_dataset.take(1)))
# print(test_dataset[0])
for test_sample in test_dataset.take(batch_size):
    # print(test_sample)
    test_sample = test_sample[0:num_examples_to_generate + 1, :, :, :]

generate_sample(test_sample)
generate_and_save_images(model, 0, test_sample)
# plt.show(block=False)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
for epoch in range(start_epoch, epochs + 1):
    start_time = time.time()
    cnt = 0
    for train_x in train_dataset:
        cnt += 1
        if cnt % 100 == 0:
            print(cnt, "itreations complete, seconds passed:", time.time() - start_time)
        model.train_step(train_x, optimizer)
    end_time = time.time()
    print("train ended, loss calculation")
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(model.compute_loss(test_x))
        # test_sample = test_x
    elbo = -loss.result()
    # display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)
    model.encoder.save("enc.h5")
    model.decoder.save("dec.h5")

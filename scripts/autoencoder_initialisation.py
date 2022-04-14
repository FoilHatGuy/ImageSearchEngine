# from IPython import display

import glob
import os
import re

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_probability as tfp
import time
import json

with open("../config.json") as f:
    config = json.load(f)


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
                [
                        tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),
                        tf.keras.layers.Conv2D(
                                filters=32, kernel_size=3, strides=1, activation='relu', padding="same"),
                        tf.keras.layers.Conv2D(
                                filters=48, kernel_size=3, strides=1, activation='relu', padding="same"),
                        tf.keras.layers.MaxPool2D(
                                pool_size=2,
                                strides=None,
                                padding='valid',
                                data_format=None),
                        tf.keras.layers.Dropout(
                                .2, noise_shape=None, seed=13029),
                        tf.keras.layers.Conv2D(
                                filters=64, kernel_size=3, strides=2, activation='relu', padding="same"),
                        tf.keras.layers.Conv2D(
                                filters=96, kernel_size=3, strides=2, activation='relu', padding="same"),
                        tf.keras.layers.MaxPool2D(
                                pool_size=2,
                                strides=None,
                                padding='valid',
                                data_format=None
                        ),
                        tf.keras.layers.Dropout(
                                .2, seed=13000),
                        tf.keras.layers.Conv2D(
                                filters=100, kernel_size=3, strides=3, activation='relu', padding="same"),
                        # tf.keras.layers.Conv2D(
                        #         filters=100, kernel_size=4, strides=2, activation='relu', padding="same"),
                        tf.keras.layers.MaxPool2D(
                                pool_size=2,
                                strides=None,
                                padding='same',
                                data_format=None
                        ),
                        tf.keras.layers.Flatten(),
                        # No activation
                        tf.keras.layers.Dense(latent_dim + latent_dim),
                ]
        )
        # self.encoder.summary()
        # assert False

        self.decoder = tf.keras.Sequential(
                [
                        tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                        tf.keras.layers.Dense(units=2 * 2 * 100, activation=tf.nn.relu),
                        tf.keras.layers.Reshape(target_shape=(2, 2, 100)),
                        tf.keras.layers.Conv2DTranspose(  # 0
                                filters=96, kernel_size=4, strides=1, padding='same',
                                activation='relu'),
                        tf.keras.layers.Conv2DTranspose(  # 1
                                filters=72, kernel_size=4, strides=2, padding='same',
                                activation='relu'),
                        tf.keras.layers.Conv2DTranspose(  # 0
                                filters=72, kernel_size=3, strides=2, padding='same',
                                activation='relu'),
                        tf.keras.layers.UpSampling2D(  # 0
                                size=(2, 2), data_format=None, interpolation='bilinear'),
                        tf.keras.layers.Dropout(
                                .2, noise_shape=None, seed=13019),
                        tf.keras.layers.Conv2DTranspose(  # 2
                                filters=64, kernel_size=4, strides=2, padding='same',
                                activation='relu'),
                        tf.keras.layers.Conv2DTranspose(  # 3
                                filters=64, kernel_size=4, strides=2, padding='same',
                                activation='relu'),
                        tf.keras.layers.Conv2DTranspose(  # 4
                                filters=48, kernel_size=4, strides=2, padding='same',
                                activation='relu'),
                        tf.keras.layers.UpSampling2D(  # 1
                                size=(2, 2), data_format=None, interpolation='bilinear'),
                        tf.keras.layers.Dropout(
                                .2, noise_shape=None, seed=13039),
                        # No activation
                        tf.keras.layers.Conv2DTranspose(  # 5
                                filters=32, kernel_size=3, strides=1, padding='same',
                                activation='relu'),
                        tf.keras.layers.Conv2DTranspose(  # 6
                                filters=3, kernel_size=3, strides=1, padding='same'),
                ]
        )
        # self.decoder.summary()
        # assert False
        print("=================================================================")

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compute_loss(self, x):
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(
                    -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                    axis=raxis)

        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def train_step(self, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))


optimizer = tf.keras.optimizers.Adam(1e-4)

epochs = 10
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 500
num_examples_to_generate = 5

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

train_size = 1000
batch_size = 5
test_size = 200

datadir = "../data/rusdata2"
dataset_params = {"directory":  datadir,
                  "shuffle":    False,
                  "image_size": (256, 256),
                  "labels":     None,
                  "batch_size": batch_size}  # ,
# "validation_split": 0.15,
# "seed":             60065

# test_images = tf.keras.utils.image_dataset_from_directory(datadir).unbatch()
images_dataset = tf.keras.utils.image_dataset_from_directory(**dataset_params).cache().repeat(5)  # tf.data.Dataset()
# train_images = tf.keras.utils.image_dataset_from_directory(**dataset_params, subset="training")
print(len(images_dataset))
print(images_dataset)
# images_dataset = images_dataset.repeat(5)
# print(len(images_dataset))
# images_dataset = images_dataset.map(
#         lambda image: tf.image.convert_image_dtype(image, tf.float32)
# ).cache().map(lambda image: tf.image.random_flip_left_right(image)
#               ).map(lambda image: tf.image.random_contrast(image, lower=0.4, upper=.6)
#                     ).shuffle(1000)
print(images_dataset)

train_count = int(len(images_dataset)*.2).__floor__()
train_images = images_dataset.take(train_count)
test_images = images_dataset.skip(train_count)

# percentage = 85
# test_images, train_images = images_dataset.as_dataset(split=[f'test[{percentage}%:]', f'train[:{percentage}%]'])

print(len(test_images))
print(len(train_images))



# def preprocess_images(images):
#     # images = images.reshape((images.shape[0], *dataset_params["image_size"], 1)) / 255.
#     return np.where(images > .5, 1.0, 0.0).astype('float32')


# train_images = preprocess_images(train_images)
# test_images = preprocess_images(test_images)

# TODO


train_dataset = (tf.data.Dataset.from_tensors(train_images)
                 .shuffle(len(train_images)).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensors(test_images)
                .shuffle(len(test_images)).batch(batch_size))
print(train_dataset)


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# Pick a sample of the test set for generating output images
# assert batch_size >= num_examples_to_generate
# print( len(test_dataset.take(1)))
# print(test_dataset[0])
# for test_sample in test_dataset[0].take(batch_size):
#     print(test_sample)
#     # test_sample = test_batch[0]  # [0:num_examples_to_generate, :, :, :]
#
#     generate_and_save_images(model, 0, test_sample.numpy())

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        model.train_step(train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(model.compute_loss(test_x))
    elbo = -loss.result()
    # display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)

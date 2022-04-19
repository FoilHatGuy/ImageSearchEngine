import tensorflow as tf
import os
import numpy as np


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, decoder=True):
        super(CVAE, self).__init__()
        # latent_dim = kwargs["latent_dim"]
        self.latent_dim = 500
        if "enc.h5" not in os.listdir('.'):

            input_layer = tf.keras.layers.InputLayer(input_shape=(256, 256, 3))
            x = tf.keras.layers.Dropout(
                    .2, noise_shape=None, seed=13019)(input_layer)
            x = tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, activation='relu', padding="same")(x)
            x = tf.keras.layers.Conv2D(
                    filters=48, kernel_size=3, strides=1, activation='relu', padding="same")(x)
            x = tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, activation='relu', padding="same")(x)
            x = tf.keras.layers.MaxPool2D(
                    pool_size=2,
                    strides=None,
                    padding='valid',
                    data_format=None)(x)
            x = tf.keras.layers.Dropout(
                    .2, noise_shape=None, seed=13029)(x)
            x = tf.keras.layers.Conv2D(
                    filters=72, kernel_size=4, strides=2, activation='relu', padding="same")(x)
            x = tf.keras.layers.Conv2D(
                    filters=80, kernel_size=3, strides=1, activation='relu', padding="same")(x)
            x = tf.keras.layers.Conv2D(
                    filters=96, kernel_size=4, strides=2, activation='relu', padding="same")(x)
            x = tf.keras.layers.MaxPool2D(
                    pool_size=2,
                    strides=None,
                    padding='valid',
                    data_format=None
            )(x)
            x = tf.keras.layers.Dropout(
                    .2, seed=13000)(x)
            x = tf.keras.layers.Conv2D(
                    filters=126, kernel_size=4, strides=2, activation='relu', padding="same")(x)
            x = tf.keras.layers.Conv2D(
                    filters=168, kernel_size=4, strides=2, activation='relu', padding="same")(x)
            x = tf.keras.layers.Conv2D(
                    filters=200, kernel_size=3, strides=1, activation='relu', padding="same")(x)
            # tf.keras.layers.Conv2D(
            #         filters=100, kernel_size=4, strides=2, activation='relu', padding="same"),
            x = tf.keras.layers.MaxPool2D(
                    pool_size=2,
                    strides=None,
                    padding='same',
                    data_format=None
            )(x)
            x = tf.keras.layers.Dropout(
                    .2, noise_shape=None, seed=13019)(x)
            output_layer = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(self.latent_dim + self.latent_dim)(x)
            self.encoder = tf.keras.Model(input_layer, output_layer, name="encoder")

        else:
            self.encoder = tf.keras.models.load_model("enc.h5")
            
        tf.keras.utils.plot_model(self.encoder, "encoder.png", show_shapes=True)

        # self.encoder.summary()

        # assert False

        if "dec.h5" not in os.listdir('.'):
            self.decoder = tf.keras.Sequential()

            input_layer = tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))(x)
            x = tf.keras.layers.Dense(units=4 * 4 * 200, activation=tf.nn.relu)(x)
            x = tf.keras.layers.Reshape(target_shape=(4, 4, 200))(x)
            x = tf.keras.layers.Dropout(
                    .2, noise_shape=None, seed=13019)(x)
            x = tf.keras.layers.Conv2DTranspose(  # 0
                    filters=200, kernel_size=4, strides=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2DTranspose(  # 1
                    filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2DTranspose(  # 2
                    filters=96, kernel_size=4, strides=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(  # 0
                    size=(2, 2), data_format=None, interpolation='bilinear')(x)
            x = tf.keras.layers.Dropout(
                    .2, noise_shape=None, seed=13019)(x)
            x = tf.keras.layers.Conv2DTranspose(  # 3
                    filters=72, kernel_size=4, strides=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2DTranspose(  # 4
                    filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2DTranspose(  # 5
                    filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(  # 1
                    size=2, data_format=None, interpolation='bilinear')(x)
            x = tf.keras.layers.Dropout(
                    .2, noise_shape=None, seed=13039)(x)
            # No activation
            x = tf.keras.layers.Conv2DTranspose(  # 6
                    filters=48, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2DTranspose(  # 7
                    filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2DTranspose(  # 8
                    filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            output_layer = tf.keras.layers.Conv2DTranspose(  # 9
                    filters=3, kernel_size=3, strides=1, padding='same')(x)
            self.decoder = tf.keras.Model(input_layer, output_layer, name="decoder")

            tf.keras.utils.plot_model(self.decoder, "decoder.png", show_shapes=True)
    

    #
    # else:
    #     self.decoder = tf.keras.models.load_model("dec.h5")
    # self.decoder.summary()
    # assert False
    print("=================================================================")


    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)


    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar


    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean


    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


    @tf.function
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


model = CVAE()

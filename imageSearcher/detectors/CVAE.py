import tensorflow as tf
import os
import numpy as np
import pydot


def inception(input_layer, f_m=1.):
    # INCEPTION BRANCH 1: 2 divides
    b1 = tf.keras.layers.AveragePooling2D(
            pool_size=4, strides=2, padding='same', data_format=None)(input_layer)
    b1 = tf.keras.layers.Conv2D(
            filters=f_m * 64, kernel_size=3, strides=1, activation='selu', padding="same")(b1)
    b1 = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=None, padding='same', data_format=None)(b1)
    i1_outb1 = tf.keras.layers.Conv2D(
            filters=f_m * 16, kernel_size=1, strides=1, activation='selu', padding="same")(b1)
    # INCEPTION BRANCH 1 END

    # INCEPTION BRANCH 2 START 1 divide
    x = tf.keras.layers.Conv2D(
            filters=f_m * 32, kernel_size=3, strides=1, activation='selu', padding="same")(input_layer)
    i1_in = tf.keras.layers.Conv2D(
            filters=f_m * 64, kernel_size=4, strides=2, activation='selu', padding="same")(x)
    # 128x128

    # INCEPTION BRANCH 2.1: 1 divide
    b2 = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=None, padding='same', data_format=None)(i1_in)
    b2 = tf.keras.layers.Conv2D(
            filters=f_m * 16, kernel_size=1, strides=1, activation='selu', padding="same")(b2)
    i1_outb2 = tf.keras.layers.Conv2D(
            filters=f_m * 32, kernel_size=5, strides=1, activation='selu', padding="same")(b2)
    # INCEPTION BRANCH 2.1 END

    # INCEPTION BRANCH 2.2:
    b3 = tf.keras.layers.Conv2D(
            filters=f_m * 16, kernel_size=1, strides=1, activation='selu', padding="same")(i1_in)
    i1_outb3 = tf.keras.layers.Conv2D(
            filters=f_m * 32, kernel_size=4, strides=2, activation='selu', padding="same")(b3)
    # INCEPTION BRANCH 2.2 END

    i1_comb_out = tf.keras.layers.Concatenate()([i1_outb1, i1_outb2, i1_outb3])
    output_layer = tf.keras.layers.Conv2D(
            filters=f_m * 32, kernel_size=1, strides=1, activation='selu', padding="same")(i1_comb_out)
    # INCEPTION END 64x64 x 32
    return output_layer


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, decoder=True, training=False):
        super(CVAE, self).__init__()
        # latent_dim = kwargs["latent_dim"]
        self.latent_dim = 300
        if "enc.h5" not in os.listdir('.'):

            input_layer = tf.keras.Input(shape=(256, 256, 3))

            x1 = tf.keras.layers.Conv2D(
                    filters=8, kernel_size=1, strides=1, activation='selu', padding="same")(input_layer)
            x2 = tf.keras.layers.Conv2D(
                    filters=24, kernel_size=3, strides=1, activation='selu', padding="same")(input_layer)
            x = tf.keras.layers.Concatenate()([x1, x2])

            b1 = tf.keras.layers.AveragePooling2D(
                    pool_size=4, strides=None, padding='same', data_format=None)(x)

            # b2 = tf.keras.layers.Conv2D(
            #         filters=16, kernel_size=4, strides=2, activation='selu', padding="same")(x)
            # b2 = tf.keras.layers.Conv2D(
            #         filters=32, kernel_size=4, strides=2, activation='selu', padding="same")(b2)

            b3 = tf.keras.layers.Conv2D(
                    filters=32, kernel_size=2, strides=2, activation='selu', padding="same")(x)
            b3 = tf.keras.layers.Conv2D(
                    filters=64, kernel_size=2, strides=2, activation='selu', padding="same")(b3)

            comb_out = tf.keras.layers.Concatenate()([b1,
                                                      # b2,
                                                      b3])
            x = tf.keras.layers.Dropout(
                    .4, noise_shape=None, seed=13029)(comb_out)

            # 2nd
            b1 = tf.keras.layers.AveragePooling2D(
                    pool_size=4, strides=None, padding='same', data_format=None)(x)
            #
            # b2 = tf.keras.layers.Conv2D(
            #         filters=16, kernel_size=1, strides=1, activation='selu', padding="same")(x)
            # b2 = tf.keras.layers.Conv2D(
            #         filters=16, kernel_size=4, strides=2, activation='selu', padding="same")(b2)
            # b2 = tf.keras.layers.Conv2D(
            #         filters=32, kernel_size=4, strides=2, activation='selu', padding="same")(b2)

            b3 = tf.keras.layers.Conv2D(
                    filters=16, kernel_size=1, strides=1, activation='selu', padding="same")(x)
            b3 = tf.keras.layers.Conv2D(
                    filters=32, kernel_size=2, strides=2, activation='selu', padding="same")(b3)
            b3 = tf.keras.layers.Conv2D(
                    filters=48, kernel_size=2, strides=2, activation='selu', padding="same")(b3)

            comb_out = tf.keras.layers.Concatenate()([b1,
                                                      # b2,
                                                      b3])
            x = tf.keras.layers.Dropout(
                    .4, noise_shape=None, seed=13029)(comb_out)

            # 3rd
            b1 = tf.keras.layers.AveragePooling2D(
                    pool_size=4, strides=None, padding='same', data_format=None)(x)
            #
            # b2 = tf.keras.layers.Conv2D(
            #         filters=16, kernel_size=1, strides=1, activation='selu', padding="same")(x)
            # b2 = tf.keras.layers.Conv2D(
            #         filters=16, kernel_size=4, strides=2, activation='selu', padding="same")(b2)
            # b2 = tf.keras.layers.Conv2D(
            #         filters=32, kernel_size=4, strides=2, activation='selu', padding="same")(b2)

            b3 = tf.keras.layers.Conv2D(
                    filters=16, kernel_size=1, strides=1, activation='selu', padding="same")(x)
            b3 = tf.keras.layers.Conv2D(
                    filters=32, kernel_size=2, strides=2, activation='selu', padding="same")(b3)
            b3 = tf.keras.layers.Conv2D(
                    filters=64, kernel_size=2, strides=2, activation='selu', padding="same")(b3)

            comb_out = tf.keras.layers.Concatenate()([b1,
                                                      # b2,
                                                      b3])
            x = tf.keras.layers.Dropout(
                    .4, noise_shape=None, seed=13029)(comb_out)

            # 4th
            b1 = tf.keras.layers.AveragePooling2D(
                    pool_size=4, strides=None, padding='same', data_format=None)(x)
            #
            # b2 = tf.keras.layers.Conv2D(
            #         filters=16, kernel_size=1, strides=1, activation='selu', padding="same")(x)
            # b2 = tf.keras.layers.Conv2D(
            #         filters=16, kernel_size=4, strides=2, activation='selu', padding="same")(b2)
            # b2 = tf.keras.layers.Conv2D(
            #         filters=32, kernel_size=4, strides=2, activation='selu', padding="same")(b2)

            b3 = tf.keras.layers.Conv2D(
                    filters=16, kernel_size=1, strides=1, activation='selu', padding="same")(x)
            b3 = tf.keras.layers.Conv2D(
                    filters=32, kernel_size=2, strides=2, activation='selu', padding="same")(b3)
            b3 = tf.keras.layers.Conv2D(
                    filters=64, kernel_size=2, strides=2, activation='selu', padding="same")(b3)

            comb_out = tf.keras.layers.Concatenate()([b1,
                                                      # b2,
                                                      b3])
            x = tf.keras.layers.Dropout(
                    .4, noise_shape=None, seed=13029)(comb_out)

            x = tf.keras.layers.Flatten()(x)
            output_layer = tf.keras.layers.Dense(self.latent_dim + self.latent_dim)(x)
            self.encoder = tf.keras.Model(input_layer, output_layer, name="encoder")

        else:
            self.encoder = tf.keras.models.load_model("enc.h5")
            print("model loaded")

        # self.encoder.summary()
        tf.keras.utils.plot_model(self.encoder, "encoder.png", show_shapes=True)

        # assert False
        if decoder:
            if "dec.h5" not in os.listdir('.'):
                # self.decoder = tf.keras.Sequential()

                input_layer = tf.keras.Input(shape=self.latent_dim)
                x = tf.keras.layers.Dropout(
                        .1, noise_shape=None, seed=13019)(input_layer)
                x = tf.keras.layers.Dense(units=2 * 2 * 400, activation=tf.nn.relu)(x)
                cmn_inp1 = tf.keras.layers.Reshape(target_shape=(2, 2, 400))(x)

                # medium detailed block, lots of convo
                x = tf.keras.layers.Conv2DTranspose(  # 0
                        filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(cmn_inp1)
                x = tf.keras.layers.Conv2DTranspose(  # 0
                        filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 1
                        filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.UpSampling2D(  # 0
                        size=2, data_format=None, interpolation='bilinear')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 1
                        filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 1
                        filters=32, kernel_size=5, strides=2, padding='same', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 1
                        filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.UpSampling2D(  # 0
                        size=2, data_format=None, interpolation='bilinear')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 1
                        filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 1
                        filters=16, kernel_size=5, strides=2, padding='same', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 2
                        filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.UpSampling2D(  # 0
                        size=2, data_format=None, interpolation='bilinear')(x)
                b1_out = tf.keras.layers.Dropout(
                        .2, noise_shape=None, seed=13019)(x)

                # color block, large upsampling
                x = tf.keras.layers.Conv2DTranspose(  # 4
                        filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(cmn_inp1)
                x = tf.keras.layers.Conv2DTranspose(  # 4
                        filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.UpSampling2D(
                        size=4, data_format=None, interpolation='bilinear')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 4
                        filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.UpSampling2D(
                        size=4, data_format=None, interpolation='bilinear')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 5
                        filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(x)
                b2_out = tf.keras.layers.UpSampling2D(
                        size=4, data_format=None, interpolation='bilinear')(x)

                # small detailed block
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=128, kernel_size=1, strides=1, padding='same', activation='relu')(cmn_inp1)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=96, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=96, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=64, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=64, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=48, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=48, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')(x)  # 16?
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=32, kernel_size=1, strides=1, padding='valid', activation='relu')(x)
                x = tf.keras.layers.UpSampling2D(  # 1
                        size=2, data_format=None, interpolation='bilinear')(x)  # 32
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.UpSampling2D(  # 1
                        size=2, data_format=None, interpolation='bilinear')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                b3_out = tf.keras.layers.UpSampling2D(  # 1
                        size=2, data_format=None, interpolation='bilinear')(x)

                i1_comb_out = tf.keras.layers.Concatenate()([b1_out, b2_out, b3_out])

                # using acquired data
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(i1_comb_out)
                x = tf.keras.layers.UpSampling2D(  # 1
                        size=2, data_format=None, interpolation='bilinear')(x)
                x = tf.keras.layers.Dropout(
                        .2, noise_shape=None, seed=13039)(x)
                x = tf.keras.layers.Conv2DTranspose(  # 6
                        filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
                out = tf.keras.layers.Conv2DTranspose(  # 7
                        filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(x)

                # output layer
                output_layer = tf.keras.layers.Conv2DTranspose(  # 9
                        filters=3, kernel_size=3, strides=1, padding='same')(out)
                self.decoder = tf.keras.Model(input_layer, output_layer, name="decoder")

            else:
                self.decoder = tf.keras.models.load_model("dec.h5")
            # self.decoder.summary()
            tf.keras.utils.plot_model(self.decoder, "decoder.png", show_shapes=True)
            self.decoder.training = training
        self.encoder.training = training

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

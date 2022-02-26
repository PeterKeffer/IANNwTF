from copy import copy

import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Input, LeakyReLU, ReLU, Concatenate, Activation

class PatchGANDiscriminator(tf.keras.Model):

    def __init__(self):
        super().__init__()

        kernel_initializer = RandomNormal(stddev=0.02)

        self.model_layers = [
            Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(), # InstanceNormalization
            LeakyReLU(alpha=0.2),

            Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(), # InstanceNormalization
            LeakyReLU(alpha=0.2),

            Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            LeakyReLU(alpha=0.2),

            Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            LeakyReLU(alpha=0.2),

            Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            LeakyReLU(alpha=0.2),

            # Patch Layer
            Conv2D(1, kernel_size=(4, 4), padding='same', kernel_initializer=kernel_initializer)

        ]

    def call(self, x, training=True):
        for layer in self.model_layers:
            x = layer(x)

        return x


    def backward_step_discriminator(self, real_data, fake_data):

        # 1. Input real data
        output_real_data = self.call(real_data)
        loss_real_data = self.loss()

        # 2. Input fake data
        output_fake_data = self.call(fake_data)
        loss_fake_data = self.loss()

        loss_combined =  (loss_real_data + loss_fake_data) * 0.5


class ResNetBlock(tf.keras.layers.Layer):

    def __init__(self, n_filters: int):
        super().__init__()

        kernel_initializer = RandomNormal(stddev=0.02)

        self.model_layers = [
            Conv2D(n_filters, kernel_size=(3,3), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2D(n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
        ]
        self.concatenate = Concatenate()

    def call(self, x):
        input = copy(x)
        for layer in self.model_layers:
            x = layer(x)
        x = self.concatenate([x, input])

        return x

class PatchGANGenerator(tf.keras.Model):

    def __init__(self, n_resnet_blocks = 9, n_resnet_filter = 256):
        super().__init__()

        kernel_initializer = RandomNormal(stddev=0.02)

        self.layer_encoding = [
            Conv2D(64, kernel_size=(7, 7), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            ReLU(),
        ]

        self.layer_interpreting_resnet_blocks = [ResNetBlock(n_resnet_filter) for resNet_block_index in range(n_resnet_blocks)]

        self.layer_decoding = [
            Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2D(3, kernel_size=(7, 7), padding="same", kernel_initializer=kernel_initializer),
            BatchNormalization(),  # InstanceNormalization
            Activation("tanh"),
        ]

        self.model_layers = self.layer_encoding + self.layer_interpreting_resnet_blocks + self.layer_decoding

    def call(self, x, training=True):
        for layer in self.model_layers:
            x = layer(x)

        return x

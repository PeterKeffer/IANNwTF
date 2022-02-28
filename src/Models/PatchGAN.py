from copy import copy

import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Input, LeakyReLU, ReLU, Concatenate, Activation, Layer
from tensorflow_addons.layers import InstanceNormalization

class PatchGANDiscriminator(tf.keras.Model):

    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        kernel_initializer = RandomNormal(stddev=0.02)

        self.model_layers = [
            Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(), # InstanceNormalization
            LeakyReLU(alpha=0.2),

            Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(), # InstanceNormalization
            LeakyReLU(alpha=0.2),

            Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            LeakyReLU(alpha=0.2),

            Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            LeakyReLU(alpha=0.2),

            Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            LeakyReLU(alpha=0.2),

            # Patch Layer
            Conv2D(1, kernel_size=(4, 4), padding='same', kernel_initializer=kernel_initializer)

        ]

    def call(self, x, training=True):
        for layer in self.model_layers:
            x = layer(x)

        return x



class ResNetBlock(tf.keras.Model):

    def __init__(self, n_filters: int):
        super(ResNetBlock, self).__init__()

        kernel_initializer = RandomNormal(stddev=0.02)

        self.resnet_model_layers = [
            Conv2D(n_filters, kernel_size=(3,3), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2D(n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
        ]
        self.concatenate = Concatenate()


    def call(self, x, training=True):
        input = copy(x) # TODO
        for layer in self.resnet_model_layers:
            x = layer(x)
        x = self.concatenate([x, input])
        input = None
        return x



class PatchGANGenerator(tf.keras.Model):

    def __init__(self, n_resnet_blocks = 9, n_resnet_filter = 256):
        super(PatchGANGenerator, self).__init__()

        kernel_initializer = RandomNormal(stddev=0.02)

        self.layer_encoding = [
            Conv2D(64, kernel_size=(7, 7), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            ReLU(),
        ]

        self.layer_interpreting_resnet_blocks = [ResNetBlock(n_resnet_filter) for resNet_block_index in range(n_resnet_blocks)]

        self.layer_decoding = [
            Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            ReLU(),

            Conv2D(3, kernel_size=(7, 7), padding="same", kernel_initializer=kernel_initializer),
            InstanceNormalization(),  # InstanceNormalization
            Activation("tanh"),
        ]

        # TODO Hier nach Bug schauen
        self.model_layers = self.layer_encoding + self.layer_interpreting_resnet_blocks + self.layer_decoding

    def call(self, x, training=True):
        for layer in self.model_layers:
            x = layer(x)

        return x

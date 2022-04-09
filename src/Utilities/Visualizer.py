import random

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class Visualizer:

    # n zufällige bilder von dataset
    # n Listen von Bildern printen
    # output von model printen

    def denormalize(self, image):
        image = (image + 1) * 128
        return image

    def show_dataset_images(self, dataset: tf.data.Dataset, amount: int, normalized: bool = True) -> None:
        """
        selects a determined amount of randomly chosen images from our dataset

        :param normalized: is the dataset already normalized?
        :param dataset: tensorflow Dataset
        :param amount: number of images to be shown
        :return: None
        """
        print_counter = 0
        for index, batch in enumerate(dataset.take(amount)):
            batch_size = batch.shape[0]
            random_index = random.randint(0, batch_size-1)

            self.show_image(batch[random_index])

            print_counter += 1
            if print_counter >= amount:
                break


    def show_image(self, image, normalized: bool = True):
        """
        Prints given numpy array or tensorflow tensor as image

        :param image: image to be printed
        :param normalized: is this a normalized image?
        :return: None
        """
        # Check if image is black&white
        if image.shape[-1] == 1:
            image = tf.squeeze(image)
            plt.gray()

        if normalized:
            image = self.denormalize(image)

        if type(image) is not np.ndarray:
            image = image.numpy()

        image = image.astype('uint8')

        plt.imshow(image)
        plt.show()




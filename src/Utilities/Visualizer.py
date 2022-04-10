import random

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class Visualizer:

    def denormalize(self, image):
        """
        "Removes" normalization from image
        :param image: normalized image
        :return: image
        """
        image = (image + 1) * 128
        return image

    def show_dataset_images(self, dataset: tf.data.Dataset, amount: int, normalized: bool = True) -> None:
        """
        Selects a determined amount of randomly chosen images from our dataset

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


    def show_image(self, image, is_normalized: bool = True):
        """
        Prints given numpy array or tensorflow tensor as image

        :param image: image to be printed
        :param is_normalized: is this a normalized image?
        :return: None
        """
        # Check if image is black&white
        if image.shape[-1] == 1:
            image = tf.squeeze(image)
            plt.gray()

        if is_normalized:
            image = self.denormalize(image)

        if type(image) is not np.ndarray:
            image = image.numpy()

        image = image.astype('uint8')

        plt.imshow(image)
        plt.show()




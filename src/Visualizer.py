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



class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, dataset: tf.data.Dataset, num_img=4):
        self.dataset = dataset
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, images in enumerate(self.dataset.take(self.num_img)):
            prediction = self.model.generator_AtoB(images)[0].numpy()
            prediction = ((prediction + 1) * 128).astype(np.uint8)
            image = ((images[0] + 1) * 128).numpy().astype(np.uint8)

            ax[i, 0].imshow(image)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = tf.keras.preprocessing.image.array_to_img(prediction)
            prediction.save(
                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            )
        plt.show()
        plt.close()

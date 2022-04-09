import os
from copy import copy
from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from matplotlib import pyplot as plt

from Data.DataPipeline import DataPipeline


class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, dataset: tf.data.Dataset, config):
        self.dataset = dataset
        self.config = config
        self.amount_generated_example_images = config["settings"]["amount_generated_example_images"]

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(self.amount_generated_example_images, 2, figsize=(3 * self.amount_generated_example_images, 3 * self.amount_generated_example_images))
        for i, images in enumerate(self.dataset.take(self.amount_generated_example_images)):
            generated_image = self.model.generator_AtoB(images)[0].numpy()
            generated_image = ((generated_image + 1) * 128).astype(np.uint8)
            input_image = ((images[0] + 1) * 128).numpy().astype(np.uint8)

            ax[i, 0].imshow(input_image)
            ax[i, 1].imshow(generated_image)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            data_pipeline = DataPipeline()

            output_generated_image_filepath = self.config["output_data"]["output_path"] + f"/Generated_Images/generated_img_{epoch+1}_{i}.png"
            output_generated_image_directory = os.path.dirname(output_generated_image_filepath)
            Path(output_generated_image_directory).mkdir(parents=True, exist_ok=True)
            generated_image_wandb = wandb.Image(generated_image, caption="Generated_Image")

            output_input_image_filepath = self.config["output_data"]["output_path"] + f"Input_Images/input_img_{epoch+1}_{i}.png"
            output_input_image_directory = os.path.dirname(output_input_image_filepath)
            Path(output_input_image_directory).mkdir(parents=True, exist_ok=True)
            input_image_wandb = wandb.Image(input_image, caption="Input_Image")

            generated_image = tf.keras.preprocessing.image.array_to_img(generated_image)
            generated_image.save(output_generated_image_filepath)

            input_image = tf.keras.preprocessing.image.array_to_img(input_image)
            input_image.save(output_input_image_filepath)

        plt.show()
        plt.close()
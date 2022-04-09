import os
from pathlib import Path

import tensorflow as tf

from src.Data.DataPipeline import DataPipeline


class ImageGenerator(tf.keras.callbacks.Callback):

    def __init__(self, config):
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.config["settings"]["printing_epoch_interval"] == 0:
            target_shape = (self.config["output_data"]["generated_image_width"], self.config["output_data"]["generated_image_height"], self.config["input_data"]["input_image_channels"])

            data_pipeline = DataPipeline()

            image = data_pipeline.deprocess_image_tensor_vgg19(logs["generated_image"].numpy(), target_shape)

            output_filename = self.config["output_data"]["output_path"] + self.config["output_data"]["output_prefix"] + "_at_iteration_%d.png" % epoch
            output_image_directory = os.path.dirname(output_filename)
            Path(output_image_directory).mkdir(parents=True, exist_ok=True)

            tf.keras.preprocessing.image.save_img(output_filename, image)


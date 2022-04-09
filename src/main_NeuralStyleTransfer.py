import logging

import tensorflow as tf
import wandb
from tensorflow.keras.applications import vgg19
import numpy as np
from src.Data.DataPipeline import DataPipeline
from src.Models.NeuralStyleTransfer import NeuralStyleTransfer
from src.Configs.Config_NeuralStyleTransfer import ConfigNeuralStyleTransfer
from src.Utilities.Callbacks.ImageGenerator import ImageGenerator


config = ConfigNeuralStyleTransfer().get_config()

logging.basicConfig(level=config["settings"]["logging_level"])

if config["settings"]["use_wandb"]:
    wandb.init("NeuralStyleTransfer")
    wandb.config = config

data_pipeline = DataPipeline()
neural_style_transfer = NeuralStyleTransfer(config)


target_image_shape = (config["output_data"]["generated_image_width"], config["output_data"]["generated_image_height"], config["input_data"]["input_image_channels"])
input_image_shape = (config["input_data"]["input_image_width"], config["input_data"]["input_image_height"], config["input_data"]["input_image_channels"])

content_image = data_pipeline.preprocess_image_vgg19(config["input_data"]["content_image_path"], target_image_shape)
style_image = data_pipeline.preprocess_image_vgg19(config["input_data"]["style_image_path"], target_image_shape)
generated_image = tf.Variable(data_pipeline.preprocess_image_vgg19(config["input_data"]["content_image_path"], target_image_shape))

optimizer = tf.keras.optimizers.Adam(learning_rate=config["hyperparameters"]["learning_rate"])

image_generator_callback = ImageGenerator(config)

for iteration in range(1, config["hyperparameters"]["iterations"] + 1):
    logs = neural_style_transfer.train_step((generated_image, content_image, style_image), optimizer)
    # generated_image = logs["generated_image"]
    if iteration % config["hyperparameters"]["printing_epoch_interval"] == 0:
        print("Iteration %d: loss=%.2f" % (iteration, logs["total_loss"]))

    image_generator_callback.on_epoch_end(iteration, logs)
    if config["settings"]["use_wandb"]:
        wandb.log(logs)





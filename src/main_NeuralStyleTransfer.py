import datetime
import logging
from copy import copy

import tensorflow as tf
import wandb
from tensorflow.keras.applications import vgg19
import numpy as np
from Data.DataPipeline import DataPipeline
from Models.NeuralStyleTransfer import NeuralStyleTransfer
from Configs.Config_NeuralStyleTransfer import ConfigNeuralStyleTransfer
from Utilities.Callbacks.ImageGenerator import ImageGenerator
from Utilities.Visualizer import Visualizer

config = ConfigNeuralStyleTransfer().get_config()

logging.basicConfig(level=config["settings"]["logging_level"])

if config["settings"]["use_wandb"]:
    wandb.init("NeuralStyleTransfer", config=config)

data_pipeline = DataPipeline()
neural_style_transfer = NeuralStyleTransfer(config)


target_image_shape = (config["output_data"]["generated_image_width"], config["output_data"]["generated_image_height"], config["input_data"]["input_image_channels"])
input_image_shape = (config["input_data"]["input_image_width"], config["input_data"]["input_image_height"], config["input_data"]["input_image_channels"])

content_image = data_pipeline.preprocess_image_vgg19(config["input_data"]["content_image_path"], target_image_shape)
style_image = data_pipeline.preprocess_image_vgg19(config["input_data"]["style_image_path"], target_image_shape)
generated_image = tf.Variable(data_pipeline.preprocess_image_vgg19(config["input_data"]["content_image_path"], target_image_shape))

visualizer = Visualizer()

optimizer = tf.keras.optimizers.Adam(learning_rate=config["hyperparameters"]["learning_rate"])

image_generator_callback = ImageGenerator(config)

# Tensorboard Preparation
current_time = datetime.datetime.now().strftime("%Y%m%d")
train_log_dir = 'Logs/Training/' + current_time
tensorboard_summary_writer = tf.summary.create_file_writer(train_log_dir)

for iteration in range(1, config["hyperparameters"]["iterations"] + 1):
    logs = neural_style_transfer.train_step((generated_image, content_image, style_image), optimizer)

        # Tensorboard usage
    with tensorboard_summary_writer.as_default():
        tf.summary.scalar('total_loss', logs["total_loss"], step=iteration)
        tf.summary.scalar('content_loss', logs["content_loss"], step=iteration)
        tf.summary.scalar('style_loss', logs["style_loss"], step=iteration)
        if iteration % config["settings"]["printing_epoch_interval"] == 0:
            tf.summary.image("Generated Image", generated_image, step=iteration)

    if iteration % config["settings"]["printing_epoch_interval"] == 0:
        print(f"Iteration {iteration}: total loss=%.2f, style loss=%.2f, content loss=%.2f" % (logs["total_loss"], logs["style_loss"], logs["content_loss"]))

        generated_image_deprocessed = data_pipeline.deprocess_image_tensor_vgg19(copy(generated_image).numpy(), target_image_shape)
        visualizer.show_image(generated_image_deprocessed, normalized=False)

    image_generator_callback.on_epoch_end(iteration, logs)
    # Delete Generated Image from Logs
    del logs["generated_image"]

    if config["settings"]["use_wandb"]:
        wandb.log(logs)
        if iteration % config["settings"]["printing_epoch_interval"] == 0:
            generated_image_wandb = data_pipeline.deprocess_image_tensor_vgg19(copy(generated_image).numpy(), target_image_shape)
            generated_image_wandb = wandb.Image(generated_image_wandb,caption="Generated")

            content_image_wandb = data_pipeline.deprocess_image_tensor_vgg19(copy(content_image).numpy(), target_image_shape)
            content_image_wandb = wandb.Image(content_image_wandb,caption="Content")

            style_image_wandb = data_pipeline.deprocess_image_tensor_vgg19(copy(style_image).numpy(), target_image_shape)
            style_image_wandb = wandb.Image(style_image_wandb,caption="Style")

            wandb.log({"Generated_Image": generated_image_wandb, "Content_Image": content_image_wandb, "Style_Image": style_image_wandb})






import tensorflow as tf
from tensorflow.keras.applications import vgg19
import numpy as np
from Data.DataPipeline import DataPipeline
from Models.NeuralStyleTransfer import NeuralStyleTransfer
from Configs.Config_NeuralStyleTransfer import ConfigNeuralStyleTransfer
from src.Utilities.Callbacks.ImageGenerator import ImageGenerator

tf.config.run_functions_eagerly(True)


config = ConfigNeuralStyleTransfer().get_config()
data_pipeline = DataPipeline()
neural_style_transfer = NeuralStyleTransfer(config)


target_shape = (config["output_data"]["generated_image_width"], config["output_data"]["generated_image_height"], config["input_data"]["input_image_channels"])
input_shape = (config["input_data"]["input_image_width"], config["input_data"]["input_image_height"], config["input_data"]["input_image_channels"])

content_image = data_pipeline.preprocess_image_vgg19(config["input_data"]["content_image_path"], target_shape)
style_image = data_pipeline.preprocess_image_vgg19(config["input_data"]["style_image_path"], target_shape)
generated_image = tf.Variable(data_pipeline.preprocess_image_vgg19(config["input_data"]["content_image_path"], target_shape))

"""for i in range(1, config["hyperparameters"]["iterations"] + 1):
    loss, gradients = neural_style_transfer((generated_image, content_image, style_image))
    optimizer.apply_gradients([(gradients, generated_image)])


    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        image = data_pipeline.deprocess_image_tensor_vgg19(generated_image.numpy(), target_shape)
        fname = config["output_data"]["output_prefix"] + "_at_iteration_%d.png" % i
        tf.keras.preprocessing.image.save_img(fname, image)"""

optimizer = tf.keras.optimizers.Adam(learning_rate=config["hyperparameters"]["learning_rate"])

stacked_images = tf.stack([generated_image, content_image, style_image], axis=1)

image_generator_callback = ImageGenerator(100, config)

neural_style_transfer.compile(optimizer=optimizer, )
neural_style_transfer.fit(
    stacked_images,
    batch_size=1,
    epochs=4000,
    callbacks=[image_generator_callback, ])

for i in range(1, config["hyperparameters"]["iterations"] + 1):
    loss, gradients = neural_style_transfer.train_step((generated_image, content_image, style_image))
    optimizer.apply_gradients([(gradients, generated_image)])





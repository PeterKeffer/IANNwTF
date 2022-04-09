import tensorflow as tf
from tensorflow.keras.applications import vgg19
import numpy as np

class NeuralStyleTransfer(tf.keras.Model):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

        self.layer_outputs_dict = dict([(layer.name, layer.output) for layer in self.model.layers])

        self.feature_extractor = tf.keras.Model(inputs=self.model.inputs, outputs=self.layer_outputs_dict)

    def gram_matrix(self, image):
        transposed_input = tf.transpose(image, (2,0,1))
        features = tf.reshape(transposed_input, (tf.shape(transposed_input)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))

        return gram

    def style_loss(self, style_image, generated_image, image_channels = 3):
        style_gram_matrix = self.gram_matrix(style_image)
        combination_gram_matrix = self.gram_matrix(generated_image)

        generated_image_size = self.config["output_data"]["generated_image_width"] * self.config["output_data"]["generated_image_height"]

        style_loss = tf.reduce_sum(tf.square(style_gram_matrix - combination_gram_matrix)) / (4.0 * (image_channels ** 2) * (generated_image_size ** 2))

        return style_loss

    def content_loss(self, content_image, generated_image):
        content_loss = tf.reduce_sum(tf.square(generated_image - content_image))

        return content_loss

    def total_variation_loss(self, generated_image):
        a = tf.square(generated_image[:, : self.config["output_data"]["generated_image_width"] - 1, : self.config["output_data"]["generated_image_height"] - 1, :] - generated_image[:, 1:, : self.config["output_data"]["generated_image_height"] - 1, :])
        b = tf.square(generated_image[:, : self.config["output_data"]["generated_image_width"] - 1, : self.config["output_data"]["generated_image_height"] - 1, :] - generated_image[:, : self.config["output_data"]["generated_image_width"] - 1, 1:, :])

        return tf.reduce_sum(tf.pow(a + b, 1.25))

    def call(self, images):
        generated_image, content_image, style_image = images

        # Initialize the loss
        total_loss = tf.zeros(shape=())

        # Extract features of all three images
        images = tf.concat([content_image, style_image, generated_image], axis=0)
        features = self.feature_extractor(images)

        # Add content loss
        layer_features = features[self.config["hyperparameters"]["content_layer_name"]]
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        weighted_content_loss = self.config["hyperparameters"]["content_weight"] * self.content_loss(content_image_features, combination_features)
        total_loss += weighted_content_loss

        # Add style loss
        for layer_name in self.config["hyperparameters"]["style_layer_names"]:
            layer_features = features[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            layer_style_loss = self.style_loss(style_features, combination_features)
            layer_contribution_factor = 1 / len(self.config["hyperparameters"]["style_layer_names"])
            weighted_layer_style_loss = (self.config["hyperparameters"]["style_weight"] * layer_contribution_factor) * layer_style_loss
            total_loss += weighted_layer_style_loss

        # Add total variation loss
        total_loss += self.config["hyperparameters"]["total_variation_weight"] * self.total_variation_loss(generated_image)

        return total_loss

    def train_step(self, images, optimizer):
        generated_image, content_image, style_image = images

        with tf.GradientTape() as tape:
            total_loss = self.call(images)
        gradients = tape.gradient(total_loss, generated_image)

        optimizer.apply_gradients([(gradients, generated_image)])

        return {"total_loss": total_loss, "generated_image": generated_image}

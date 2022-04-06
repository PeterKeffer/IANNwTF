import tensorflow as tf
import numpy as np
from keras.applications import vgg19


class DataPipeline:

    def __init__(self):
        pass

    def pipeline(self):
        pass

    def change_dataset_image_sizes(self, images, size: tuple = (28, 28)):
        for image in images:
            image = self.change_image_size(image, size)
        return images

    def change_image_size(self, image, size: tuple):
        """

        :param image: [(batch), height, width, channels]
        :param size:
        :return:
        """
        image = tf.image.resize(image, [size[0], size[1]])
        return image

    def normalize_dataset(self, dataset: tf.data.Dataset):
        dataset = dataset.map(lambda image: self.normalize_image(image))
        return dataset

    def normalize_image(self, image):
        image = image / 128.0 - 1.0
        return image

    def reshape_dataset(self, dataset: tf.data.Dataset, size: tuple = (28, 28, 1)):
        dataset = dataset.map(lambda image: self.reshape_image(image, size))
        return dataset

    def reshape_image(self, image, size: tuple):
        image = tf.reshape(image, size)
        return image

    def convert_to_tensorflow_dataset(self, images) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(images)
        return dataset

    def batch_preparation(self, dataset: tf.data.Dataset, shuffle_size=1000, batch_size=64,
                          prefetch_size=10) -> tf.data.Dataset:
        dataset = dataset.cache().shuffle(shuffle_size).batch(batch_size).prefetch(prefetch_size)
        return dataset

    def split_dataset(self, dataset: tf.data.Dataset, test_ratio: float, validation_ratio: float) -> [tf.data.Dataset,
        tf.data.Dataset, tf.data.Dataset]:
        dataset_size = len(list(dataset))
        test_dataset_size = int(dataset_size * test_ratio)
        validation_dataset_size = int(dataset_size * validation_ratio)
        train_dataset_size = dataset_size - (test_dataset_size + validation_dataset_size)
        train_dataset = dataset.take(train_dataset_size)
        test_dataset = dataset.skip(train_dataset_size)
        validation_dataset = dataset.skip(validation_dataset_size)
        test_dataset = dataset.take(test_dataset_size)
        return train_dataset, validation_dataset, test_dataset

    def augment_data(self, image, rotation_range):
        image = tf.keras.preprocessing.image.random_rotation(image, rotation_range)
        return image

    def convert_tensor_to_image(self, image_tensor, image_width, image_height, image_channels):
        image = image_tensor.reshape((image_width, image_height, image_channels))

        return image

    def remove_zero_center_imagenet(self, image):
        image[:, :, 0] += 103.939
        image[:, :, 1] += 116.779
        image[:, :, 2] += 123.68

        return image

    def set_zero_center_imagenet(self, image):
        image[:, :, 0] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 2] -= 123.68

        return image

    def convert_bgr_to_rgb(self, image):
        image = image[:, :, ::-1]
        image = np.clip(image, 0, 255).astype("uint8")

        return image

    def preprocess_image_vgg19(self, image_path, target_image_shape):
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(target_image_shape[0], target_image_shape[1]))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = vgg19.preprocess_input(image)
        image_tensor = tf.convert_to_tensor(image)

        return image_tensor

    def deprocess_image_tensor_vgg19(self, image_tensor, image_shape):
        image = self.convert_tensor_to_image(image_tensor, image_width=image_shape[0], image_height=image_shape[1], image_channels=image_shape[2])
        image = self.remove_zero_center_imagenet(image)
        image = self.convert_bgr_to_rgb(image)

        return image


import tensorflow as tf
import numpy as np
from keras.applications import vgg19


class DataPipeline:
    """
    Toolbox for Datapreprocessing or -postprocessing
    """

    def __init__(self):
        pass

    def pipeline(self):
        pass

    def normalize_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Normalized every image of a dataset between 0 to -1
        :param dataset: Tensorflow Dataset
        :return: Changed Tensorflow Dataset
        """
        dataset = dataset.map(lambda image: self.normalize_image(image))
        return dataset

    def normalize_image(self, image):
        """
        Normalized every image of a dataset between 0 to -1
        :param image: Image
        :return: Image
        """
        image = image / 128.0 - 1.0
        return image

    def reshape_dataset(self, dataset: tf.data.Dataset, size: tuple = (28, 28, 1)):
        """
        Reshapes all images of the dataset with a given size (28, 28, 1)
        :param dataset:
        :param size:
        :return:
        """
        dataset = dataset.map(lambda image: self.reshape_image(image, size))
        return dataset

    def reshape_image(self, image, size: tuple):
        image = tf.reshape(image, size)
        return image

    def convert_to_tensorflow_dataset(self, images) -> tf.data.Dataset:
        """
        Converts array of images to TF Dataset
        :param images: array of images
        :return: Tensorflow Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices(images)
        return dataset

    def batch_preparation(self, dataset: tf.data.Dataset, shuffle_size: int=1000, batch_size: int=64,
                          prefetch_size: int=10) -> tf.data.Dataset:
        """
        Applies multiple preprocessing steps of TF to TF Dataset
        :param dataset: TF Dataset
        :param shuffle_size: Shuffle Size
        :param batch_size: Batch Size
        :param prefetch_size: Prefetch Size
        :return: TF Dataset
        """
        dataset = dataset.cache().shuffle(shuffle_size).batch(batch_size).prefetch(prefetch_size)
        return dataset

    def split_dataset(self, dataset: tf.data.Dataset, test_ratio: float, validation_ratio: float) -> [tf.data.Dataset,
        tf.data.Dataset, tf.data.Dataset]:
        """
        Splits Dataset into traun, validation and test set.
        :param dataset: TF Dataset
        :param test_ratio: ratio of the main dataset for test set: 0.0 - 1.0
        :param validation_ratio: ratio of the main dataset for validation set: 0.0 - 1.0
        :return: List of Datasets
        """
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
        """
        Rotates image(and can also be extended with other augmentations)
        :param image: Image
        :param rotation_range: how much rotation is allowed
        :return: Image
        """
        image = tf.keras.preprocessing.image.random_rotation(image, rotation_range)
        return image

    def convert_tensor_to_image(self, image_tensor, image_width, image_height, image_channels):
        """

        :param image_tensor: Image as Tensor
        :param image_width: width of the target image
        :param image_height: height of the target image
        :param image_channels: channels of the target image
        :return: Image in image format instead of "Tensor" format - but still a Tensor in it's datatype
        """
        image = image_tensor.reshape((image_width, image_height, image_channels))

        return image

    def remove_zero_center_imagenet(self, image):
        """
        Removes Zero Center of Color Channels (for ImageNet Zero Centering)
        :param image: image with shape (width, height, channels)
        :return: image
        """
        image[:, :, 0] += 103.939
        image[:, :, 1] += 116.779
        image[:, :, 2] += 123.68

        return image

    def set_zero_center_imagenet(self, image):
        """
        Sets Zero Center of Color Channels (for ImageNet as Dataset)
        :param image: image with shape (width, height, channels)
        :return: image
        """
        image[:, :, 0] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 2] -= 123.68

        return image

    def convert_bgr_to_rgb(self, image):
        """
        Converts from BGR color space to RGB and clips between 0 and 255
        :param image: image
        :return: RGB Image
        """
        image = image[:, :, ::-1]
        image = np.clip(image, 0, 255).astype("uint8")

        return image

    def preprocess_image_vgg19(self, image_path, target_image_shape):
        """
        Summary of preprocessing steps for VGG19
        :param image_path: Path of image
        :param target_image_shape: Target shape of the image
        :return: Image Tensor
        """
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(target_image_shape[0], target_image_shape[1]))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = vgg19.preprocess_input(image)
        image_tensor = tf.convert_to_tensor(image)

        return image_tensor

    def deprocess_image_tensor_vgg19(self, image_tensor, image_shape):
        """
        Summary of Deprocessing steps for VGG19 Preprocessing
        :param image_tensor: Image Tensor
        :param target_image_shape: Target shape of the image
        :return: Image
        """
        image = self.convert_tensor_to_image(image_tensor, image_width=image_shape[0], image_height=image_shape[1], image_channels=image_shape[2])
        image = self.remove_zero_center_imagenet(image)
        image = self.convert_bgr_to_rgb(image)

        return image


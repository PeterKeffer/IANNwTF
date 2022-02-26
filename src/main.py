import logging
import tensorflow as tf
import wandb

from src.Configs.Config import config
from DataPipeline import DataPipeline
from DatasetDownloader import DatasetDownloader
from Visualizer import Visualizer

wandb.init(project='IANNwTF')
wandb.config = config
logging.basicConfig(encoding='utf-8', level=logging.INFO)

dataset_downloader = DatasetDownloader(url="https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ukiyoe2photo.zip", output_filepath="Data/Downloads/ukiyoe-dataset.zip")
dataset_downloader.unpack(output_filepath="Data/")

dataset = tf.keras.utils.image_dataset_from_directory(
    "/src/Data/ukiyoe2photo/trainA",
    labels=None,
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=64,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None, subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False)


dataPipeline = DataPipeline()
dataset = dataPipeline.normalize_dataset(dataset)
train_dataset, validation_dataset, test_dataset = dataPipeline.split_dataset(dataset=dataset, test_ratio=0.2, validation_ratio=0.1)

visualizer = Visualizer()
visualizer.show_dataset_images(train_dataset, 5)
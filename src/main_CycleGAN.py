import logging
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbCallback

from src.Data.DataPipeline import DataPipeline
from src.Data.DatasetDownloader import DatasetDownloader
from src.Utilities.Visualizer import Visualizer
from src.Utilities.Callbacks.GANMonitor import GANMonitor
from src.Models.CycleGAN import CycleGAN
from src.Configs.Config_CycleGAN import ConfigCycleGAN

config = ConfigCycleGAN().get_config()

logging.basicConfig(level=config["settings"]["logging_level"])

if config["settings"]["use_wandb"]:
    wandb.init("CycleGAN", config=config)

dataset_downloader = DatasetDownloader(url=config["input_data"]["dataset_url"], output_filepath=config["input_data"]["dataset_download_filepath"])
dataset_downloader.unpack(output_filepath=config["input_data"]["dataset_download_unpack_path"])

datasetA = tf.keras.utils.image_dataset_from_directory(
    config["input_data"]["datasetA_data_path"],
    labels=None,
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=config["hyperparameters"]["batch_size"],
    image_size=(config["output_data"]["target_image_width"], config["output_data"]["target_image_height"]),
    shuffle=True,
    seed=None,
    validation_split=None, subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False)

datasetB = tf.keras.utils.image_dataset_from_directory(
    config["input_data"]["datasetB_data_path"],
    labels=None,
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=config["hyperparameters"]["batch_size"],
    image_size=(config["output_data"]["target_image_width"], config["output_data"]["target_image_height"]),
    shuffle=True,
    seed=None,
    validation_split=None, subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False)

dataPipeline = DataPipeline()

datasetA = dataPipeline.normalize_dataset(datasetA)
datasetB = dataPipeline.normalize_dataset(datasetB)

trainA_dataset, validationA_dataset, testA_dataset = dataPipeline.split_dataset(dataset=datasetA, test_ratio=config["settings"]["test_ratio"], validation_ratio=config["settings"]["validation_ratio"])
trainB_dataset, validationB_dataset, testB_dataset = dataPipeline.split_dataset(dataset=datasetB, test_ratio=config["settings"]["test_ratio"], validation_ratio=config["settings"]["validation_ratio"])

visualizer = Visualizer()

visualizer.show_dataset_images(trainA_dataset, config["settings"]["preview_amount_images"])
visualizer.show_dataset_images(trainB_dataset, config["settings"]["preview_amount_images"])

callbacks = []

ganMonitor = GANMonitor(testA_dataset, config)
callbacks.append(ganMonitor)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='Logs/', write_graph=True,
    write_images=True)
callbacks.append(tensorboard_callback)

if config["settings"]["use_wandb"]:
    callbacks.append(WandbCallback())

model = CycleGAN()
model.compile()
model.fit(
    tf.data.Dataset.zip((trainA_dataset, trainB_dataset)),
    epochs=config["hyperparameters"]["epochs"],
    callbacks=callbacks)
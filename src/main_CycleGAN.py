import logging
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbCallback

from Data.DataPipeline import DataPipeline
from Data.DatasetDownloader import DatasetDownloader
from Utilities.Visualizer import Visualizer
from Utilities.Callbacks.GANMonitor import GANMonitor
from Models.CycleGAN import CycleGAN
from Configs.ConfigCycleGAN import ConfigCycleGAN

# Get and Create Config
config = ConfigCycleGAN().get_config()

logging.basicConfig(level=config["settings"]["logging_level"])

# Init wandb(Weights & Biases)
if config["settings"]["use_wandb"]:
    wandb.init("CycleGAN", config=config)

# Downloads and unpacks Dataset
dataset_downloader = DatasetDownloader(url=config["input_data"]["dataset_url"], output_filepath=config["input_data"]["dataset_download_filepath"])
dataset_downloader.unpack(output_filepath=config["input_data"]["dataset_download_unpack_path"])


# Preprocessing
################

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

# Normalize Datasets
datasetA = dataPipeline.normalize_dataset(datasetA)
datasetB = dataPipeline.normalize_dataset(datasetB)

# Split Datasets into train, validation and test datasets
trainA_dataset, validationA_dataset, testA_dataset = dataPipeline.split_dataset(dataset=datasetA, test_ratio=config["settings"]["test_ratio"], validation_ratio=config["settings"]["validation_ratio"])
trainB_dataset, validationB_dataset, testB_dataset = dataPipeline.split_dataset(dataset=datasetB, test_ratio=config["settings"]["test_ratio"], validation_ratio=config["settings"]["validation_ratio"])

################

visualizer = Visualizer()

# Previewing just some images from both datasets
print("Preview of Train Dataset A:")
visualizer.show_dataset_images(trainA_dataset, config["settings"]["preview_amount_images"])
print("Preview of Train Dataset B:")
visualizer.show_dataset_images(trainB_dataset, config["settings"]["preview_amount_images"])

callbacks = []

# Adding GANMonitor to Callbacks
ganMonitor = GANMonitor(testA_dataset, config)
callbacks.append(ganMonitor)

# Adding Tensorboard to Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='Logs/Training', write_graph=True,
    write_images=True)
callbacks.append(tensorboard_callback)

# Adding Weights and Biases to Callbacks
if config["settings"]["use_wandb"]:
    callbacks.append(WandbCallback())


# Model init and training loop start
model = CycleGAN(config)
model.compile()
model.fit(
    tf.data.Dataset.zip((trainA_dataset, trainB_dataset)),
    epochs=config["hyperparameters"]["epochs"],
    callbacks=callbacks)
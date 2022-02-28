import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
#import wandb
#from wandb.keras import WandbCallback

from src.Configs.Config import config
from DataPipeline import DataPipeline
from DatasetDownloader import DatasetDownloader
from Visualizer import Visualizer, GANMonitor
from src.Models.CycleGAN import CycleGAN

tf.keras.backend.clear_session()

adverserial_loss = tf.keras.losses.MeanAbsoluteError()
def generator_fake_adverserial_loss(discriminator_fake_output):
    fake_loss = adverserial_loss(tf.ones_like(discriminator_fake_output), discriminator_fake_output)
    return fake_loss


def discriminator_loss(real, fake):
    real_loss = adverserial_loss(tf.ones_like(real), real)
    fake_loss = adverserial_loss(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

#wandb.init(project='IANNwTF')
#wandb.config = config
from src.Models.PatchGAN import PatchGANDiscriminator, PatchGANGenerator

logging.basicConfig(encoding='utf-8', level=logging.INFO)

dataset_downloader = DatasetDownloader(url="https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ukiyoe2photo.zip", output_filepath="Data/Downloads/ukiyoe-dataset.zip")
dataset_downloader.unpack(output_filepath="Data/")

datasetB = tf.keras.utils.image_dataset_from_directory(
    "Data/ukiyoe2photo/trainA",
    labels=None,
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=1,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None, subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False)

datasetA = tf.keras.utils.image_dataset_from_directory(
    "Data/ukiyoe2photo/trainB",
    labels=None,
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=1,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None, subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False)

#datasetA = tf.data.Dataset.from_tensor_slices([tf.random.normal([256,256,3]) for i in range(100)])
#datasetB = tf.data.Dataset.from_tensor_slices([tf.random.normal([256,256,3]) for i in range(100)])


dataPipeline = DataPipeline()

# trainA_dataset = dataPipeline.batch_preparation(datasetA)
# trainB_dataset = dataPipeline.batch_preparation(datasetB)

trainA_dataset = dataPipeline.normalize_dataset(datasetA)
trainB_dataset = dataPipeline.normalize_dataset(datasetB)

#trainA_dataset, validationA_dataset, testA_dataset = dataPipeline.split_dataset(dataset=datasetA, test_ratio=0.2, validation_ratio=0.1)
#trainB_dataset, validationB_dataset, testB_dataset = dataPipeline.split_dataset(dataset=datasetB, test_ratio=0.2, validation_ratio=0.1)

visualizer = Visualizer()

visualizer.show_dataset_images(trainA_dataset, 5)
visualizer.show_dataset_images(trainB_dataset, 5)

ganMonitor = GANMonitor(trainA_dataset)

generator_AtoB = PatchGANGenerator()
generator_BtoA = PatchGANGenerator()

discriminator_A = PatchGANDiscriminator()
discriminator_B = PatchGANDiscriminator()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

model = CycleGAN(generator_AtoB, generator_BtoA, discriminator_A, discriminator_B)
model.compile(tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5), tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5), tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5), tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5), tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.MeanAbsoluteError(), generator_fake_adverserial_loss, discriminator_loss)
model.fit(
    tf.data.Dataset.zip((trainA_dataset, trainB_dataset)),
    epochs=25,
    callbacks=[ganMonitor, ])
print("Ende")




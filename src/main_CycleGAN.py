import tensorflow as tf

from src.Data.DataPipeline import DataPipeline
from src.Data.DatasetDownloader import DatasetDownloader
from src.Utilities.Visualizer import Visualizer, GANMonitor
from src.Models.CycleGAN import CycleGAN

# logging.basicConfig(level=logging.INFO)
# wandb.init()
# wandb.config = config

dataset_downloader = DatasetDownloader(url="https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ukiyoe2photo.zip", output_filepath="Data/Downloads/ukiyoe-dataset.zip")
dataset_downloader.unpack(output_filepath="Data/")


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

dataPipeline = DataPipeline()

datasetA = dataPipeline.normalize_dataset(datasetA)
datasetB = dataPipeline.normalize_dataset(datasetB)

trainA_dataset, validationA_dataset, testA_dataset = dataPipeline.split_dataset(dataset=datasetA, test_ratio=0.2, validation_ratio=0.1)
trainB_dataset, validationB_dataset, testB_dataset = dataPipeline.split_dataset(dataset=datasetB, test_ratio=0.2, validation_ratio=0.1)

visualizer = Visualizer()

visualizer.show_dataset_images(trainA_dataset, 5)
visualizer.show_dataset_images(trainB_dataset, 5)

ganMonitor = GANMonitor(testA_dataset)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', write_graph=True,
    write_images=True)

model = CycleGAN()
model.compile()
model.fit(
    tf.data.Dataset.zip((trainA_dataset, trainB_dataset)),
    epochs=100,
    callbacks=[ganMonitor])
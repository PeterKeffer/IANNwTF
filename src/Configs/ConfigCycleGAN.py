import tensorflow as tf
import logging
from Configs.Config import Config

class ConfigCycleGAN(Config):
    config = {
        "settings": {
            "use_wandb": True,
            "logging_level": logging.INFO,

            "preview_amount_images": 5,
            "test_ratio": 0.2,
            "validation_ratio": 0.1,
            "amount_generated_example_images": 5
    },
        "input_data": {
            "dataset_url": "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ukiyoe2photo.zip",
            "dataset_download_filepath": "Data/Downloads/ukiyoe-dataset.zip",
            "dataset_download_unpack_path": "Data/",

            "datasetA_data_path": "Data/ukiyoe2photo/trainB",
            "datasetB_data_path": "Data/ukiyoe2photo/trainA",
        },
        "output_data": {
            "target_image_width": 256,
            "target_image_height": 256,

            "output_path": "Logs/CycleGAN/",
        },

        "hyperparameters": {
            "learning_rate_generator_AtoB":2e-4,
            "beta_1_generator_AtoB":0.5,
            "learning_rate_generator_BtoA":2e-4,
            "beta_1_generator_BtoA":0.5,
        
            "learning_rate_discriminator_A":2e-4,
            "beta_1_discriminator_A":0.5,
            "learning_rate_discriminator_B":2e-4,
            "beta_1_discriminator_B":0.5,

            "cycle_weight":10.0,
            "identity_weight":0.5,

            "epochs": 100,
            "batch_size": 1,

        }
    }

    def __init__(self):
        super().__init__()

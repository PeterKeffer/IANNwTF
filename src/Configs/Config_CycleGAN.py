import tensorflow as tf
from src.Configs.Config import Config

class ConfigCycleGAN(Config):
    config = {
        "settings": {
            "use_wandb": False,
            "logging_level": logging.INFO
    },
        "input_data": {

        },
        "output_data": {
        },

        "hyperparameters": {

        }
    }

    def __init__(self):
        super().__init__()

    def setup(self):

        return self.config
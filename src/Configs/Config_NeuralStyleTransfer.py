import tensorflow as tf
from Configs.Config import Config

class ConfigNeuralStyleTransfer(Config):
    config = {
        "input_data": {
            "content_image_path": tf.keras.utils.get_file(fname="content.jpg", origin="https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=1.00xw:0.669xh;0,0.190xh&resize=1200:*"),
            "style_image_path": tf.keras.utils.get_file(fname="style.jpg", origin="https://camo.githubusercontent.com/4ed1f0a52c522eadfe1b95e5555de2209fbc8b34c4b816c1b1e9c2d4277b1652/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f646f776e6c6f61642e74656e736f72666c6f772e6f72672f6578616d706c655f696d616765732f56617373696c795f4b616e64696e736b792532435f313931335f2d5f436f6d706f736974696f6e5f372e6a7067"),
            "input_image_width": None,
            "input_image_height": None,
            "input_image_channels": None

        },
        "output_data": {
            "output_path": "",
            "output_prefix": "doggo_generated",
            "generated_image_resize_factor": None,
            "generated_image_width": 400,
            "generated_image_height": None
        },

        "hyperparameters": {
            "style_layer_names": [
                                "block1_conv1",
                                "block2_conv1",
                                "block3_conv1",
                                "block4_conv1",
                                "block5_conv1",
                                ],
            "content_layer_name": "block4_conv2",
            "total_variation_weight": 1e-4,
            "style_weight": 1e-4,
            "content_weight": 1e-7,
            "learning_rate": 10.0,

            "iterations": 5000
        }
    }

    def __init__(self):
        super().__init__()

    def setup(self, config):
        config["input_data"]["input_image_width"] = tf.keras.preprocessing.image.load_img(config["input_data"]["content_image_path"]).size[0]
        config["input_data"]["input_image_height"] = tf.keras.preprocessing.image.load_img(config["input_data"]["content_image_path"]).size[1]
        config["input_data"]["input_image_channels"] = 3 if tf.keras.preprocessing.image.load_img(config["input_data"]["content_image_path"]).mode == "RGB" else 1

        config["output_data"]["generated_image_width"] = int(config["input_data"]["input_image_width"] * config["output_data"]["generated_image_resize_factor"]) if config["output_data"]["generated_image_resize_factor"] else config["output_data"]["generated_image_width"]
        config["output_data"]["generated_image_height"] = int(config["input_data"]["input_image_width"] * config["output_data"]["generated_image_width"] / config["input_data"]["input_image_height"])

        return config



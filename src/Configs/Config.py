class Config:

    config = {
        "sub_config": {
            "key": "value",
        }
    }

    def __init__(self):
        self.config = self.setup(self.config)

    def setup(self, config):

        return config

    def get_config(self):

        return self.config

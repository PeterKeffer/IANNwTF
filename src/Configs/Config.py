class Config:

    config = {
        "sub_config": {
            "key": "value",
        }
    }

    def __init__(self):
        self.config = self.setup()

    def setup(self):

        return self.config

    def get_config(self):

        return self.config

class Config:

    # Using a dictionary, to easily use experiment tracker tools like Weights&Biases (Don't like the usage of dicts, but in the end it's easier)
    config = {
        "sub_config": {
            "key": "value",
        }
    }

    def __init__(self):
        """
        Automatically runs setup() on init. (Not sure if this is a code smell.)
        """
        self.config = self.setup()

    def setup(self):
        """
        Here some dynamic settings can be calculated and saved.
        :return: config
        """

        return self.config

    def get_config(self) -> dict:
        """
        Getter for Config, so we don't need to use a global variable for config

        :return: config
        """

        return self.config

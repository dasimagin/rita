import yaml


class Config(object):
    """
    A configuration class that allows to refer parameters as object attributes
    """

    def __init__(self, config):
        """
        config: dict like config parameters
        """

        assert isinstance(config, dict), 'Required dict like object'

        for key, value in config.items():
            if isinstance(value, dict):
                value = Config(value)

            self.__dict__[key] = value

    @staticmethod
    def fromYamlFile(path):
        """
        path: path to json file
        """
        with open(path) as lines:
            return Config(yaml.safe_load(lines))


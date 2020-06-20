import pickle
from datetime import datetime
import os


class SimpleLogger:
    """Train data logger using dictionary.
    Easy to save and load.
    """
    def __init__(self, keys: list):
        """
        Params
        keys: (list) list of items to save
        """
        self._data = {k: [] for k in keys}

    def append(self, key: str, val):
        """Append a new datapoint to key.
        """
        if key not in self._data.keys():
            raise ValueError(
                "Item {} does not registered on initialization. Registered keys are {}".format(
                    key, self._data.keys()))

        self._data[key].append(val)

    def extend(self, key: str, vals: list):
        """Append a new datapoint to key.
        """
        assert isinstance(vals, (list, tuple)), "vals {}".format(vals)
        if key not in self._data.keys():
            raise ValueError(
                "Item {} does not registered on initialization. Registered keys are {}".format(
                    key, self._data.keys()))

        self._data[key].extend(vals)

    def save(self, filepath, add_timestamp=True):
        if add_timestamp:
            tms = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = "{}_{}".format(filepath, tms)
        
        if os.path.isfile(filepath):
            raise ValueError("File {} already exists".format(filepath))
            
        with open(filepath, 'wb') as f:
            pickle.dump(self._data, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, (dict,)):
            raise ValueError("Data format of {} does not corresponds to this logger class.".format(filepath))

        self._data = data

    

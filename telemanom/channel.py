import numpy as np
import os
import logging

logger = logging.getLogger('telemanom')


class Channel:
    def __init__(self, config, chan_id):
        """
        Load and reshape channel values (predicted and actual).

        Args:
            config (obj): Config object containing parameters for processing
            chan_id (str): channel id

        Attributes:
            id (str): channel id
            config (obj): see Args
            X_train (arr): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (arr): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (arr): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (arr): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (arr): train data loaded from .npy file
            test(arr): test data loaded from .npy file
        """

        self.id = chan_id
        self.config = config
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        self.train = None
        self.test = None

    def shape_data(self, arr, boundaries=None, train=True):
        data = []
        window_len = self.config.l_s + self.config.n_predictions
        for i in range(len(arr) - window_len):
            end = i + window_len
            if boundaries is not None and any(i < b < end for b in boundaries):
                continue  # skip windows that cross a file boundary
            data.append(arr[i:end])
        data = np.array(data)

        assert len(data.shape) == 3

        if train:
            np.random.shuffle(data)
            self.X_train = data[:, :-self.config.n_predictions, :]
            self.y_train = data[:, -self.config.n_predictions:, 0]
        else:
            self.X_test = data[:, :-self.config.n_predictions, :]
            self.y_test = data[:, -self.config.n_predictions:, 0]

    def load_data(self):
        try:
            self.train = np.load(os.path.join("data", "train", "{}.npy".format(self.id)))
            self.test = np.load(os.path.join("data", "test", "{}.npy".format(self.id)))

            train_b_path = os.path.join("data", "train", "_boundaries.npy")
            test_b_path = os.path.join("data", "test", "_boundaries.npy")
            train_boundaries = np.load(train_b_path) if os.path.exists(train_b_path) else None
            test_boundaries = np.load(test_b_path) if os.path.exists(test_b_path) else None

        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")

        self.shape_data(self.train, boundaries=train_boundaries)
        self.shape_data(self.test, boundaries=test_boundaries, train=False)
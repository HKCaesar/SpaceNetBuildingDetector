import os

class DataSet(Object):

    def __init__(self, images_raw3, images_raw8, labels=None):
        """
        Args:
            images: the data set images
            test: if True than create data set with test data
        """
        self._num_examples = images_raw3.shape[0]
        self._images_raw3 = images_raw3
        self._images_raw8 = images_raw8
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

def load_data_set(dataDir, testData=False):

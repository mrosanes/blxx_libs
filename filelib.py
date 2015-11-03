from specpython.filespec import FileSpec
import numpy as np


class FileData(object):
    """
    Base class to read data files.
    """

    def get_channels(self, scan_id):
        """
        :param scan_id: Scan ID
        :return: dictionary {channel_id: numpy array}
        """

        raise NotImplemented('You should implement the method.')

    def get_channel(self, channel_id, scan_id):
        """
        :param channel_id: Channel name
        :param scan_id: Scan ID
        :return: numpy array
        """
        return self.get_channels(scan_id)[channel_id]


class SpecData(FileData):
    """
    Class to get information from a spec file.
    """

    def __init__(self, filename):
        self.spec_file = FileSpec(filename)


    def get_channels(self, scan_id):
        scan = self.spec_file.getScanByNumber(scan_id)
        scan_data = scan.getData().transpose()
        labels = self.scan.getLabels()
        data = {}
        for channel_id, data in zip(labels, scan_data):
            data[channel_id] = data
        return data

class HDF5Data(FileData):
    """
    Class to get information from a hdf5 file.
    """
    pass

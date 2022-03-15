from . import exceptions as ex
from . import detectors
from . import preprocessing
import cv2 as cv

class Detector:
    def __init__(self, config):
        self.config = config
        self.preprocessing = preprocessing.imageSearchAndRestore
        self.detectors = {"AE": detectors.autoencoder.DetectorAE(),
                          "TM": detectors.template.DetectorTM(),
                          "KP": detectors.keypoint.DetectorKP()}
        self.currentDetector = self.detectors["AE"]
        print(self.currentDetector)  # AE, KM, TM

    def reindex(self):
        pass

    def add(self):
        pass

    def search(self, data):
        if data["type"] in self.detectors.keys():
            self.currentDetector = self.detectors.get(data["type"])
            image = self.preprocessing(data["image"], self.config) #self.currentDetector.process()
            # cv.imshow('Source', image)
            # print("image")
        else:
            raise ex.DetectorMissing(data["type"], self.detectors.keys())

        pass




import os
import time

from . import exceptions as ex
from . import detectors as det
from . import preprocessing
import cv2 as cv
import pandas as pd
from os.path import join as pathJoin


class ImageSearcher:
    def __init__(self, config):
        self.data = pd.read_csv(pathJoin(config["workingFolder"], "index.csv"), index_col="id").fillna('')
        # print(self.data)
        self.config = config
        self.preprocessing = preprocessing.imageSearchAndRestore
        self.detectors = {  # "AE": detectors.autoencoder.DetectorAE(self.config, self.data),
                          # "TM": detectors.template.DetectorTM(self.config, self.data),
                          "KP": det.keypoint.DetectorKP(self.config, self.data)}
        # self.currentDetector = self.detectors["AE"]
        # print(self.currentDetector)  # AE, KM, TM
        # self.reindex()

    def reindex(self):
        for val in self.detectors.values():
            val.reindex(self.data.index)
        # self.detectors["KP"].reindex(self.data.index)
        pass

    def getDets(self):
        return self.detectors.keys()

    def add(self):
        pass

    def search(self, data):
        if data["det_type"] in self.detectors.keys():
            if data["image"]:
                startingTime = time.time()
                img = cv.imread(data["image"])
                img = cv.resize(img, self.config["inputSizes"])
                # image =  self.preprocessing(data["image"], self.config) #
                # cv.imshow('Source', data["image"])
                # print(self.data)
                try:
                    try:
                        response = self.detectors.get(data["det_type"]).search(self.preprocessing(img, self.config))
                    except ex.PPError:
                        response = self.detectors.get(data["det_type"]).search(img)
                    response["desc"] = self.data.loc[response["best"]]["desc"]
                    # print(response["desc"])
                    response["name"] = self.data.loc[response["best"]]["name"]
                    response["time"] = time.time() - startingTime

                    return response
                except det.exceptions.NoMatchesFound:
                    pass
            else:
                raise ex.PPDataMissing("image not received in Module")
        else:
            raise ex.DetectorMissing(data["det_type"], self.detectors.keys())

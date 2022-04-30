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
        self.data = pd.read_csv(pathJoin(config["workingFolder"], '0', "index.csv"),
                                index_col="id").fillna('')
        # self.data["folder"] = "0"
        try:
            for dir in os.listdir(config["workingFolder"])[1:]:
                loaded_data = pd.read_csv(pathJoin(config["workingFolder"], dir, "index.csv"),
                            index_col="id").fillna('')
                # loaded_data["folder"] = str(dir)
                self.data = self.data.append(loaded_data)
        except FileNotFoundError:
            pass
        # print(self.data.columns)
        self.config = config
        self.preprocessing = preprocessing.imageSearchAndRestore
        self.detectors = {"AE": det.autoencoder.DetectorAE(self.config, self.data),
                          # "TM": det.template.DetectorTM(self.config, self.data),
                          "KP": det.keypoint.DetectorKP(self.config, self.data)}

    def reindex(self):
        for val in self.detectors.values():
            val.reindex(self.data)

    def getDets(self):
        return self.detectors.keys()

    def search(self, data):
        if data["det_type"] in self.detectors.keys():
            if data["image"]:
                startingTime = time.time()
                img = cv.imread(data["image"])
                img = cv.resize(img, self.config["inputSizes"])
                try:
                    try:
                        # raise ex.PPError
                        response = self.detectors.get(data["det_type"]).search(self.preprocessing(img, self.config))
                    except ex.PPError:
                        response = self.detectors.get(data["det_type"]).search(img)
                    # print(self.data.loc[response["best"]])
                    response["desc"] = self.data.loc[response["best"]]["desc"]
                    response["name"] = self.data.loc[response["best"]]["name"]
                    response["time"] = time.time() - startingTime
                    response["folder"] = self.data.loc[response["best"]]["folder"]
                    return response
                except det.exceptions.NoMatchesFound:
                    pass
            else:
                raise ex.PPDataMissing("image not received in Module")
        else:
            raise ex.DetectorMissing(data["det_type"], self.detectors.keys())

    def add(self):
        pass


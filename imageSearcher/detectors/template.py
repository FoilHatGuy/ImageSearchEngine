# from imageSearcher.detectors import logger
import os
from . import exceptions as ex
import pandas as pd
import cv2 as cv
import numpy as np
import sys


class DetectorTM:
    def __init__(self, config, data):

        self.config = config
        self.db = pd.DataFrame()

        setname = self.config["workingFolder"].split("/")[-1]
        self.csvName = os.path.normpath(os.path.join(self.config["cwd"],
                                                     self.config["localDBFolder"],
                                                     setname + "_TM.pkl"))
        try:
            self.db = pd.read_pickle(self.csvName)
        except FileNotFoundError:
            self.reindex(data)
        self.db = self.db.set_index("id")
        # self.db.applymap(lambda x: x.astype(np.uint8))
        print("TM:")
        self.db.info(memory_usage="deep")

    def reindex(self, data):
        cnt = 0
        new_data = []
        for id, data in data.iterrows():
            # print(data['folder'])
            img = cv.imread(os.path.normpath(os.path.join(self.config["cwd"],
                                                          self.config["workingFolder"],
                                                          str(data['folder']),
                                                          id + ".jpeg")))

            new_data.append({"id": id, "img": self.prepare(img)})
            cnt += 1
        new_data = pd.DataFrame(new_data, columns=["id", "img"])
        # self.db.applymap(lambda x: x.astype(np.uint8))
        new_data.to_pickle(self.csvName)
        self.db = new_data
        self.db = self.db.set_index("id")
        print("TM Reindex done!")
        print("TM:")
        self.db.info(memory_usage="deep")

    def search(self, img):
        img = self.prepare(img)
        batch = self.findInDB(img)
        # print("batch", batch)
        return {  # "bestBatch": list([x[0] for x in batch[:self.config["numOfBest"]]]),
                "best":       batch[0][0],
                "confidence": batch[0][1]}

    def findInDB(self, query):
        conf_grade = sorted(list(self.db.applymap(lambda x: self.match(query, x))
                                 .itertuples(name=None)), key=lambda x: x[1], reverse=True)
        # print("KP:", conf_grade)
        return conf_grade

    def prepare(self, img):
        # cv.normalize(img, None, alpha=255, norm_type=cv.NORM_MINMAX)
        img = cv.resize(img, (128, 128))
        # cv.imshow("RGB", img)
        img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
        # cv.imshow("HSV", img2)
        # cv.imshow("S", img2[:, :, 0])
        # cv.imshow("H", img2[:, :, 1])
        # cv.imshow("V", img2[:, :, 2])
        # cv.waitKey()
        # img2 = cv.equalizeHist(img2)
        return [img2[:, :, 0], img2[:, :, 2]]

    def match(self, query_img, img2):
        # print(query_img)
        # print(img2)
        matchH = cv.matchTemplate(img2[0], query_img[0], cv.TM_CCOEFF_NORMED)
        matchS = cv.matchTemplate(img2[1], query_img[1], cv.TM_CCOEFF_NORMED)
        # print(matchH)
        # print(matchS)

        return matchH * matchS

    def add(self):
        pass

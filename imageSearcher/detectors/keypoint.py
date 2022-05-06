# from imageSearcher.detectors import logger
import os
from . import exceptions as ex
import pandas as pd
import cv2 as cv
import numpy as np
import sys


class DetectorKP:
    def __init__(self, config, data):
        self.kpDescriptor = cv.ORB_create()
        self.config = config
        self.db = pd.DataFrame()

        setname = self.config["workingFolder"].split("/")[-1]
        self.csvName = os.path.normpath(os.path.join(self.config["cwd"],
                                                     self.config["localDBFolder"],
                                                     setname+"_KP.pkl"))
        try:
            self.db = pd.read_pickle(self.csvName)
        except FileNotFoundError:
            self.reindex(data)
        self.db = self.db.set_index("id")
        # self.db.applymap(lambda x: x.astype(np.uint8))
        self.db.info(memory_usage="deep")

        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_LSH = 6
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.matcher = cv.FlannBasedMatcher(index_params, search_params)

    def reindex(self, data):
        cnt = 0
        new_data = []
        for id, data in data.iterrows():
            # print(data['folder'])
            img = cv.imread(os.path.normpath(os.path.join(self.config["cwd"],
                                                          self.config["workingFolder"],
                                                          str(data['folder']),
                                                          id + ".jpeg")))

            new_data.append({"id": id, "des": self.searchKP(img)})
            cnt += 1
        new_data = pd.DataFrame(new_data, columns=["id", "des"])
        self.db.applymap(lambda x: x.astype(np.uint8))
        new_data.to_pickle(self.csvName)
        self.db = new_data
        print("KP Reindex done!")

    def search(self, img):
        batch = self.findInDB(self.searchKP(img))
        # print("batch", batch)
        return {#"bestBatch": list([x[0] for x in batch[:self.config["numOfBest"]]]),
                "best": batch[0][0],
                "confidence": batch[0][1]}

    def findInDB(self, des1):
        conf_grade = sorted(list(self.db.applymap(lambda x: self.KPMatch(des1, x))
                                 .itertuples(name=None)), key=lambda x: x[1], reverse=True)
        # print("KP:", conf_grade)
        return conf_grade

    def searchKP(self, img):
        # cv.normalize(img, None, alpha=255, norm_type=cv.NORM_MINMAX)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.equalizeHist(img)
        kp = self.kpDescriptor.detect(img, None)
        return self.kpDescriptor.compute(img, kp)[1].astype(np.uint8)  # des .astype(np.uint8)

    def KPMatch(self, des1, des2):
        cnt = 0
        # print(des1)
        # print(des2)
        # knn_matches = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        knn_matches = self.matcher.knnMatch(des1, des2, k=2)

        for i, pair in enumerate(knn_matches):
            if len(pair) < 2:
                continue
            else:
                m, n = pair
                if m.distance < 0.6 * n.distance:
                    cnt += 1
        confidence = (cnt / len(knn_matches)) if len(knn_matches) > 0 else 0
        return confidence

    def add(self):
        pass

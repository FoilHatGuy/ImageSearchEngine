import tensorflow
import pandas as pd
import numpy as np
import os
import cv2 as cv


class DetectorAE:
    def __init__(self, config, data):
        pass
        # self.model = None
        #
        # self.config = config
        # self.db = pd.DataFrame()
        # self.csvName = os.path.normpath(os.path.join(self.config["cwd"], self.config["localDBFolder"], "AE.pkl"))
        # try:
        #     self.db = pd.read_pickle(self.csvName)
        # except FileNotFoundError:
        #     self.reindex(data.index)
        # self.db = self.db.set_index("id")
        # # self.db.applymap(lambda x: x.astype(np.uint8))
        # self.db.info(memory_usage="deep")


    def reindex(self, data):
        cnt = 0
        new_data = []
        for id in data:
            img = cv.imread(os.path.normpath(os.path.join(self.config["cwd"],
                                                          self.config["workingFolder"],
                                                          id + ".jpeg")))

            new_data.append({"id": id, "des": self.searchKP(img)})
            cnt += 1
        new_data = pd.DataFrame(new_data, columns=["id", "des"])
        self.db.applymap(lambda x: x.astype(np.uint8))
        new_data.to_pickle(self.csvName)
        self.db = new_data

    def search(self, img):
        batch = self.findInDB(self.searchKP(img))
        return {"bestBatch": list([x[1] for x in batch[:self.config["numOfBest"]]]),
                "best": batch[0][0]}

    def findInDB(self, des1):
        conf_grade = sorted(list(self.db.applymap(lambda x: self.KPMatch(des1, x))
                                 .itertuples(name=None)), key=lambda x: x[1], reverse=True)
        return conf_grade

    def searchKP(self, img):
        # cv.normalize(img, None, alpha=255, norm_type=cv.NORM_MINMAX)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.equalizeHist(img)
        kp = self.kpDescriptor.detect(img, None)
        return self.kpDescriptor.compute(img, kp)[1].astype(np.uint8)  # des .astype(np.uint8)

    def KPMatch(self, des1, des2):
        cnt = 0
        knn_matches = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        for i, (m, n) in enumerate(knn_matches):
            if m.distance < .8 * n.distance:
                cnt += 1
        confidence = (cnt / len(knn_matches)) if len(knn_matches) > 0 else 0
        return confidence


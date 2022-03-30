# from imageSearcher.detectors import logger
import os
from . import exceptions as ex
import pandas as pd
import cv2 as cv
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt


class DetectorKP:
    def __init__(self, config, data):
        self.kpDescriptor = cv.ORB_create()
        self.config = config
        self.db = pd.DataFrame()
        self.csvName = os.path.normpath(os.path.join(self.config["cwd"], self.config["localDBFolder"], "KP.pkl"))
        try:
            self.db = pd.read_pickle(self.csvName)
        except FileNotFoundError:
            self.reindex(data.index)
            # raise ex.ReindexNeededError("reindex launching for KP detector")
        self.db = self.db.set_index("id")
        # print(self.db["des"].apply(print))
        # self.db["des"] = self.db["des"].apply(np.fromstring)
        # print(self.db["des"].apply(np.typename))
        # self.db[["des"]] = self.db[["des"]].applymap(literal_eval)
        self.db.applymap(lambda x: x.astype(np.float32))

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.matcher = cv.FlannBasedMatcher(index_params, search_params)

    def search(self, img):
        batch = self.findInDB(self.searchKP(img))
        # print(batch)
        # print(len(batch))
        # if batch is not None:
        return {"bestBatch": list([x[1] for x in batch[:self.config["numOfBest"]]]),
                "best": batch[0][0]}
        # else:
        #     raise ex.NoMatchesFound("KP method haven't found any match")

    def searchKP(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp = self.kpDescriptor.detect(img, None)
        return self.kpDescriptor.compute(img, kp)[1].astype(np.float32)  # des

    def findInDB(self, des1):
        # print(list(self.db.iterrows()))
        # conf_grade = list(map(lambda line: (line.index, self.KPMatch(des1, line["des"])).sorted(lambda x: x[1]),
        #                       self.db.iterrows()))
        # print(self.db.applymap(lambda x: x).itertuples(name=None))
        # self.db.itertuples()
        conf_grade = sorted(list(self.db.applymap(lambda x: self.KPMatch(des1, x))
                                 .itertuples(name=None)), key=lambda x: x[1], reverse=True)
        # print("conf_grade\n", conf_grade)
        # img_id = ""
        return conf_grade

    def KPMatch(self, des1, des2):
        cnt = 0
        # print(des1)
        # print(des2)
        # print(np.type(np.array(des1)))
        # print(np.type(des2))
        # des2 = des2["des"]
        knn_matches = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        # matchesMask = [[0, 0] for i in range(len(knn_matches))]
        for i, (m, n) in enumerate(knn_matches):
            if m.distance < .8 * n.distance:
                # matchesMask[i] = [1, 0]
                cnt +=1
        # print(cnt)
        # print(len(knn_matches))
        confidence = (cnt / len(knn_matches)) if len(knn_matches) > 0 else 0
        # print(confidence)
        return confidence

    def add(self):
        pass

    def reindex(self, data):
        # print("data in reindex:", data)
        cnt = 0
        new_data = []
        for id in data:
            # print(id)
            img = cv.imread(os.path.normpath(os.path.join(self.config["cwd"],
                                                          self.config["workingFolder"],
                                                          id+".jpeg")))
            # print(os.path.normpath(os.path.join(self.config["cwd"],
            #                                               self.config["workingFolder"],
            #                                               id+".jpeg")))

            new_data.append({"id": id, "des": self.searchKP(img)})
            # print(cnt)
            cnt += 1
        # print(new_data)
        new_data = pd.DataFrame(new_data, columns=["id", "des"])
        new_data.to_pickle(self.csvName)
        # new_data = new_data.set_index("id")
        # print(new_data)
        self.db = new_data

# image = cv.imread("../../result-3.jpg")
# inst = DetectorKP()
# inst.process(image)
# cv.waitKey()

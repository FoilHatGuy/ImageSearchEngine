import tensorflow
import pandas as pd
import numpy as np
import os
import cv2 as cv
from . import CVAE


class DetectorAE:
    def __init__(self, config, data):
        self.searching = False

        self.model = CVAE.CVAE(decoder=False, training=False)

        self.config = config
        self.db = pd.DataFrame()
        setname = self.config["workingFolder"].split("/")[-1]
        self.csvName = os.path.normpath(os.path.join(self.config["cwd"],
                                                     self.config["localDBFolder"],
                                                     setname+"_AE.pkl"))

        try:
            self.db = pd.read_pickle(self.csvName)
            # self.db = self.db.set_index("id")
            # self.reindex(data)
        except Exception:
            self.reindex(data)
        # self.db.applymap(lambda x: x.astype(np.uint8))
        print(self.db)

    def reindex(self, data):
        cnt = 0
        new_data = []
        # print(data["folder"].iteritems())
        for idx, data in data.iterrows():
            # print(idx, data)
            img = cv.imread(os.path.normpath(os.path.join(self.config["cwd"],
                                                          self.config["workingFolder"],
                                                          str(data['folder']),
                                                          idx + ".jpeg")))

            # print(id, folder)
            # print(os.path.normpath(os.path.normpath(os.path.join(self.config["cwd"],
            #                                               self.config["workingFolder"],
            #                                               str(folder),
            #                                               id + ".jpeg"))))
            # print(img)
            # print("img")
            params = self.encode(img)
            # if idx == "5c3e0bd393fa687ca4df9156":
            #     self.searching = False
            #     cv.imshow("in db", img)
            #     cv.waitKey()
            #     print(params)
            new_data.append({"id": idx, "par": params})

            cnt += 1
        new_data = pd.DataFrame(new_data, columns=["id", "par"]).set_index("id")
        # print(list(new_data.loc["5c3e0bd393fa687ca4df9156"]))
        # self.db.applymap(lambda x: (x.numpy()*255).astype(np.uint8))
        new_data.to_pickle(self.csvName)
        self.db = new_data
        print("REINDEX AE DONE")

    def search(self, img):
        self.searching = True
        # input_id = "5c3e0a6393fa687ca4cade53" #input("id: ")
        params = self.encode(img)
        # with open("ids.txt", "w") as f:
            # p
        # print(list(self.db.loc[input_id]))
        # print(params)
        batch = self.findInDB(params)
        # print(batch)
        # print(list(self.db.loc[batch[0][0]]))
        self.searching = False
        return {#"bestBatch": list([x[0] for x in batch[:self.config["numOfBest"]]]),
                "best":      batch[0][0],
                "confidence": batch[0][1]}

    def findInDB(self, des1):
        # print(self.db)
        # print(des1)
        # print(self.db.loc["5c3e0bd393fa687ca4df9156"])

        f = self.db.applymap(lambda x: x-des1)
        # print(f)
        # r = f.applymap(lambda x: np.linalg.norm(x*(x-np.min(x) > np.mean(x)-np.min(x)*0.3), 2))
        r = f.applymap(lambda x: np.linalg.norm(x, 2))
        # print(r)
        # print(list(r.itertuples(name=None)))
        conf_grade = sorted(list(r.itertuples(name=None)), key=lambda x: x[1], reverse=False)

        # print("AE:", conf_grade)

        # print(des1)
        # print(self.db.loc[conf_grade[0][0]])
        return conf_grade

    def encode(self, img):
        # cv.normalize(img, None, alpha=255, norm_type=cv.NORM_MINMAX)
        img = np.resize((cv.resize(img, (256, 256)) / 255).astype("float32"), (1, 256, 256, 3))
        # print(img.shape)
        # if self.searching:
        #     print(img[0].shape)
        #     cv.imshow("name", img[0])
        #     cv.waitKey()
        parameters, _ = self.model.encode(img)
        return parameters.numpy().flatten()

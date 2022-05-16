import tensorflow
import pandas as pd
import numpy as np
import os
import cv2 as cv
from . import CVAE


class DetectorAE:
    def __init__(self, config, data):
        self.model = CVAE.CVAE(decoder=False, training=False)

        self.config = config
        self.db = pd.DataFrame()
        setname = self.config["workingFolder"].split("/")[-1]
        self.csvName = os.path.normpath(os.path.join(self.config["cwd"],
                                                     self.config["localDBFolder"],
                                                     setname + "_AE.pkl"))

        try:
            self.db = pd.read_pickle(self.csvName)
            # self.db = self.db.set_index("id")
            # self.reindex(data)
        except Exception:
            self.reindex(data)
        # self.db.applymap(lambda x: x.astype(np.uint8))
        print("AE:")
        self.db.info(memory_usage="deep")
        # print(self.db)

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
        # self.db.applymap(lambda x: (x.numpy()).astype(np.int16))
        new_data.to_pickle(self.csvName)
        self.db = new_data
        print("AE Reindex done!")
        print("AE:")
        self.db.info(memory_usage="deep")

    def search(self, img):
        params = self.encode(img)
        batch = self.findInDB(params)

        # import matplotlib.pyplot as plt
        # import matplotlib.colors as col
        #
        # print(os.path.normpath(os.path.join(self.config["cwd"],
        #                    self.config["workingFolder"],
        #                    "4",
        #                    batch[0][0] + ".jpeg")).replace("\\", "/"))
        #
        #
        # img1 = None
        # folder = 0
        # while img1 is None:
        #     img1 = cv.imread(os.path.normpath(os.path.join(self.config["cwd"],
        #                                                              self.config["workingFolder"],
        #                                                              str(folder),
        #                                                              batch[0][0] + ".jpeg")))
        #     folder += 1
        #     if folder > 10:
        #         break
        # # print(img1)
        # img2 = None
        # folder = 0
        # while img2 is None:
        #     img2 = cv.imread(os.path.normpath(os.path.join(self.config["cwd"],
        #                                                              self.config["workingFolder"],
        #                                                              str(folder),
        #                                                              batch[10][0] + ".jpeg")))
        #     folder += 1
        #     if folder > 10:
        #         break
        # img1 = cv.cvtColor(cv.resize(img1, (256, 256)), cv.COLOR_BGR2RGB)
        # img = cv.cvtColor(cv.resize(img, (256, 256)), cv.COLOR_BGR2RGB)
        # img2 = cv.cvtColor(cv.resize(img2, (256, 256)), cv.COLOR_BGR2RGB)
        # # fig = plt.figure()
        # plt.subplot(3, 3, 1)
        # plt.imshow(img1)
        # plt.subplot(3, 3, 4)
        # plt.imshow(img)
        # plt.subplot(3, 3, 7)
        # plt.imshow(img2)
        #
        #
        # plt.subplot(3, 3, 2)
        # params1 = self.db.loc[batch[0][0]]["par"]
        # print(params1)
        # print(params)
        # plt.imshow(np.resize(params1, (15, 20)), vmin=-1, vmax=1)
        # plt.subplot(3, 3, 5)
        # plt.imshow(np.resize(params, (15, 20)), vmin=-1, vmax=1)
        # plt.subplot(3, 3, 8)
        # params2 = self.db.loc[batch[10][0]]["par"]
        # plt.imshow(np.resize(params2, (15, 20)), vmin=-1, vmax=1)
        #
        # plt.subplot(2, 3, 3)
        # plt.imshow(np.resize(params1 - params, (15, 20)), vmin=-1, vmax=1)
        # plt.subplot(2, 3, 6)
        # plt.imshow(np.resize(params2 - params, (15, 20)), vmin=-1, vmax=1)
        #
        # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        # plt.colorbar(cax=cax)
        # plt.show()
        # cv.waitKey()

        return {  # "bestBatch": list([x[0] for x in batch[:self.config["numOfBest"]]]),
                "best":       batch[0][0],
                "confidence": batch[0][1]}

    def findInDB(self, des1):
        # print(self.db)
        # print(des1)
        # print(self.db.loc["5c3e0bd393fa687ca4df9156"])

        f = self.db.applymap(lambda x: x - des1)
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
        img = np.resize((cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (256, 256)) / 255).astype("float32"),
                        (1, 256, 256, 3))
        # print(img.shape)
        # if self.searching:
        #     print(img[0].shape)
        #     cv.imshow("name", img[0])
        #     cv.waitKey()
        parameters, _ = self.model.encode(img)
        return parameters.numpy().flatten()

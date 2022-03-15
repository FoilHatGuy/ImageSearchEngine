# from imageSearcher.detectors import logger
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt


class DetectorKP:
    def __init__(self):
        try:
            self.db = pd.read_csv("KP.csv")
        except FileNotFoundError:
            pass
        self.kpDescriptor = cv.ORB_create()

    def search(self, image):
        pass

    def searchKP(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp = self.kpDescriptor.detect(img, None)
        return self.kpDescriptor.compute(img, kp)  # kp, des

    def process(self, image):
        pass

    def add(self):
        pass

    def reindex(self):
        pass


image = cv.imread("../../result-3.jpg")
inst = DetectorKP()
inst.process(image)
cv.waitKey()

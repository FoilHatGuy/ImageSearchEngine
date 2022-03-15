import imageSearcher
import imageSearcher.exceptions as ex
import flask
import logging
import cv2 as cv
import datetime
import base64
import os
import re
import sys
from datetime import datetime

from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"


class FlaskServer:
    app = flask.Flask(__name__)

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        self.searchEngine = imageSearcher.detector.Detector(config)

        self.app.run(host=config['host'])

    def tester(self, query):
        try:
            self.searchEngine.search(query)
        except ex.InterfaceError as e:
            self.logger.exception(e.message)
        # self.app = flask.Flask(__name__)
        pass

    @app.route("/", methods=["POST"])
    def upload_info(self):
        name = int(datetime.utcnow().timestamp() * 1000000)
        # print(name)
        # assert request.files['upload']
        f = request.files['file']
        # print(str(f))
        f.save(f'{C.PREROOT}imgs/{name}.{f.filename.split(".")[1]}')
        imgs = [str(base64.b64encode(open(f"{C.PREROOT}imgs/{C.IMG_SRC}/src_imgs/{f}", "rb").read()))[2:-1] for f in
                os.listdir(f'{C.PREROOT}imgs/{C.IMG_SRC}/src_imgs') if re.match(".*\.jpeg", f)]
        catalogue = [f.split('.')[0] for f in os.listdir(f'{C.PREROOT}imgs/{C.IMG_SRC}/src_imgs') if
                     re.match(".*\.jpeg", f)]
        # catalogue.encode('utf-8')
        result, time, img_name, desc, src = keras_part.model.predict(name, f.filename.split(".")[1])
        if desc != '':
            desc = 'no description'
        # name = '5c3e1a0093fa687ca4ad3401.jpeg'
        return render_template('result.html', temp=src, img_name=img_name, desc=desc, pics=imgs, ids=catalogue,
                               result_route=f'/{C.IMG_SRC}/src_imgs/{result}', time=time, IMG_SRC=C.IMG_SRC)


parameters = {
    "PP-resultSizes": [240, 240],
    "workingFolder": "./data/testData",
    "host": 'localhost'
}


if __name__ == "__main__":
    img = cv.imread("3.jpg")
    query = {"type": "AE", "image": img}
    instance = FlaskServer(parameters)
    instance.tester(query)
    cv.waitKey()

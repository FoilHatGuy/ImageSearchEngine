import datetime
import random
import sys
from datetime import datetime
import json
import flask
from flask import render_template
from flask import request
from flask import send_from_directory
from ServerDetectorInterface import ServerDetectorInterface
import cv2 as cv
import os
app = flask.Flask(__name__)


@app.route("/", methods=["POST"])
def upload_info():
    response = instance.search(request)

    return render_template('result.html',
                           # status=response["status"],
                           src_file=response["src_file"],
                           img_name=response["name"],
                           desc=response["desc"],
                           # pics=instance.imgs,
                           # ids=instance.catalogue,
                           result_file=response["result_file"],
                           time=response["time"],
                           IMG_SRC=instance.config['workingFolder'],
                           detectors=instance.getDets())


@app.route("/")
def hello():
    # catalogue.encode('utf-16')
    return render_template('start.html',
                           # pics=instance.imgs,
                           # ids=instance.catalogue,
                           IMG_SRC=instance.config['workingFolder'],
                           detectors=instance.getDets())

# @app.route('/<path:path>')
# def send_js(path):
#     print(path)
#     return send_from_directory('../imgs', path)


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    instance = ServerDetectorInterface(config)
    img = cv.imread("3.jpg")
    query = {"type": "AE", "image": img}
    app.config["UPLOAD_FOLDER"] = config["workingFolder"]
    app.run(host=instance.config['host'])
    instance.search(query)
    cv.waitKey()

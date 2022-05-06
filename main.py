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
def img_load():
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


@app.route("/testing", methods=["POST"])
def testing():
    # print(request)
    # return {"I": "Love you"}
    response = instance.search(request, testing=True)
    return str(response)


@app.route("/check", methods=["GET", "POST"])
def check():
    return request

@app.route("/reindex", methods=["GET"])
def reindex():
    instance.reindex()
    return flask.redirect("/")


@app.route("/")
def index():
    # catalogue.encode('utf-16')
    return render_template('start.html',
                           # pics=instance.imgs,
                           # ids=instance.catalogue,
                           IMG_SRC=instance.config['workingFolder'],
                           detectors=instance.getDets())


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    instance = ServerDetectorInterface(config)
    app.config["UPLOAD_FOLDER"] = config["workingFolder"]
    app.run(host=instance.config['host'])


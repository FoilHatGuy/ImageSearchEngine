import imageSearcher
import pandas as pd
import imageSearcher.exceptions as ex
import logging
import base64
import os
import datetime
import random
import pathlib
import re
from os import sep as s
# from appMethods import *


class ServerDetectorInterface:
    def __init__(self, config):  # image data storage
        self.config = config  # importing config from outer scope
        self.config["cwd"] = os.getcwd()
        self.logger = logging.getLogger()  # logger initialisation
        self.searchEngine = imageSearcher.imageSearcher.ImageSearcher(config)  # imageSearcher instancing
        # self.imgs = [str(base64.b64encode(open(f"{self.config['workingFolder']}{s}{f}",
        #                                        "rb").read()))[2:-1]
        #              for f in os.listdir(self.config['workingFolder']) if re.match(".*\.jpeg", f)]
        # self.catalogue = [f.split('.')[0] for f in os.listdir(self.config['workingFolder']) if
        #                   re.match(".*\.jpeg", f)]
        # print(self.catalogue)
        #
        # self.cleanup_data()  # database cleanup, no picture - no data

    def add(self):
        pass

    def reindex(self):
        self.searchEngine.reindex()

    def getDets(self):
        return self.searchEngine.getDets()

    def search(self, request, testing=False):
        # JSON = request.get_json()
        # print(request.form)
        name = int(datetime.datetime.utcnow().timestamp() * 1000000) + random.randint(0, 1000)
        f = request.files['file']
        response = {}
        if f:
            filename = os.path.normpath(os.path.join(self.config["cwd"],
                                                     self.config["tempFolder"],
                                                     f"{name}.jpeg"))  #{f.filename.split('.')[1]}
            f.save(filename)
            # print(filename)
            response = self.searchEngine.search({"image": filename, **request.form})  # respond with {id: ""}
            # print("response", response["folder"])
            # print("normpath:", os.path.normpath(
            #             os.path.join(
            #                 self.config["cwd"],
            #                 self.config["workingFolder"],
            #                 str(response["folder"]),
            #                 f"{response['best']}.jpeg")
            #         ))
            if not testing:
                response["src_file"] = str(base64.b64encode(
                    open(
                        os.path.normpath(
                            os.path.join(
                                self.config["cwd"], self.config["tempFolder"], filename
                            )
                        ), "rb").read()))[2:-1]
                response["result_file"] = str(base64.b64encode(
                    open(
                        os.path.normpath(
                            os.path.join(
                                self.config["cwd"],
                                self.config["workingFolder"],
                                str(response["folder"]),
                                f"{response['best']}.jpeg")
                        ), "rb").read()))[2:-1]
                if len(response["desc"]) < 5:
                    response["desc"] = 'no description'
            # os.remove(os.path.normpath(os.path.join(self.config["cwd"], self.config["tempFolder"], filename)))
            print(response["best"], response["name"], response["time"], response["confidence"])
            return response
        else:
            return {}  #make exception raise bc no file

    def tester(self, query):
        try:
            self.searchEngine.search(query)
        except ex.InterfaceError as e:
            self.logger.exception(e.message)
        # self.app = flask.Flask(__name__)
        pass

    def cleanup_data(self):

        # for dir in os.listdir(self.config["workingFolder"]):
        dir = ""
        data = pd.read_csv(os.path.join(self.config["workingFolder"], dir, "index.csv"), index_col="id")
        dirContents = os.listdir(os.path.join(self.config["workingFolder"], dir))
        print(dir)
        # print(dir_contents)

        for line_id, line in data.iterrows():
            # print(line.index())
            filename = line_id + ".jpeg"
            if filename not in dirContents:
                data = data.drop(line_id)

        data.to_csv(os.path.join(self.config["workingFolder"], dir, "index.csv"))
        print(data)



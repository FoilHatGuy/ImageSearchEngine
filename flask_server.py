import base64
import os
import re
import sys
from datetime import datetime

from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory

sys.path.insert(0, os.getcwd())
from scripts import CONST as C
# from  import *
from server import keras_part

app = Flask(__name__)


# base_url = "/api"


# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         f = request.files['the_file']
#         f.save('/var/www/uploads/uploaded_file.txt')


@app.route("/")
def hello():
    imgs = [str(base64.b64encode(open(f"{C.PREROOT}imgs/{C.IMG_SRC}/src_imgs/{f}", "rb").read()))[2:-1] for f in os.listdir(f'{C.PREROOT}imgs/{C.IMG_SRC}/src_imgs') if re.match(".*\.jpeg", f)]
    catalogue = [f.split('.')[0] for f in os.listdir(f'{C.PREROOT}imgs/{C.IMG_SRC}/src_imgs') if re.match(".*\.jpeg", f)]
    # catalogue.encode('utf-16')
    return render_template('start.html', pics=imgs, ids=catalogue, IMG_SRC=C.IMG_SRC)


@app.route('/<path:path>')
def send_js(path):
    print(path)
    return send_from_directory('../imgs', path)


# @app.route("/result")
# # def hello():
# #     return """
# #   <form action="/result"
# # enctype="multipart/form-data" method="post">
# # <p>
# # Please specify a file, or a set of files:<br>
# # <input type="file" name="datafile" size="40">
# # </p>
# # <div>
# # <input type="submit" value="Send">
# # </div>
# # </form>"""


@app.route("/", methods=["POST"])
def upload_info():
    name = int(datetime.utcnow().timestamp() * 1000000)
    # print(name)
    # assert request.files['upload']
    f = request.files['file']
    # print(str(f))
    f.save(f'{C.PREROOT}imgs/{name}.{f.filename.split(".")[1]}')
    imgs = [str(base64.b64encode(open(f"{C.PREROOT}imgs/{C.IMG_SRC}/src_imgs/{f}", "rb").read()))[2:-1] for f in os.listdir(f'{C.PREROOT}imgs/{C.IMG_SRC}/src_imgs') if re.match(".*\.jpeg", f)]
    catalogue = [f.split('.')[0] for f in os.listdir(f'{C.PREROOT}imgs/{C.IMG_SRC}/src_imgs') if re.match(".*\.jpeg", f)]
    # catalogue.encode('utf-8')
    result, time, img_name, desc, src = keras_part.model.predict(name, f.filename.split(".")[1])
    if desc != '':
        desc = 'no description'
    # name = '5c3e1a0093fa687ca4ad3401.jpeg'
    return render_template('result.html', temp=src, img_name=img_name, desc=desc, pics=imgs, ids=catalogue,
                           result_route=f'/{C.IMG_SRC}/src_imgs/{result}', time=time, IMG_SRC=C.IMG_SRC)


if __name__ == "__main__":
    app.run(host='localhost')

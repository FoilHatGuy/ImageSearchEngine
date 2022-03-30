import pandas as pd
import json
import os

with open("config.json") as f:
    config = json.load(f)

data = pd.read_csv(config["workingFolder"]+"/index.csv", index_col="id")
dirContents = os.listdir(config["workingFolder"])
print(data)

for line_id, line in data.iterrows():
    # print(line.index())
    filename = line_id + ".jpeg"
    if filename not in dirContents:
        data = data.drop(line_id)

data.to_csv(config["workingFolder"]+"/index.csv")
print(data)

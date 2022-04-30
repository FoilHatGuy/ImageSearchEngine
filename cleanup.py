import pandas as pd
import json
import os

with open("config.json") as f:
    config = json.load(f)

for dir in os.listdir(config["workingFolder"]):
    data = pd.read_csv(os.path.join(config["workingFolder"], dir, "index.csv"), index_col="id")
    dir_contents = os.listdir(os.path.join(config["workingFolder"], dir))
    # print(data)
    print(dir_contents)

    for line_id, line in data.iterrows():
        # print(line.index())
        filename = line_id + ".jpeg"
        if filename not in dir_contents:
            data = data.drop(line_id)

    data.to_csv(os.path.join(config["workingFolder"],dir,"index.csv"))
    print(data)

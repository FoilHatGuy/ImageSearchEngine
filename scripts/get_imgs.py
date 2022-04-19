import os
import shutil

import pandas as pd
import requests
import re
from PIL import Image

SIZE = (500, 500)
path = '../data/rusdata/'
for dir in [int(f) for f in os.listdir(path) if os.path.isdir(path + f) and re.match(r'\d*', f)][1:]:
    print(dir)
    df = pd.read_csv(path+str(dir)+'/index.csv')
    # print(df.head())
    for x in list(df.iterrows()):
        # print(x[1])
        # print(x[1]['url'])
        r = requests.get(x[1]['url'], stream=True)
        print(df.index.size, r.headers)
        # print()
        if 'Content-Type' in r.headers:
            try:
                print(int(r.headers['Content-Length']) == 0)
                if int(r.headers['Content-Length']) == 0:
                    print(x[1]['id'])
                    df = df.drop(labels=x[1]['id'], axis=0)

                    continue
            except:
                # print(df.index.size)
                pass
            # print(r.headers["Content-Length"])
            with open(path+f"{str(dir)}/{x[1]['id']}.jpeg", 'wb') as f: #{r.headers['Content-Type'].split('/')[1]}
                r.raw.decode_content = True
                with open(path+'tmp.jpeg', 'w+b') as tmp:
                    shutil.copyfileobj(r.raw, tmp)
                    img = Image.open(tmp).convert('RGB')

                    quantifier = min(SIZE[0] / max(SIZE[0], img.size[0]),
                                     SIZE[1] / max(SIZE[1], img.size[1]))
                    # print(quantifier)
                    sizes = int(img.size[0] * quantifier), int(img.size[1] * quantifier)
                    img.resize(sizes).save(f)
                os.remove(path+ 'tmp.jpeg')
    df.to_csv(path+str(dir)+'/index.csv')



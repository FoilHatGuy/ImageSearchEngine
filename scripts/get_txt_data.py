import pandas as pd
import requests
import os
import re
import sys
path = '../data/rusmuseum/'
NUMBER = 1000
headers = {'Accept': 'application/json',
           'X-API-KEY': 'd02270983aea10827ccd92e7fa874b03f556784ebff02839faa71bdc713a3490'}
initial_url = 'https://opendata.mkrf.ru/v2/museum-exhibits/$'
url = initial_url
data = {"eventType": "AAS_PORTAL_START", "data": {"uid": "hfe3hf45huf33545", "aid": "1", "vid": "1"}}
values = {'f': '{"data.museum.name":{"$search":"Государственный русский музей"},"data.typology.name":{"$search":"Живопись"}}', 'l': str(NUMBER)}
stop = False
if 'last.txt' not in os.listdir('.'):
    response = requests.get(url, params=values, headers=headers).json()
else:
    with open('last.txt', 'r') as f:
        response = requests.get(f.read(), headers=headers).json()

files = [int(f) for f in os.listdir('../data') if
         os.path.isdir(path + f) and re.match(r'\d*', f)]
print(files)
# num = max(files)+1 if len(files) > 0 else 0
num = 0
while not stop and num >= 0:
    df = pd.DataFrame(columns=['id', 'name', 'artist', 'url', 'height', 'width', 'desc', 'folder'])
    try:
        os.mkdir(path+str(num))
    except:
        pass

    if 'nextPage' in response.keys():
        url = response['nextPage']
    df.set_index('id')
    for item in response['data']:
        # print(item['data'].keys())
        # print(item.keys())
        if 'images' in item['data'] and 'url' in item['data']['images'][0]:
            df = df.append({'id': item['_id'], 'url': item['data']['images'][0]['url'],
                            'artist': item['data']['authors'][0] if 'authors' in item['data'].keys() else None,
                            'name': item['data']['name'] if 'name' in item['data'].keys() else None,
                            'desc': item['data']['description'] if 'description' in item['data'].keys() else None,
                            'height': item['data']['height'] if 'height' in item['data'].keys() else None,
                            'width': item['data']['width'] if 'width' in item['data'].keys() else None,
                            'folder': num},
                           ignore_index=True)
    print(df)
    # stop = True
    print(num)
    df.to_csv(path+str(num)+'/index.csv', index=False)
    if len(df)==0:
        break
    with open('last.txt', 'w') as f:
        f.write(url)
    response = requests.get(url, headers=headers).json()
    num += 1

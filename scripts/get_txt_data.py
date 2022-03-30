import pandas as pd
import requests
import os
import re
import sys

NUMBER = 1000
headers = {'Accept': 'application/json',
           'X-API-KEY': 'd02270983aea10827ccd92e7fa874b03f556784ebff02839faa71bdc713a3490'}
initial_url = 'https://opendata.mkrf.ru/v2/museum-exhibits/$'
url = initial_url
data = {"eventType": "AAS_PORTAL_START", "data": {"uid": "hfe3hf45huf33545", "aid": "1", "vid": "1"}}
values = {'f': '{"data.museum.name":{"$search":"русский"},"data.typology.name":{"$search":"Живопись"}}', 'l': str(NUMBER)}
stop = False
if 'last.txt' not in os.listdir('../scripts'):
    response = requests.get(url, params=values, headers=headers).json()
else:
    with open('last.txt', 'r') as f:
        response = requests.get(f.read(), headers=headers).json()

files = [int(f) for f in os.listdir('../data') if
         os.path.isdir('../rusdata/' + f) and re.match(r'\d*', f)]
print(files)
num = max(files)+1 if len(files) > 0 else 0

while not stop and num > -1:
    df = pd.DataFrame(columns=['id', 'name', 'artist', 'url', 'height', 'width', 'desc'])
    os.mkdir('../rusdata/'+str(num))

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
                            'width': item['data']['width'] if 'width' in item['data'].keys() else None},
                           ignore_index=True)
    # print(df)
    # stop = True
    print(num)
    df.to_csv('../rusdata/'+str(num)+'/index.csv', index=False)
    with open('last.txt', 'w') as f:
        f.write(url)
    response = requests.get(url, headers=headers).json()
    num += 1
    if len(df)==0:
        break

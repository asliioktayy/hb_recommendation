import json

import pandas as pd


with open('./data/events.json','r') as f:
    event_data = json.loads(f.read())
# Flatten data
events = pd.json_normalize(event_data, record_path =['events'])

with open('./data/meta.json','r') as f:
    meta_data = json.loads(f.read())
# Flatten data
meta = pd.json_normalize(meta_data, record_path =['meta'])
events = events.merge(meta, on='productid')

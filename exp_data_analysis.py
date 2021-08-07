import json

import pandas as pd
import plotly as py

from get_data import events, meta


events.info()

events["eventtime"] = pd.to_datetime(events["eventtime"])
events['price'] = events['price'].astype(float)

events.isnull().sum()

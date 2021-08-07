import json
import os 
import re
from string import punctuation

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from get_data import events, meta

nltk.download("stopwords")
pd.options.display.max_colwidth = 20000


events["brand"] = events["brand"].fillna("eksik bilgi")
events.drop(columns=['event', 'eventtime', 'sessionid'], inplace=True)

events['price'] = events['price'].astype(float)
#events['price_cat'] = events.groupby(['category'])['price'].apply(lambda x: pd.qcut(x, 3, labels=["ucuz", "normal", "pahalı"]))
#events['price_cat'] = events['price_cat'].astype(str)
events['name2']=events['name']

regular_expression = '[' + re.escape (''. join (list(punctuation))) + ']'
numb='[1234567890]'
stopw_turkish = stopwords.words('turkish')
stop_turkish = r'\b(?:{})\b'.format('|'.join(stopw_turkish))

cols = ['name', 'category', 'subcategory', 'brand']

for col in cols:
    events[col] = events[col].astype(str)
    events[col] = events[col].str.lower()
    events[col] = events[col].str.replace(stop_turkish, '')
    events[col] = events[col].str.replace(numb, '', regex=True)
    events[col] = events[col].str.replace(regular_expression, '', regex=True)
    events[col] = events[col].str.strip()


# remove word from name if less than 3 letter like 'li', 'ml'
events['name'] = events['name'].str.replace(r'\b(\w{1,2})\b', '')

def create_soup(column):
    return column['brand'] + ' '+ column['category'] + ' ' + column['subcategory'] + ' ' + column['name']# + ' ' + column['price_cat']

events=events.dropna().reset_index(drop=True)

events['soup'] = events.apply(create_soup, axis=1)

events=events[["name2", "productid", "soup"]]
events=events.drop_duplicates().reset_index(drop=True)
events.to_csv("./extracted_df/events_tfidf.csv", index=False)

print(events.head())

stopw_turkish = stopwords.words('turkish')
tf = TfidfVectorizer(stop_words=stopw_turkish)

tfidf_matrix = tf.fit_transform(events['soup'])

from sklearn.metrics.pairwise import linear_kernel
cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
np.save(os.path.join("./extracted_model/", "linear_kernel_similarity.npy"), cos_sim)

print(type(cos_sim))
ind = pd.Series(events['productid'].index, index=events['productid']).drop_duplicates()

def get_recommendations(name, cosine_sim=cos_sim):
    idx = ind[name]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    print(sim_scores)
    product_indices = [i[0] for i in sim_scores]
    recommends = pd.DataFrame(events[['productid', 'name2']].iloc[product_indices])
    recommends["sim_score"] = [i[1] for i in sim_scores]

    return recommends[["name2", "sim_score"]]

print(get_recommendations("HBV00000NE0TW"))

import json
import re
from string import punctuation

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pickle

from get_data import events, meta

nltk.download("stopwords")

events.drop(columns=['event', 'brand','eventtime', 'sessionid'], inplace=True)
events.dropna(axis=0, inplace=True)

events['price'] = events['price'].astype(float)
#events['price_cat'] = events.groupby(['category'])['price'].apply(lambda x: pd.qcut(x, 3, labels=["ucuz", "normal", "pahalı"]))
#events['price_cat'] = events['price_cat'].astype(str)
events['product_name']=events['name']

# some variables to remove unnecassary words from data
regular_expression = '[' + re.escape (''. join (list(punctuation))) + ']'
numb='[1234567890]'
stopw_turkish = stopwords.words('turkish')
stop_turkish = r'\b(?:{})\b'.format('|'.join(stopw_turkish))
cols = ['name', 'category', 'subcategory']

for col in cols:
    events[col] = events[col].astype(str)
    events[col] = events[col].str.lower()
    events[col] = events[col].str.replace(stop_turkish, '')
    events[col] = events[col].str.replace(numb, '', regex=True)
    events[col] = events[col].str.replace(regular_expression, '', regex=True)
    events[col] = events[col].str.strip()

# remove word from name if less than 3 letter like 'li', 'ml'
events['name'] = events['name'].str.replace(r'\b(\w{1,2})\b', '')

# get new feature as text
def create_soup(column):
    return column['category'] + ' ' + column['subcategory'] + ' ' + column['name'] #+ ' ' + column['price_cat'] 

events=events.dropna().reset_index(drop=True)

events['soup'] = events.apply(create_soup, axis=1)

events=events[["product_name", "productid", "soup"]]
events=events.drop_duplicates().reset_index(drop=True)
events['soup'] = events['soup'].apply(lambda x: list(x.split(" ")))
events['soup'] =events['soup'].apply(lambda str_list: list(filter(None, str_list)))
events.to_csv("./extracted_df/events_doc2vec.csv", index=False)

# get tag for every row in the new text feature
tagged_data = []
for i, doc in enumerate(events['soup']):
    tagged = TaggedDocument(doc, [i])
    tagged_data.append(tagged)

model = Doc2Vec(tagged_data, vector_size=200, workers=4)
model.build_vocab(tagged_data)

# extract the model for api 
with open("./extracted_model/doc2vec.pkl", 'wb') as file:  
    pickle.dump(model, file)

product_name = "Pınar Piliç Şnitzel 415 gr".split(" ")
product_name_vectorized = model.infer_vector(product_name)

# Calculate cosine similarity. 
similar_product = model.docvecs.most_similar(positive=[product_name_vectorized])

# Output
recommend_product = []
for i, v in enumerate(similar_product):
    index = v[0]
    recommend_product.append([tagged_data[index], v[1]])

recommend_df = pd.DataFrame(recommend_product, columns=["tagged_data", "sim_score"])
recommend_df["index"] = [tag[1][0] for tag in recommend_df["tagged_data"]]
recommend_df["product_name"]= [i for i in events.loc[recommend_df['index'].tolist(), "product_name"]]
recommend_df = recommend_df[["product_name", "sim_score"]]

print(recommend_df.sort_values("sim_score", ascending=False))
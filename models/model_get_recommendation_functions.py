import json
import os

import numpy as np 
import pandas as pd 
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def get_recommendations_tfidf(prd_id):
    """ get product id 
        load model and preprocessed data
        find cosine similarity for this product
        sort and select top 10 recommendation 
        return pd dataframe includes name and similarity score """

    events = pd.read_csv("./extracted_df/events_tfidf.csv")
    cosine_similarity = np.load(os.path.join("./extracted_model/", "linear_kernel_similarity.npy"))
    ind = pd.Series(events['productid'].index, index=events['productid']).drop_duplicates()
    idx = ind[prd_id]
    sim_scores = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]
    recommends = pd.DataFrame(events[['productid', 'product_name']].iloc[product_indices])
    recommends["sim_score"] = [i[1].round(3) for i in sim_scores]

    return recommends[["product_name", "sim_score"]]


def get_recommendations_doc2vec(prd_id):

    """ get product id 
        load model and preprocessed data
        find similarity for this product
        sort and select top 10 recommendation 
        return pd dataframe includes name and similarity score """

    events_doc2vec = pd.read_csv("./extracted_df/events_doc2vec.csv")
    with open("./extracted_model/doc2vec.pkl", 'rb') as file:  
        doc2vec = pickle.load(file)
    
    product_name = events_doc2vec.loc[events_doc2vec['productid'] == prd_id, 'product_name'].unique()[0]
    product_name = product_name.split(" ")
    product_name_vectorized = doc2vec.infer_vector(product_name)
    similar_product = doc2vec.docvecs.most_similar(positive=[product_name_vectorized])

    tagged_data = []
    for i, doc in enumerate(events_doc2vec['soup']):
        tagged = TaggedDocument(doc, [i])
        tagged_data.append(tagged)

    # Output
    recommend_product = []
    for i, v in enumerate(similar_product):
        index = v[0]
        recommend_product.append([tagged_data[index], v[1]])

    recommend_df = pd.DataFrame(recommend_product, columns=["tagged_data", "sim_score"])
    recommend_df["index"] = [tag[1][0] for tag in recommend_df["tagged_data"]]

    recommend_df["product_name"]= [i for i in events_doc2vec.loc[recommend_df['index'].tolist(), "product_name"]]
    recommend_df = recommend_df[["product_name", "sim_score"]].sort_values("sim_score", ascending=False)
    recommend_df["sim_score"] = recommend_df["sim_score"].round(3)
    
    return recommend_df


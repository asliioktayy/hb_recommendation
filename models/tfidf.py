import json
import os

import numpy as np 
import pandas as pd 


events = pd.read_csv("./extracted_df/events_tfidf.csv")
cosine_similarity = np.load(os.path.join("./extracted_model/", "linear_kernel_similarity.npy"))
ind = pd.Series(events['productid'].index, index=events['productid']).drop_duplicates()


def get_recommendations(name, cosine_sim=cosine_similarity):
    idx = ind[name]
    sim_scores = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]
    recommends = pd.DataFrame(events[['productid', 'name2']].iloc[product_indices])
    recommends["sim_score"] = [i[1] for i in sim_scores]

    return recommends[["name2", "sim_score"]]

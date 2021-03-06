# hb_recommendation


## Content
* About Project
* Data
* Technologies used
* What is there inside the box?
* Running of model and api
* Packacing


### About Project
In this project, I developed a recommendation system for an online grocery store. When a customer is landed on the cart page, I provided a recommended product list which is related to the items in the cart.


### Data 
events.json : add2cart events for a period of time
meta.json : meta info of products in the events 

### Technologies used
uvicorn, fastapi, pandas, nltk, numpy, matplotlib, scikit-learn, mlxtend, plotly, gensim, Jinja2

### What is there inside the box?
Folder or File | Function
-------------- | --------
data | events.json and meta.json
data analysis-visualization | exp_data_analysis.ipynb file in here
exp_data_analysis.ipynb | exploration data analysis
extracted_df | there are preprocessed data different models in here.
events_doc2vec.csv | preprocessed data for doc2vec model. But it is not loaded to github because of github memory. You can access when you run models/doc2vec.py
events_tfidf.csv | preprocessed data for tfidf model. But it is not loaded to github because of github memory. You can access when you run models/tfidf_cos_sim.py
extracted_model | There are saved model to run api in here. 
linear_kernel_similarity.npy | similarities for tfidf model. But it is not loaded to github because of github memory. You can access when you run models/tfidf_cos_sim.py
doc2vec.pkl |  doc2vec model pickle. But it is not loaded to github because of github memory. You can access when you run models/doc2vec.py
models | There are getting data and model file in here
get_data.py | getting data from source
doc2vec.py | doc2vec model.
tfidf_cos_sim.py | tfidf_cos_sim model.
model_get_recommendation_functions.py | this code provide load saved models and running. It was used in api 
association_rules.py | apriopri model
api.py | api for doc2vec and tfidf models
templates | html file in here for api
recommendation_case_result.txt | results


### Running of model and api
You should run firstly tfidf_cos_sim.py or doc2vec.py file. Because model should be created. After you could run api

To run models :
`python3 doc2vec.py`
`python3 tfidf_cos_sim.py`

To run api :
`python3 api.py`

link for tf_idf : http://0.0.0.0:8000/tfidf_cos_sim/{productid}
link for doc2vec : http://0.0.0.0:8000/doc2vec/{productid}

For example : http://0.0.0.0:8000/tfidf_cos_sim/HBV00000NE0TW
### Packacing
`docker build -t hb_recommendation .`
`docker run -p 8000:8000 -d --name api hb_recommendation`







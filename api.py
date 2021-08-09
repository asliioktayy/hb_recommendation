from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from models.model_get_recommendation_functions import get_recommendations_tfidf
from models.model_get_recommendation_functions import get_recommendations_doc2vec


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# get request for doc2vec model
@app.get("/doc2vec/{product_id}", response_class=HTMLResponse)
async def root(request: Request, product_id):
    products = get_recommendations_doc2vec(product_id)
    product_names = products['product_name'].to_list()
    sim_scores = products['sim_score'].to_list()
    response = list(zip(product_names, sim_scores))
    
    return templates.TemplateResponse("product.html", {"response": response, "request": request})

# get request for doc2vec tfidf
@app.get("/tfidf_cos_sim/{product_id}", response_class=HTMLResponse)
async def root(request: Request, product_id):
    products = get_recommendations_tfidf(product_id)
    product_names = products['product_name'].to_list()
    sim_scores = products['sim_score'].to_list()
    response = list(zip(product_names, sim_scores))
    
    return templates.TemplateResponse("product.html", {"response": response, "request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
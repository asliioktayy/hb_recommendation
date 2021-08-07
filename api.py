from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# from models.association_rules import get_recommendations
# from models.doc2vec import get_recommendations
# from models.tfidf_cos_sim import get_recommendations
from models.tfidf import get_recommendations


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/association_rules/{product_id}", response_class=HTMLResponse)
async def root(request: Request, product_id):
    products = get_recommendations(product_id)
    product_names = products['name2'].to_list()
    sim_scores = products['sim_score'].to_list()
    response = list(zip(product_names, sim_scores))
    
    return templates.TemplateResponse("product.html", {"response": response, "request": request})


@app.get("/doc2vec/{product_id}", response_class=HTMLResponse)
async def root(request: Request, product_id):
    products = get_recommendations(product_id)
    product_names = products['name2'].to_list()
    sim_scores = products['sim_score'].to_list()
    response = list(zip(product_names, sim_scores))
    
    return templates.TemplateResponse("product.html", {"response": response, "request": request})


@app.get("/tfidf_cos_sim/{product_id}", response_class=HTMLResponse)
async def root(request: Request, product_id):
    products = get_recommendations(product_id)
    product_names = products['name2'].to_list()
    sim_scores = products['sim_score'].to_list()
    response = list(zip(product_names, sim_scores))
    
    return templates.TemplateResponse("product.html", {"response": response, "request": request})


@app.get("/tfidf/{product_id}", response_class=HTMLResponse)
async def root(request: Request, product_id):
    products = get_recommendations(product_id)
    product_names = products['name2'].to_list()
    sim_scores = products['sim_score'].to_list()
    response = list(zip(product_names, sim_scores))
    
    return templates.TemplateResponse("product.html", {"response": response, "request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
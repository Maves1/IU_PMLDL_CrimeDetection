from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from assignment import ROOT_DIR
from .api import main_router, joke_router

from ..model.joke_generation import load_gpt_model, load_gpt2_model

app = FastAPI()
app.mount("/static", StaticFiles(directory=f"{ROOT_DIR}/assignment/app/static"), name="static")

app.include_router(main_router, prefix="/home")
app.include_router(joke_router, prefix="/joke")


@app.on_event("startup")
async def startup_event():
    load_gpt_model()
    load_gpt2_model()

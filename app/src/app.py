from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api import main_router, joke_router

app = FastAPI()
app.mount("/static", StaticFiles(directory="src/static"), name="static")

app.include_router(main_router, prefix="/home")
app.include_router(joke_router, prefix="/joke")

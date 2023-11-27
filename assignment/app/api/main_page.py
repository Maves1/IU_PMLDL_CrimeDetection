from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from assignment import ROOT_DIR

router = APIRouter()
templates = Jinja2Templates(directory=f"{ROOT_DIR}/assignment/app/templates")


@router.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

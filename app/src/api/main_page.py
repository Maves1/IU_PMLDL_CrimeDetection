from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="/code/src/templates")


@router.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

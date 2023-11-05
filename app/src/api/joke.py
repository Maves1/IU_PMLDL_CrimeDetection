from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.model.joke_generation import generate_joke

templates = Jinja2Templates(directory="src/templates")
router = APIRouter()


@router.post("/new_joke", response_class=HTMLResponse)
def complete_joke(request: Request, start: str = Form(...)):
    try:
        full_joke = generate_joke(start)
        return templates.TemplateResponse("index.html", {"request": request, "joke": full_joke})
    except Exception as e:
        return HTMLResponse(content=f"<p>Error: {str(e)}</p>", status_code=500)

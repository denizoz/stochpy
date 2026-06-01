from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from research import get_research_company, search_companies, to_markdown
from stochpy.exceptions import StochPyException

app = FastAPI()

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    matches: list[dict]


class ResearchRequest(BaseModel):
    linkedin_url: str


class ResearchResponse(BaseModel):
    markdown: str


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        result = search_companies(query=query)
        return SearchResponse(matches=[m.model_dump() for m in result.matches])
    except StochPyException as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    url = req.linkedin_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="linkedin_url is required")
    try:
        result = get_research_company(linkedin_url=url)
        return ResearchResponse(markdown=to_markdown(result))
    except StochPyException as e:
        raise HTTPException(status_code=502, detail=str(e))

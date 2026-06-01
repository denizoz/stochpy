from pydantic import BaseModel
from stochpy import context, interface, ModelRegistry, PromptRegistry
from pathlib import Path

registry = ModelRegistry(quality="gemini/gemini-2.5-pro")

prompts = PromptRegistry(root=str(Path(__file__).parent / "prompts"))


@interface
class CompanyMatch(BaseModel):
    name: str
    linkedin_url: str
    tagline: str
    location: str


@interface
class SearchResult(BaseModel):
    matches: list[CompanyMatch]


@context(
    model=registry.quality,
    prompt=prompts.load("research.company.search"),
    retries=1,
)
def search_companies(query: str) -> SearchResult:
    ...


@interface
class CompanyResearch(BaseModel):
    about: str
    how_money: str
    strategic_priorities: str
    business_challenges: str
    competitive_landscape: str


@context(
    model=registry.quality,
    prompt=prompts.load("research.company.profile"),
    retries=2,
)
def get_research_company(linkedin_url: str) -> CompanyResearch:
    ...


def to_markdown(result: CompanyResearch) -> str:
    return f"""## About
{result.about}

## How the Company Makes Money
{result.how_money}

## Strategic Priorities
{result.strategic_priorities}

## Business Challenges
{result.business_challenges}

## Competitive Landscape
{result.competitive_landscape}
"""

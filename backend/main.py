from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from orchestrator import run_orchestrator

app = FastAPI(title="VentureIQ — Multi-Agent Due Diligence System")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartupRequest(BaseModel):
    idea: str
    groq_api_key: str
    google_search_api_key: str
    google_search_cx: str


@app.get("/")
def root():
    return {"status": "VentureIQ API is running"}


@app.get("/health")
def health():
    """Health check endpoint — confirms system is live."""
    return {
        "status": "ok",
        "system": "VentureIQ Multi-Agent Due Diligence System",
        "agents": 3,
        "orchestrator": "active",
        "llm": "llama-3.3-70b-versatile (Groq)",
        "search_tool": "DuckDuckGo"
    }


@app.get("/tools")
def get_tools():
    """
    Returns all registered agents, their design patterns,
    and the tools available to each agent.
    Mirrors the /tools endpoint pattern from Lab 06.
    """
    return {
        "total_agents": 3,
        "orchestrator": "VentureIQ Orchestrator — coordinates all agents, synthesizes final report",
        "agents": [
            {
                "name": "Market Analyst Agent",
                "pattern": "ReAct",
                "description": "Evaluates market size, timing, and competition using autonomous tool selection",
                "tools": [
                    {
                        "name": "search_market_data",
                        "description": "Searches for market size, growth trends, and industry data",
                        "type": "DuckDuckGo Web Search"
                    },
                    {
                        "name": "search_competitors",
                        "description": "Searches for existing competitors and similar solutions",
                        "type": "DuckDuckGo Web Search"
                    }
                ]
            },
            {
                "name": "Research Agent",
                "pattern": "Tool Use",
                "description": "Searches the web for real competitor and market intelligence",
                "tools": [
                    {
                        "name": "duckduckgo_search",
                        "description": "General web search for startup research and data gathering",
                        "type": "DuckDuckGo Web Search"
                    }
                ]
            },
            {
                "name": "Critic Agent",
                "pattern": "Reflection",
                "description": "Searches for real-world risk data, generates initial critique, then reflects and refines it",
                "tools": [
                    {
                        "name": "search_risks",
                        "description": "Searches for real-world startup failures, risks, and challenges to ground the critique",
                        "type": "DuckDuckGo Web Search"
                    }
                ]
            }
        ]
    }


@app.post("/analyze")
def analyze(request: StartupRequest):
    """
    Main endpoint — receives startup idea + the caller's own API keys,
    runs all agents, returns full report.
    """
    if not request.groq_api_key.strip():
        raise HTTPException(status_code=400, detail="Missing Groq API key. Add it in Settings.")
    if not request.google_search_api_key.strip() or not request.google_search_cx.strip():
        raise HTTPException(status_code=400, detail="Missing Google Custom Search API key/CX. Add them in Settings.")

    keys = {
        "groq_api_key": request.groq_api_key,
        "google_search_api_key": request.google_search_api_key,
        "google_search_cx": request.google_search_cx,
    }
    result = run_orchestrator(request.idea, keys)
    return result
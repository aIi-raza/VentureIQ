# VentureIQ — Multi-Agent Due Diligence System

## Setup

1. Add your API keys in Settings
2. Install dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   ```
3. Run the backend:
   ```
   uvicorn main:app --reload
   ```
4. Open `frontend/index.html` in your browser

## Architecture

- **Market Analyst Agent** → ReAct pattern
- **Research Agent** → Tool Use pattern  
- **Critic Agent** → Reflection pattern
- **Orchestrator** → Dispatches all agents, synthesizes final report

## For n8n version
Replace the API_URL in `frontend/index.html` to your n8n Webhook URL.

import os
from groq import Groq
from dotenv import load_dotenv
from agents.market_agent import run_market_agent
from agents.research_agent import run_research_agent
from agents.critic_agent import run_critic_agent

load_dotenv()


def run_orchestrator(startup_idea: str, keys: dict) -> dict:
    """
    ORCHESTRATOR PATTERN
    Dispatches to all 3 specialist agents, collects outputs,
    synthesizes final verdict. No agent talks to another directly.

    `keys` contains the caller's own API keys (BYOK):
      groq_api_key, google_search_api_key, google_search_cx
    """

    print("[Orchestrator] Dispatching to Market Analyst Agent...")
    market_result = run_market_agent(startup_idea, keys["groq_api_key"])

    print("[Orchestrator] Dispatching to Research Agent...")
    research_result = run_research_agent(
        startup_idea, keys["groq_api_key"], keys["google_search_api_key"], keys["google_search_cx"]
    )

    print("[Orchestrator] Dispatching to Critic Agent...")
    critic_result = run_critic_agent(startup_idea, keys["groq_api_key"])

    print("[Orchestrator] Synthesizing final report...")
    final_report = _synthesize(startup_idea, market_result, research_result, critic_result, keys["groq_api_key"])

    return {
        "startup_idea": startup_idea,
        "agents": {
            "market_analyst": market_result,
            "research_agent": research_result,
            "critic_agent": critic_result,
        },
        "final_report": final_report
    }


def _synthesize(idea, market, research, critic, groq_api_key):
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""
You are VentureIQ's Orchestrator. Synthesize 3 agent reports into a final executive verdict.

STARTUP IDEA: {idea}

MARKET ANALYST OUTPUT:
{market['output']}

RESEARCH AGENT OUTPUT:
{research['output']}

CRITIC AGENT OUTPUT:
{critic['output']}

Respond in this EXACT format:

**EXECUTIVE SUMMARY:**
[4-5 sentences giving an overall picture of this startup's viability]

**STRENGTHS:**
• [Strength 1 — specific]
• [Strength 2 — specific]
• [Strength 3 — specific]

**WEAKNESSES:**
• [Weakness 1 — specific]
• [Weakness 2 — specific]
• [Weakness 3 — specific]

**OVERALL SCORE:** [X/10]

**FINAL VERDICT:** [STRONG GO | CONDITIONAL GO | PIVOT RECOMMENDED | DO NOT PROCEED]

**RECOMMENDATION:**
[3-4 actionable sentences telling the founder exactly what to do next]
"""}]
    )

    raw = response.choices[0].message.content
    return {
        "raw": raw,
        "market_score": market.get("score", "N/A"),
        "risk_score": critic.get("score", "N/A"),
    }

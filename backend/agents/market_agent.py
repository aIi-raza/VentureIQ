import os
import re
from groq import Groq
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

# ── TOOLS ────────────────────────────────────────────────────────────────────

def search_market_data(query: str) -> str:
    """Search for market size, industry trends, growth rates."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query + " market size trends", max_results=4):
                results.append(f"- {r.get('title','')}: {r.get('body','')}")
        return "\n".join(results) if results else "No market data found."
    except Exception as e:
        return f"Search error: {str(e)}"


def search_competitors(query: str) -> str:
    """Search for existing competitors and similar companies."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query + " competitors existing companies startups", max_results=4):
                results.append(f"- {r.get('title','')}: {r.get('body','')}")
        return "\n".join(results) if results else "No competitor data found."
    except Exception as e:
        return f"Search error: {str(e)}"


TOOLS = {
    "search_market_data": search_market_data,
    "search_competitors": search_competitors,
}

TOOL_DESCRIPTIONS = """
You have access to these tools:
- search_market_data(query): searches for market size, trends, TAM, growth rates
- search_competitors(query): searches for existing competitors and similar startups
"""

# ── ReAct LOOP ───────────────────────────────────────────────────────────────

def run_market_agent(startup_idea: str, groq_api_key: str) -> dict:
    """
    DESIGN PATTERN: ReAct (Reason + Act)

    Manual implementation of the ReAct loop:
      1. LLM reasons about what to do next  (Thought)
      2. LLM decides which tool to call     (Action)
      3. Tool runs and returns real data    (Observation)
      4. LLM reasons again with new data   (repeat)
      5. LLM produces final answer when done
    """
    client = Groq(api_key=groq_api_key)

    system_prompt = f"""You are a Market Analyst Agent evaluating startup ideas.
{TOOL_DESCRIPTIONS}

Follow this EXACT format for every step:

Thought: [your reasoning about what to do next]
Action: search_market_data OR search_competitors
Action Input: [the search query string, nothing else]

When you have enough data, end with:
Thought: I now have enough information to give a complete analysis.
Final Answer: [your full market analysis]

Rules:
- Always start with a Thought
- Action Input must be a plain string, no quotes or brackets
- After each Observation you will get new data — use it
- End with Final Answer that includes: target customer, market size, competitors, timing, and MARKET SCORE: X/10
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Evaluate the market opportunity for: {startup_idea}"}
    ]

    tool_trace = []
    max_iterations = 6

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stop=["Observation:"],   # stop when it expects tool output
            max_tokens=600,
        )

        llm_output = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": llm_output})

        # Check if the LLM is done
        if "Final Answer:" in llm_output:
            final_text = llm_output.split("Final Answer:")[-1].strip()
            return {
                "agent": "Market Analyst Agent",
                "pattern": "ReAct",
                "output": final_text,
                "tool_trace": tool_trace,
                "tools_used": list(set(t["tool"] for t in tool_trace)),
                "score": _extract_score(final_text),
                "fallback_used": False,
                "iterations": i + 1,
            }

        # Parse Action and Action Input
        action_match = re.search(r"Action:\s*(\w+)", llm_output)
        input_match  = re.search(r"Action Input:\s*(.+)", llm_output)

        if not action_match or not input_match:
            # LLM didn't follow format — nudge it
            messages.append({
                "role": "user",
                "content": "Please follow the format: Thought / Action / Action Input, or write Final Answer."
            })
            continue

        tool_name  = action_match.group(1).strip()
        tool_input = input_match.group(1).strip()

        # Call the actual tool
        if tool_name in TOOLS:
            observation = TOOLS[tool_name](tool_input)
            tool_trace.append({
                "tool": tool_name,
                "input": tool_input,
                "output_preview": observation[:300] + ("..." if len(observation) > 300 else "")
            })
        else:
            observation = f"Unknown tool '{tool_name}'. Use search_market_data or search_competitors."

        # Feed observation back to LLM
        messages.append({"role": "user", "content": f"Observation: {observation}\n\nContinue:"})

    # Ran out of iterations — ask for final answer with whatever data we have
    messages.append({"role": "user", "content": "Provide your Final Answer now based on what you have found."})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=800,
    )
    final_text = response.choices[0].message.content.strip()

    return {
        "agent": "Market Analyst Agent",
        "pattern": "ReAct",
        "output": final_text,
        "tool_trace": tool_trace,
        "tools_used": list(set(t["tool"] for t in tool_trace)),
        "score": _extract_score(final_text),
        "fallback_used": False,
        "iterations": max_iterations,
    }


def _extract_score(text: str) -> str:
    match = re.search(r'MARKET SCORE.*?(\d+(?:\.\d+)?)/10', text, re.IGNORECASE)
    return match.group(1) + "/10" if match else "N/A"
import os
import re
from groq import Groq
 
from duckduckgo_search import DDGS




def _search_risks(query: str) -> str:
    """Search for real-world risks, failures, and challenges related to the startup idea."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query + " startup failure risks challenges problems", max_results=4):
                results.append(f"- {r.get('title','')}: {r.get('body','')}")
        return "\n".join(results) if results else "No risk data found."
    except Exception as e:
        return f"Search error: {str(e)}"


def run_critic_agent(startup_idea: str, groq_api_key: str) -> dict:
    """
    DESIGN PATTERN: Reflection
    Agent searches for real risk data, generates initial critique grounded
    in that data, then reflects and refines it.
    """
    client = Groq(api_key=groq_api_key)

    # Step 0: Search for real-world risk data (Tool Use within Reflection pattern)
    search_query = startup_idea[:80]  # trim long ideas
    risk_data = _search_risks(search_query)

    # Step 1: Initial critique — grounded in real search data
    initial_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""
You are a Devil's Advocate Agent. Critically attack this startup idea.
Use the real-world data below to make your critique specific and grounded.
Be harsh but fair. Cover: flawed assumptions, market risks, execution risks, financial risks.
Write 5-6 bullet points. Each point should reference the real data where possible.

Startup Idea: {startup_idea}

REAL-WORLD RISK DATA FROM SEARCH:
{risk_data}
"""}]
    )
    initial_output = initial_response.choices[0].message.content

    # Step 2: Reflection — agent reflects on its own critique and refines it
    reflection_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""
You are the same Devil's Advocate Agent. You wrote this initial critique:

{initial_output}

Now REFLECT on it and produce a refined version.

**REFLECTION NOTES:**
[2-3 sentences: what did you miss or what was too harsh/not harsh enough?]

**FINAL CRITIQUE:**
• [Risk point 1 — specific and detailed]
• [Risk point 2 — specific and detailed]
• [Risk point 3 — specific and detailed]
• [Risk point 4 — specific and detailed]
• [Risk point 5 — specific and detailed]

**BIGGEST RISK:** [The single most fatal flaw in one clear sentence]

**RISK SCORE:** [X/10] — [one line justification. 10 = extremely risky]
"""}]
    )

    final_output = reflection_response.choices[0].message.content
    score = _extract_score(final_output)

    return {
        "agent": "Critic Agent",
        "pattern": "Reflection",
        "search_query": search_query,
        "risk_data_found": len(risk_data) > 30,
        "initial_critique": initial_output,
        "output": final_output,
        "score": score
    }


def _extract_score(text: str) -> str:
    match = re.search(r'RISK SCORE.*?(\d+(?:\.\d+)?)/10', text, re.IGNORECASE)
    return match.group(1) + "/10" if match else "N/A"
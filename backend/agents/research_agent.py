import os
from groq import Groq
 
from tools.search_tool import google_search




def run_research_agent(startup_idea: str, groq_api_key: str, google_search_api_key: str, google_search_cx: str) -> dict:
    """
    DESIGN PATTERN: Tool Use
    Agent decides what to search, calls the tool,
    receives raw results, then processes into useful intelligence.
    """
    client = Groq(api_key=groq_api_key)

    # Step 1: Agent decides search queries
    query_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""
Given this startup idea, generate exactly 2 search queries to find competitor and market data.
Return ONLY 2 search queries, one per line. No numbering, no extra text.

Startup Idea: {startup_idea}
"""}]
    )
    queries = [q.strip() for q in query_response.choices[0].message.content.strip().split("\n") if q.strip()][:2]

    # Step 2: TOOL USE — call search tool
    all_results = []
    for query in queries:
        results = google_search(query, google_search_api_key, google_search_cx, num_results=4)
        all_results.extend(results)

    raw_data = ""
    for i, r in enumerate(all_results, 1):
        raw_data += f"{i}. {r['title']}\n   {r['snippet']}\n   {r['link']}\n\n"

    # Step 3: LLM processes raw tool output
    final_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""
You are a Research Agent. You used a search tool and got these results:

{raw_data}

Analyze in context of this startup idea: {startup_idea}

Respond in this exact format:

**SEARCH QUERIES USED:**
• {queries[0] if len(queries) > 0 else 'N/A'}
• {queries[1] if len(queries) > 1 else 'N/A'}

**KEY FINDINGS:**
• [Finding 1 — specific insight from search results]
• [Finding 2 — specific insight from search results]
• [Finding 3 — specific insight from search results]

**COMPETITOR LANDSCAPE:**
[3-4 sentences describing existing players, their approach, and market gaps]

**RESEARCH SUMMARY:**
[3-4 sentences on what this data means for the startup's viability]
"""}]
    )

    return {
        "agent": "Research Agent",
        "pattern": "Tool Use",
        "queries_used": queries,
        "raw_results_count": len(all_results),
        "output": final_response.choices[0].message.content,
        "score": "N/A"
    }

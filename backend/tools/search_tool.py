import requests


def google_search(query: str, api_key: str, cx: str, num_results: int = 5) -> list[dict]:
    """
    Tool Use Pattern — this is the TOOL that the Research Agent calls.
    Takes a search query, hits Google Custom Search API, returns results.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": num_results,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            })
        return results

    except Exception as e:
        return [{"title": "Search Error", "snippet": str(e), "link": ""}]

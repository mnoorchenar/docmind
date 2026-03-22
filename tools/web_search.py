# Added retry logic with exponential backoff for DuckDuckGo rate limits.

import time
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException

def web_search(query: str, max_results: int = 4) -> str:
    last_error = None
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=max_results))
            if not hits:
                return "No results found."
            lines = []
            for h in hits:
                lines.append(f"Title: {h.get('title','')}\nSnippet: {h.get('body','')}\nURL: {h.get('href','')}\n")
            return "\n".join(lines)
        except RatelimitException as e:
            last_error = e
            wait = (attempt + 1) * 4   # 4s, 8s, 12s
            time.sleep(wait)
        except Exception as e:
            return f"Search error: {e}"
    return f"Search rate limited after 3 attempts. Try again in a few seconds. ({last_error})"


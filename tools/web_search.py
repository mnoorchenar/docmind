from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 4) -> str:
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        if not hits:
            return "No results found."
        lines = []
        for h in hits:
            lines.append(f"Title: {h.get('title','')}\nSnippet: {h.get('body','')}\nURL: {h.get('href','')}\n")
        return "\n".join(lines)
    except Exception as exc:
        return f"Search error: {exc}"
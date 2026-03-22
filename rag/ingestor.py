# SearchIngestor updated with retry logic for DuckDuckGo rate limits.

import os, re, time, requests
from pypdf import PdfReader
from bs4 import BeautifulSoup
from duckduckgo_search.exceptions import RatelimitException

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
MAX_PDF_BYTES = 10 * 1024 * 1024


class PDFIngestor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def _extract_text(self, path: str) -> list:
        reader = PdfReader(path)
        pages  = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "page": i + 1})
        return pages

    def _chunk(self, page_data: list, source: str) -> list:
        chunks = []
        for pd in page_data:
            text  = re.sub(r"\s+", " ", pd["text"])
            words = text.split()
            start = 0
            while start < len(words):
                end   = min(start + self.chunk_size, len(words))
                chunk = " ".join(words[start:end])
                chunks.append({"page_content": chunk, "page": pd["page"], "source": source})
                start += self.chunk_size - self.chunk_overlap
        return chunks

    def ingest(self, path: str) -> list:
        size = os.path.getsize(path)
        if size > MAX_PDF_BYTES:
            raise ValueError(f"File exceeds 10 MB limit ({size/1024/1024:.1f} MB).")
        filename = os.path.basename(path)
        pages    = self._extract_text(path)
        return self._chunk(pages, filename)


class URLIngestor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def _fetch_text(self, url: str) -> str:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script","style","nav","footer","header","aside","form","noscript","iframe"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        text = main.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()

    def _chunk(self, text: str, source: str) -> list:
        words  = text.split()
        chunks = []
        start  = 0
        page   = 1
        while start < len(words):
            end   = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append({"page_content": chunk, "page": page, "source": source})
            start += self.chunk_size - self.chunk_overlap
            page  += 1
        return chunks

    def ingest(self, url: str) -> list:
        text = self._fetch_text(url)
        if len(text) < 100:
            raise ValueError("Could not extract meaningful content. The page may require JavaScript or block bots.")
        words = text.split()
        if len(words) > 15000:
            text = " ".join(words[:15000])
        from urllib.parse import urlparse
        source = urlparse(url).netloc or url
        return self._chunk(text, source)


class SearchIngestor:
    def __init__(self):
        self._url_ingestor = URLIngestor()

    def _ddg_search(self, query: str, max_results: int = 5) -> list:
        """DuckDuckGo search with retry on rate limit."""
        from duckduckgo_search import DDGS
        last_error = None
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))
            except RatelimitException as e:
                last_error = e
                time.sleep((attempt + 1) * 5)   # 5s, 10s, 15s
            except Exception as e:
                raise ValueError(f"Search failed: {e}")
        raise ValueError(f"DuckDuckGo rate limited after 3 attempts. Wait a few seconds and try again. ({last_error})")

    def search_and_ingest(self, query: str, site: str = "") -> dict:
        full_query = f"site:{site} {query}" if site.strip() else query
        hits       = self._ddg_search(full_query)
        if not hits:
            raise ValueError("No search results found for this query.")
        last_error = None
        for hit in hits:
            url = hit.get("href", "")
            if not url:
                continue
            try:
                chunks = self._url_ingestor.ingest(url)
                return {"url": url, "title": hit.get("title", url), "chunks": chunks}
            except Exception as e:
                last_error = e
                continue
        raise ValueError(f"Could not fetch any search result. Last error: {last_error}")
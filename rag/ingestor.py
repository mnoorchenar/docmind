import os, re, requests
from pypdf import PdfReader
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
MAX_PDF_BYTES = 10 * 1024 * 1024   # 10 MB hard limit


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
            raise ValueError(f"File exceeds 10 MB limit ({size / 1024 / 1024:.1f} MB). Please upload a smaller file.")
        filename = os.path.basename(path)
        pages    = self._extract_text(path)
        return self._chunk(pages, filename)


class URLIngestor:
    """Fetches a public URL, strips HTML boilerplate, and returns text chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def _fetch_text(self, url: str) -> str:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe", "ads"]):
            tag.decompose()
        # Prefer main content area if it exists
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        text = main.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text

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
            raise ValueError("Could not extract meaningful content from this URL. The page may require JavaScript or block bots.")
        # Truncate very large pages to keep indexing fast
        words = text.split()
        if len(words) > 15000:
            words = words[:15000]
            text  = " ".join(words)
        # Use domain as source label
        from urllib.parse import urlparse
        source = urlparse(url).netloc or url
        return self._chunk(text, source)


class SearchIngestor:
    """Search DuckDuckGo with an optional site: operator, then fetch the top result."""

    def __init__(self):
        self._url_ingestor = URLIngestor()

    def search_and_ingest(self, query: str, site: str = "") -> dict:
        from duckduckgo_search import DDGS
        full_query = f"site:{site} {query}" if site.strip() else query
        with DDGS() as ddgs:
            hits = list(ddgs.text(full_query, max_results=5))
        if not hits:
            raise ValueError("No search results found for this query.")
        # Try results in order until one fetches successfully
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


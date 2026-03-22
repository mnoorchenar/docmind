import os, re, requests
from pypdf import PdfReader
from bs4 import BeautifulSoup

MAX_PDF_BYTES = 10 * 1024 * 1024

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

_BLOCKED_DOMAINS = {"amazon.com", "www.amazon.com", "amazon.ca", "amazon.co.uk"}

_PDF_HINT = (
    "\n\nTip: This page cannot be fetched automatically. "
    "Open it in your browser → Print → Save as PDF → upload the PDF here."
)


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
        pages = self._extract_text(path)
        if not pages:
            raise ValueError("Could not extract any text from this PDF. It may be scanned or image-only.")
        return self._chunk(pages, os.path.basename(path))


class URLIngestor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def _fetch_text(self, url: str) -> str:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        if domain in _BLOCKED_DOMAINS:
            raise ValueError(f"⛔ {domain} blocks all automated access." + _PDF_HINT)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20, allow_redirects=True)
            if resp.status_code == 403:
                raise ValueError("403 Forbidden — this site blocks automated access." + _PDF_HINT)
            if resp.status_code == 404:
                raise ValueError("404 Not Found — the URL does not exist.")
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script","style","nav","footer","header","aside","form","noscript","iframe"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("body") or soup
            text = main.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 200:
                raise ValueError("Page content is too short or empty." + _PDF_HINT)
            return text
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Could not fetch this page: {e}." + _PDF_HINT)

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
        words = text.split()
        if len(words) > 15000:
            text = " ".join(words[:15000])
        from urllib.parse import urlparse
        source = urlparse(url).netloc or url
        return self._chunk(text, source)

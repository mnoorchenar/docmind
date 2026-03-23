"""
Document ingestors — PDF and URL sources.

Both ingestors use LangChain's RecursiveCharacterTextSplitter for chunking
and return lists of LangChain Document objects, making them compatible with
any LangChain-based vector store or retrieval pipeline.
"""
import os
import re
import requests
from urllib.parse import urlparse

from pypdf import PdfReader
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

_BLOCKED_DOMAINS = {"amazon.com", "www.amazon.com", "amazon.ca", "amazon.co.uk"}

_PDF_HINT = (
    "\n\nTip: This page cannot be fetched automatically. "
    "Open it in your browser → Print → Save as PDF → upload the PDF here."
)


def _make_splitter(chunk_size: int = 1500, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """Character-based splitter — standard LangChain RAG configuration."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


class PDFIngestor:
    """Ingest a local PDF file into LangChain Document objects.

    Uses pypdf for text extraction and LangChain's RecursiveCharacterTextSplitter
    for chunking, preserving page-number metadata for source citations.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.splitter = _make_splitter(chunk_size, chunk_overlap)

    def _extract_pages(self, path: str) -> list[dict]:
        reader = PdfReader(path)
        pages  = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": re.sub(r"\s+", " ", text), "page": i + 1})
        return pages

    def ingest(self, path: str) -> list[Document]:
        if os.path.getsize(path) > MAX_PDF_BYTES:
            size_mb = os.path.getsize(path) / 1024 / 1024
            raise ValueError(f"File exceeds 10 MB limit ({size_mb:.1f} MB).")
        pages = self._extract_pages(path)
        if not pages:
            raise ValueError(
                "Could not extract any text from this PDF. It may be scanned or image-only."
            )
        source = os.path.basename(path)
        docs: list[Document] = []
        for pd in pages:
            chunks = self.splitter.create_documents(
                [pd["text"]],
                metadatas=[{"source": source, "page": pd["page"]}],
            )
            docs.extend(chunks)
        return docs


class URLIngestor:
    """Fetch a public URL and ingest its text into LangChain Document objects.

    Strips navigation, scripts, and boilerplate via BeautifulSoup before
    splitting with RecursiveCharacterTextSplitter.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.splitter = _make_splitter(chunk_size, chunk_overlap)

    def _fetch_text(self, url: str) -> str:
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
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form",
                              "noscript", "iframe"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("body") or soup
            text = re.sub(r"\s+", " ", main.get_text(separator=" ", strip=True)).strip()
            if len(text) < 200:
                raise ValueError("Page content is too short or empty." + _PDF_HINT)
            return text
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not fetch this page: {exc}." + _PDF_HINT)

    def ingest(self, url: str) -> list[Document]:
        text   = self._fetch_text(url)
        # Cap at ~15k words to stay within context limits
        text   = " ".join(text.split()[:15_000])
        source = urlparse(url).netloc or url
        return self.splitter.create_documents(
            [text],
            metadatas=[{"source": source, "page": 1}],
        )

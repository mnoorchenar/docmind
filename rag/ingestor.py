import os, re
from pypdf import PdfReader
from rag.embeddings import embed


class PDFIngestor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def _extract_text(self, path: str) -> list:
        """Returns list of {text, page} dicts."""
        reader = PdfReader(path)
        pages  = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "page": i + 1})
        return pages

    def _chunk(self, page_data: list) -> list:
        """Splits pages into overlapping chunks."""
        chunks = []
        for pd in page_data:
            text   = re.sub(r"\s+", " ", pd["text"])
            words  = text.split()
            start  = 0
            while start < len(words):
                end   = min(start + self.chunk_size, len(words))
                chunk = " ".join(words[start:end])
                chunks.append({"page_content": chunk, "page": pd["page"]})
                start += self.chunk_size - self.chunk_overlap
        return chunks

    def ingest(self, path: str) -> list:
        """Returns list of chunk dicts with page_content, page, source."""
        filename   = os.path.basename(path)
        pages      = self._extract_text(path)
        chunks     = self._chunk(pages)
        for c in chunks:
            c["source"] = filename
        return chunks
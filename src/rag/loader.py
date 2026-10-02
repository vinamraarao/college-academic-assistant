"""PDF discovery and text extraction with page-level metadata and quality checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import DATA_DIR, MIN_CHARS_PER_PAGE

logger = logging.getLogger(__name__)


class DataDirectoryError(RuntimeError):
    """Raised when the knowledge-base folder is missing or holds no PDFs."""


@dataclass
class ExtractionReport:
    """What the loader managed to read, and what looked unusable."""

    files_loaded: list[str] = field(default_factory=list)
    pages_total: int = 0
    pages_sparse: list[tuple[str, int]] = field(default_factory=list)
    files_failed: dict[str, str] = field(default_factory=dict)

    @property
    def has_warnings(self) -> bool:
        return bool(self.pages_sparse or self.files_failed)

    def summary(self) -> str:
        lines = [
            f"Loaded {len(self.files_loaded)} PDF(s), {self.pages_total} page(s) of text."
        ]
        if self.files_failed:
            lines.append(
                "Failed files: "
                + "; ".join(f"{name} ({err})" for name, err in self.files_failed.items())
            )
        if self.pages_sparse:
            listed = ", ".join(f"{name} p.{page}" for name, page in self.pages_sparse[:10])
            extra = "" if len(self.pages_sparse) <= 10 else f" (+{len(self.pages_sparse) - 10} more)"
            lines.append(
                f"Sparse pages (possible scans/poor extraction, answers may be incomplete): "
                f"{listed}{extra}"
            )
        return "\n".join(lines)


def discover_pdfs(data_dir: Path | str = DATA_DIR) -> list[Path]:
    """Return every PDF under data_dir, sorted for deterministic indexing."""
    directory = Path(data_dir)
    if not directory.exists():
        raise DataDirectoryError(
            f"Knowledge-base folder not found at {directory}. "
            "Create it and place your college PDFs inside."
        )
    pdfs = sorted(p for p in directory.rglob("*.pdf") if p.is_file())
    if not pdfs:
        raise DataDirectoryError(
            f"No PDF files found in {directory}. Add your college documents and retry."
        )
    return pdfs


def _clean_text(text: str) -> str:
    """Collapse whitespace while keeping paragraph breaks readable."""
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    # Collapse runs of blank lines; repeated spaces carry no meaning in PDFs.
    cleaned = cleaned.replace("\t", " ")
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def load_pdfs(
    data_dir: Path | str = DATA_DIR,
    min_chars_per_page: int = MIN_CHARS_PER_PAGE,
) -> tuple[list[dict], ExtractionReport]:
    """Extract text from every PDF, one record per page.

    Returns page records (dict with page_content/source/page) and an
    ExtractionReport flagging corrupt files and sparse (likely scanned) pages.
    """
    pdfs = discover_pdfs(data_dir)
    report = ExtractionReport()
    pages: list[dict] = []

    for pdf_path in pdfs:
        try:
            reader = PdfReader(str(pdf_path))
            file_pages = 0
            for page_index, page in enumerate(reader.pages):
                try:
                    raw = page.extract_text() or ""
                except Exception as exc:  # a single bad page must not kill the file
                    logger.warning("Page %d of %s failed to extract: %s",
                                   page_index + 1, pdf_path.name, exc)
                    raw = ""
                text = _clean_text(raw)
                if len(text) < min_chars_per_page:
                    report.pages_sparse.append((pdf_path.name, page_index + 1))
                if not text:
                    continue
                pages.append(
                    {
                        "page_content": text,
                        "source": pdf_path.name,
                        "page": page_index + 1,
                    }
                )
                file_pages += 1
            if file_pages == 0:
                report.files_failed[pdf_path.name] = "no extractable text (likely scanned)"
            else:
                report.files_loaded.append(pdf_path.name)
            report.pages_total += file_pages
        except (PdfReadError, OSError, ValueError) as exc:
            logger.warning("Could not read %s: %s", pdf_path.name, exc)
            report.files_failed[pdf_path.name] = f"corrupted or unreadable ({exc})"

    if not pages:
        details = "; ".join(f"{n}: {e}" for n, e in report.files_failed.items())
        raise DataDirectoryError(
            "No usable text could be extracted from any PDF. "
            + (f"Details: {details}" if details else
               "The files may be scanned images; OCR them first (e.g. OCRmyPDF).")
        )

    if report.has_warnings:
        logger.warning("\n%s", report.summary())

    return pages, report
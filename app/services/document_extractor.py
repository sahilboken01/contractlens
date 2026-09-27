from dataclasses import dataclass
from io import BytesIO

from docx import Document as DocxDocument
from pypdf import PdfReader


class DocumentExtractionError(Exception):
    """Raised when a supported document cannot be read or has no text."""


@dataclass
class ExtractedSegment:
    locator: str
    text: str


@dataclass
class ExtractionResult:
    text: str
    segments: list[ExtractedSegment]


def extract_document(filename: str, content: bytes) -> ExtractionResult:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        if suffix == "pdf":
            result = _extract_pdf(content)
        elif suffix == "docx":
            result = _extract_docx(content)
        elif suffix == "txt":
            result = _extract_txt(content)
        else:
            raise DocumentExtractionError("Unsupported file type.")
    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError("The file could not be read as a valid document.") from exc

    if not result.text.strip():
        raise DocumentExtractionError("No extractable text was found in the document.")
    return result


def _extract_pdf(content: bytes) -> ExtractionResult:
    reader = PdfReader(BytesIO(content), strict=False)
    segments = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            segments.append(ExtractedSegment(locator=f"Page {index}", text=page_text))
    return ExtractionResult("\n\n".join(segment.text for segment in segments), segments)


def _extract_docx(content: bytes) -> ExtractionResult:
    document = DocxDocument(BytesIO(content))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells]
            if any(values):
                paragraphs.append(" | ".join(values))
    text = "\n".join(paragraphs)
    segments = [ExtractedSegment(locator="Document", text=text)] if text else []
    return ExtractionResult(text, segments)


def _extract_txt(content: bytes) -> ExtractionResult:
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        text = content.decode("cp1252").strip()
    segments = [ExtractedSegment(locator="Document", text=text)] if text else []
    return ExtractionResult(text, segments)

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.document import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentTextResponse,
    DocumentUploadResponse,
    AnswerSource,
    AskRequest,
    AskResponse,
    ProcessDocumentResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TextSegment,
)
from app.schemas.analysis import ContractAnalysisResponse
from app.services.chunking import chunk_segments
from app.services.document_extractor import DocumentExtractionError, extract_document
from app.services.embeddings import EmbeddingConfigurationError, embed_texts
from app.services.llm import LLMConfigurationError, LLMProviderError, create_llm_provider
from app.services.rag import answer_question
from app.services.contract_analysis import ContractAnalysisError, analyze_contract
from app.services.retrieval import retrieve_chunks
from app.services import vector_store
from app.services import document_store
from app.services.document_store import StoredDocument

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    settings = get_settings()
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported file types are PDF, DOCX, and TXT.")

    content = await file.read(settings.max_upload_size_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds the {settings.max_upload_size_mb} MB limit.")

    try:
        extracted = extract_document(filename, content)
    except DocumentExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected document extraction failure")
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc

    document_id = str(uuid4())
    try:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        (settings.upload_dir / f"{document_id}{extension}").write_bytes(content)
    except OSError as exc:
        logger.exception("Unable to persist uploaded document id=%s", document_id)
        raise HTTPException(status_code=500, detail="Unable to store uploaded document.") from exc

    logger.info("Document uploaded id=%s type=%s bytes=%d", document_id, extension, len(content))
    try:
        document_store.create_document(StoredDocument(
            document_id=document_id, filename=filename, content_type=file.content_type,
            size_bytes=len(content), character_count=len(extracted.text), text=extracted.text,
            segments=[{"locator": segment.locator, "text": segment.text} for segment in extracted.segments],
            processing_status="uploaded", chunk_count=0, embedding_model=None, embedding_dimensions=None,
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        ))
    except Exception as exc:
        (settings.upload_dir / f"{document_id}{extension}").unlink(missing_ok=True)
        logger.exception("Unable to persist document metadata id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to save document metadata. Check database setup.") from exc
    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        content_type=file.content_type,
        size_bytes=len(content),
        character_count=len(extracted.text),
        text=extracted.text,
        segments=[TextSegment(locator=segment.locator, text=segment.text) for segment in extracted.segments],
    )


def _metadata(document: StoredDocument) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document.document_id, filename=document.filename, content_type=document.content_type,
        size_bytes=document.size_bytes, character_count=document.character_count,
        processing_status=document.processing_status, chunk_count=document.chunk_count,
        embedding_model=document.embedding_model, embedding_dimensions=document.embedding_dimensions,
        created_at=document.created_at, updated_at=document.updated_at,
    )


@router.get("", response_model=DocumentListResponse, summary="List saved documents")
def list_documents(limit: int = 50, offset: int = 0) -> DocumentListResponse:
    if limit < 1 or limit > 100 or offset < 0:
        raise HTTPException(status_code=422, detail="limit must be 1–100 and offset cannot be negative.")
    try:
        documents = document_store.list_documents(limit, offset)
        total = document_store.count_documents()
    except Exception as exc:
        logger.exception("Unable to list documents")
        raise HTTPException(status_code=503, detail="Unable to access document history.") from exc
    return DocumentListResponse(documents=[_metadata(item) for item in documents], total=total)


@router.get("/{document_id}", response_model=DocumentMetadata, summary="Get document metadata and status")
def get_document(document_id: UUID) -> DocumentMetadata:
    try:
        document = document_store.get_document(str(document_id))
    except Exception as exc:
        logger.exception("Unable to retrieve document metadata id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to access document.") from exc
    if not document:
        raise HTTPException(status_code=404, detail="Uploaded document was not found.")
    return _metadata(document)


@router.get("/{document_id}/text", response_model=DocumentTextResponse, summary="Get extracted document text")
def get_document_text(document_id: UUID) -> DocumentTextResponse:
    try:
        document = document_store.get_document(str(document_id))
    except Exception as exc:
        logger.exception("Unable to retrieve document text id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to access document text.") from exc
    if not document:
        raise HTTPException(status_code=404, detail="Uploaded document was not found.")
    return DocumentTextResponse(
        document_id=document.document_id, text=document.text,
        segments=[TextSegment(**segment) for segment in document.segments],
    )


@router.post("/{document_id}/process", response_model=ProcessDocumentResponse)
def process_document(document_id: UUID) -> ProcessDocumentResponse:
    settings = get_settings()
    try:
        document = document_store.get_document(str(document_id))
    except Exception as exc:
        logger.exception("Unable to access document id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to access document.") from exc
    if not document:
        raise HTTPException(status_code=404, detail="Uploaded document was not found.")
    try:
        document_store.update_status(str(document_id), "processing", chunk_count=0)
        chunks = chunk_segments(
            str(document_id),
            [(segment["locator"], segment["text"]) for segment in document.segments],
            settings.chunk_size_chars,
            settings.chunk_overlap_chars,
        )
        vectors = embed_texts([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, vectors, strict=True):
            chunk.embedding = embedding
        vector_store.replace_document_chunks(str(document_id), chunks)
        document_store.update_status(
            str(document_id), "processed", chunk_count=len(chunks),
            embedding_model=settings.embedding_model, embedding_dimensions=settings.embedding_dimensions,
        )
    except EmbeddingConfigurationError as exc:
        try:
            document_store.update_status(str(document_id), "failed")
        except Exception:
            logger.exception("Unable to record document failure id=%s", document_id)
        logger.exception("Embedding failed document_id=%s", document_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        try:
            document_store.update_status(str(document_id), "failed")
        except Exception:
            logger.exception("Unable to record document failure id=%s", document_id)
        logger.exception("Document processing failed document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to process document. Check model and database setup.") from exc
    return ProcessDocumentResponse(
        document_id=str(document_id),
        chunk_count=len(chunks),
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
    )


@router.post("/{document_id}/search", response_model=SearchResponse)
def search_document(document_id: UUID, payload: SearchRequest) -> SearchResponse:
    settings = get_settings()
    try:
        document = document_store.get_document(str(document_id))
    except Exception as exc:
        logger.exception("Unable to access document for search id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to access document.") from exc
    if not document:
        raise HTTPException(status_code=404, detail="Uploaded document was not found.")
    if document.processing_status != "processed" or document.chunk_count == 0:
        raise HTTPException(status_code=409, detail="Document has not been processed yet.")
    try:
        results = retrieve_chunks(
            str(document_id),
            payload.query,
            payload.top_k or settings.retrieval_top_k,
            vector_store,
        )
    except EmbeddingConfigurationError as exc:
        logger.exception("Query embedding failed document_id=%s", document_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document search failed document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to search document. Check model and database setup.") from exc
    return SearchResponse(
        document_id=str(document_id),
        query=payload.query,
        results=[
            SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                source_locator=result.source_locator,
                page_number=result.page_number,
                similarity=result.similarity,
                metadata=result.metadata,
            )
            for result in results
        ],
    )


@router.post(
    "/{document_id}/ask",
    response_model=AskResponse,
    summary="Ask a question about one uploaded contract",
    description=(
        "Retrieves relevant chunks scoped to this document, generates an answer from those chunks, "
        "and returns validated source citations. Requires the document to have been processed."
    ),
)
def ask_document(document_id: UUID, payload: AskRequest) -> AskResponse:
    settings = get_settings()
    try:
        document = document_store.get_document(str(document_id))
    except Exception as exc:
        logger.exception("Unable to access document id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to access document.") from exc
    if not document:
        raise HTTPException(status_code=404, detail="Uploaded document was not found.")

    chunk_count = document.chunk_count
    if document.processing_status != "processed" or chunk_count == 0:
        raise HTTPException(status_code=409, detail="Document has not been processed yet.")

    try:
        result = answer_question(
            document_id=str(document_id),
            question=payload.question,
            top_k=settings.retrieval_top_k,
            minimum_similarity=settings.rag_min_similarity,
            store=vector_store,
            provider=create_llm_provider(),
        )
    except LLMConfigurationError as exc:
        logger.warning("RAG provider is not configured for document id=%s", document_id)
        raise HTTPException(status_code=503, detail="Question answering is not configured.") from exc
    except LLMProviderError as exc:
        logger.warning("RAG provider failed for document id=%s", document_id)
        raise HTTPException(status_code=502, detail="Answer generation failed.") from exc
    except EmbeddingConfigurationError as exc:
        logger.exception("Question embedding failed document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="Question embedding is unavailable.") from exc
    except Exception as exc:
        logger.exception("Question answering failed document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to answer question from document.") from exc

    return AskResponse(
        document_id=str(document_id),
        question=payload.question,
        answer=result.answer,
        sources=[
            AnswerSource(
                chunk_id=source.chunk_id,
                source_locator=source.source_locator,
                page_number=source.page_number,
                similarity=source.similarity,
            )
            for source in result.sources
        ],
    )


@router.post(
    "/{document_id}/analyze",
    response_model=ContractAnalysisResponse,
    summary="Analyze a processed contract",
    description=(
        "Extracts structured contract facts from document-scoped retrieved chunks. "
        "Every returned finding includes source chunk metadata; unsupported fields remain null or empty."
    ),
)
def analyze_document(document_id: UUID) -> ContractAnalysisResponse:
    settings = get_settings()
    try:
        document = document_store.get_document(str(document_id))
    except Exception as exc:
        logger.exception("Unable to access document id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to access document.") from exc
    if not document:
        raise HTTPException(status_code=404, detail="Uploaded document was not found.")
    if not document.text.strip():
        raise HTTPException(status_code=422, detail="Contract text is empty or could not be extracted.")

    chunk_count = document.chunk_count
    if document.processing_status != "processed" or chunk_count == 0:
        raise HTTPException(status_code=409, detail="Document has not been processed yet.")

    try:
        return analyze_contract(
            document_id=str(document_id),
            store=vector_store,
            provider=create_llm_provider(),
            top_k=settings.analysis_top_k,
            minimum_similarity=settings.rag_min_similarity,
        )
    except EmbeddingConfigurationError as exc:
        logger.exception("Analysis query embedding failed document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="Contract analysis retrieval is unavailable.") from exc
    except ContractAnalysisError as exc:
        logger.warning("LLM returned malformed contract analysis document_id=%s", document_id)
        raise HTTPException(status_code=502, detail="The analysis provider returned an invalid response.") from exc
    except LLMConfigurationError as exc:
        logger.warning("Contract analysis provider is not configured document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="Contract analysis is not configured.") from exc
    except LLMProviderError as exc:
        logger.warning("Contract analysis provider failed document_id=%s", document_id)
        raise HTTPException(status_code=502, detail="Contract analysis generation failed.") from exc
    except Exception as exc:
        logger.exception("Contract analysis failed document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="Unable to analyze contract.") from exc

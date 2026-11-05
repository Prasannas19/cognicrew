from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..document_pipeline import extract_pdf_pages, extract_pdf_pages_and_structure
from ..document_profiler import clear_document_profile
from ..models import Document, DocumentLine, DocumentPage, DocumentSection, ResearchQueryDocument, ResearchSource, User
from ..settings import settings
from .research import _doc_cache_invalidate

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
def list_documents(
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, object]:
    """Return all documents owned by the authenticated user."""
    docs = database.scalars(
        select(Document)
        .where(Document.owner_id == user.id)
        .order_by(Document.created_at.desc())
    ).all()
    return {
        "documents": [
            {
                "id": str(doc.id),
                "document_id": str(doc.id),
                "name": doc.original_name,
                "page_count": doc.page_count,
                "byte_size": doc.byte_size,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            }
            for doc in docs
        ]
    }


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_pdf(
    file: UploadFile = File(...),
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, object]:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF uploads are accepted.")
    original_name = Path(file.filename or "document.pdf").name
    doc_id = uuid.uuid4()
    storage_key = f"{user.id}/{doc_id}.pdf"
    destination = settings.upload_directory / storage_key
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    size = destination.stat().st_size
    if size == 0 or size > settings.max_upload_bytes:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="PDF is empty or exceeds the upload limit.")
    try:
        pages, doc_structure = extract_pdf_pages_and_structure(destination, original_name)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="The PDF could not be read page by page.") from exc
    document = Document(
        id=doc_id,
        owner_id=user.id, original_name=original_name, storage_key=storage_key,
        mime_type="application/pdf", byte_size=size, page_count=len(pages),
    )
    database.add(document)
    database.flush()

    page_id_by_num = {}
    for page in pages:
        p_id = uuid.uuid4()
        page_id_by_num[page.pdf_page_number] = p_id
        database.add(DocumentPage(
            id=p_id,
            document_id=document.id, pdf_page_number=page.pdf_page_number,
            printed_page_label=page.printed_page_label, page_kind=page.page_kind.value,
            extracted_text=page.text,
        ))
    database.flush()

    for page in pages:
        p_id = page_id_by_num[page.pdf_page_number]
        for line in page.lines:
            database.add(DocumentLine(
                id=uuid.uuid4(),
                document_id=document.id,
                page_id=p_id,
                pdf_page_number=page.pdf_page_number,
                line_number=line.line_number,
                original_line_order=line.original_line_order,
                line_text=line.line_text,
                font_size=line.font_size,
                is_bold=line.is_bold,
                bbox={"bbox": list(line.bbox)} if line.bbox else None,
            ))

    for s in doc_structure.sections:
        database.add(DocumentSection(
            id=uuid.uuid4(),
            document_id=document.id,
            entity_type=s.entity_type,
            entity_number=s.entity_number,
            title=s.title,
            start_page=s.start_page,
            start_line=s.start_line,
            end_page=s.end_page,
            end_line=s.end_line,
            parent_id=None,
            content=s.content,
            metadata_json={"sub_entities_count": len(s.sub_entities)},
        ))

    database.commit()
    return {
        "id": str(document.id),
        "document_id": str(document.id),
        "name": document.original_name,
        "page_count": document.page_count,
        "byte_size": document.byte_size,
    }


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_document(
    document_id: uuid.UUID,
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Response:
    doc = database.scalar(
        select(Document).where(Document.id == document_id, Document.owner_id == user.id)
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        # Remove sources that reference this document's pages first, otherwise
        # cascading the pages would NULL them out and violate the check
        # constraint demanding document sources keep a page reference.
        page_ids = select(DocumentPage.id).where(DocumentPage.document_id == doc.id)
        database.execute(
            delete(ResearchSource).where(ResearchSource.document_page_id.in_(page_ids))
        )
        # Detach research-history links that would otherwise block deletion.
        database.execute(
            delete(ResearchQueryDocument).where(
                ResearchQueryDocument.document_id == doc.id
            )
        )
        # Remove the file from storage
        storage_path = settings.upload_directory / doc.storage_key
        storage_path.unlink(missing_ok=True)
        database.delete(doc)  # document_pages cascade via ondelete="CASCADE"
        database.commit()
        _doc_cache_invalidate(doc.id)
        clear_document_profile(str(doc.id))
    except Exception:
        database.rollback()
        raise HTTPException(
            status_code=500, detail="Could not delete the document. Please try again."
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..models import ChatMessage, Report, ResearchQuery, ResearchQueryDocument, ResearchSource, User

logger = logging.getLogger("cognicrew.history")

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/reports")
def list_reports(
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """List the authenticated user's past reports, newest first.

    TASK 3A (v5): lightweight metadata only. The old implementation loaded the
    full Report entities (including the potentially large ``body`` JSONB) for
    every row AND issued one extra ``database.get(ResearchQuery, ...)`` round
    trip PER report (N+1) just to render a list. This endpoint now selects only
    the columns the list view needs in a single query; full content (body,
    sources, messages) is fetched per-report by GET /history/reports/{id}.
    """
    import time as _time

    t0 = _time.perf_counter()
    rows = database.execute(
        select(
            Report.id,
            Report.summary,
            Report.not_found,
            Report.created_at,
            ResearchQuery.question,
            ResearchQuery.mode,
        )
        .join(ResearchQuery, Report.query_id == ResearchQuery.id)
        .where(Report.owner_id == user.id)
        .order_by(Report.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    items = [
        {
            "report_id": str(row.id),
            "question": row.question or "",
            "mode": row.mode or "",
            "summary": row.summary,
            "not_found": row.not_found,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]
    logger.info("history list: %d item(s) in %.3fs", len(items), _time.perf_counter() - t0)
    return {"reports": items, "total": len(items), "offset": offset}


@router.get("/reports/{report_id}")
def get_report(
    report_id: uuid.UUID,
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, object]:
    """Return a full report with all sources and chat messages."""
    report = database.scalar(
        select(Report).where(Report.id == report_id, Report.owner_id == user.id)
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    query = database.get(ResearchQuery, report.query_id)
    sources_raw = database.scalars(
        select(ResearchSource).where(ResearchSource.report_id == report.id)
    ).all()
    doc_page_ids = [
        src.document_page_id
        for src in sources_raw
        if src.source_type == "document" and src.document_page_id
    ]
    pages_by_id = {}
    if doc_page_ids:
        from ..models import DocumentPage
        pages_found = database.scalars(
            select(DocumentPage).where(DocumentPage.id.in_(doc_page_ids))
        ).all()
        pages_by_id = {p.id: p for p in pages_found}

    sources = []
    for src in sources_raw:
        source_item: dict[str, object] = {
            "id": str(src.id),
            "source_type": src.source_type,
            "title": src.title,
            "url": src.url,
            "excerpt": src.excerpt,
        }
        if src.source_type == "document" and src.document_page_id:
            page = pages_by_id.get(src.document_page_id)
            if page:
                source_item["pdf_page_number"] = page.pdf_page_number
                source_item["printed_page_label"] = page.printed_page_label
        sources.append(source_item)
    messages_raw = database.scalars(
        select(ChatMessage)
        .where(ChatMessage.report_id == report.id)
        .order_by(ChatMessage.created_at)
    ).all()
    messages = [
        {
            "id": str(msg.id),
            "role": msg.role,
            "text": msg.text,
            "citations": msg.citations,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        }
        for msg in messages_raw
    ]
    return {
        "report_id": str(report.id),
        "question": query.question if query else "",
        "mode": query.mode if query else "",
        "summary": report.summary,
        "body": report.body,
        "not_found": report.not_found,
        "sources": sources,
        "messages": messages,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


@router.delete("/reports/{report_id}", status_code=204)
def delete_report(
    report_id: uuid.UUID,
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Response:
    """TASK 3D (v5): delete one research report from the authenticated user's history.

    Ownership is enforced (like every other history endpoint) by filtering on
    ``Report.owner_id == user.id``; other users' reports are untouched and 404
    for this caller. Deletion order: chat messages and sources first (they are
    linked to the report), then the report, then the now-orphaned research
    query (and its document links). Uploaded DOCUMENTS are never touched.
    """
    report = database.scalar(
        select(Report).where(Report.id == report_id, Report.owner_id == user.id)
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    report_id_val = report.id
    query_id_val = report.query_id

    # Fast path (single round trip): the schema declares ON DELETE CASCADE from
    # reports, research_sources, chat_messages and research_query_documents up
    # to their parents (database/migrations/001_initial.sql and the ORM
    # ondelete= declarations), so deleting the research query removes the whole
    # report chain server-side. Report.query_id is UNIQUE (1:1), so exactly one
    # report is affected. Ownership was verified above.
    try:
        database.execute(delete(ResearchQuery).where(ResearchQuery.id == query_id_val))
        database.commit()
    except Exception:
        database.rollback()

    still_there = database.scalar(select(Report.id).where(Report.id == report_id_val))
    if still_there is not None:
        # The live schema lacks (part of) the cascades: fall back to explicit
        # child-first deletes in dependency order.
        database.execute(delete(ChatMessage).where(ChatMessage.report_id == report_id_val))
        database.execute(delete(ResearchSource).where(ResearchSource.report_id == report_id_val))
        database.execute(delete(Report).where(Report.id == report_id_val))
        database.execute(delete(ResearchQueryDocument).where(ResearchQueryDocument.query_id == query_id_val))
        database.execute(delete(ResearchQuery).where(ResearchQuery.id == query_id_val))
        database.commit()
    logger.info("history delete: report=%s owner=%s", report_id_val, user.id)
    return Response(status_code=204)

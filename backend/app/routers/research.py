from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

# Phase 19/20: Page-number query detection
_PAGE_QUERY_RX = re.compile(
    r"(?i)\b(?:page|pg|p)\.?\s*(\d{1,4})\b|\b(\d{1,4})(?:st|nd|rd|th)?\s+page\b"
)

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

logger = logging.getLogger("cognicrew.research")

from ..answer_engine import get_answer_engine, source_display_name
from ..database import SessionLocal, get_db
from ..dependencies import current_user
from ..intent_classifier import UserRequestIntent, classify_user_intent, extract_requested_word_count
from ..response_safety import normalize_server_exception
from ..document_pipeline import (
    DocumentMetadata,
    DocumentStructureIndex,
    Evidence,
    LineRecord,
    PageKind,
    PageRecord,
    QueryIntent,
    SectionEntity,
    _answer_from_evidence,
    _format_evidence_for_frontend,
    _source_count_label,
    build_document_structure_index,
    classify_query,
    closest_match_evidence,
    extract_pdf_pages_and_structure,
    is_poor_extraction,
    ocr_pdf_page,
    retrieve_document_evidence,
    sanitize_line_range,
)
from ..models import (
    ChatMessage,
    Document,
    DocumentLine,
    DocumentPage,
    DocumentSection,
    Report,
    ResearchQuery,
    ResearchQueryDocument,
    ResearchSource,
    User,
)
from ..settings import settings
from ..web_search import search_exa

router = APIRouter(prefix="/research", tags=["research"])

# ---------------------------------------------------------------------------
# Per-document page/line/section cache.
#
# Why: the live database is a remote Neon Postgres (us-east-1) reached over a
# lossy, high-latency path (~320ms RTT, 100% ICMP loss, 3.9s-56.9s swings for
# the SAME 702-row SELECT on one warm connection — see scripts/controlled_probe.py
# output in the bug-fix notes). Re-loading every document's pages/lines on each
# request made Document Research take 30-60s+.
#
# Uploaded documents are immutable (no update endpoint) and document isolation
# is unaffected: the owner check against the live documents table still runs on
# every request BEFORE the cache is consulted, and deleted documents are never
# served from the cache. Cache entries hold the same read-only PageRecord /
# LineRecord / SectionEntity objects that the request path builds today; every
# consumer only reads them.
# ---------------------------------------------------------------------------
_DOC_CACHE_TTL_SECONDS = 900.0
_DOC_CACHE_MAX_ENTRIES = 32
_doc_pages_cache: dict[uuid.UUID, tuple[float, tuple[PageRecord, ...], dict[tuple[str, int], uuid.UUID], dict[str, DocumentStructureIndex]]] = {}
_doc_cache_lock = threading.Lock()


def _doc_cache_get(doc_id: uuid.UUID):
    with _doc_cache_lock:
        entry = _doc_pages_cache.get(doc_id)
        if entry is None:
            return None
        expires_at, pages, page_ids, structures = entry
        if time.monotonic() > expires_at:
            del _doc_pages_cache[doc_id]
            return None
        return pages, page_ids, structures


def _doc_cache_put(
    doc_id: uuid.UUID,
    pages: tuple[PageRecord, ...],
    page_ids: dict[tuple[str, int], uuid.UUID],
    structures: dict[str, DocumentStructureIndex],
) -> None:
    with _doc_cache_lock:
        if len(_doc_pages_cache) >= _DOC_CACHE_MAX_ENTRIES and doc_id not in _doc_pages_cache:
            oldest = min(_doc_pages_cache, key=lambda k: _doc_pages_cache[k][0])
            del _doc_pages_cache[oldest]
        _doc_pages_cache[doc_id] = (time.monotonic() + _DOC_CACHE_TTL_SECONDS, pages, page_ids, structures)


def _doc_cache_invalidate(doc_id: uuid.UUID) -> None:
    with _doc_cache_lock:
        _doc_pages_cache.pop(doc_id, None)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    mode: str = Field(pattern="^(documents|topic|combined)$")
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=12)
    web_search_enabled: bool = Field(default=True)
    depth: str | None = Field(default="detailed")


class FollowUpRequest(BaseModel):
    question: str | None = Field(default=None, max_length=4000)
    query: str | None = Field(default=None, max_length=4000)
    web_search_enabled: bool | None = Field(default=None)
    depth: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_question_or_query(self) -> FollowUpRequest:
        if not self.question and not self.query:
            raise ValueError("Either question or query must be provided.")
        if not self.question and self.query:
            self.question = self.query
        return self


# ---------------------------------------------------------------------------
# TASK 3 (v11): follow-up context resolution helpers.
#
# Detects pronouns / vague references in follow-up queries that require
# preceding thread context (previous question, previous answer, original
# summary) to resolve to a concrete retrieval query.
# ---------------------------------------------------------------------------

# Pronouns and anaphoric references that signal the query depends on prior
# context. Anchored to word boundaries so 'item' or 'iteration' don't trigger
# retrieval: 'it', 'that', 'this', 'they', or bare anaphora like 'explain
# further' / 'why is that important'. The user's original phrasing is still
# what gets stored and displayed; resolution only rewrites the RETRIEVAL query.
_VAGUE_SUBJECT_RX = re.compile(
    r"\b(it|that|this|they|them|those|these|he|she|one|which)\b|"
    r"^(explain|elaborate|expand|tell me more|go on|continue|and)\b|"
    r"\b(example of it|for example)\b",
    re.I,
)


def _is_vague_followup(question: str) -> bool:
    """TASK 3 (v11): does this follow-up need thread context to resolve?"""
    return bool(_VAGUE_SUBJECT_RX.search(question))


def _resolve_followup_subject(payload_question: str, bundle: dict[str, object]) -> str | None:
    """TASK 3 (v11): rewrite a vague follow-up into a concrete retrieval query.

    Reads ONLY the context bundle (previous question, previous answer, original
    summary). The resolved subject is the strongest salient noun phrase from
    that bundle: a quoted phrase, or the longest Title-Case run, or a keyword
    overlap with the previous answer, or the previous question's content words.
    Returns None when the bundle holds nothing usable (caller falls back to the
    verbatim question — never worse than pre-v11 behavior).
    """
    candidates: list[str] = []
    summary = str(bundle.get("original_summary") or "")
    prev_answer = str(bundle.get("previous_answer") or "")
    prev_question = str(bundle.get("previous_question") or "")
    # TASK 3B (v11): evidence excerpts ALREADY surfaced in this report are part
    # of the bundle — the subject usually appears verbatim in them ('The
    # Discipline of Perception (how we see ...)').
    raw_excerpts = bundle.get("evidence_excerpts")
    excerpt_text = " ".join(
        str(e) for e in (raw_excerpts if isinstance(raw_excerpts, list) else [])[:3]
    )

    # 1) Quoted phrases from any bundle text ("The Discipline of Perception").
    # A trailing period inside the quotes marks a quoted SENTENCE/gloss rather
    # than a concept name ('"how we see and perceive the world around us."'),
    # so keep it for the candidate pool but let salience prefer clean names.
    for text in (prev_answer, prev_question, summary):
        for m in re.finditer(r'"([^"\n]{3,80})"', text):
            candidates.append(m.group(1).strip())

    # 2) Title-Case runs of 2+ words from the summary / previous answer head /
    #    stored evidence excerpts
    for text in (summary, prev_answer[:600], excerpt_text[:1200]):
        for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+(?:of|the|and)?\s*[A-Z][a-z]{2,}){1,4})\b", text):
            candidates.append(m.group(1).strip())

    # 3) Capitalized acronym or distinctive single tokens from the summary
    for m in re.finditer(r"\b([A-Z]{2,}[a-z0-9]*)\b", summary):
        candidates.append(m.group(1))

    # 4) Previous question content words (>= 4 chars, not interrogatives, or digits/identifiers)
    stopish = {"what", "when", "where", "which", "who", "why", "how", "does", "is", "are",
               "mean", "meaning", "explain", "about", "tell", "with",
               "example", "detail", "more", "that", "this", "they", "them"}
    pq_tokens = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", prev_question)
                 if w.lower() not in stopish and (len(w) >= 4 or w.isdigit() or w.lower() in {"q", "q1", "q2", "q3", "q4", "q5"})]
    if pq_tokens:
        candidates.append(" ".join(pq_tokens))

    if not candidates:
        return None

    # The thread's ORIGINAL question is the canonical subject anchor: a
    # candidate that appears verbatim in it (e.g. 'The Discipline of
    # Perception' in 'what do you mean by The Discipline of Perception') is
    # what the user already established as the topic of this thread.
    orig_q = str(bundle.get("original_question") or "").lower()

    def _salience(c: str) -> int:
        score = 0
        low = c.lower().rstrip(".!? ")
        words = low.split()
        # how specific is it? moderately long, Title-Case, corroborated by the
        # stored bundle. Length bonus is capped at 3 words so a quoted 9-word
        # gloss can't outscore the 3-word concept name it came from.
        score += min(len(words), 3) * 2
        # Title-Case CONCEPT NAME: first word capitalized, no terminal period.
        if c[:1].isupper():
            score += 2
        if c.endswith((".", "!", "?")):
            score -= 2
        # every CONTENT word capitalized (a proper noun phrase) is stronger
        # evidence of a concept name than a sentence fragment. Small function
        # words ('of', 'the', ...) inside a Title-Case run don't disqualify it
        # ('The Discipline of Perception' IS a concept name).
        function_words = {"of", "the", "and", "a", "an", "to", "in", "for"}
        content_words = [w for w in c.split() if w.lower() not in function_words]
        if len(content_words) > 1 and all(w[:1].isupper() for w in content_words):
            score += 3
        if summary and low in summary.lower():
            score += 3
        if orig_q and low in orig_q:
            score += 4
        if excerpt_text and low in excerpt_text.lower():
            score += 2
        if prev_answer and low in prev_answer.lower():
            score += 1
        if any(ch.isdigit() for ch in c):
            score += 5
        return score

    best = max(candidates, key=_salience)
    return best.strip() or None


def _rewritten_followup_query(question: str, resolved_subject: str) -> str:
    """TASK 3 (v11): splice the resolved subject into the follow-up text.

    'explain it with an example in detail' + subject 'The Discipline of
    Perception' -> 'explain The Discipline of Perception with an example in
    detail'. Unresolved pronouns are replaced; otherwise the subject is
    appended so retrieval sees BOTH the user's words and the subject.
    """
    rewritten = re.sub(
        r"\b(that\s+question|this\s+question|that|this|it|they|them|those|these)\b",
        resolved_subject,
        question,
        flags=re.I,
    )
    if rewritten == question:
        rewritten = f"{question} {resolved_subject}"
    return re.sub(r"\s{2,}", " ", rewritten).strip()


def _is_standalone_topic_question(question: str, bundle: dict[str, object] | str | None) -> bool:
    """Determine if a question in a Topic Research thread is a standalone query rather than a follow-up."""
    if not bundle:
        return True

    q_lower = question.strip().lower()

    # 1. Distinct tasks (code, creative, math) are standalone unless explicitly referring to previous context
    has_deictic = bool(re.search(r"\b(this|that|it|its|above|previous|earlier)\b", q_lower))
    from ..intent_classifier import _CODE_RX, _CREATIVE_RX, _MATH_RX
    if (_CODE_RX.search(question) or _CREATIVE_RX.search(question) or _MATH_RX.search(question)) and not has_deictic:
        return True

    # 2. Pronoun / anaphora check -> genuine follow-up
    if _VAGUE_SUBJECT_RX.search(question):
        return False

    continuation_cues = (
        "can i", "can we", "am i", "should i", "when does", "what is its", "what are its",
        "how much", "prize", "prizes", "deadline", "submission", "eligibility", "eligible",
        "requirements", "who can", "where is it", "dates", "when did", "how to apply",
        "apply for this", "participate", "rules"
    )
    if any(cue in q_lower for cue in continuation_cues):
        return False

    # 3. Check for standalone definition/knowledge patterns:
    standalone_patterns = (
        r"^(?:tell\s+me\s+)?what\s+(?:do\s+you\s+mean\s+by|is|are)\s+([a-z0-9\s\-]+)",
        r"^what\s+does\s+([a-z0-9\s\-]+)\s+mean",
        r"^explain\s+([a-z0-9\s\-]+)",
        r"^define\s+([a-z0-9\s\-]+)",
        r"^who\s+is\s+([a-z0-9\s\-]+)",
    )
    if isinstance(bundle, str):
        prior_text = bundle.strip().lower()
    else:
        prev_q = str(bundle.get("previous_question") or "").lower()
        orig_q = str(bundle.get("original_question") or "").lower()
        prior_text = f"{prev_q} {orig_q}".strip()

    for pat in standalone_patterns:
        m = re.match(pat, q_lower)
        if m:
            subject = m.group(1).strip()
            if subject and not any(part in prior_text for part in subject.split() if len(part) >= 4):
                return True

    # 4. Check entity overlap against prior questions

    stop_words = {
        "what", "when", "where", "which", "who", "why", "how", "does", "mean",
        "tell", "explain", "about", "detail", "more", "with", "this", "that", "from",
        "write", "code", "python", "make", "user", "file", "data", "list", "give"
    }
    q_words = [w for w in re.findall(r"[a-z0-9]{4,}", q_lower) if w not in stop_words]

    if q_words and any(w in prior_text for w in q_words):
        return False

    return True


def _direct_answer_lead(ai_answer: str | None, limit: int = 400) -> str | None:
    """TASK 2 (v5): derive a SHORT lead-in from the single AI generation.

    The AI provider is invoked exactly ONCE per research run (proven by the
    STAGE6 logs: exactly one "STAGE6 nvidia call" line per request). The old
    code assigned that same full markdown text to BOTH the top-level
    ``direct_answer`` field AND the "Synthesized Answer" section body, so the
    frontend rendered the identical content twice under two headings.
    ``direct_answer`` is now a brief excerpt of the same single generation
    (first non-heading paragraph) while the full text stays only in the
    "Synthesized Answer" section.
    """
    if not ai_answer:
        return None
    lines = ai_answer.splitlines()
    for paragraph in lines:
        text = paragraph.strip()
        if not text:
            continue
        # Skip structural/formatting lines so the lead is real prose: headings,
        # horizontal rules, table rows, blockquotes, bullets and bold labels.
        if text.startswith(("#", "---", "**", "|", ">", "- ", "* ")):
            continue
        lead = text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
        return lead
    # The answer was all headings/tables/bullets: fall back to the first
    # non-empty line (still a short excerpt, never the full section body).
    for paragraph in lines:
        text = paragraph.strip()
        if text:
            lead = text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
            return lead
    lead = ai_answer.strip()
    return lead if len(lead) <= limit else lead[: limit - 1].rstrip() + "…"


def _ocr_fallback_for_page(
    database: Session,
    document_ids: Sequence[uuid.UUID],
    page_rec: PageRecord,
) -> tuple[str, list[LineRecord]]:
    """Selective local OCR fallback for a scanned/image-only PDF page."""
    for doc_id in document_ids:
        doc_rec = database.scalar(select(Document).where(Document.id == doc_id))
        if doc_rec:
            doc_path = settings.upload_directory / doc_rec.storage_key
            if doc_path.exists():
                text, lines = ocr_pdf_page(doc_path, page_rec.pdf_page_number)
                if text and not is_poor_extraction(text):
                    p_row = database.scalar(select(DocumentPage).where(
                        DocumentPage.document_id == doc_id,
                        DocumentPage.pdf_page_number == page_rec.pdf_page_number
                    ))
                    if p_row:
                        p_row.extracted_text = text
                        # Delete existing lines for this page to prevent unique constraint violation
                        database.execute(
                            delete(DocumentLine).where(
                                DocumentLine.document_id == doc_id,
                                DocumentLine.pdf_page_number == page_rec.pdf_page_number,
                            )
                        )
                        for ol in lines:
                            database.add(DocumentLine(
                                id=uuid.uuid4(),
                                document_id=doc_id,
                                page_id=p_row.id,
                                pdf_page_number=page_rec.pdf_page_number,
                                line_number=ol.line_number,
                                original_line_order=ol.original_line_order,
                                line_text=ol.line_text,
                                font_size=ol.font_size,
                                is_bold=ol.is_bold,
                                bbox=None
                            ))
                        database.commit()
                        _doc_cache_invalidate(doc_id)
                    return text, lines
    return "", []


def _document_pages(
    database: Session, user_id: uuid.UUID, ids: Sequence[uuid.UUID]
) -> tuple[list[PageRecord], dict[tuple[str, int], uuid.UUID], dict[str, DocumentStructureIndex]]:
    stage_start = time.perf_counter()
    logger.info("STAGE2 _document_pages: entering for %d document(s)", len(ids))
    if not ids:
        return [], {}, {}
    documents = database.scalars(
        select(Document).where(Document.id.in_(ids), Document.owner_id == user_id)
    ).all()
    if len(documents) != len(set(ids)):
        raise HTTPException(status_code=404, detail="One or more selected documents were not found.")

    doc_structures: dict[str, DocumentStructureIndex] = {}
    doc_page_maps: dict[uuid.UUID, dict[tuple[str, int], uuid.UUID]] = {}
    doc_pages_by_id: dict[uuid.UUID, tuple[PageRecord, ...]] = {}
    doc_structs_by_id: dict[uuid.UUID, dict[str, DocumentStructureIndex]] = {}
    doc_order: list[uuid.UUID] = []
    cached_docs: dict[uuid.UUID, tuple[tuple[PageRecord, ...], dict[tuple[str, int], uuid.UUID], dict[str, DocumentStructureIndex]]] = {}

    # Fast path: serve per-document loads from the immutable-document cache.
    # Ownership was already verified above against the live documents table.
    remaining: list[Document] = []
    for doc in documents:
        doc_order.append(doc.id)
        entry = _doc_cache_get(doc.id)
        if entry is None:
            remaining.append(doc)
        else:
            cached_docs[doc.id] = entry
    if cached_docs:
        logger.info("STAGE2 _document_pages: served %d document(s) from cache, %d to load", len(cached_docs), len(remaining))
    documents = remaining

    for doc in documents:
        # Check if lines exist in document_lines table
        lines_count = database.scalar(
            select(func.count(DocumentLine.id)).where(DocumentLine.document_id == doc.id)
        )
        storage_path = settings.upload_directory / doc.storage_key
        if (lines_count == 0 or lines_count is None) and storage_path.exists():
            # Backfill on the fly for previously uploaded documents
            try:
                extracted_pages, struct_idx = extract_pdf_pages_and_structure(storage_path, doc.original_name)
                p_rows = database.scalars(
                    select(DocumentPage).where(DocumentPage.document_id == doc.id)
                ).all()
                p_map = {row.pdf_page_number: row.id for row in p_rows}
                for ep in extracted_pages:
                    pid = p_map.get(ep.pdf_page_number)
                    if not pid:
                        continue
                    for l in ep.lines:
                        database.add(DocumentLine(
                            id=uuid.uuid4(),
                            document_id=doc.id,
                            page_id=pid,
                            pdf_page_number=ep.pdf_page_number,
                            line_number=l.line_number,
                            original_line_order=l.original_line_order,
                            line_text=l.line_text,
                            font_size=l.font_size,
                            is_bold=l.is_bold,
                            bbox={"bbox": list(l.bbox)} if l.bbox else None,
                        ))
                for s in struct_idx.sections:
                    database.add(DocumentSection(
                        id=uuid.uuid4(),
                        document_id=doc.id,
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
            except Exception:
                database.rollback()

        # Load DocumentPages
        p_rows = database.scalars(
            select(DocumentPage)
            .where(DocumentPage.document_id == doc.id)
            .order_by(DocumentPage.pdf_page_number.asc())
        ).all()
        # Load DocumentLines
        line_rows = database.scalars(
            select(DocumentLine)
            .where(DocumentLine.document_id == doc.id)
            .order_by(DocumentLine.pdf_page_number.asc(), DocumentLine.line_number.asc())
        ).all()
        lines_by_page: dict[int, list[LineRecord]] = {}
        for lr in line_rows:
            raw_bbox = lr.bbox.get("bbox") if (lr.bbox and "bbox" in lr.bbox) else None
            lines_by_page.setdefault(lr.pdf_page_number, []).append(LineRecord(
                line_number=lr.line_number,
                line_text=lr.line_text,
                original_line_order=lr.original_line_order,
                font_size=lr.font_size or 0.0,
                is_bold=lr.is_bold,
                bbox=(float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])) if raw_bbox else None,
            ))

        # Load DocumentSections
        sec_rows = database.scalars(
            select(DocumentSection)
            .where(DocumentSection.document_id == doc.id)
            .order_by(DocumentSection.start_page.asc(), DocumentSection.start_line.asc())
        ).all()
        sections = [
            SectionEntity(
                entity_type=sr.entity_type,
                entity_number=sr.entity_number,
                title=sr.title,
                start_page=sr.start_page,
                start_line=sr.start_line,
                end_page=sr.end_page,
                end_line=sr.end_line,
                content=sr.content,
            )
            for sr in sec_rows
        ]

        doc_pages: list[PageRecord] = []
        doc_page_ids: dict[tuple[str, int], uuid.UUID] = {}
        for pr in p_rows:
            p_lines = tuple(lines_by_page.get(pr.pdf_page_number, []))
            if not p_lines and pr.extracted_text:
                p_lines = tuple(
                    LineRecord(i + 1, lt.strip(), i + 1)
                    for i, lt in enumerate(pr.extracted_text.splitlines()) if lt.strip()
                )
            p_sections = tuple(s for s in sections if s.start_page <= pr.pdf_page_number <= s.end_page)
            rec = PageRecord(
                document_name=doc.original_name,
                pdf_page_number=pr.pdf_page_number,
                text=pr.extracted_text,
                page_kind=PageKind(pr.page_kind),
                printed_page_label=pr.printed_page_label,
                lines=p_lines,
                sections=p_sections,
                document_id=str(doc.id),
            )
            doc_pages.append(rec)
            doc_page_ids[(str(doc.id), pr.pdf_page_number)] = pr.id
            doc_page_ids[(doc.original_name, pr.pdf_page_number)] = pr.id

        # Always build complete structure index to ensure all sections and sub_entities are populated
        struct_idx = build_document_structure_index(doc.original_name, doc_pages)
        doc_structures[str(doc.id)] = struct_idx
        doc_structures[doc.original_name] = struct_idx
        _doc_cache_put(
            doc.id,
            tuple(doc_pages),
            dict(doc_page_ids),
            {str(doc.id): struct_idx, doc.original_name: struct_idx},
        )
        doc_page_maps[doc.id] = dict(doc_page_ids)
        doc_pages_by_id[doc.id] = tuple(doc_pages)
        doc_structs_by_id[doc.id] = {str(doc.id): struct_idx, doc.original_name: struct_idx}

    # Merge results in the caller's original document order.
    all_pages: list[PageRecord] = []
    page_ids: dict[tuple[str, int], uuid.UUID] = {}
    for doc_id in doc_order:
        if doc_id in cached_docs:
            c_pages, c_page_ids, c_structs = cached_docs[doc_id]
        else:
            c_pages = doc_pages_by_id[doc_id]
            c_page_ids = doc_page_maps[doc_id]
            c_structs = doc_structs_by_id[doc_id]
        all_pages.extend(c_pages)
        page_ids.update(c_page_ids)
        doc_structures.update(c_structs)
    return all_pages, page_ids, doc_structures


@router.post("/run")
def run_research(payload: ResearchRequest, database: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, object]:
    user_id = user.id
    user_email = user.email
    t0 = time.perf_counter()
    logger.info(
        "STAGE1 request: user=%s mode=%s docs=%d question=%r",
        user_email, payload.mode, len(payload.document_ids), payload.question[:120],
    )
    if payload.mode in {"documents", "combined"} and not payload.document_ids:
        raise HTTPException(status_code=422, detail="Document and Combined Research require at least one uploaded PDF.")

    web_search_active = payload.web_search_enabled if payload.mode == "topic" else True
    exa_called = False
    exa_key = settings.exa_api_key

    # Topic research with web search ON requires EXA_API_KEY
    if payload.mode == "topic" and web_search_active and not exa_key:
        raise HTTPException(status_code=503, detail="EXA_API_KEY is not configured.")
    if payload.mode == "combined" and not exa_key:
        raise HTTPException(status_code=503, detail="EXA_API_KEY is not configured.")

    stage_start = time.perf_counter()
    pages, page_ids, doc_structures = _document_pages(database, user_id, payload.document_ids)
    logger.info("STAGE3 document pages loaded: pages=%d elapsed=%.3fs", len(pages), time.perf_counter() - stage_start)
    query_id = uuid.uuid4()
    query = ResearchQuery(id=query_id, owner_id=user_id, question=payload.question.strip(), mode=payload.mode, status="running")
    database.add(query)
    database.flush()
    database.add_all(ResearchQueryDocument(query_id=query_id, document_id=document_id) for document_id in payload.document_ids)
    database.commit()

    stage_start = time.perf_counter()

    # Phase 19/20: Intercept direct page queries for documents mode.
    # Pattern: "tell me info on page 100", "what is on page 5", etc.
    _page_query_num: int | None = None
    _page_direct_result: str | None = None
    _exact_page: PageRecord | None = None
    if pages:
        _m = _PAGE_QUERY_RX.search(payload.question)
        if _m:
            _page_query_num = int(_m.group(1) or _m.group(2))
            # Find the exact page
            _exact_page = next(
                (p for p in pages if p.pdf_page_number == _page_query_num), None
            )
            if _exact_page is None:
                # Try printed page label match
                _exact_page = next(
                    (p for p in pages if p.printed_page_label and str(p.printed_page_label).strip() == str(_page_query_num)),
                    None
                )
            if _exact_page:
                page_text = (_exact_page.text or "").strip()
                if is_poor_extraction(page_text):
                    logger.info("PAGE QUERY: page=%d poor extraction (%d chars), attempting selective OCR fallback", _page_query_num, len(page_text))
                    ocr_t, ocr_l = _ocr_fallback_for_page(database, payload.document_ids, _exact_page)
                    if ocr_t and not is_poor_extraction(ocr_t):
                        _exact_page = PageRecord(
                            document_name=_exact_page.document_name,
                            pdf_page_number=_exact_page.pdf_page_number,
                            text=ocr_t,
                            page_kind=_exact_page.page_kind,
                            printed_page_label=_exact_page.printed_page_label,
                            lines=tuple(ocr_l),
                            sections=_exact_page.sections,
                        )
                        page_text = ocr_t
                if page_text:
                    _page_direct_result = page_text
                    logger.info(
                        "PAGE QUERY: direct retrieval page=%d printed=%s text_len=%d",
                        _page_query_num, _exact_page.printed_page_label, len(page_text)
                    )
                else:
                    logger.info(
                        "PAGE QUERY: page=%d exists but has no extractable text and OCR could not extract text",
                        _page_query_num
                    )
                    _page_direct_result = f"PDF page {_page_query_num} appears to be a scanned or image-only page and OCR could not extract readable text."
            else:
                _page_direct_result = f"PDF page {_page_query_num} was not found in this document. The document has {len(pages)} pages."

    try:
        database.close()
    except Exception:
        pass

    if _page_direct_result is not None and _exact_page is not None:
        document_evidence = [
            Evidence(
                document_name=_exact_page.document_name,
                pdf_page_number=_exact_page.pdf_page_number,
                printed_page_label=_exact_page.printed_page_label,
                excerpt=(_page_direct_result or "")[:500],
                page_kind=_exact_page.page_kind,
                key_sentences=tuple(l.strip() for l in (_page_direct_result or "").splitlines()[:2] if l.strip()),
                matched_heading=f"Page {_exact_page.pdf_page_number}",
                relevance_score=1.0,
                start_line=1,
                end_line=len(_exact_page.lines) if _exact_page.lines else 1,
                section_name=f"Page {_exact_page.pdf_page_number}",
            )
        ]
    else:
        document_evidence = (
            retrieve_document_evidence(pages, payload.question, doc_structures=doc_structures)
            if bool(payload.document_ids)
            else []
        )
    t_doc_retrieval = time.perf_counter() - stage_start
    logger.info("STAGE4 retrieval done: evidence=%d direct_page=%s elapsed=%.3fs", len(document_evidence), bool(_page_direct_result), t_doc_retrieval)
    t_exa_elapsed = 0.0
    web_sources: list[dict[str, object]] = []
    if payload.mode == "topic" and web_search_active and exa_key:
        t_exa_start = time.perf_counter()
        logger.info("STAGE5_EXA start: query=%r", payload.question[:80])
        web_sources = search_exa(payload.question, exa_key)
        t_exa_elapsed = time.perf_counter() - t_exa_start
        exa_called = True
        logger.info("STAGE5_EXA done: sources=%d elapsed=%.3fs", len(web_sources), t_exa_elapsed)
    elif payload.mode == "combined" and exa_key:
        t_exa_start = time.perf_counter()
        logger.info("STAGE5_EXA start: query=%r", payload.question[:80])
        web_sources = search_exa(payload.question, exa_key)
        t_exa_elapsed = time.perf_counter() - t_exa_start
        exa_called = True
        logger.info("STAGE5_EXA done: sources=%d elapsed=%.3fs", len(web_sources), t_exa_elapsed)

    direct_answer, page_reference = _answer_from_evidence(document_evidence)
    doc_sources_formatted = _format_evidence_for_frontend(document_evidence)
    sources = list(doc_sources_formatted)
    sources.extend(web_sources)

    doc_count_label = _source_count_label(document_evidence, web_sources)
    sections: list[dict[str, object]] = []
    answer_engine = get_answer_engine()
    user_intent = classify_user_intent(payload.question)
    not_found = False
    ai_succeeded: bool = False
    summary: str = ""
    closest_items: list = []  # TASK 7 (v8): closest-match candidates (persisted after report flush)
    t_ai_elapsed = 0.0

    if payload.mode == "documents":
        # Phase 19/20: Direct page query response - bypass RAG
        if _page_direct_result is not None and _page_query_num is not None:
            printed_label = ""
            for p in pages:
                if p.pdf_page_number == _page_query_num and p.printed_page_label:
                    printed_label = f" (Book page: {p.printed_page_label})"
                    break
            page_answer = (
                f"**PDF Page {_page_query_num}{printed_label}**\n\n"
                + _page_direct_result
            )
            direct_answer = page_answer[:400] if len(page_answer) > 400 else page_answer
            sections.append({
                "heading": f"Page {_page_query_num} Content{printed_label}",
                "body": _page_direct_result,
                "source_type": "document",
            })
            summary = f"Direct page retrieval: PDF page {_page_query_num}{printed_label}."
            ai_succeeded = True
            not_found = (_page_direct_result.startswith("PDF page ") and "not found" in _page_direct_result)
            if not not_found and _exact_page is not None:
                ev = Evidence(
                    document_name=_exact_page.document_name,
                    pdf_page_number=_exact_page.pdf_page_number,
                    printed_page_label=_exact_page.printed_page_label,
                    excerpt=_page_direct_result[:800],
                    page_kind=_exact_page.page_kind,
                    key_sentences=(),
                    matched_heading=f"Page {_exact_page.pdf_page_number}",
                    relevance_score=100.0,
                    start_line=1,
                    end_line=len(_exact_page.lines) if _exact_page.lines else 1,
                    section_name=f"Page {_exact_page.pdf_page_number}",
                )
                document_evidence = [ev]
                sources = list(_format_evidence_for_frontend(document_evidence))
                doc_count_label = _source_count_label(document_evidence, web_sources)
                page_reference = f"PDF Page {_exact_page.pdf_page_number}" + (f" (Book page: {_exact_page.printed_page_label})" if _exact_page.printed_page_label else "")
        elif not document_evidence:
            # MANDATORY NO-EVIDENCE SAFETY GATE: Never call AI to hallucinate.
            # TASK 7 (v8): before refusing outright, surface the closest
            # candidate sections found below the acceptance threshold so the
            # user sees where to look next (clearly labeled, never styled as
            # a confirmed grounded answer).
            closest = closest_match_evidence(pages, payload.question, doc_structures=doc_structures)
            if closest:
                not_found = True
                summary = "I couldn't find an exact answer. Here are the closest related sections I found:"
                sections.append({
                    "heading": "Closest match — not confirmed",
                    "body": "\n\n".join(
                        f"[{item.document_name} · PDF p.{item.pdf_page_number}"
                        f"{' · ' + item.printed_page_label if item.printed_page_label else ''}"
                        f" · similarity {item.relevance_score:.0f}%]\n{item.excerpt}"
                        for item in closest
                    ),
                    "source_type": "closest_match",
                })
                closest_items = list(closest)
                closest_formatted = _format_evidence_for_frontend(closest)
                for entry in closest_formatted:
                    entry["source_type"] = "closest_match"
                sources.extend(closest_formatted)
                doc_count_label = "0 confirmed sources · closest matches shown"
                logger.info("STAGE4b closest-match fallback served %d candidate(s)", len(closest))
            else:
                not_found = True
                summary = "I couldn't reliably locate explanatory evidence in the selected documents."
                direct_answer = None
        elif document_evidence:
            stage_start = time.perf_counter()
            logger.info("STAGE5 answer_engine.generate_document_answer: entering with %d evidence items", len(document_evidence))
            t_ai_start = time.perf_counter()
            ai_answer, ai_succeeded = answer_engine.generate_document_answer(
                payload.question,
                document_evidence,
                user_intent=user_intent,
                depth_mode=payload.depth,
            )
            t_ai_elapsed = time.perf_counter() - t_ai_start
            logger.info("STAGE5 answer_engine done: ai_succeeded=%s answer_len=%d elapsed=%.3fs", ai_succeeded, len(ai_answer or ""), t_ai_elapsed)
            # TASK 2 (v5): ONE AI call total. direct_answer is a SHORT lead derived
            # from the same single generation; the full text lives only in the
            # "Synthesized Answer" section (no second generation, no duplicate copy).
            if not ai_answer or not ai_answer.strip():
                ai_succeeded = False
                ai_answer = "Document evidence verified, but AI generation is currently unavailable."
            direct_answer = _direct_answer_lead(ai_answer)
            summary = "Document evidence verified." if ai_succeeded else "Document evidence verified. (AI generation currently unavailable)"
            sections.append({
                "heading": "Synthesized Answer",
                "body": ai_answer,
                "source_type": "ai",
            })
            sections.append({
                "heading": "Evidence from your documents",
                "body": "\n\n".join(item.excerpt for item in document_evidence),
                "source_type": "document",
            })
    elif payload.mode == "topic":
        if web_search_active and not web_sources:
            not_found = True
            summary = "Topic Research completed; no web sources found."
            direct_answer = None
        else:
            t_ai_start = time.perf_counter()
            ai_answer, ai_succeeded = answer_engine.generate_topic_answer(
                payload.question,
                web_sources,
                intent=user_intent,
                web_search_enabled=web_search_active,
                depth_mode=payload.depth,
            )
            t_ai_elapsed = time.perf_counter() - t_ai_start
            page_reference = None
            if not web_search_active:
                # Web Search OFF: 0 web sources, 0 document evidence, no stale references
                if user_intent == UserRequestIntent.DIRECT_QA:
                    # Phase 4: Clean concise direct answer ONLY, no duplicate synthesized answer section, no Executive Summary
                    direct_answer = ai_answer.strip()
                    summary = ""
                    # sections remains empty
                else:
                    direct_answer = _direct_answer_lead(ai_answer)
                    summary = "Topic Research completed using direct AI knowledge (Web Search OFF)."
                    heading_title = "Detailed Explanation" if user_intent == UserRequestIntent.DETAILED_EXPLANATION else "Direct AI Response"
                    sections.append({
                        "heading": heading_title,
                        "body": ai_answer,
                        "source_type": "ai",
                    })
            else:
                direct_answer = _direct_answer_lead(ai_answer)
                if ai_succeeded:
                    heading_title = "Synthesized Answer" if user_intent == UserRequestIntent.DIRECT_QA else "Synthesized Findings"
                    summary = "Topic Research completed with verified web sources."
                    sections.append({
                        "heading": heading_title,
                        "body": ai_answer,
                        "source_type": "ai",
                    })
                else:
                    summary = "Report synthesis failed after all configured AI attempts."
                    sections.append({
                        "heading": "Synthesis Status",
                        "body": "Report synthesis failed after all configured AI attempts.",
                        "source_type": "synthesis_unavailable",
                    })
            if web_sources:
                sections.append({
                    "heading": "Web Sources Retrieved",
                    "body": "\n\n".join(
                        f"### {s.get('title', 'Source')} [{source_display_name(s)}]\n"
                        f"**URL:** {s.get('url', '')}\n\n"
                        f"{str(s.get('excerpt', '')).strip()}"
                        for s in web_sources
                    ),
                    "source_type": "web",
                })
    else:  # combined
        if not document_evidence and not web_sources:
            not_found = True
            summary = "Combined Research completed; no verified evidence found."
            direct_answer = None
        else:
            t_ai_start = time.perf_counter()
            ai_answer, ai_succeeded = answer_engine.generate_combined_answer(
                payload.question,
                document_evidence,
                web_sources,
                depth_mode=payload.depth,
            )
            t_ai_elapsed = time.perf_counter() - t_ai_start
            # TASK 2 (v5): same single-generation short-lead treatment as document mode.
            direct_answer = _direct_answer_lead(ai_answer)
            summary = "Combined document and web evidence synthesized." if ai_succeeded else "Combined evidence verified. (AI generation currently unavailable)"
            sections.append({
                "heading": "Integrated Analysis",
                "body": ai_answer,
                "source_type": "ai",
            })
            if document_evidence:
                sections.append({
                    "heading": "Evidence from your documents",
                    "body": "\n\n".join(item.excerpt for item in document_evidence),
                    "source_type": "document",
                })
            if web_sources:
                sections.append({
                    "heading": "Web Sources",
                    "body": "\n\n".join(str(s.get("excerpt", "")) for s in web_sources),
                    "source_type": "web",
                })

    report_id = uuid.uuid4()
    sources_to_add: list[ResearchSource] = []
    for item in closest_items:
        doc_page_id = (page_ids.get((str(item.document_id), item.pdf_page_number)) if item.document_id else None) or page_ids.get((item.document_name, item.pdf_page_number))
        sources_to_add.append(ResearchSource(
            id=uuid.uuid4(), report_id=report_id, source_type="document",
            document_page_id=doc_page_id,
            title=item.document_name, url=None, excerpt=item.excerpt,
            start_line=item.start_line, end_line=item.end_line,
            section_name=f"Closest match — not confirmed · similarity {item.relevance_score:.0f}%",
            relevance_score=item.relevance_score,
        ))

    for item in document_evidence:
        doc_page_id = (page_ids.get((str(item.document_id), item.pdf_page_number)) if item.document_id else None) or page_ids.get((item.document_name, item.pdf_page_number))
        sources_to_add.append(ResearchSource(
            id=uuid.uuid4(),
            report_id=report_id,
            source_type="document",
            document_page_id=doc_page_id,
            title=item.document_name,
            url=None,
            excerpt=item.excerpt,
            start_line=item.start_line,
            end_line=item.end_line,
            section_name=item.section_name,
            relevance_score=item.relevance_score,
        ))

    for w_item in web_sources:
        sources_to_add.append(ResearchSource(
            id=uuid.uuid4(),
            report_id=report_id,
            source_type="web",
            document_page_id=None,
            title=str(w_item.get("title") or "Web Source"),
            url=str(w_item.get("url") or ""),
            excerpt=str(w_item.get("excerpt") or ""),
            start_line=None,
            end_line=None,
            section_name=None,
            relevance_score=w_item.get("relevance_score"),
        ))

    for attempt in range(3):
        try:
            with SessionLocal() as persist_db:
                q_row = persist_db.scalar(select(ResearchQuery).where(ResearchQuery.id == query_id))
                if q_row:
                    q_row.status = "complete"

                report = Report(id=report_id, query_id=query_id, owner_id=user_id, summary=summary, body=sections, not_found=not_found)
                persist_db.add(report)
                if sources_to_add:
                    persist_db.add_all(sources_to_add)
                persist_db.commit()
                break
        except Exception as persist_exc:
            logger.warning("Attempt %d to persist research report failed: %s", attempt + 1, persist_exc)
            if attempt == 2:
                logger.error("Failed to persist report to DB after retries: %s. Preserving AI answer.", persist_exc)

    t_end = time.perf_counter()
    logger.info("STAGE7 response ready: total_elapsed=%.3fs report_id=%s", t_end - t0, report_id)

    routing_res = getattr(answer_engine, "last_routing_result", None)
    answering_final = getattr(answer_engine, "last_answering_model", None)
    if not ai_succeeded or not answering_final:
        answering_final = None

    attempts_records = getattr(routing_res, "attempts", []) or []
    attempt_count = getattr(routing_res, "attempt_count", 1) if routing_res else 1

    model_attempt_1_ms = round(attempts_records[0].actual_latency * 1000, 1) if len(attempts_records) > 0 else 0.0
    model_attempt_2_ms = round(attempts_records[1].actual_latency * 1000, 1) if len(attempts_records) > 1 else 0.0
    gemini_fallback_ms = round(getattr(routing_res, "gemini_fallback_ms", 0.0), 1)
    if gemini_fallback_ms == 0.0 and len(attempts_records) > 1:
        gemini_fallback_ms = round(sum(a.actual_latency for a in attempts_records[1:] if "gemini" in a.model_id.lower()) * 1000, 1)
    fallback_ms = round(sum(a.actual_latency for a in attempts_records[1:]) * 1000, 1) if len(attempts_records) > 1 else 0.0
    validation_ms = getattr(answer_engine, "last_validation_ms", 0.0) or 0.0

    t_total_ms = round((t_end - t0) * 1000, 1)
    t_exa_ms = round(t_exa_elapsed * 1000, 1)
    t_doc_ms = round(t_doc_retrieval * 1000, 1)
    t_retrieval_ms = round(t_doc_ms + t_exa_ms, 1)
    t_ai_ms = round(t_ai_elapsed * 1000, 1)
    t_formatting_ms = round(max(0.0, t_total_ms - (t_retrieval_ms + t_ai_ms + validation_ms)), 1)

    logger.info(
        "[RESEARCH TELEMETRY] request_id=%s mode=%s outcome=%s model=%s attempts=%d "
        "total_ms=%.1f retrieval_ms=%.1f attempt_1_ms=%.1f attempt_2_ms=%.1f fallback_ms=%.1f ai_ms=%.1f",
        report_id, payload.mode, "SUCCESS" if ai_succeeded else "SYNTHESIS_UNAVAILABLE",
        answering_final or "none", attempt_count,
        t_total_ms, t_retrieval_ms, model_attempt_1_ms, model_attempt_2_ms, fallback_ms, t_ai_ms,
    )

    related_topics: list[str] = []
    if payload.mode == "topic":
        related_topics = answer_engine.generate_related_topics(payload.question, direct_answer or summary or "", web_sources)

    return {
        "report_id": str(report_id),
        "summary": summary,
        "sections": sections,
        "sources": sources,
        "related_topics": related_topics,
        "document_source_count": len(document_evidence),
        "web_source_count": len(web_sources),
        "direct_answer": direct_answer,
        "page_reference": page_reference,
        "source_count_label": doc_count_label,
        "not_found": not_found,
        "answering_model": answering_final,
        "answering_model_final": answering_final,
        "routing_attempts": attempt_count,
        "timings": {
            "total_elapsed_s": round(t_end - t0, 3),
            "total_backend_ms": t_total_ms,
            "total_ms": t_total_ms,
            "retrieval_ms": t_retrieval_ms,
            "doc_retrieval_ms": t_doc_ms,
            "exa_elapsed_s": round(t_exa_elapsed, 3),
            "exa_ms": t_exa_ms,
            "ai_ms": t_ai_ms,
            "model_attempt_1_ms": model_attempt_1_ms,
            "model_attempt_2_ms": model_attempt_2_ms,
            "gemini_fallback_ms": gemini_fallback_ms,
            "fallback_ms": fallback_ms,
            "validation_ms": validation_ms,
            "formatting_ms": t_formatting_ms,
            "provider": getattr(routing_res, "final_provider", None) or (settings.ai_provider if ai_succeeded else None),
            "model": answering_final,
            "model_used": answering_final,
            "success": ai_succeeded,
            "status": "success" if ai_succeeded else "failure",
            "failure_classification": getattr(routing_res, "failure_classification", None),
            "timeout_reason": getattr(routing_res, "timeout_reason", None),
            "attempt_count": attempt_count,
            "web_search_enabled": web_search_active,
            "exa_called": exa_called,
            "exa_latency_ms": t_exa_ms,
            "exa_result_count": len(web_sources),
            "intent": getattr(user_intent, "value", str(user_intent)),
            "requested_word_count": extract_requested_word_count(payload.question),
            "attempts_summary": [
                {"model": a.model_id, "status": a.status, "latency_ms": round(a.actual_latency * 1000, 1)}
                for a in attempts_records
            ],
        },
    }


def _web_citation(web_source: dict[str, object]) -> dict[str, object]:
    """TASK 4 (v8): citation shape for a fresh web source used in follow-ups."""
    return {
        "title": str(web_source.get("title") or "Web Source"),
        "url": str(web_source.get("url") or ""),
        "excerpt": str(web_source.get("excerpt") or ""),
    }


_GENERATIVE_REQUEST_RX = re.compile(
    r"\b(viva|quiz|practice|revision|solve|draft|write|generate|create|give me|suggest|"
    r"summar(y|ize)|explain|prepare|possible questions|expected questions|"
    r"questions (would|could|can) be asked|examples?)\b",
    re.I,
)


def _stored_source_as_evidence(database: Session, src: ResearchSource) -> Evidence | None:
    """TASK 10 (v8): rebuild an Evidence object from the report's own stored
    document source, so generative follow-ups (low keyword overlap) can still
    synthesize from the evidence the report was built on."""
    page = database.get(DocumentPage, src.document_page_id) if src.document_page_id else None
    if not src.excerpt:
        return None
    return Evidence(
        document_name=src.title or "document",
        pdf_page_number=page.pdf_page_number if page else 0,
        printed_page_label=page.printed_page_label if page else None,
        excerpt=src.excerpt,
        page_kind=PageKind.CONTENT,
        start_line=src.start_line,
        end_line=src.end_line,
        section_name=src.section_name,
        relevance_score=float(src.relevance_score or 0.0),
    )


def _clamp_chat_text(text: str) -> str:
    """TASK 4 (v8): keep stored chat messages within the DB CHECK constraint
    (chat_messages.text: length between 1 and 4000). The API response still
    returns the full text; only the persisted copy is truncated."""
    return (text or "").strip()[:4000] or "(empty)"


def _answer_basis(answer: str, doc_count: int, web_count: int) -> str:
    """TASK 5 (v8): explicit label of which evidence type backed a follow-up."""
    if doc_count and web_count:
        return "Answer from both"
    if doc_count:
        return "Answer from document evidence"
    if web_count:
        return "Answer from web sources (not in your document)"
    return answer or "No evidence available for this follow-up."


@router.post("/{report_id}/follow-ups")
def follow_up(report_id: uuid.UUID, payload: FollowUpRequest, database: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, object]:
    report = database.scalar(select(Report).where(Report.id == report_id, Report.owner_id == user.id))
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    question_text = (payload.question or payload.query or "").strip()
    if not question_text:
        raise HTTPException(status_code=422, detail="Question is required.")

    # Load original query for context building
    orig_query = database.scalar(select(ResearchQuery).where(ResearchQuery.id == report.query_id))
    original_mode = (orig_query.mode if orig_query else "") or "documents"
    query_document_ids = database.scalars(select(ResearchQueryDocument.document_id).where(ResearchQueryDocument.query_id == report.query_id)).all()

    context: dict[str, object] = {}
    # TASK 3 (v11): the follow-up context bundle — assembled BEFORE retrieval
    # for EVERY follow-up, so pronouns/vague references can be resolved.
    bundle: dict[str, object] = {
        "followup_question": question_text,
        "original_question": None,
        "original_summary": None,
        "previous_question": None,
        "previous_answer": None,
        "document_scope": [str(d) for d in query_document_ids],
        "evidence_excerpts": [],
    }
    # TASK 3B: the original report's direct answer — prefer the stored
    # 'Synthesized Answer' section body over the status-label summary.
    body_sections = report.body if isinstance(report.body, list) else []
    for sec in body_sections:
        if isinstance(sec, dict) and str(sec.get("heading") or "") == "Synthesized Answer":
            bundle["original_summary"] = str(sec.get("body") or "")[:1500]
            break
    if bundle["original_summary"] is None:
        bundle["original_summary"] = (report.summary or "")[:1500]
    if orig_query:
        context["initial_question"] = orig_query.question
        bundle["original_question"] = orig_query.question
        # Infer last structural entity from prior sources
        prior_sources = database.scalars(
            select(ResearchSource)
            .where(ResearchSource.report_id == report.id)
            .order_by(ResearchSource.retrieved_at.desc())
        ).all()
        excerpt_pool: list[str] = []
        for src in prior_sources:
            if src.section_name:
                context["last_section_name"] = src.section_name
            if src.excerpt:
                excerpt_pool.append(src.excerpt)
        # Evidence excerpts already surfaced in this report (TASK 3B bundle)
        bundle["evidence_excerpts"] = excerpt_pool[:4]
        # Extract entity type/number from original question classification
        orig_q_class = classify_query(orig_query.question)
        if orig_q_class.target_entity:
            context["last_entity_type"] = orig_q_class.target_entity
        if orig_q_class.target_identifier:
            context["last_entity_number"] = orig_q_class.target_identifier
        if orig_q_class.parent_entity:
            context["last_parent_entity"] = orig_q_class.parent_entity
        if orig_q_class.parent_identifier:
            context["last_parent_identifier"] = orig_q_class.parent_identifier
        # Also check section_extraction/question_extraction in prior chat messages
        prior_msgs = database.scalars(
            select(ChatMessage)
            .where(ChatMessage.report_id == report.id)
            .order_by(ChatMessage.created_at.desc())
        ).all()
        for msg in prior_msgs:
            if msg.text != question_text:
                prior_class = classify_query(msg.text or "")
                if prior_class.target_entity and prior_class.target_identifier:
                    context["last_entity_type"] = prior_class.target_entity
                    context["last_entity_number"] = prior_class.target_identifier
                if prior_class.parent_entity and prior_class.parent_identifier:
                    context["last_parent_entity"] = prior_class.parent_entity
                    context["last_parent_identifier"] = prior_class.parent_identifier
                    break
        # TASK 3B: previous question + previous assistant answer in this thread
        if prior_msgs:
            for msg in prior_msgs:
                if msg.role == "assistant" and bundle["previous_answer"] is None:
                    bundle["previous_answer"] = (msg.text or "")[:2000]
                if msg.role == "user" and msg.text != question_text and bundle["previous_question"] is None:
                    bundle["previous_question"] = msg.text
                if bundle["previous_answer"] and bundle["previous_question"]:
                    break

    # TASK 3B: resolve vague references BEFORE retrieval. Retrieval gets the
    # rewritten query; the UI/thread keep the user's verbatim question.
    retrieval_question = question_text
    resolved_subject: str | None = None
    vague_followup = _is_vague_followup(question_text)
    if vague_followup:
        resolved_subject = _resolve_followup_subject(question_text, bundle)
        if resolved_subject:
            retrieval_question = _rewritten_followup_query(question_text, resolved_subject)
            bundle["resolved_subject"] = resolved_subject
        logger.info(
            "FOLLOWUP resolve: question=%r vague=%s subject=%r retrieval_query=%r",
            question_text[:80], vague_followup, resolved_subject, retrieval_question[:120],
        )

    database.add(ChatMessage(id=uuid.uuid4(), report_id=report.id, owner_id=user.id, role="user", text=question_text, citations=[]))

    answer_engine = get_answer_engine()
    followup_intent = classify_user_intent(question_text)
    exa_key = settings.exa_api_key

    # ---------------- TASK 4 (v8): Topic Research follow-ups ----------------
    # A Topic Research report has NO documents.
    if original_mode == "topic":
        is_standalone = _is_standalone_topic_question(question_text, bundle)
        active_bundle: dict[str, object] | None = None if is_standalone else bundle
        topic_retrieval_q = question_text if is_standalone else retrieval_question

        web_search_active = payload.web_search_enabled if payload.web_search_enabled is not None else (
            False if (report.summary and "web search off" in report.summary.lower()) else True
        )
        if not web_search_active:
            # Web Search OFF: 0 Exa calls, 0 web sources, direct AI answer
            answer, _ = answer_engine.generate_topic_followup_answer(
                topic_retrieval_q,
                bundle=active_bundle,
                web_sources=None,
                web_search_enabled=False,
                depth=payload.depth,
                intent=followup_intent,
            )
            citations = []
            basis = "Direct AI knowledge (Web Search OFF)"
            stored_text = _clamp_chat_text(answer)
            database.add(ChatMessage(id=uuid.uuid4(), report_id=report.id, owner_id=user.id, role="assistant", text=stored_text, citations=citations))
            database.commit()
            related_topics = answer_engine.generate_related_topics(question_text, answer, [])
            return {"answer": answer, "citations": citations, "answer_basis": basis, "sources": [], "related_topics": related_topics}

        # Web Search ON: check if existing sources can answer or search Exa
        db_sources = database.scalars(
            select(ResearchSource).where(ResearchSource.report_id == report.id, ResearchSource.source_type == "web")
        ).all()
        prior_sources_web: list[dict[str, object]] = [{"title": s.title, "url": s.url, "excerpt": s.excerpt, "source_type": "web"} for s in db_sources]

        web_sources: list[dict[str, object]] = []
        if not is_standalone and vague_followup and prior_sources_web:
            # Exa conservation: reuse prior verified sources only when continuing the same topic
            web_sources = prior_sources_web
        elif exa_key:
            web_sources = search_exa(topic_retrieval_q, exa_key)

        if web_sources:
            answer, _ = answer_engine.generate_topic_followup_answer(
                topic_retrieval_q,
                bundle=active_bundle,
                web_sources=web_sources,
                web_search_enabled=True,
                depth=payload.depth,
                intent=followup_intent,
            )
            citations = [_web_citation(s) for s in web_sources]
            if web_sources != prior_sources_web:
                database.add_all(
                    ResearchSource(
                        id=uuid.uuid4(), report_id=report.id, source_type="web",
                        document_page_id=None, title=str(s.get("title") or "Web Source"),
                        url=str(s.get("url") or ""), excerpt=str(s.get("excerpt") or ""),
                        start_line=None, end_line=None, section_name=None,
                        relevance_score=s.get("relevance_score"),
                    )
                    for s in web_sources
                )
            basis = _answer_basis(answer, 0, len(web_sources))
            formatted_sources = [
                {"title": str(s.get("title") or "Web Source"), "url": str(s.get("url") or ""), "excerpt": str(s.get("excerpt") or ""), "source_type": "web"}
                for s in web_sources
            ]
        else:
            # Web search returned no sources or unavailable: fall back to direct AI knowledge without polluting with previous sources
            answer, _ = answer_engine.generate_topic_followup_answer(
                topic_retrieval_q,
                bundle=active_bundle,
                web_sources=None,
                web_search_enabled=False,
                depth=payload.depth,
                intent=followup_intent,
            )
            citations = []
            basis = "Direct AI knowledge"
            formatted_sources = []

        stored_text = _clamp_chat_text(answer if basis == answer else f"{basis}\n\n{answer}")
        database.add(ChatMessage(id=uuid.uuid4(), report_id=report.id, owner_id=user.id, role="assistant", text=stored_text, citations=citations))
        database.commit()
        related_topics = answer_engine.generate_related_topics(question_text, answer, web_sources)
        return {"answer": answer, "citations": citations, "answer_basis": basis, "sources": formatted_sources, "related_topics": related_topics}

    # ------------- TASK 5 (v8): Combined Research web fallback -------------
    pages, _, doc_structures = _document_pages(database, user.id, query_document_ids)
    evidence = retrieve_document_evidence(pages, retrieval_question, doc_structures=doc_structures, context=context)

    if original_mode == "combined":
        web_sources: list[dict[str, object]] = []
        if not evidence and exa_key:
            # The uploaded file(s) do not cover the follow-up: fall back to a
            # fresh Exa web search for the follow-up's own text instead of
            # refusing. Never refuse without trying web first.
            web_sources = search_exa(retrieval_question, exa_key)
        if evidence and web_sources:
            answer, _ = answer_engine.generate_combined_answer(retrieval_question, evidence, web_sources)
            basis = _answer_basis(answer, len(evidence), len(web_sources))
        elif evidence:
            answer, _ = answer_engine.generate_follow_up_answer(retrieval_question, evidence, context=context, user_intent=followup_intent)
            basis = _answer_basis(answer, len(evidence), 0)
        elif web_sources:
            answer, _ = answer_engine.generate_topic_answer(retrieval_question, web_sources, intent=followup_intent)
            basis = _answer_basis(answer, 0, len(web_sources))
        else:
            answer = "I couldn't reliably locate explanatory evidence for that follow-up in the documents attached to this research, and the web fallback found no reliable sources either."
            basis = answer
        citations: list[dict[str, object]] = []
        for item in evidence:
            citations.append({
                "document_name": item.document_name,
                "pdf_page_number": item.pdf_page_number,
                "printed_page_label": item.printed_page_label,
                "start_line": sanitize_line_range(item.start_line, item.end_line)[0],
                "end_line": sanitize_line_range(item.start_line, item.end_line)[1],
                "section_name": item.section_name,
            })
        citations.extend(_web_citation(s) for s in web_sources)
        if web_sources:
            database.add_all(
                ResearchSource(
                    id=uuid.uuid4(), report_id=report.id, source_type="web",
                    document_page_id=None, title=str(s.get("title") or "Web Source"),
                    url=str(s.get("url") or ""), excerpt=str(s.get("excerpt") or ""),
                    start_line=None, end_line=None, section_name=None,
                    relevance_score=s.get("relevance_score"),
                )
                for s in web_sources
            )
        stored_text = _clamp_chat_text(answer if basis == answer else f"{basis}\n\n{answer}")
        database.add(ChatMessage(id=uuid.uuid4(), report_id=report.id, owner_id=user.id, role="assistant", text=stored_text, citations=citations))
        database.commit()
        return {
            "answer": answer,
            "citations": citations,
            "answer_basis": basis,
            "answering_model_final": getattr(answer_engine, "last_answering_model", None),
        }

    # -------- Document Research follow-ups (STRICTLY grounded, unchanged) --------
    # TASK 10 (v8): the gate is 'is there sufficient grounded evidence to work
    # from', NOT 'did the question text keyword-match the evidence'. Log the
    # retrieval outcome so gating decisions are auditable (STEP 10B).
    top_score = max((e.relevance_score for e in evidence), default=None)
    logger.info(
        "FOLLOWUP gate: report=%s mode=%s question=%r evidence=%d top_score=%s",
        report.id, original_mode, question_text[:80], len(evidence), top_score,
    )
    if not evidence and original_mode == "documents" and _GENERATIVE_REQUEST_RX.search(retrieval_question):
        # Generative/synthesis requests ('give me viva questions…') naturally
        # have low keyword overlap with the source text. The report's OWN
        # stored document evidence is still fully usable — pass it through.
        # Factual lookup questions about unrelated topics get NO fallback:
        # Document Research must keep refusing ungroundable questions.
        stored_sources = database.scalars(
            select(ResearchSource)
            .where(ResearchSource.report_id == report.id, ResearchSource.source_type == "document")
            .order_by(ResearchSource.relevance_score.desc().nulls_last())
        ).all()
        rebuilt = [e for e in (_stored_source_as_evidence(database, s) for s in stored_sources) if e]
        if rebuilt:
            logger.info("FOLLOWUP gate: zero keyword hits — falling back to %d stored document evidence item(s)", len(rebuilt))
            evidence = rebuilt
    if evidence:
        citations = [{
            "document_name": item.document_name,
            "pdf_page_number": item.pdf_page_number,
            "printed_page_label": item.printed_page_label,
            "start_line": sanitize_line_range(item.start_line, item.end_line)[0],
            "end_line": sanitize_line_range(item.start_line, item.end_line)[1],
            "section_name": item.section_name,
        } for item in evidence]
        answer, _ = answer_engine.generate_follow_up_answer(
            retrieval_question,
            evidence,
            context=context,
            bundle=bundle,
            user_intent=followup_intent,
            depth_mode=payload.depth,
        )
    else:
        citations, answer = [], "I couldn't reliably locate explanatory evidence for that follow-up in the documents attached to this research."
    # Document path keeps its exact historical text (no label prefix): strictly
    # grounded by contract. basis is still reported for API completeness.
    # v11 Group 2 FIX: the PERSISTED copy must go through _clamp_chat_text —
    # chat_messages.text has a DB CHECK (1..4000) and long solve/code answers
    # caused CheckViolation 500s on this path (API response keeps full text).
    basis = _answer_basis(answer, len(evidence), 0)
    database.add(ChatMessage(id=uuid.uuid4(), report_id=report.id, owner_id=user.id, role="assistant", text=_clamp_chat_text(answer), citations=citations))
    database.commit()
    return {
        "answer": answer,
        "citations": citations,
        "answer_basis": basis,
        "answering_model_final": getattr(answer_engine, "last_answering_model", None),
    }


class StreamResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: str = Field(default="topic", pattern="^(documents|topic|combined)$")
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=12)
    web_search_enabled: bool = Field(default=True)
    depth: str | None = Field(default="detailed")
    report_id: uuid.UUID | None = Field(default=None)


@router.post("/stream")
async def stream_research(
    request: Request,
    payload: StreamResearchRequest,
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    user_id = user.id

    async def event_generator():
        try:
            is_topic = payload.mode == "topic"
            web_search_active = payload.web_search_enabled if is_topic else True

            # Step 1: Initial status events
            if is_topic and web_search_active:
                yield f"data: {json.dumps({'type': 'status', 'status': 'Searching the web...'})}\n\n"
            elif payload.document_ids or payload.mode in {"documents", "combined"}:
                yield f"data: {json.dumps({'type': 'status', 'status': 'Searching document pages...'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'status', 'status': 'Synthesizing knowledge...'})}\n\n"

            if await request.is_disconnected():
                logger.info("Client disconnected during initial stream setup")
                return

            answer_engine = get_answer_engine()

            if payload.report_id:
                # ---------------- Follow-Up Streaming ----------------
                report = database.scalar(
                    select(Report).where(
                        Report.id == payload.report_id,
                        Report.owner_id == user_id,
                    )
                )
                if not report:
                    yield f"data: {json.dumps({'type': 'error', 'detail': 'Report not found.'})}\n\n"
                    return

                orig_query = database.scalar(
                    select(ResearchQuery).where(ResearchQuery.id == report.query_id)
                )
                original_mode = orig_query.mode if orig_query else "documents"

                is_topic_chat = (payload.mode == "topic") or (not payload.document_ids and original_mode == "topic")

                bundle: dict[str, object] = {
                    "previous_question": None,
                    "previous_answer": None,
                    "original_summary": report.summary,
                    "resolved_subject": None,
                }
                prior_msgs = database.scalars(
                    select(ChatMessage)
                    .where(ChatMessage.report_id == report.id)
                    .order_by(ChatMessage.created_at.desc())
                ).all()
                for msg in prior_msgs:
                    if msg.role == "assistant" and bundle["previous_answer"] is None:
                        bundle["previous_answer"] = (msg.text or "")[:2000]
                    if msg.role == "user" and msg.text != payload.question and bundle["previous_question"] is None:
                        bundle["previous_question"] = msg.text
                    if bundle["previous_answer"] and bundle["previous_question"]:
                        break

                retrieval_question = payload.question
                if _is_vague_followup(payload.question):
                    resolved_sub = _resolve_followup_subject(payload.question, bundle)
                    if resolved_sub:
                        retrieval_question = _rewritten_followup_query(payload.question, resolved_sub)
                        bundle["resolved_subject"] = resolved_sub

                if is_topic_chat:
                    try:
                        database.close()
                    except Exception:
                        pass
                    # Multi-question in same Topic Research chat (continuation or new topic)
                    followup_intent = classify_user_intent(payload.question, mode="topic", is_followup=False)
                    exa_key = settings.exa_api_key
                    topic_web_sources: list[dict[str, Any]] = []
                    if web_search_active and exa_key:
                        yield f"data: {json.dumps({'type': 'status', 'status': 'Searching the web...'})}\n\n"
                        topic_web_sources = search_exa(retrieval_question, exa_key)
                        yield f"data: {json.dumps({'type': 'status', 'status': f'Verified {len(topic_web_sources)} web sources...'})}\n\n"

                    yield f"data: {json.dumps({'type': 'status', 'status': 'Composing answer...'})}\n\n"
                    accumulated_text = ""
                    token_stream = answer_engine.stream_topic_answer(
                        payload.question,
                        web_sources=topic_web_sources,
                        intent=followup_intent,
                        web_search_enabled=web_search_active,
                        depth_mode=payload.depth,
                    )
                    for chunk in token_stream:
                        if await request.is_disconnected():
                            logger.info("Client cancelled stream topic follow-up")
                            return
                        accumulated_text += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

                    validated_answer = answer_engine._validated(accumulated_text.strip())
                    user_msg = ChatMessage(id=uuid.uuid4(), report_id=report.id, owner_id=user_id, role="user", text=payload.question, citations=[])
                    asst_msg = ChatMessage(id=uuid.uuid4(), report_id=report.id, owner_id=user_id, role="assistant", text=validated_answer, citations=topic_web_sources)
                    for attempt in range(3):
                        try:
                            with SessionLocal() as follow_db:
                                follow_db.add(user_msg)
                                follow_db.add(asst_msg)
                                follow_db.commit()
                                break
                        except Exception as follow_exc:
                            logger.warning("Attempt %d to persist topic follow-up chat messages failed: %s", attempt + 1, follow_exc)

                    related_topics = answer_engine.generate_related_topics(payload.question, validated_answer, web_sources=topic_web_sources)
                    yield f"data: {json.dumps({'type': 'done', 'report_id': str(payload.report_id), 'sources': topic_web_sources, 'related_topics': related_topics, 'citations': []})}\n\n"
                    return

                # Otherwise: Document Follow-up
                followup_intent = classify_user_intent(payload.question)
                doc_evidence: list[Evidence] = []
                doc_ids = database.scalars(
                    select(ResearchQueryDocument.document_id).where(
                        ResearchQueryDocument.query_id == report.query_id
                    )
                ).all()
                if payload.document_ids:
                    doc_ids = list(payload.document_ids)

                pages: list[PageRecord] = []
                doc_structures: dict[str, DocumentStructureIndex] = {}
                if doc_ids:
                    pages, _, doc_structures = _document_pages(database, user_id, doc_ids)

                _page_direct_result = None
                _exact_page = None
                _page_query_num = None
                if pages:
                    _m = _PAGE_QUERY_RX.search(payload.question)
                    if _m:
                        _page_query_num = int(_m.group(1) or _m.group(2))
                        _exact_page = next((p for p in pages if p.pdf_page_number == _page_query_num), None)
                        if _exact_page is None:
                            _exact_page = next((p for p in pages if p.printed_page_label and str(p.printed_page_label).strip() == str(_page_query_num)), None)
                        if _exact_page:
                            page_text = (_exact_page.text or "").strip()
                            if is_poor_extraction(page_text):
                                ocr_t, ocr_l = _ocr_fallback_for_page(database, doc_ids, _exact_page)
                                if ocr_t and not is_poor_extraction(ocr_t):
                                    _exact_page = PageRecord(
                                        document_name=_exact_page.document_name,
                                        pdf_page_number=_exact_page.pdf_page_number,
                                        text=ocr_t,
                                        page_kind=_exact_page.page_kind,
                                        printed_page_label=_exact_page.printed_page_label,
                                        lines=tuple(ocr_l),
                                        sections=_exact_page.sections,
                                    )
                                    page_text = ocr_t
                            if page_text:
                                _page_direct_result = page_text

                try:
                    database.close()
                except Exception:
                    pass

                if _page_direct_result is not None and _exact_page is not None and _page_query_num is not None:
                    doc_evidence = [
                        Evidence(
                            document_name=_exact_page.document_name,
                            pdf_page_number=_exact_page.pdf_page_number,
                            printed_page_label=_exact_page.printed_page_label,
                            excerpt=(_page_direct_result or "")[:500],
                            page_kind=_exact_page.page_kind,
                            key_sentences=tuple(l.strip() for l in (_page_direct_result or "").splitlines()[:2] if l.strip()),
                            matched_heading=f"Page {_exact_page.pdf_page_number}",
                            relevance_score=1.0,
                            start_line=1,
                            end_line=len(_exact_page.lines) if _exact_page.lines else 1,
                            section_name=f"Page {_exact_page.pdf_page_number}",
                        )
                    ]
                    accumulated_text = f"**PDF Page {_page_query_num}**\n\n{_page_direct_result}"
                    yield f"data: {json.dumps({'type': 'chunk', 'text': accumulated_text})}\n\n"
                elif pages:
                    doc_evidence = retrieve_document_evidence(pages, retrieval_question, doc_structures=doc_structures)
                    yield f"data: {json.dumps({'type': 'status', 'status': 'Preparing answer...'})}\n\n"
                    accumulated_text = ""
                    token_stream = answer_engine.stream_follow_up_answer(
                        payload.question,
                        evidence=doc_evidence,
                        bundle=bundle,
                        user_intent=followup_intent,
                    )
                    for chunk in token_stream:
                        if await request.is_disconnected():
                            logger.info("Client cancelled stream follow-up")
                            return
                        accumulated_text += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                else:
                    accumulated_text = "I couldn't locate documents attached to this research."
                    yield f"data: {json.dumps({'type': 'chunk', 'text': accumulated_text})}\n\n"

                validated_answer = answer_engine._validated(accumulated_text.strip())

                citations: list[dict[str, object]] = []
                for item in doc_evidence:
                    citations.append({
                        "document_name": item.document_name,
                        "pdf_page_number": item.pdf_page_number,
                        "printed_page_label": item.printed_page_label,
                        "start_line": sanitize_line_range(item.start_line, item.end_line)[0],
                        "end_line": sanitize_line_range(item.start_line, item.end_line)[1],
                        "section_name": item.section_name,
                    })

                user_msg = ChatMessage(id=uuid.uuid4(), report_id=report.id, owner_id=user_id, role="user", text=payload.question, citations=[])
                asst_msg = ChatMessage(id=uuid.uuid4(), report_id=report.id, owner_id=user_id, role="assistant", text=validated_answer, citations=citations)
                for attempt in range(3):
                    try:
                        with SessionLocal() as follow_db:
                            follow_db.add(user_msg)
                            follow_db.add(asst_msg)
                            follow_db.commit()
                            break
                    except Exception as follow_exc:
                        logger.warning("Attempt %d to persist doc follow-up chat messages failed: %s", attempt + 1, follow_exc)

                frontend_sources = _format_evidence_for_frontend(doc_evidence)
                related_topics = answer_engine.generate_related_topics(payload.question, validated_answer)
                yield f"data: {json.dumps({'type': 'done', 'report_id': str(payload.report_id), 'sources': frontend_sources, 'related_topics': related_topics, 'citations': citations})}\n\n"

            else:
                # ---------------- New Research Streaming ----------------
                if payload.mode in {"documents", "combined"} and not payload.document_ids:
                    yield f"data: {json.dumps({'type': 'error', 'detail': 'Document and Combined Research require at least one uploaded PDF.'})}\n\n"
                    return

                exa_key = settings.exa_api_key
                if payload.mode == "topic" and web_search_active and not exa_key:
                    yield f"data: {json.dumps({'type': 'error', 'detail': 'EXA_API_KEY is not configured.'})}\n\n"
                    return
                if payload.mode == "combined" and not exa_key:
                    yield f"data: {json.dumps({'type': 'error', 'detail': 'EXA_API_KEY is not configured.'})}\n\n"
                    return

                pages, page_ids, doc_structures = _document_pages(database, user_id, payload.document_ids)

                if payload.mode in {"documents", "combined"}:
                    yield f"data: {json.dumps({'type': 'status', 'status': 'Extracting pages and retrieving explanatory evidence...', 'step': 2})}\n\n"

                _page_query_num: int | None = None
                _page_direct_result: str | None = None
                _exact_page: PageRecord | None = None
                if pages:
                    _m = _PAGE_QUERY_RX.search(payload.question)
                    if _m:
                        _page_query_num = int(_m.group(1) or _m.group(2))
                        _exact_page = next((p for p in pages if p.pdf_page_number == _page_query_num), None)
                        if _exact_page is None:
                            _exact_page = next((p for p in pages if p.printed_page_label and str(p.printed_page_label).strip() == str(_page_query_num)), None)
                        if _exact_page:
                            page_text = (_exact_page.text or "").strip()
                            if is_poor_extraction(page_text):
                                ocr_t, ocr_l = _ocr_fallback_for_page(database, payload.document_ids, _exact_page)
                                if ocr_t and not is_poor_extraction(ocr_t):
                                    _exact_page = PageRecord(
                                        document_name=_exact_page.document_name,
                                        pdf_page_number=_exact_page.pdf_page_number,
                                        text=ocr_t,
                                        page_kind=_exact_page.page_kind,
                                        printed_page_label=_exact_page.printed_page_label,
                                        lines=tuple(ocr_l),
                                        sections=_exact_page.sections,
                                    )
                                    page_text = ocr_t
                            if page_text:
                                _page_direct_result = page_text
                            else:
                                _page_direct_result = f"PDF page {_page_query_num} appears to be a scanned or image-only page and OCR could not extract readable text."
                        else:
                            _page_direct_result = f"PDF page {_page_query_num} was not found in this document. The document has {len(pages)} pages."

                try:
                    database.close()
                except Exception:
                    pass

                if _page_direct_result is not None and _exact_page is not None:
                    document_evidence = [
                        Evidence(
                            document_name=_exact_page.document_name,
                            pdf_page_number=_exact_page.pdf_page_number,
                            printed_page_label=_exact_page.printed_page_label,
                            excerpt=(_page_direct_result or "")[:500],
                            page_kind=_exact_page.page_kind,
                            key_sentences=tuple(l.strip() for l in (_page_direct_result or "").splitlines()[:2] if l.strip()),
                            matched_heading=f"Page {_exact_page.pdf_page_number}",
                            relevance_score=1.0,
                            start_line=1,
                            end_line=len(_exact_page.lines) if _exact_page.lines else 1,
                            section_name=f"Page {_exact_page.pdf_page_number}",
                        )
                    ]
                else:
                    document_evidence = (
                        retrieve_document_evidence(pages, payload.question, doc_structures=doc_structures)
                        if bool(payload.document_ids)
                        else []
                    )

                web_sources: list[dict[str, object]] = []
                if (payload.mode == "topic" and web_search_active and exa_key) or (payload.mode == "combined" and exa_key):
                    web_sources = search_exa(payload.question, exa_key)
                    if payload.mode == "topic":
                        yield f"data: {json.dumps({'type': 'status', 'status': f'Sources found ({len(web_sources)} results)...', 'step': 2})}\n\n"

                if payload.mode in {"documents", "combined"}:
                    yield f"data: {json.dumps({'type': 'status', 'status': 'Verifying references and cross-checking page evidence...', 'step': 3})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'status', 'status': 'Verifying and cross-checking sources...', 'step': 3})}\n\n"

                if await request.is_disconnected():
                    return

                # Mandatory no-evidence safety gate
                if payload.mode == "documents" and not document_evidence and _page_direct_result is None:
                    closest = closest_match_evidence(pages, payload.question, doc_structures=doc_structures)
                    closest_formatted = _format_evidence_for_frontend(closest)
                    yield f"data: {json.dumps({'type': 'done', 'report_id': '', 'sources': closest_formatted, 'related_topics': [], 'direct_answer': None, 'sections': []})}\n\n"
                    return

                if payload.mode in {"documents", "combined"}:
                    yield f"data: {json.dumps({'type': 'status', 'status': 'Preparing and composing grounded report with traceable citations...', 'step': 4})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'status', 'status': 'Preparing and composing comprehensive report...', 'step': 4})}\n\n"

                user_intent = classify_user_intent(payload.question)
                accumulated_text = ""

                stream_query_id = uuid.uuid4()
                try:
                    with SessionLocal() as init_s_db:
                        init_s_q = ResearchQuery(
                            id=stream_query_id,
                            owner_id=user_id,
                            question=payload.question.strip(),
                            mode=payload.mode,
                            status="running",
                        )
                        init_s_db.add(init_s_q)
                        init_s_db.flush()
                        if payload.document_ids:
                            init_s_db.add_all(
                                ResearchQueryDocument(query_id=stream_query_id, document_id=doc_id)
                                for doc_id in payload.document_ids
                            )
                        init_s_db.commit()
                except Exception as init_exc:
                    logger.warning("Could not persist initial stream query: %s", init_exc)

                if _page_direct_result is not None:
                    page_ans = f"**PDF Page {_page_query_num}**\n\n{_page_direct_result}"
                    accumulated_text = page_ans
                    yield f"data: {json.dumps({'type': 'chunk', 'text': page_ans})}\n\n"
                elif payload.mode == "documents":
                    token_stream = answer_engine.stream_document_answer(
                        payload.question,
                        document_evidence,
                        user_intent=user_intent,
                        depth_mode=payload.depth,
                    )
                    for chunk in token_stream:
                        if await request.is_disconnected():
                            logger.info("Client disconnected during document stream")
                            return
                        accumulated_text += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                elif payload.mode == "topic":
                    token_stream = answer_engine.stream_topic_answer(
                        payload.question,
                        web_sources=web_sources,
                        intent=user_intent,
                        web_search_enabled=web_search_active,
                        depth_mode=payload.depth,
                    )
                    for chunk in token_stream:
                        if await request.is_disconnected():
                            logger.info("Client disconnected during topic stream")
                            return
                        accumulated_text += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                else:  # combined
                    token_stream = answer_engine.stream_document_answer(
                        payload.question,
                        document_evidence,
                        user_intent=user_intent,
                        depth_mode=payload.depth,
                    )
                    for chunk in token_stream:
                        if await request.is_disconnected():
                            logger.info("Client disconnected during combined stream")
                            return
                        accumulated_text += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

                validated_answer = answer_engine._validated(accumulated_text.strip())
                has_stream_answer = bool(validated_answer and validated_answer.strip())
                if not has_stream_answer and _page_direct_result is None:
                    validated_answer = "Document evidence verified, but AI generation produced no answer."
                    yield f"data: {json.dumps({'type': 'chunk', 'text': validated_answer})}\n\n"

                report_obj_id = uuid.uuid4()
                sections_out: list[dict[str, object]] = []
                if _page_direct_result is not None:
                    sections_out.append({
                        "heading": f"Page {_page_query_num} Content",
                        "body": _page_direct_result,
                        "source_type": "document",
                    })
                    summary = f"Direct page retrieval: PDF page {_page_query_num}."
                    direct_ans = validated_answer[:400]
                else:
                    sections_out.append({
                        "heading": "Synthesized Answer",
                        "body": validated_answer,
                        "source_type": "ai",
                    })
                    if document_evidence:
                        sections_out.append({
                            "heading": "Evidence from your documents",
                            "body": "\n\n".join(item.excerpt for item in document_evidence),
                            "source_type": "document",
                        })
                    if web_sources:
                        sections_out.append({
                            "heading": "Web Sources",
                            "body": "\n\n".join(str(s.get("excerpt", "")) for s in web_sources),
                            "source_type": "web",
                        })
                    summary = "Evidence verified." if has_stream_answer else "Document evidence verified. (AI generation currently unavailable)"
                    direct_ans = _direct_answer_lead(validated_answer) if has_stream_answer else ""

                sources_to_add: list[ResearchSource] = []
                for item in document_evidence:
                    doc_page_id = (page_ids.get((str(item.document_id), item.pdf_page_number)) if item.document_id else None) or page_ids.get((item.document_name, item.pdf_page_number))
                    sources_to_add.append(ResearchSource(
                        id=uuid.uuid4(),
                        report_id=report_obj_id,
                        source_type="document",
                        document_page_id=doc_page_id,
                        title=item.document_name,
                        url=None,
                        excerpt=item.excerpt,
                        start_line=item.start_line,
                        end_line=item.end_line,
                        section_name=item.section_name,
                        relevance_score=item.relevance_score,
                    ))

                for w_item in web_sources:
                    sources_to_add.append(ResearchSource(
                        id=uuid.uuid4(),
                        report_id=report_obj_id,
                        source_type="web",
                        document_page_id=None,
                        title=str(w_item.get("title") or "Web Source"),
                        url=str(w_item.get("url") or ""),
                        excerpt=str(w_item.get("excerpt") or ""),
                        start_line=None,
                        end_line=None,
                        section_name=None,
                        relevance_score=w_item.get("relevance_score"),
                    ))

                for attempt in range(3):
                    try:
                        with SessionLocal() as persist_db:
                            q_row = persist_db.scalar(select(ResearchQuery).where(ResearchQuery.id == stream_query_id))
                            if q_row:
                                q_row.status = "complete"
                            else:
                                q_row = ResearchQuery(
                                    id=stream_query_id,
                                    owner_id=user.id,
                                    question=payload.question.strip(),
                                    mode=payload.mode,
                                    status="complete",
                                )
                                persist_db.add(q_row)
                                persist_db.flush()
                                if payload.document_ids:
                                    persist_db.add_all(
                                        ResearchQueryDocument(query_id=stream_query_id, document_id=doc_id)
                                        for doc_id in payload.document_ids
                                    )

                            report_obj = Report(
                                id=report_obj_id,
                                query_id=stream_query_id,
                                owner_id=user_id,
                                summary=summary,
                                body=sections_out,
                                not_found=False,
                            )
                            persist_db.add(report_obj)
                            if sources_to_add:
                                persist_db.add_all(sources_to_add)
                            persist_db.commit()
                            break
                    except Exception as persist_exc:
                        logger.warning("Attempt %d to persist stream report failed: %s", attempt + 1, persist_exc)
                        if attempt == 2:
                            logger.error("Failed to persist stream report after retries: %s. Preserving AI answer for user.", persist_exc)

                doc_sources_formatted = _format_evidence_for_frontend(document_evidence)
                all_sources = doc_sources_formatted + (web_sources if web_sources else [])
                related_topics = answer_engine.generate_related_topics(payload.question, validated_answer, web_sources=web_sources)
                yield f"data: {json.dumps({'type': 'done', 'report_id': str(report_obj_id), 'sources': all_sources, 'related_topics': related_topics, 'direct_answer': direct_ans, 'sections': sections_out})}\n\n"

        except Exception as exc:
            logger.exception("Error in stream_research: %s", exc)
            status_code, user_msg = normalize_server_exception(exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': user_msg})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

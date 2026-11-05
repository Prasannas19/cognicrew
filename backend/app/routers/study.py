from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..answer_engine import get_answer_engine
from ..database import get_db
from ..dependencies import current_user
from ..document_profiler import get_or_build_document_profile
from ..models import Document, User
from .research import _document_pages

logger = logging.getLogger("cognicrew.study")

router = APIRouter(prefix="/study", tags=["study"])


class QuizRequest(BaseModel):
    document_id: uuid.UUID
    num_questions: int | str = Field(default=5)
    focus: str | None = Field(default=None, max_length=500)


class FlashcardsRequest(BaseModel):
    document_id: uuid.UUID
    num_cards: int | str | None = Field(default=None)
    focus: str | None = Field(default=None, max_length=500)


class StudyGuideRequest(BaseModel):
    document_id: uuid.UUID
    focus: str | None = Field(default=None, max_length=500)


class MindMapRequest(BaseModel):
    document_id: uuid.UUID
    focus: str | None = Field(default=None, max_length=500)


class CheatSheetRequest(BaseModel):
    document_id: uuid.UUID
    focus: str | None = Field(default=None, max_length=500)


def _load_document_and_profile(
    document_id: uuid.UUID,
    database: Session,
    user: User,
) -> tuple[Document, Any]:
    """Verify ownership, load pages, and retrieve/build cached DocumentProfile."""
    doc = database.scalar(
        select(Document).where(Document.id == document_id, Document.owner_id == user.id)
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    pages, _, structures = _document_pages(database, user.id, [document_id])
    if not pages:
        raise HTTPException(status_code=422, detail="Selected document contains no readable text pages.")

    profile = get_or_build_document_profile(
        document_id=str(doc.id),
        document_name=doc.original_name,
        pages=list(pages),
        structures=structures,
    )
    return doc, profile


@router.post("/quiz")
def generate_quiz(
    payload: QuizRequest,
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Generate structured multiple-choice quiz questions (up to 50) from verified document evidence."""
    t0 = time.perf_counter()
    doc, profile = _load_document_and_profile(payload.document_id, database, user)

    target_num = payload.num_questions
    is_auto = str(target_num).lower() == "auto"
    if is_auto:
        from ..document_profiler import compute_auto_quiz_count
        final_target = compute_auto_quiz_count(profile)
    else:
        try:
            final_target = int(target_num)
        except (ValueError, TypeError):
            final_target = 5
        final_target = max(1, min(50, final_target))

    answer_engine = get_answer_engine()
    quiz_items = answer_engine.generate_scaled_quiz(
        profile=profile,
        target_count=final_target,
        focus=payload.focus,
    )

    t_elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not quiz_items:
        raise HTTPException(
            status_code=503,
            detail="AI quiz generation is temporarily unavailable. Please try again or verify document readability.",
        )

    logger.info(
        "[STUDY QUIZ] doc=%s requested=%s target=%d generated=%d pages=%d total_ms=%.1f",
        doc.original_name, str(payload.num_questions), final_target, len(quiz_items), profile.page_count, t_elapsed_ms
    )

    telemetry = getattr(answer_engine, "last_quiz_telemetry", {}) or {}

    return {
        "document_id": str(doc.id),
        "document_name": doc.original_name,
        "count": len(quiz_items),
        "requested_count": final_target,
        "is_auto": is_auto,
        "quiz": quiz_items,
        "questions": quiz_items,
        "timings": {
            "total_ms": t_elapsed_ms,
            "document_pages": profile.page_count,
            "section_count": len(profile.sections),
            "concept_count": len(profile.all_concepts),
            "requested_count": final_target,
            "generated_count": len(quiz_items),
            "validated_count": len(quiz_items),
            "duplicate_count": telemetry.get("duplicate_count", 0),
            "batch_count": telemetry.get("batch_count", 1),
            "failed_batches": telemetry.get("failed_batches", 0),
            "retry_count": telemetry.get("retry_count", 0),
            "cache_hit": getattr(profile, "cached", True),
        },
    }


@router.post("/flashcards")
def generate_flashcards(
    payload: FlashcardsRequest,
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Generate structured study flashcards with adaptive document coverage."""
    t0 = time.perf_counter()
    doc, profile = _load_document_and_profile(payload.document_id, database, user)

    target_val = payload.num_cards
    is_auto = target_val is None or str(target_val).lower() == "auto"
    if is_auto or target_val is None:
        final_target = None
    else:
        try:
            final_target = int(target_val)
            final_target = max(1, min(50, final_target))
        except (ValueError, TypeError):
            final_target = None

    answer_engine = get_answer_engine()
    flashcards = answer_engine.generate_scaled_flashcards(
        profile=profile,
        target_count=final_target,
        focus=payload.focus,
    )

    t_elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not flashcards:
        raise HTTPException(
            status_code=503,
            detail="AI flashcard generation is temporarily unavailable. Please try again.",
        )

    logger.info(
        "[STUDY FLASHCARDS] doc=%s generated=%d pages=%d total_ms=%.1f",
        doc.original_name, len(flashcards), profile.page_count, t_elapsed_ms
    )

    telemetry = getattr(answer_engine, "last_flashcards_telemetry", {}) or {}

    return {
        "document_id": str(doc.id),
        "document_name": doc.original_name,
        "count": len(flashcards),
        "requested_count": final_target or len(flashcards),
        "is_auto": is_auto,
        "flashcards": flashcards,
        "timings": {
            "total_ms": t_elapsed_ms,
            "document_pages": profile.page_count,
            "section_count": len(profile.sections),
            "concept_count": len(profile.all_concepts),
            "requested_count": final_target or len(flashcards),
            "generated_count": len(flashcards),
            "validated_count": len(flashcards),
            "duplicate_count": telemetry.get("duplicate_count", 0),
            "batch_count": telemetry.get("batch_count", 1),
            "failed_batches": telemetry.get("failed_batches", 0),
            "retry_count": telemetry.get("retry_count", 0),
            "cache_hit": getattr(profile, "cached", True),
        },
    }


@router.post("/study-guide")
def generate_study_guide(
    payload: StudyGuideRequest,
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Generate a comprehensive academic study guide grounded in document structure."""
    t0 = time.perf_counter()
    doc, profile = _load_document_and_profile(payload.document_id, database, user)

    answer_engine = get_answer_engine()
    guide = answer_engine.generate_study_guide(
        profile=profile,
        focus=payload.focus,
    )

    t_elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "document_id": str(doc.id),
        "document_name": doc.original_name,
        "title": guide["title"],
        "content": guide["content"],
        "sections_covered": guide["sections_covered"],
        "pages": guide["pages"],
        "timings": {
            "total_ms": t_elapsed_ms,
            "document_pages": profile.page_count,
            "sections_covered": guide["sections_covered"],
        },
    }


@router.post("/mind-map")
def generate_mind_map(
    payload: MindMapRequest,
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Generate a hierarchical concept tree for the Mind Map viewer."""
    t0 = time.perf_counter()
    doc, profile = _load_document_and_profile(payload.document_id, database, user)

    answer_engine = get_answer_engine()
    mind_map = answer_engine.generate_mind_map(
        profile=profile,
        focus=payload.focus,
    )

    t_elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "document_id": str(doc.id),
        "document_name": doc.original_name,
        "mind_map": mind_map,
        "timings": {
            "total_ms": t_elapsed_ms,
            "document_pages": profile.page_count,
            "section_count": len(profile.sections),
            "concept_count": len(profile.all_concepts),
        },
    }


@router.post("/cheat-sheet")
def generate_cheat_sheet(
    payload: CheatSheetRequest,
    database: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Generate a high-density revision cheat sheet containing formulas, definitions, and key models."""
    t0 = time.perf_counter()
    doc, profile = _load_document_and_profile(payload.document_id, database, user)

    answer_engine = get_answer_engine()
    sheet = answer_engine.generate_cheat_sheet(
        profile=profile,
        focus=payload.focus,
    )

    t_elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "document_id": str(doc.id),
        "document_name": doc.original_name,
        "title": sheet["title"],
        "content": sheet["content"],
        "formula_count": sheet["formula_count"],
        "algorithm_count": sheet["algorithm_count"],
        "timings": {
            "total_ms": t_elapsed_ms,
            "document_pages": profile.page_count,
            "formula_count": sheet["formula_count"],
            "algorithm_count": sheet["algorithm_count"],
        },
    }


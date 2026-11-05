"""Document Profiler & Adaptive Coverage Engine for CogniCrew.

Analyzes document structure, extracts concepts, definitions, formulas, and algorithms,
and builds an in-memory cached DocumentProfile to drive adaptive Study Tools:
- AI Quiz (scalable 5 to 50 items with section-aware batching)
- AI Flashcards (adaptive document coverage, no hardcoded counts)
- Study Guide (document-structured academic learning guide)
- Mind Map (hierarchical concept tree)
- Cheat Sheet (high-density revision sheet with formulas & key models)
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .document_pipeline import DocumentStructureIndex, PageRecord

logger = logging.getLogger("cognicrew.profiler")

_DEFINITION_RX = re.compile(
    r"\b([A-Z][A-Za-z0-9_\-\s]{2,30})\s+(?:is defined as|refers to|is an?|means|represents|is the process of|consists of)\s+([^.\n]{15,200}\.)",
    re.M,
)
_ALGORITHM_RX = re.compile(
    r"\b([A-Z][A-Za-z0-9_\-\s]{2,30}\s+(?:algorithm|method|classifier|regression|clustering|network|tree|forest|model|decomposition|filter))\b",
    re.I,
)
_FORMULA_RX = re.compile(
    r"(?:\b[A-Za-z_]{1,10}\s*=\s*[^.\n]{4,80}|∑|√|π|\b(?:mean|variance|standard deviation|silhouette|pagerank|euclidean distance|loss|accuracy|precision|recall|f1-score)\b)",
    re.I,
)


@dataclass
class SectionProfile:
    index: int
    title: str
    start_page: int
    end_page: int
    page_count: int
    text_excerpt: str
    concepts: list[str] = field(default_factory=list)
    definitions: list[dict[str, str]] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    algorithms: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class DocumentProfile:
    document_id: str
    document_name: str
    page_count: int
    sections: list[SectionProfile]
    all_concepts: list[str]
    all_definitions: list[dict[str, str]] = field(default_factory=list)
    all_formulas: list[str] = field(default_factory=list)
    all_algorithms: list[str] = field(default_factory=list)
    coverage_summary: str = ""
    hierarchy: list[dict[str, Any]] = field(default_factory=list)
    concept_density: float = 0.0
    keywords: list[str] = field(default_factory=list)
    coverage_map: dict[str, Any] = field(default_factory=dict)
    document_version: str = "v1"
    created_at: float = field(default_factory=time.monotonic)


_PROFILE_CACHE: dict[str, DocumentProfile] = {}
_PROFILE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = 1800.0  # 30 minutes


def clear_document_profile(document_id: str) -> None:
    """Invalidate cached profile when document changes or is deleted."""
    with _PROFILE_LOCK:
        keys_to_del = [k for k in _PROFILE_CACHE if k.startswith(document_id)]
        for k in keys_to_del:
            _PROFILE_CACHE.pop(k, None)


def compute_auto_quiz_count(profile: DocumentProfile) -> int:
    """Phase 20: Deterministically determine target quiz count from document profile.

    Policy:
    - small document (<=5 pages): 5-10 questions
    - medium document (<=25 pages): 10-20 questions
    - large academic document (<=100 pages): 20-35 questions
    - very rich document (>100 pages): up to 40-50 questions
    Maximum: 50. Target is strictly deterministic for the same document profile.
    """
    p = profile.page_count
    num_secs = len(profile.sections)
    num_defs = len(profile.all_definitions)
    num_concepts = len(profile.all_concepts)

    if p <= 5:
        target = min(10, max(5, num_secs * 2 + num_defs))
    elif p <= 25:
        target = min(20, max(10, num_secs * 3 + min(8, num_defs // 2)))
    elif p <= 80:
        target = min(35, max(20, num_secs * 3 + min(15, num_concepts // 3)))
    elif p <= 165:
        target = min(45, max(25, num_secs * 3 + min(20, num_concepts // 2)))
    else:
        target = 50

    return min(50, max(5, target))


def get_or_build_document_profile(
    document_id: str,
    document_name: str,
    pages: list[PageRecord],
    structures: dict[str, DocumentStructureIndex] | None = None,
    document_version: str = "v1",
) -> DocumentProfile:
    """Retrieve cached DocumentProfile (keyed by doc_id + version) or build through structure analysis."""
    cache_key = f"{document_id}:{document_version}:{len(pages)}"
    now = time.monotonic()
    with _PROFILE_LOCK:
        cached = _PROFILE_CACHE.get(cache_key)
        if cached and (now - cached.created_at) < _CACHE_TTL_SECONDS:
            return cached

    profile = _build_document_profile(document_id, document_name, pages, structures, document_version=document_version)

    with _PROFILE_LOCK:
        _PROFILE_CACHE[cache_key] = profile

    return profile


def _build_document_profile(
    document_id: str,
    document_name: str,
    pages: list[PageRecord],
    structures: dict[str, DocumentStructureIndex] | None = None,
    document_version: str = "v1",
) -> DocumentProfile:
    """Build a comprehensive profile by analyzing pages and document structure."""
    total_pages = len(pages)
    doc_struct = None
    if structures:
        doc_struct = structures.get(document_id) or structures.get(document_name)

    # Step 1: Identify sections
    raw_sections = doc_struct.sections if doc_struct and doc_struct.sections else []
    section_profiles: list[SectionProfile] = []

    if raw_sections and len(raw_sections) >= 2:
        # Use verified structural index
        for idx, s in enumerate(raw_sections):
            start_p = s.start_page
            end_p = s.end_page if s.end_page and s.end_page >= start_p else min(start_p + 10, total_pages)
            sec_pages = [p for p in pages if start_p <= p.pdf_page_number <= end_p]
            sec_text = "\n".join(p.text for p in sec_pages[:6])
            section_profiles.append(
                _analyze_section(
                    index=idx + 1,
                    title=s.title,
                    start_page=start_p,
                    end_page=end_p,
                    pages=sec_pages,
                    text_sample=sec_text[:4000],
                )
            )
    else:
        # Adaptive partition based on total pages
        chunk_size = max(4, min(20, total_pages // 8 or 4))
        for idx, start_idx in enumerate(range(0, total_pages, chunk_size)):
            chunk_pages = pages[start_idx : start_idx + chunk_size]
            if not chunk_pages:
                continue
            start_p = chunk_pages[0].pdf_page_number
            end_p = chunk_pages[-1].pdf_page_number
            # Try to infer section title from the first heading on the first page
            first_text = chunk_pages[0].text
            title = f"Section {idx + 1} (Pages {start_p}–{end_p})"
            for line in first_text.splitlines()[:15]:
                line_clean = line.strip()
                if line_clean.startswith("#"):
                    title = line_clean.lstrip("#").strip()
                    break
                elif re.match(r"^(?:Chapter|Unit|Section|Slip|Part|Module|Lecture)\s*[:#\s]*\w+", line_clean, re.I):
                    title = line_clean
                    break
                elif 4 <= len(line_clean) <= 45 and (line_clean.isupper() or line_clean.istitle()) and not line_clean.lower().startswith(("page ", "http", "www", "copyright")):
                    title = f"{line_clean} (Pages {start_p}–{end_p})"
                    break
            sec_text = "\n".join(p.text for p in chunk_pages[:4])
            section_profiles.append(
                _analyze_section(
                    index=idx + 1,
                    title=title,
                    start_page=start_p,
                    end_page=end_p,
                    pages=chunk_pages,
                    text_sample=sec_text[:3500],
                )
            )

    # Step 2: Aggregate global concepts, definitions, formulas, algorithms
    all_concepts: list[str] = []
    seen_concepts: set[str] = set()
    all_definitions: list[dict[str, str]] = []
    seen_defs: set[str] = set()
    all_formulas: list[str] = []
    seen_formulas: set[str] = set()
    all_algorithms: list[str] = []
    seen_algos: set[str] = set()

    for sec in section_profiles:
        for c in sec.concepts:
            low = c.lower()
            if low not in seen_concepts and len(c) > 2:
                seen_concepts.add(low)
                all_concepts.append(c)
        for d in sec.definitions:
            term = d["term"].strip()
            low = term.lower()
            if low not in seen_defs:
                seen_defs.add(low)
                all_definitions.append(d)
        for f in sec.formulas:
            low = f.strip().lower()
            if low not in seen_formulas:
                seen_formulas.add(low)
                all_formulas.append(f)
        for a in sec.algorithms:
            low = a.strip().lower()
            if low not in seen_algos:
                seen_algos.add(low)
                all_algorithms.append(a)

    summary_lines = [
        f"Document: {document_name} ({total_pages} pages, {len(section_profiles)} major sections)",
        f"Key Concept Domains: {len(all_concepts)} terms identified",
    ]
    if all_algorithms:
        summary_lines.append(f"Identified Models & Algorithms: {', '.join(all_algorithms[:12])}")
    if all_definitions:
        summary_lines.append(f"Core Defined Entities: {', '.join(d['term'] for d in all_definitions[:10])}")

    concept_density = round(len(all_concepts) / max(1, total_pages), 2)
    keywords = list(dict.fromkeys(all_concepts[:25] + all_algorithms[:15]))
    coverage_map = {
        s.title: {
            "start_page": s.start_page,
            "end_page": s.end_page,
            "concepts": len(s.concepts),
            "definitions": len(s.definitions),
            "formulas": len(s.formulas),
        }
        for s in section_profiles
    }
    hierarchy = [
        {
            "index": s.index,
            "title": s.title,
            "start_page": s.start_page,
            "end_page": s.end_page,
            "concepts": s.concepts[:6],
            "definitions": [d["term"] for d in s.definitions[:4]],
        }
        for s in section_profiles
    ]

    logger.info(
        "Built DocumentProfile for '%s': pages=%d sections=%d concepts=%d defs=%d algos=%d",
        document_name,
        total_pages,
        len(section_profiles),
        len(all_concepts),
        len(all_definitions),
        len(all_algorithms),
    )

    return DocumentProfile(
        document_id=document_id,
        document_name=document_name,
        page_count=total_pages,
        sections=section_profiles,
        all_concepts=all_concepts,
        all_definitions=all_definitions,
        all_formulas=all_formulas,
        all_algorithms=all_algorithms,
        coverage_summary="\n".join(summary_lines),
        hierarchy=hierarchy,
        concept_density=concept_density,
        keywords=keywords,
        coverage_map=coverage_map,
        document_version=document_version,
    )


def sanitize_concept(raw: str) -> str:
    """Sanitize and deduplicate noisy technical concept phrases."""
    s = " ".join(raw.strip().split())
    words = s.split()
    if len(words) > 1 and len(set(w.lower() for w in words)) == 1:
        return words[0]
    if s.lower().startswith("clustering ") and len(words) > 1:
        s = words[1]
    return s


def _analyze_section(
    index: int,
    title: str,
    start_page: int,
    end_page: int,
    pages: list[PageRecord],
    text_sample: str,
) -> SectionProfile:
    """Analyze a single section chunk to detect concepts, definitions, formulas, and weights."""
    concepts: list[str] = []
    definitions: list[dict[str, str]] = []
    formulas: list[str] = []
    algorithms: list[str] = []

    # Detect definitions
    for m in _DEFINITION_RX.finditer(text_sample):
        term = m.group(1).strip()
        defn = m.group(2).strip()
        if len(term) <= 35 and len(defn) >= 15:
            definitions.append({"term": term, "definition": defn})
            concepts.append(term)

    # Detect algorithms & models
    for m in _ALGORITHM_RX.finditer(text_sample):
        algo = m.group(1).strip()
        if len(algo) <= 40:
            algorithms.append(algo)
            concepts.append(algo)

    # Detect formulas
    for m in _FORMULA_RX.finditer(text_sample):
        form = m.group(0).strip()
        if len(form) >= 3:
            formulas.append(form)

    concepts = [sanitize_concept(c) for c in concepts if len(sanitize_concept(c)) >= 3]

    # Detect prominent concepts (Title-Case technical terms, distinctive technical phrases)
    if len(concepts) < 8:
        for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})\b", text_sample):
            phrase = sanitize_concept(m.group(1))
            if phrase.lower() not in {"this is", "there are", "for example", "such as", "in order", "as well", "based on", "due to", "according to", "in this"}:
                if phrase not in concepts and len(phrase) <= 35 and len(phrase) >= 3:
                    concepts.append(phrase)
                if len(concepts) >= 12:
                    break

    # Detect distinctive acronyms or uppercase technical terms (e.g. JVM, JDK, JRE, OOP, SQL, API)
    if len(concepts) < 8:
        for m in re.finditer(r"\b([A-Z]{2,6})\b", text_sample):
            acr = m.group(1)
            if acr not in {"THE", "AND", "FOR", "NOT", "CAN", "BUT", "ALL", "ANY", "ARE", "HAS", "HAD", "PAGE", "UNIT", "NOTE", "FILE"}:
                if acr not in concepts:
                    concepts.append(acr)
                if len(concepts) >= 15:
                    break

    # Detect prominent single-word capitalized technical terms (e.g. Java, Encapsulation, Inheritance, Polymorphism, Abstraction, Constructor, Interface)
    if len(concepts) < 8:
        _STOP_CAPS = {
            "This", "That", "These", "Those", "There", "Here", "What", "When", "Where", "Which", "While", "With",
            "From", "Into", "Only", "Some", "Each", "Every", "Most", "Many", "Also", "Then", "Thus", "Unit",
            "Page", "Note", "Notes", "File", "Chapter", "Section", "Title", "Text",
        }
        for m in re.finditer(r"\b([A-Z][a-z]{3,20})\b", text_sample):
            word = m.group(1)
            if word not in _STOP_CAPS:
                if word not in concepts:
                    concepts.append(word)
                if len(concepts) >= 12:
                    break

    # If section still has 0 concepts, extract words from title or section text
    if not concepts:
        title_words = [w.strip() for w in title.split() if len(w.strip()) >= 4 and w.strip().lower() not in {"unit", "chapter", "section", "part", "notes"}]
        for tw in title_words[:4]:
            if tw not in concepts:
                concepts.append(tw)

    # Weight proportional to length and entity density
    weight = max(1.0, len(text_sample) / 1000.0 + len(definitions) * 0.5 + len(algorithms) * 0.5)

    return SectionProfile(
        index=index,
        title=title,
        start_page=start_page,
        end_page=end_page,
        page_count=len(pages),
        text_excerpt=text_sample,
        concepts=list(dict.fromkeys(concepts)),
        definitions=definitions[:10],
        formulas=list(dict.fromkeys(formulas))[:10],
        algorithms=list(dict.fromkeys(algorithms))[:10],
        weight=round(weight, 2),
    )


def plan_coverage_allocation(profile: DocumentProfile, target_count: int) -> list[tuple[SectionProfile, int]]:
    """Allocate question/card counts across sections proportionally to weight."""
    if not profile.sections:
        return []

    num_secs = len(profile.sections)
    if target_count <= 0:
        target_count = min(50, max(5, num_secs * 3))

    total_weight = sum(s.weight for s in profile.sections) or float(num_secs)
    allocations: list[tuple[SectionProfile, int]] = []
    allocated_sum = 0

    for s in profile.sections:
        portion = int(round((s.weight / total_weight) * target_count))
        # Ensure at least 1 item per section if target allows
        if target_count >= num_secs:
            portion = max(1, portion)
        allocations.append((s, portion))
        allocated_sum += portion

    # Adjust rounding differences
    diff = target_count - allocated_sum
    if diff != 0 and allocations:
        # Add/subtract from highest weight section
        sorted_indices = sorted(range(len(allocations)), key=lambda i: allocations[i][0].weight, reverse=True)
        idx = sorted_indices[0]
        s, current_p = allocations[idx]
        allocations[idx] = (s, max(1, current_p + diff))

    return [a for a in allocations if a[1] > 0]

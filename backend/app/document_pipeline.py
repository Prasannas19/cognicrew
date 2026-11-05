"""Line-aware, structure-aware extraction, classification, and evidence retrieval.

This module deliberately has no database or web dependencies. That makes its
evidence rules directly testable and prevents a missing web key from affecting
Document Research.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


def clean_extracted_text(text: str) -> str:
    """Normalize unicode ligatures (e.g. fi, fl) and sanitize text for storage."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\x00", "")
    try:
        text.encode("cp1252")
    except UnicodeEncodeError:
        text = text.encode("cp1252", errors="replace").decode("cp1252")
    return text


class PageKind(StrEnum):
    CONTENT = "content"
    TOC = "toc"
    INDEX = "index"
    BIBLIOGRAPHY = "bibliography"
    NAVIGATION = "navigation"
    UNKNOWN = "unknown"


class QueryIntent(StrEnum):
    COUNT = "count"
    STRUCTURED_LIST = "structured_list"
    SECTION_EXTRACTION = "section_extraction"
    QUESTION_EXTRACTION = "question_extraction"
    METADATA = "metadata"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    EXACT_FACT = "exact_fact"
    SUMMARY = "summary"
    MULTI_PART = "multi_part"
    FOLLOW_UP = "follow_up"
    PAGE_LOOKUP = "page_lookup"
    STRUCTURAL_NAVIGATION = "structural_navigation"
    SHOW_ALL_ITEMS_IN_PARENT = "show_all_items_in_parent"


@dataclass(frozen=True)
class LineRecord:
    line_number: int  # 1-based per PDF page
    line_text: str
    original_line_order: int
    font_size: float = 0.0
    is_bold: bool = False
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class SectionEntity:
    entity_type: str  # e.g., 'slip', 'chapter', 'section', 'part', 'unit', 'question', 'heading'
    entity_number: str | None
    title: str
    start_page: int
    start_line: int
    end_page: int
    end_line: int
    content: str
    parent_number: str | None = None
    sub_entities: tuple[SectionEntity, ...] = ()


@dataclass(frozen=True)
class DocumentMetadata:
    title: str | None = None
    author: str | None = None
    publisher: str | None = None
    edition: str | None = None
    year: str | None = None
    page_count: int = 0
    printed_page_count: int | None = None


@dataclass(frozen=True)
class PageRecord:
    document_name: str
    pdf_page_number: int
    text: str
    page_kind: PageKind
    printed_page_label: str | None = None
    lines: tuple[LineRecord, ...] = ()
    sections: tuple[SectionEntity, ...] = ()
    document_id: str | None = None


@dataclass(frozen=True)
class DocumentStructureIndex:
    document_name: str
    metadata: DocumentMetadata
    pages: list[PageRecord]
    sections: list[SectionEntity]
    document_id: str | None = None


def sanitize_line_range(
    start_line: int | None,
    end_line: int | None,
    page_line_count: int | None = None,
) -> tuple[int | None, int | None]:
    """Sanitize and validate start_line and end_line according to canonical citation rules:

    1 <= start_line <= end_line <= actual_line_count

    If line information cannot be determined or violates constraints, returns (None, None).
    Never creates inverted ranges (e.g. 6-5), zero/negative ranges, or ranges beyond line count.
    """
    if start_line is None or end_line is None:
        return None, None
    s_line = start_line
    e_line = end_line

    if s_line < 1:
        return None, None
    if s_line > e_line:
        return None, None

    if page_line_count is not None and page_line_count > 0:
        if s_line > page_line_count:
            return None, None
        if e_line > page_line_count:
            e_line = page_line_count

    return s_line, e_line


@dataclass(frozen=True)
class Evidence:
    document_name: str
    pdf_page_number: int
    printed_page_label: str | None
    excerpt: str
    page_kind: PageKind
    key_sentences: tuple[str, ...] = ()
    matched_heading: str | None = None
    relevance_score: float = 0.0
    start_line: int | None = None
    end_line: int | None = None
    section_name: str | None = None
    entity_type: str | None = None
    entity_count: int | None = None
    document_id: str | None = None

    def __post_init__(self) -> None:
        s_line, e_line = sanitize_line_range(self.start_line, self.end_line)
        object.__setattr__(self, "start_line", s_line)
        object.__setattr__(self, "end_line", e_line)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to", "was", "with",
    # TASK 2 (v11): 'will' removed from stopwords — it is a CONTENT noun in
    # 'The Discipline of Will' and its removal deleted the query's subject
    # word entirely. Ubiquity filtering already demotes auxiliary 'will'.
    "book", "document", "page", "pdf", "information", "stated", "stated in",
    "what", "when", "where", "which", "who", "why", "how", "does", "do", "did",
    "explain", "explains", "explained", "explaining",
    "tell", "me", "find", "give", "detail", "details", "about", "used",
}

_SCAFFOLD = {
    "some", "something", "any", "anyone", "anything", "else",
    "say", "said", "tell", "told",
    "other", "others", "went", "was", "were", "be", "been", "so",
    "many", "much", "few", "several",
}

_TOKENS = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.I)
_PAGE_EDGE = re.compile(r"^(?:page\s+)?(\d{1,5})(?:\s+of\s+\d{1,5})?$", re.I)

_ANSWER_MARKERS = (
    "is called", "are called", "is defined as", "refers to", "known as",
    "means", "is the process of", "is used to", "is used for",
    "the purpose of", "is a technique", "is a form of", "is a type of",
    "is a kind of", "is the study of", "describes how", "is often called",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(]*[A-Z0-9(])")
_MARKER_RX = tuple(re.compile(rf"\b{re.escape(m)}\b[\s,;:—–\-\.]", re.I) for m in _ANSWER_MARKERS)

_CODE_LINE = re.compile(
    r"[{};]\s*$|\b(public|private|protected|static|final|void|int|long|return|import|package)\b.*[;(]"
)

# Generic pattern matching labeled document units: e.g. "Slip No.:-1", "Chapter 26: Images",
# "Section 2", "Part II", "Unit 3", "Assignment 4", "Paper 1", "Module 5".
# Requires whitespace or delimiter before the unit number/identifier to prevent matching plain words.
_GENERIC_UNIT_RX = re.compile(
    r"^\s*([A-Za-z]{2,20})(?:\s+(?:No\.?|Number|#))?(?:[\s:\.\-–—]+(?:No\.?|Number|#)?[\s:\.\-–—]*(\d+|[ivxlcdm]+|[A-Za-z]\b))(?:\s*[:\.\-–—]\s*(.*))?$",
    re.I
)

# Generic question header pattern: e.g. "Q.1. Attempt any four", "Question 2", "Question No. 1", "Problem 3"
_GENERIC_QUESTION_HEADER_RX = re.compile(
    r"^\s*(?:Q\.?\s*(?:No\.?|Number|#)?\s*[:\.\-–—]?\s*(\d+|[ivxlcdm]+|[a-z]\b)|Question\s*(?:No\.?|Number|#)?\s*[:\.\-–—]?\s*(\d+|[ivxlcdm]+|[a-z]\b)|Problem\s*(?:No\.?|Number|#)?\s*[:\.\-–—]?\s*(\d+|[ivxlcdm]+|[a-z]\b))\s*[:\.\-–—]?\s*(.*)$",
    re.I
)

# Numbered sub-questions or numbered items: e.g. "1. Represent the following", "2) Plot the graph", "(a) Scaling"
_GENERIC_NUMBERED_ITEM_RX = re.compile(
    r"^\s*(?:(?:\(?\s*(\d{1,3}|[a-z])\s*[\.\)]|\(\s*(\d{1,3}|[a-z])\s*\)))\s*(.*)$",
    re.I
)


def _has_answer_marker(text: str) -> bool:
    return any(rx.search(text) for rx in _MARKER_RX)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


# -------------------------------------------------------------------------
# TASK 2 (v11): query-type detection — structural vs explanatory
# -------------------------------------------------------------------------

# "what/explain/mean by/why/how does" phrasing -> the user wants an
# explanation, so short heading-only matches must NOT win outright.
_EXPLANATION_QUERY_RX = re.compile(
    r"\b(what|explain|explained|explaination|meaning|mean|means|why|how|define|definition|describe|difference)\b",
    re.I,
)

# A number/identifier reference ("slip 3", "question 12", "unit 4", "page 833",
# "Q.1", "chapter iv") -> the user wants a structural lookup, where a short
# exact heading match IS the right answer even if it carries little text.
_STRUCTURAL_QUERY_RX = re.compile(
    r"\b(slip|question|q|unit|chapter|section|part|module|assignment|problem|exercise|page|fig|figure|table|appendix)\s*"
    r"(?:no\.?|number|#)?\s*[\s:\.\-–—]*\s*(\d{1,4}|[ivxlcdm]{1,7})\b",
    re.I,
)


def is_explanatory_query(question: str) -> bool:
    """TASK 2 (v11): simple, testable query-type rule.

    A structural identifier (numbered slip/question/unit/page...) wins even
    when explanation words are also present ('explain slip 3 question 1').
    Otherwise, explanation phrasing ('what do you mean by X', 'explain X')
    makes the query explanatory. Everything else defaults to explanatory so
    the stricter evidence standard applies to unknown phrasing.
    """
    if _STRUCTURAL_QUERY_RX.search(question):
        return False
    return bool(_EXPLANATION_QUERY_RX.search(question))


def _cross_line_phrase_hit(
    page_text: str,
    phrases: list[str],
) -> tuple[str, int, int] | None:
    """TASK 2 (v11): first page line index where a phrase completes.

    Walks the page's token stream matching each phrase word-for-word IN ORDER,
    allowing at most ONE intervening connector word between phrase words
    ('discipline of will' matches; 'discipline. only then will' does NOT —
    the gap is 2 words, so cross-SENTENCE co-occurrence can never match) and
    a span of at most 2 adjacent lines. Returns (phrase, start_idx, end_idx)
    for scoring/anchoring, or None. Original word order only — this never
    manufactures a match from unrelated co-occurring words.
    """
    if not phrases:
        return None
    raw_lines = [l for l in page_text.splitlines()]
    tokens: list[tuple[str, int]] = []
    for li, line in enumerate(raw_lines):
        for w in _TOKENS.findall(line.lower()):
            tokens.append((w, li))
    for p in phrases:
        words = p.split()
        n = len(words)
        if n < 2:
            continue
        for start, (tw, tli) in enumerate(tokens):
            if tw != words[0]:
                continue
            wi = 1
            j = start + 1
            gap = 0
            end_li = tli
            while j < len(tokens) and wi < n:
                w, li = tokens[j]
                if w == words[wi]:
                    if li - end_li > 1:  # span at most 2 adjacent lines
                        break
                    end_li = li
                    wi += 1
                    gap = 0
                else:
                    gap += 1
                    if gap > 1:  # at most ONE connector word between phrase words
                        break
                    if li - end_li > 1:
                        break
                j += 1
            if wi == n:
                return (p, tli, end_li)
    return None


def _term_rx(term: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(term)}\b", re.I)


def _phrase_rx(phrase: str) -> re.Pattern:
    words = [re.escape(w) for w in phrase.split()]
    return re.compile(r"\b" + r"[\s\-]+".join(words) + r"\b", re.I)


def _phrase_rx_bridged(phrase: str) -> re.Pattern:
    """TASK 2 (v11): phrase RX tolerating ONE small function word between words.

    Query bigrams are built from stopword-stripped tokens, so 'discipline
    perception' can never match the actual text 'discipline of perception'.
    Allowing up to one connector word ('of', 'the', ...) between content words
    restores the match while keeping original word order (no reordering).
    """
    words = [re.escape(w) for w in phrase.split()]
    return re.compile(r"\b" + r"(?:\s+\w+)?\s+".join(words) + r"\b", re.I)


def _page_line_index_of_match(lines: Sequence[LineRecord], low: str, char_pos: int) -> int:
    """TASK 2 (v11): map a char position in normalized page text to a line index.

    `low` is page text with every whitespace run collapsed to one space, so a
    running character count over the same collapse maps char positions back to
    the line that contains them. Returns the LAST line index whose original
    text contributed on/before char_pos (clamped to the final line).
    """
    consumed = 0
    last_idx = 0
    for i, l in enumerate(lines):
        norm = re.sub(r"\s+", " ", l.line_text.strip())
        if not norm:
            continue
        if consumed + len(norm) >= char_pos:
            return i
        consumed += len(norm) + 1  # the single collapsed separator space
        last_idx = i
    return last_idx


def _concept(word: str) -> str:
    """Singular-ish normalized form so Interfaces~interface, classes~class, slips~slip."""
    w = word.lower()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _heading_matches(heading_text: str, phrase: str) -> bool:
    h = re.sub(r"\s+", " ", heading_text.strip().lower())
    h = re.sub(r"\s+\d{1,4}$", "", h)  # drop a trailing running page number
    p = re.sub(r"\s+", " ", phrase.strip().lower())
    if not h or not p:
        return False
    if h == p:
        return True
    hw = [_concept(w) for w in h.split()]
    pw = [_concept(w) for w in p.split()]
    n = len(pw)
    for start in range(len(hw) - n + 1):
        if all(hw[start + k] == pw[k] for k in range(n)):
            return True
    content_words = [w for w in hw if w not in _STOPWORDS]
    if content_words == pw or set(content_words) == set(pw):
        return True
    return False


def _looks_like_code(line: str) -> bool:
    return bool(_CODE_LINE.search(line))


def _unit_heading_matches(heading_text: str, phrase: str) -> bool:
    """TASK 7 (v8): structural-unit numbers must match across separators.

    'UNIT-4', 'Unit 4:', 'UNIT 4' and roman 'UNIT IV' all match the phrase
    'unit 4'. Plain _heading_matches fails because '-' tokenizes differently.
    """
    h = re.sub(r"\s+", " ", heading_text.strip().lower())
    p = re.sub(r"\s+", " ", phrase.strip().lower())
    unit_rx = r"^(unit|chapter|part|section)\b[\s\-:.]*([0-9]+|[ivxlc]+)\b"
    m_h = re.match(unit_rx, h)
    m_p = re.match(unit_rx, p)
    if not (m_h and m_p):
        return False
    if m_h.group(1) != m_p.group(1):
        return False
    return _unit_number(m_h.group(2)) == _unit_number(m_p.group(2))


def _unit_number(num: str) -> str:
    if num.isdigit():
        return str(int(num))
    roman = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
    total = 0
    for i, ch in enumerate(num):
        val = roman.get(ch, 0)
        nxt = roman.get(num[i + 1], 0) if i + 1 < len(num) else 0
        total += -val if nxt > val else val
    return str(total)


def _is_probable_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 72 or len(s.split()) > 10:
        return False
    if s.endswith((".", ",", ";", "{", "}")):
        return False
    if _looks_like_code(s):
        return False
    tokens_match = _TOKENS.match(s.lstrip('\"“\'(`[{'))
    first_word = tokens_match.group(0) if tokens_match else ""
    if first_word and first_word[0].islower():
        return False
    return True


_UNIT_HEADER_RX = re.compile(r"^(unit|chapter|part)\b[\s\-:.]*[0-9ivxlc]+\b", re.I)


_UNIT_HEADER_RX = re.compile(r"^(unit|chapter|part)\b[\s\-:.]*[0-9ivxlc]+\b", re.I)


def _is_section_boundary(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if re.match(r"^(table|chapter|part|figure|appendix|listing|slip|section|unit)\b\s*[0-9ivxlc]", s, re.I):
        return True
    if re.match(r"^(ask the expert|progress check|critical skill|self test|viva voce)\b", s, re.I):
        return True
    # TASK 1 (v11): a line with UNBALANCED parentheses is mid-parenthetical
    # text, not a boundary. Daily Stoic p.12 has real content lines like
    # 'around us)' and 'to what end)' (short, no terminal punctuation) that
    # terminated an explanatory window after ONE line; the parenthetical they
    # close opened on an earlier line.
    if s.count("(") != s.count(")"):
        return False
    if s.endswith("("):
        return False
    if len(s) <= 60 and len(s.split()) <= 6 and not s.endswith((".", ",", ";", ":")):
        return True
    return False


def classify_page(text: str) -> PageKind:
    """Classify non-evidence pages conservatively from their own text."""
    compact = re.sub(r"\s+", " ", text).strip().lower()
    if not compact:
        return PageKind.UNKNOWN
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]

    # Table of Contents
    dot_leaders = len(re.findall(r"(?:\.\s*){3,}\s*\d{1,5}\b", text))
    if "table of contents" in compact or dot_leaders >= 3:
        return PageKind.TOC
    for line in lines[:3]:
        if re.search(r"\bcontents\b", line) and len(line) < 35:
            return PageKind.TOC

    # Index
    for line in lines[:3]:
        if re.search(r"\bindex\b", line) and len(line) < 35:
            return PageKind.INDEX
    if len(re.findall(r"\b\d{1,5}\s*[-–—]\s*\d{1,5}\b", text)) >= 8:
        return PageKind.INDEX

    # Bibliography / References
    if any(marker in compact[:700] for marker in ("bibliography", "references")):
        return PageKind.BIBLIOGRAPHY

    if len(compact) < 100 and any(marker in compact for marker in ("contents", "next chapter", "previous chapter")):
        return PageKind.NAVIGATION

    return PageKind.CONTENT


def _candidate_from_line(line: str) -> int | None:
    line = line.strip()
    match = _PAGE_EDGE.match(line)
    if match:
        return int(match.group(1))
    match = re.match(r"^(\d{1,5})\s+[A-Za-z]", line)
    if match:
        return int(match.group(1))
    match = re.search(r"[A-Za-z]\s+(\d{1,5})$", line)
    if match:
        return int(match.group(1))
    return None


def _edge_page_candidate(lines: list[LineRecord]) -> int | None:
    text_lines = [l.line_text.strip() for l in lines if l.line_text.strip()]
    for line in [*text_lines[:2], *text_lines[-2:]]:
        cand = _candidate_from_line(line)
        if cand is not None:
            return cand
    return None


# -------------------------------------------------------------------------
# Page and Line Level Extraction with Layout Awareness (PyMuPDF)
# -------------------------------------------------------------------------

def _extract_page_lines_from_fitz_page(page, pdf_page_number: int) -> list[LineRecord]:
    """Extract line records with 1-based line numbers, preserving original reading order."""
    lines: list[LineRecord] = []
    line_number = 0
    order = 0

    try:
        page_dict = page.get_text("dict")
        blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
        # Sort blocks by vertical position then horizontal to guarantee reading order
        blocks.sort(key=lambda b: (round(b.get("bbox", [0, 0])[1], -1), b.get("bbox", [0, 0])[0]))

        for b in blocks:
            b_lines = b.get("lines", [])
            for bl in b_lines:
                spans = bl.get("spans", [])
                raw_text = "".join(s.get("text", "") for s in spans)
                clean = clean_extracted_text(raw_text).strip()
                if not clean:
                    continue
                line_number += 1
                order += 1
                font_size = max((s.get("size", 0.0) for s in spans), default=0.0)
                is_bold = any(
                    (s.get("flags", 0) & 16 != 0) or ("bold" in s.get("font", "").lower()) or ("black" in s.get("font", "").lower())
                    for s in spans
                )
                bbox = tuple(bl.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                lines.append(LineRecord(
                    line_number=line_number,
                    line_text=clean,
                    original_line_order=order,
                    font_size=round(font_size, 1),
                    is_bold=is_bold,
                    bbox=bbox,
                ))
    except Exception:
        # Fallback to plain text split if dict extraction has issues
        raw_lines = clean_extracted_text(page.get_text("text")).splitlines()
        for raw in raw_lines:
            s = raw.strip()
            if not s:
                continue
            line_number += 1
            order += 1
            lines.append(LineRecord(
                line_number=line_number,
                line_text=s,
                original_line_order=order,
                font_size=0.0,
                is_bold=False,
                bbox=None,
            ))

    return lines


def is_poor_extraction(text: str | None) -> bool:
    """Conservative helper to detect poor or malformed text extraction:
    - empty/very short text
    - very low alphabetic ratio
    - broken words / excessive isolated characters
    - excessive line breaks
    - malformed extraction / replacement chars
    """
    if not text:
        return True
    cleaned = text.strip()
    if len(cleaned) < 20:
        return True

    # Malformed extraction / replacement character check
    if cleaned.count("\ufffd") >= 3 or (cleaned.count("\ufffd") / len(cleaned)) > 0.05:
        return True

    # Low alphabetic / alphanumeric ratio
    total_len = len(cleaned)
    alpha_chars = sum(c.isalpha() for c in cleaned)
    alnum_chars = sum(c.isalnum() for c in cleaned)
    if total_len >= 30 and (alnum_chars / total_len) < 0.25:
        return True
    if total_len >= 40 and (alpha_chars / total_len) < 0.15:
        return True

    words = cleaned.split()
    if len(words) >= 6:
        # Broken words / excessive isolated characters (e.g. single letters spaced out)
        single_chars = sum(1 for w in words if len(w) == 1 and w.isalpha())
        if (single_chars / len(words)) > 0.40:
            return True

    # Excessive line breaks relative to content length
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if len(lines) >= 6:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if avg_line_len < 3.5:
            return True

    return False


def ocr_pdf_page(pdf_path: Path, page_number: int) -> tuple[str, list[LineRecord]]:
    """Perform selective OCR on a physical PDF page (1-based page_number).

    Uses PyMuPDF's built-in get_textpage_ocr with local tesseract runtime if available.
    Returns normalized extracted text and a list of LineRecords.
    """
    import os
    import fitz  # PyMuPDF

    tessdata_candidates = [
        Path(__file__).resolve().parent.parent / "bin" / "tesseract_env" / "usr" / "share" / "tesseract-ocr" / "5" / "tessdata",
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tesseract-ocr/4.00/tessdata"),
    ]
    tessdata_path = None
    for cand in tessdata_candidates:
        if cand.exists():
            tessdata_path = str(cand)
            break

    try:
        with fitz.open(pdf_path) as doc:
            p_idx = page_number - 1
            if p_idx < 0 or p_idx >= len(doc):
                return "", []
            page = doc[p_idx]
            page_obj: Any = page
            if tessdata_path:
                tp = page_obj.get_textpage_ocr(tessdata=tessdata_path)
            else:
                tp = page_obj.get_textpage_ocr()
            raw_text = tp.extractText()
            clean_text = clean_extracted_text(raw_text).strip()
            if not clean_text:
                return "", []

            lines: list[LineRecord] = []
            for idx, raw_line in enumerate(clean_text.splitlines(), start=1):
                sl = raw_line.strip()
                if sl:
                    lines.append(LineRecord(
                        line_number=idx,
                        line_text=sl,
                        original_line_order=idx,
                        font_size=10.0,
                        is_bold=False,
                        bbox=None,
                    ))
            return clean_text, lines
    except Exception:
        return "", []


def extract_pdf_pages(pdf_path: Path, document_name: str) -> list[PageRecord]:
    """Extract pages and lines from a PDF, preserving page and line representations."""
    pages, _ = extract_pdf_pages_and_structure(pdf_path, document_name)
    return pages


def extract_pdf_pages_and_structure(pdf_path: Path, document_name: str) -> tuple[list[PageRecord], DocumentStructureIndex]:
    """Extract full page records, line records, and generic document structure from PDF."""
    import fitz  # PyMuPDF

    pages: list[PageRecord] = []
    doc_metadata = DocumentMetadata(page_count=0)

    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        pdf_meta = doc.metadata or {}
        raw_page_lines: list[list[LineRecord]] = []
        raw_page_texts: list[str] = []

        for p_idx in range(page_count):
            p = doc[p_idx]
            lines = _extract_page_lines_from_fitz_page(p, p_idx + 1)
            raw_page_lines.append(lines)
            page_text = "\n".join(l.line_text for l in lines)
            raw_page_texts.append(page_text)

        # Printed page label detection
        candidates = [_edge_page_candidate(lines) for lines in raw_page_lines]

        for idx in range(page_count):
            lines = raw_page_lines[idx]
            text = raw_page_texts[idx]
            candidate = candidates[idx]
            neighbour = candidates[idx - 1] if idx else None
            following = candidates[idx + 1] if idx + 1 < page_count else None
            corroborated = (
                candidate is not None
                and ((neighbour is not None and candidate - neighbour == 1)
                     or (following is not None and following - candidate == 1))
            )
            kind = classify_page(text)
            pages.append(PageRecord(
                document_name=document_name,
                pdf_page_number=idx + 1,
                text=text,
                page_kind=kind,
                printed_page_label=str(candidate) if corroborated and kind == PageKind.CONTENT else None,
                lines=tuple(lines),
            ))

    # Detect generic document structure across pages and lines
    structure_index = build_document_structure_index(document_name, pages, pdf_meta)

    # Attach detected sections to pages for convenient lookup
    enriched_pages: list[PageRecord] = []
    for p in pages:
        p_sections = [s for s in structure_index.sections if s.start_page <= p.pdf_page_number <= s.end_page]
        enriched_pages.append(PageRecord(
            document_name=p.document_name,
            pdf_page_number=p.pdf_page_number,
            text=p.text,
            page_kind=p.page_kind,
            printed_page_label=p.printed_page_label,
            lines=p.lines,
            sections=tuple(p_sections),
        ))

    return enriched_pages, structure_index


# -------------------------------------------------------------------------
# Generic Document Structure Detection & Boundary Enforcement
# -------------------------------------------------------------------------

def build_document_structure_index(
    document_name: str,
    pages: list[PageRecord],
    pdf_meta: dict | None = None,
) -> DocumentStructureIndex:
    """Detect structural units (chapters, sections, slips, questions, lists) generically from patterns."""
    raw_sections: list[dict] = []
    detected_questions: list[dict] = []

    seen_entities: set[tuple[str, str]] = set()

    for p in pages:
        # Avoid treating TOC or Index pages as the primary declaration of content entities
        is_navigation = p.page_kind in (PageKind.TOC, PageKind.INDEX, PageKind.BIBLIOGRAPHY)

        for l in p.lines:
            text = l.line_text.strip()
            if not text:
                continue

            # 1. Match generic labeled units (Slip 1, Chapter 26, Section 2, Part II, Unit 3, etc.)
            m_unit = _GENERIC_UNIT_RX.match(text)
            if m_unit:
                entity_label = m_unit.group(1).strip()
                entity_num = m_unit.group(2).strip()
                norm_type = _concept(entity_label).lower()
                key = (norm_type, entity_num.lower())

                # Discard duplicate running headers on subsequent pages
                if key not in seen_entities and not is_navigation:
                    seen_entities.add(key)
                    raw_sections.append({
                        "entity_type": norm_type,
                        "entity_number": entity_num,
                        "title": text,
                        "start_page": p.pdf_page_number,
                        "start_line": l.line_number,
                        "end_page": p.pdf_page_number,
                        "end_line": len(p.lines),
                    })
                continue

            # 2. Match question headers (e.g. Q.1., Question 1, Problem 1)
            m_q = _GENERIC_QUESTION_HEADER_RX.match(text)
            if m_q and not is_navigation:
                q_num = m_q.group(1) or m_q.group(2) or m_q.group(3) or "1"
                detected_questions.append({
                    "entity_type": "question",
                    "entity_number": q_num.strip(),
                    "title": text,
                    "start_page": p.pdf_page_number,
                    "start_line": l.line_number,
                    "end_page": p.pdf_page_number,
                    "end_line": len(p.lines),
                })
                continue

    # Set boundaries for each structural section:
    # A section ends immediately before the next section of equal hierarchy begins.
    sections: list[SectionEntity] = []
    for i, s in enumerate(raw_sections):
        start_page = s["start_page"]
        start_line = s["start_line"]
        if i + 1 < len(raw_sections):
            next_s = raw_sections[i + 1]
            if next_s["start_page"] == start_page:
                end_page = start_page
                end_line = max(start_line, next_s["start_line"] - 1)
            else:
                end_page = next_s["start_page"] - 1
                end_line = len(pages[end_page - 1].lines) if (pages and 1 <= end_page <= len(pages)) else start_line
        else:
            # Last section continues to the end of the document or its page
            end_page = pages[-1].pdf_page_number if pages else start_page
            end_line = len(pages[end_page - 1].lines) if (pages and 1 <= end_page <= len(pages)) else start_line

        if end_page == start_page and end_line < start_line:
            end_line = start_line

        # Collect bounded content
        bounded_lines: list[str] = []
        for p in pages:
            if start_page <= p.pdf_page_number <= end_page:
                for l in p.lines:
                    if p.pdf_page_number == start_page and l.line_number < start_line:
                        continue
                    if p.pdf_page_number == end_page and l.line_number > end_line:
                        continue
                    bounded_lines.append(l.line_text)

        # Collect sub-entities (e.g. questions belonging to this section) with precise boundaries
        sec_questions = [
            q for q in detected_questions
            if (start_page <= q["start_page"] <= end_page)
            and not (q["start_page"] == start_page and q["start_line"] < start_line)
            and not (q["start_page"] == end_page and q["start_line"] > end_line)
        ]
        sec_questions.sort(key=lambda q: (q["start_page"], q["start_line"]))

        sub_entities: list[SectionEntity] = []
        for j, q in enumerate(sec_questions):
            q_start_p = q["start_page"]
            q_start_l = q["start_line"]
            if j + 1 < len(sec_questions):
                next_q = sec_questions[j + 1]
                if next_q["start_page"] == q_start_p:
                    q_end_p = q_start_p
                    q_end_l = max(q_start_l, next_q["start_line"] - 1)
                else:
                    q_end_p = next_q["start_page"] - 1
                    q_end_l = len(pages[q_end_p - 1].lines) if (pages and 1 <= q_end_p <= len(pages)) else q_start_l
            else:
                q_end_p = end_page
                q_end_l = end_line

            if q_end_p == q_start_p and q_end_l < q_start_l:
                q_end_l = q_start_l

            q_lines: list[str] = []
            for p in pages:
                if q_start_p <= p.pdf_page_number <= q_end_p:
                    for l in p.lines:
                        if p.pdf_page_number == q_start_p and l.line_number < q_start_l:
                            continue
                        if p.pdf_page_number == q_end_p and l.line_number > q_end_l:
                            continue
                        q_lines.append(l.line_text)

            q_content = "\n".join(q_lines) if q_lines else q["title"]
            sub_entities.append(SectionEntity(
                entity_type=q["entity_type"],
                entity_number=q["entity_number"],
                title=q["title"],
                start_page=q_start_p,
                start_line=q_start_l,
                end_page=q_end_p,
                end_line=q_end_l,
                content=q_content,
                parent_number=s["entity_number"],
            ))

        sections.append(SectionEntity(
            entity_type=s["entity_type"],
            entity_number=s["entity_number"],
            title=s["title"],
            start_page=start_page,
            start_line=start_line,
            end_page=end_page,
            end_line=end_line,
            content="\n".join(bounded_lines),
            sub_entities=tuple(sub_entities),
        ))

    # Document metadata extraction from PDF metadata and front matter
    pdf_meta = pdf_meta or {}
    meta_author = pdf_meta.get("author") or None
    meta_title = pdf_meta.get("title") or None
    meta_year = None
    meta_edition = None

    # Inspect front matter for bibliographic details if not in PDF metadata
    front_pages = pages[:8]
    for p in front_pages:
        t = p.text
        if not meta_author:
            m_auth = re.search(r"copyright[^\n]*?by\s+([A-Z][\w'\u2019.-]+(?:\s+[A-Z][\w'\u2019.-]+)+)", t, re.I)
            if m_auth:
                meta_author = m_auth.group(1).strip()
            else:
                m_names = re.search(r"Names\s*:\s*([A-Z][^.]*?(?:,\s*author\b[^.]*)+)", t, re.I)
                if m_names:
                    meta_author = m_names.group(1).strip()
        if not meta_year:
            m_year = re.search(r"\b(19\d\d|20\d\d)\b", t)
            if m_year:
                meta_year = m_year.group(1)
        if not meta_edition:
            m_ed = re.search(r"\b(\d+(?:st|nd|rd|th)?\s+edition)\b", t, re.I)
            if m_ed:
                meta_edition = m_ed.group(1)

    doc_metadata = DocumentMetadata(
        title=meta_title,
        author=meta_author,
        publisher=pdf_meta.get("publisher"),
        edition=meta_edition,
        year=meta_year,
        page_count=len(pages),
        printed_page_count=None,
    )

    doc_id = pages[0].document_id if pages and pages[0].document_id else None
    return DocumentStructureIndex(
        document_name=document_name,
        metadata=doc_metadata,
        pages=pages,
        sections=sections,
        document_id=doc_id,
    )


# -------------------------------------------------------------------------
# Query Understanding & Intent Classification (12 Intents)
# -------------------------------------------------------------------------

_COUNT_RX = re.compile(r"\b(how\s+many|number\s+of|total\s+(?:count|number|no\.?|quantity)\s+of|count\s+of)\b", re.I)
_STRUCTURED_LIST_RX = re.compile(r"\b(?:list\s+all|list\s+the|show\s+all|name\s+all|enumerate)\s+([a-z]+)\b", re.I)
_CONTAINER_EXTRACT_RX = re.compile(r"\b(?:from|in|of)\s+([a-z]{2,15})\s*(?:no\.?|number|#)?[:\.\-–—#\s]+(\d+|[ivxlcdm]+\b|[a-z]\b)", re.I)
_DIRECT_EXTRACT_RX = re.compile(r"\b(?:give|show|get|retrieve|fetch)\s+(?:me\s+)?([a-z]{2,15})\s*(?:no\.?|number|#)?[:\.\-–—#\s]+(\d+|[ivxlcdm]+\b|[a-z]\b)", re.I)
_PREFIX_EXTRACT_RX = re.compile(r"\b([a-z]{2,15})\s*(?:no\.?|number|#)?[:\.\-–—#\s]+(\d+|[ivxlcdm]+\b|[a-z]\b)\s+(?:content|questions?|problems?)\b", re.I)
_METADATA_RX = re.compile(
    r"\b(who\s+(?:is\s+the\s+author|wrote|authored)|authors?|writer|what\s+is\s+the\s+title|book\s+title|page\s+count|number\s+of\s+pages|when\s+was\s+(?:this\s+)?(?:published|written)|publication\s+year|which\s+edition|what\s+edition)\b",
    re.I
)
_DEF_RX = re.compile(r"\b(?:what\s+is|what\s+are|what\s+does\s+(.+)\s+mean|meaning\s+of|definition\s+of|define)\b", re.I)
_EXPLAIN_RX = re.compile(r"\b(?:explain|how\s+does\s+(.+)\s+work|describe|how\s+to|why\s+does)\b", re.I)
_COMPARE_RX = re.compile(r"\b(?:difference\s+between|compare|comparison\s+between|versus|vs\.?|how\s+does\s+(.+)\s+differ)\b", re.I)
_EXACT_FACT_RX = re.compile(r"\b(?:which\s+page|on\s+which\s+page|on\s+what\s+page|where\s+is\s+(.+)\s+(?:stated|mentioned|found))\b", re.I)
_SUMMARY_RX = re.compile(
    r"\b(?:summarize|summary\s+of|brief\s+overview\s+of|overview\s+of|"
    r"profile\s+(?:this\s+)?(?:entire\s+)?(?:document|pdf|file|book|text)|"
    r"profile\s+the\s+(?:entire\s+)?(?:document|pdf|file|book|text)|"
    r"what\s+is\s+this\s+(?:entire\s+)?(?:pdf|file|document|book)\s+about|"
    r"what\s+(?:skills|topics)\s+does\s+this\s+(?:document|book|pdf)\s+teach|"
    r"key\s+takeaways\s+from\s+(?:this|the)\s+(?:document|book|pdf|paper)|"
    r"executive\s+summary\s+of\s+(?:this|the)\s+(?:uploaded\s+)?(?:document|paper|pdf|book)|"
    r"explain\s+what\s+this\s+(?:entire\s+)?(?:document|pdf|book)\s+is\s+about)\b",
    re.I,
)
_FOLLOW_UP_RX = re.compile(r"\b(?:it|its|this|that|next\s+one|previous\s+one|tell\s+me\s+more|simpler\s+terms)\b", re.I)


_PDF_PAGE_RX = re.compile(
    r"^(?:show\s+|what(?:'s|\s+is)\s+on\s+|go\s+to\s+|view\s+)?(?:pdf\s+page|physical\s+page)\s*[:#\s]*(\d+)\??$",
    re.I
)
_BOOK_PAGE_RX = re.compile(
    r"^(?:show\s+|what(?:'s|\s+is)\s+on\s+|go\s+to\s+|view\s+)?(?:book\s+page|printed\s+page)\s*[:#\s]*(\d+)\??$",
    re.I
)
_GENERIC_PAGE_RX = re.compile(
    r"^(?:show\s+|what(?:'s|\s+is)\s+on\s+|go\s+to\s+|view\s+)?page\s*[:#\s]*(\d+)\??$",
    re.I
)
_PAGE_LOOKUP_COMPREHENSIVE_RX = re.compile(
    r"\b(?:(?:tell\s+me\s+(?:the\s+)?(?:info|information)\s+(?:on|about|from)|what(?:\s+is|'s)?\s+(?:written\s+)?(?:on|in)|info\s+(?:on|from)|content\s+(?:on|of|from)|notes\s+on|show|view|go\s+to|explain|summarize)\s+)?"
    r"(?:(?:pdf|physical|book|printed)\s+)?page\s*(?:number|no\.?|#)?\s*[:\s]*(\d+)\b|"
    r"\b(?:on|from|at)\s+page\s*(?:number|no\.?|#)?\s*[:\s]*(\d+)\b",
    re.I
)

# Nested: "Question 3 of Slip 2", "Q.3 in Slip 2", "Q3 in Slip 2", "Give Question 3 from Slip 2"
_NESTED_CHILD_PARENT_RX = re.compile(
    r"\b(?:give\s+(?:me\s+)?|show\s+)?(?:q(?:\.?\s*)?(?![a-z])|question\b|problem\b)\s*[:.\-–—#\s]*(\d+|[a-z])\s+(?:of|in|from)\s+([a-z]{2,15})\s*(?:no\.?|number|#)?[:.\-–—#\s]*(\d+|[ivxlcdm]+|[a-z])\b",
    re.I
)

# Nested: "Slip 2 Question 3", "Slip 2, Q.3", "Slip 2 Question No. 3"
_NESTED_PARENT_CHILD_RX = re.compile(
    r"\b([a-z]{2,15})\b\s*(?:no\.?|number|#)?[:.\-–—#\s]*(\d+|[ivxlcdm]+|[a-z])\b\s*[,:\.\-–—\s]+\s*(?:q\.?|question|problem)\b\s*(?:no\.?|number|#)?\s*[:.\-–—#\s]*(\d+|[a-z])\b",
    re.I
)

# Concise question: "question 3", "q3", "q.3", "give question 3", "what is question 1"
_CONCISE_QUESTION_RX = re.compile(
    r"^(?:what\s+(?:is|was)\s+(?:the\s+)?|give\s+(?:me\s+)?|show\s+|view\s+|tell\s+(?:me\s+)?(?:about\s+)?|explain\s+(?:the\s+)?)?(?:q\.?|question|problem)\s*[:.\-–—#\s]*(\d+|[a-z])\??$",
    re.I
)

# Concise section: "slip 3", "chapter 4", "section 2"
_CONCISE_UNIT_RX = re.compile(
    r"^(?:what\s+(?:is|was)\s+(?:the\s+)?|give\s+(?:me\s+)?|show\s+|view\s+)?(slip|chapter|section|unit|part|module|act|appendix|exercise|assignment)\s*(?:(?:no\.?|number|#)[:.\-–—#\s]*(\d+|[ivxlcdm]+|[a-z])|[:.\-–—#\s]+(\d+|[ivxlcdm]+|[a-z])|(\d+|[ivxlcdm]+))\??$",
    re.I
)

_ORDINAL_MAP = {
    "first": "1",
    "1st": "1",
    "first one": "1",
    "second": "2",
    "2nd": "2",
    "second one": "2",
    "third": "3",
    "3rd": "3",
    "third one": "3",
    "fourth": "4",
    "4th": "4",
    "fifth": "5",
    "5th": "5",
}
_ORDINAL_QUESTION_RX = re.compile(
    r"\b(?:what\s+(?:was|is)\s+the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)\s+question\s+(?:in|of|from)\s+([a-z]{2,15})\s*(?:no\.?|number|#)?[:.\-–—#\s]*(\d+|[ivxlcdm]+|[a-z])\b",
    re.I,
)
_ORDINAL_CONCISE_QUESTION_RX = re.compile(
    r"^(?:what\s+(?:was|is)\s+the\s+|explain\s+(?:the\s+)?|show\s+(?:the\s+)?|give\s+(?:the\s+)?)(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)\s+question\??$|"
    r"^(?:explain\s+(?:the\s+)?|what\s+is\s+(?:the\s+)?)(first\s+one|second\s+one|third\s+one|fourth\s+one|fifth\s+one)\??$",
    re.I,
)

_PARENT_ALL_ITEMS_RX = re.compile(
    r"\b(?:give\s+(?:all\s+)?|show\s+(?:all\s+)?|what\s+are\s+the\s+|list\s+(?:all\s+)?)(?:questions?|topics?|problems?|items?)\s+(?:from|in|of)\s+([a-z]{2,15})\s*(?:no\.?|number|#)?[:\.\-–—#\s]+(\d+|[ivxlcdm]+\b|[a-z]\b)",
    re.I,
)
_PREFIX_ALL_ITEMS_RX = re.compile(
    r"\b([a-z]{2,15})\s*(?:no\.?|number|#)?[:\.\-–—#\s]+(\d+|[ivxlcdm]+\b|[a-z]\b)\s+(?:questions?|topics?|problems?|items?|all\s+questions?)\b",
    re.I,
)
_DIRECT_SHOW_PARENT_RX = re.compile(
    r"^(?:show\s+|give\s+|view\s+|list\s+|tell\s+(?:me\s+)?)?([a-z]{2,15})\s*(?:no\.?|number|#)?[:\.\-–—#\s]+(\d+|[ivxlcdm]+\b|[a-z]\b)\s*(?:all\s+)?(?:questions?|topics?|problems?|content)?\??$",
    re.I,
)
_SOLVE_REQ_RX = re.compile(
    r"\b(solve|calculate|compute|solution|evaluate|find\s+the\s+(?:distance|slope|equation|centroid|angle))\b",
    re.I,
)


@dataclass(frozen=True)
class QueryClassification:
    intent: QueryIntent
    target_entity: str | None = None
    target_identifier: str | None = None
    parent_entity: str | None = None
    parent_identifier: str | None = None
    page_number: int | None = None
    page_type: str | None = None  # "pdf" or "book"
    phrases: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)


def classify_query(question: str) -> QueryClassification:
    """Classify the user's query intent into one of 14 distinct categories."""
    q = question.strip().lower()
    phrases, terms = _analyse_question(question)

    # 1. Multi-part check FIRST (e.g. "What is double buffering, and on which page is it stated?")
    if " and " in q and (q.count("what") + q.count("which") + q.count("how") + q.count("where") >= 2 or ", and " in q):
        return QueryClassification(intent=QueryIntent.MULTI_PART, phrases=phrases, terms=terms)

    # 2. Direct page navigation queries ("page 201", "PDF page 201", "book page 833", "show page 201", "tell me the info on page 100", "what is written on page 100")
    m_pdf_page = _PDF_PAGE_RX.match(q)
    if m_pdf_page:
        return QueryClassification(
            intent=QueryIntent.PAGE_LOOKUP,
            page_number=int(m_pdf_page.group(1)),
            page_type="pdf",
            phrases=phrases,
            terms=terms,
        )
    m_book_page = _BOOK_PAGE_RX.match(q)
    if m_book_page:
        return QueryClassification(
            intent=QueryIntent.PAGE_LOOKUP,
            page_number=int(m_book_page.group(1)),
            page_type="book",
            phrases=phrases,
            terms=terms,
        )
    m_gen_page = _GENERIC_PAGE_RX.match(q)
    if m_gen_page:
        return QueryClassification(
            intent=QueryIntent.PAGE_LOOKUP,
            page_number=int(m_gen_page.group(1)),
            page_type="pdf",
            phrases=phrases,
            terms=terms,
        )
    m_page_comp = _PAGE_LOOKUP_COMPREHENSIVE_RX.search(q)
    if m_page_comp and not (_NESTED_CHILD_PARENT_RX.search(q) or _NESTED_PARENT_CHILD_RX.search(q)):
        raw_num = m_page_comp.group(1) or m_page_comp.group(2)
        if raw_num and raw_num.isdigit():
            p_type = "book" if ("book" in q or "printed" in q) else "pdf"
            return QueryClassification(
                intent=QueryIntent.PAGE_LOOKUP,
                page_number=int(raw_num),
                page_type=p_type,
                phrases=phrases,
                terms=terms,
            )

    # 3. Nested Question Extraction ("Question 3 of Slip 2", "Slip 2 Question 3", "Q3 in Slip 2")
    m_ncp = _NESTED_CHILD_PARENT_RX.search(q)
    if m_ncp:
        child_id = m_ncp.group(1).strip()
        parent_ent = _concept(m_ncp.group(2)).lower()
        parent_id = m_ncp.group(3).strip()
        return QueryClassification(
            intent=QueryIntent.QUESTION_EXTRACTION,
            target_entity="question",
            target_identifier=child_id,
            parent_entity=parent_ent,
            parent_identifier=parent_id,
            phrases=phrases,
            terms=terms,
        )

    m_npc = _NESTED_PARENT_CHILD_RX.search(q)
    if m_npc:
        parent_ent = _concept(m_npc.group(1)).lower()
        parent_id = m_npc.group(2).strip()
        child_id = m_npc.group(3).strip()
        return QueryClassification(
            intent=QueryIntent.QUESTION_EXTRACTION,
            target_entity="question",
            target_identifier=child_id,
            parent_entity=parent_ent,
            parent_identifier=parent_id,
            phrases=phrases,
            terms=terms,
        )

    # 3.5 Ordinal Question Extraction ("what was the second question in slip 1?", "third question from slip 2")
    m_ord = _ORDINAL_QUESTION_RX.search(q)
    if m_ord:
        child_ord = m_ord.group(1).lower()
        child_id = _ORDINAL_MAP.get(child_ord, "1")
        parent_ent = _concept(m_ord.group(2)).lower()
        parent_id = m_ord.group(3).strip()
        return QueryClassification(
            intent=QueryIntent.QUESTION_EXTRACTION,
            target_entity="question",
            target_identifier=child_id,
            parent_entity=parent_ent,
            parent_identifier=parent_id,
            phrases=phrases,
            terms=terms,
        )

    m_ord_concise = _ORDINAL_CONCISE_QUESTION_RX.match(q)
    if m_ord_concise:
        child_ord = (m_ord_concise.group(1) or m_ord_concise.group(2) or "1").lower().strip()
        child_id = _ORDINAL_MAP.get(child_ord, "1")
        return QueryClassification(
            intent=QueryIntent.QUESTION_EXTRACTION,
            target_entity="question",
            target_identifier=child_id,
            parent_entity=None,
            parent_identifier=None,
            phrases=phrases,
            terms=terms,
        )

    # 3.6 Parent Question Extraction ("Slip 1 questions", "give all questions from slip 1", "show slip 1", "list slip 1 questions", "Unit 4 topics")
    is_solve = bool(_SOLVE_REQ_RX.search(q))
    m_all = _PARENT_ALL_ITEMS_RX.search(q) or _PREFIX_ALL_ITEMS_RX.search(q) or _DIRECT_SHOW_PARENT_RX.match(q)
    if not is_solve and m_all:
        parent_ent = _concept(m_all.group(1)).lower()
        parent_id = m_all.group(2).strip()
        return QueryClassification(
            intent=QueryIntent.QUESTION_EXTRACTION,
            parent_entity=parent_ent,
            parent_identifier=parent_id,
            target_entity="question",
            target_identifier=None,
            phrases=phrases,
            terms=terms,
        )

    # 4. Concise Question / Unit ("question 3", "q3", "slip 3", "chapter 4")
    m_cq = _CONCISE_QUESTION_RX.match(q)
    if m_cq:
        child_id = m_cq.group(1).strip()
        return QueryClassification(
            intent=QueryIntent.QUESTION_EXTRACTION,
            target_entity="question",
            target_identifier=child_id,
            parent_entity=None,
            parent_identifier=None,
            phrases=phrases,
            terms=terms,
        )

    m_cu = _CONCISE_UNIT_RX.match(q)
    if m_cu and _concept(m_cu.group(1)).lower() != "page":
        ent_type = _concept(m_cu.group(1)).lower()
        ent_id = (m_cu.group(2) or m_cu.group(3) or m_cu.group(4) or "").strip()
        return QueryClassification(
            intent=QueryIntent.SECTION_EXTRACTION,
            target_entity=ent_type,
            target_identifier=ent_id,
            phrases=phrases,
            terms=terms,
        )

    # 5. Count queries ("how many slips...", "how many chapters...", "how many pages...")
    m_count = _COUNT_RX.search(q)
    if m_count:
        after = q[m_count.end():].strip()
        tokens = [t for t in _TOKENS.findall(after) if t not in _STOPWORDS]
        target = _concept(tokens[0]) if tokens else "entity"
        return QueryClassification(intent=QueryIntent.COUNT, target_entity=target, phrases=phrases, terms=terms)

    # 6. Metadata queries ("who is the author", "how many pages", "when published", "what title")
    if _METADATA_RX.search(q) or ("pages" in q and ("how many" in q or "count" in q)):
        target = "author"
        if "title" in q:
            target = "title"
        elif "page" in q:
            target = "page_count"
        elif "publish" in q or "year" in q:
            target = "year"
        elif "edition" in q:
            target = "edition"
        return QueryClassification(intent=QueryIntent.METADATA, target_entity=target, phrases=phrases, terms=terms)

    # 7. Section / Question extraction for a specific target ("Give all questions from Slip 1", "Show Section 2")
    m_sec = _CONTAINER_EXTRACT_RX.search(q) or _DIRECT_EXTRACT_RX.search(q) or _PREFIX_EXTRACT_RX.search(q)
    if m_sec:
        ent_type = _concept(m_sec.group(1)).lower()
        ent_id = m_sec.group(2).strip()
        is_question_req = any(w in q for w in ("question", "questions", "q.1", "q1", "problems"))
        intent = QueryIntent.QUESTION_EXTRACTION if is_question_req else QueryIntent.SECTION_EXTRACTION
        return QueryClassification(
            intent=intent,
            target_entity=ent_type,
            target_identifier=ent_id,
            phrases=phrases,
            terms=terms
        )

    # 8. Structured list ("List all chapters", "List all sections")
    m_list = _STRUCTURED_LIST_RX.search(q)
    if m_list:
        target = _concept(m_list.group(1)).lower()
        return QueryClassification(intent=QueryIntent.STRUCTURED_LIST, target_entity=target, phrases=phrases, terms=terms)

    # 9. Comparison
    if _COMPARE_RX.search(q):
        return QueryClassification(intent=QueryIntent.COMPARISON, phrases=phrases, terms=terms)

    # 10. Exact Fact / Location ("on which page is double buffering stated")
    if _EXACT_FACT_RX.search(q) and not _DEF_RX.search(q):
        return QueryClassification(intent=QueryIntent.EXACT_FACT, phrases=phrases, terms=terms)

    # 11. Summary
    if _SUMMARY_RX.search(q):
        return QueryClassification(intent=QueryIntent.SUMMARY, phrases=phrases, terms=terms)

    # 12. Follow-up
    if _FOLLOW_UP_RX.search(q) and len(q.split()) <= 7:
        return QueryClassification(intent=QueryIntent.FOLLOW_UP, phrases=phrases, terms=terms)

    # 13. Definition
    if _DEF_RX.search(q):
        return QueryClassification(intent=QueryIntent.DEFINITION, phrases=phrases, terms=terms)

    # 14. Explanation
    if _EXPLAIN_RX.search(q):
        return QueryClassification(intent=QueryIntent.EXPLANATION, phrases=phrases, terms=terms)

    # Default to semantic explanation
    return QueryClassification(intent=QueryIntent.EXPLANATION, phrases=phrases, terms=terms)


def _analyse_question(question: str) -> tuple[list[str], list[str]]:
    tokens = _TOKENS.findall(question)
    kept = [(i, t.lower()) for i, t in enumerate(tokens)
            if len(t) > 1 and t.lower() not in _STOPWORDS and t.lower() not in _SCAFFOLD]
    phrases: list[str] = []
    for (i1, t1), (i2, t2), (i3, t3) in zip(kept, kept[1:], kept[2:]):
        phrases.append(f"{t1} {t2} {t3}")
    for (i1, t1), (i2, t2) in zip(kept, kept[1:]):
        phrases.append(f"{t1} {t2}")
    terms = sorted({t for _, t in kept})
    # TASK 2 (v11): interrogative furniture ('you', 'mean', 'please', ...) is
    # kept as scoring terms but must not appear inside PHRASES. 'what do you
    # mean by The Discipline of Will' previously produced 'you mean' and
    # 'mean discipline' glue phrases that matched nothing useful while the
    # subject pair never survived. Furniture-free phrases keep only content
    # words ('discipline will', 'discipline perception').
    furniture = {"you", "your", "yours", "mean", "means", "meaning", "please",
                 "tell", "exactly", "really", "actually", "just", "kind", "sort"}
    cleaned: list[str] = []
    for p in phrases:
        content_words = [w for w in p.split() if w not in furniture]
        if len(content_words) == len(p.split()):
            cleaned.append(p)  # no furniture: keep as-is
        elif len(content_words) >= 2:
            cleaned.append(" ".join(content_words))
        # furniture-only or furniture+1 content phrases are dropped (the
        # content word remains available as a term)
    phrases = list(dict.fromkeys(cleaned))
    return phrases, terms


# -------------------------------------------------------------------------
# Structural Count Engine (True Structural Counting)
# -------------------------------------------------------------------------

def execute_count_retrieval(
    pages: list[PageRecord],
    target_entity: str,
    structure: DocumentStructureIndex | None = None,
) -> list[Evidence]:
    """Execute true structural count handling. Never counts search results."""
    if not pages:
        return []

    doc_name = pages[0].document_name

    # Physical / printed page count
    if target_entity in ("page", "pages"):
        count = len(pages)
        first_page = pages[0]
        excerpt = f"This document contains {count} physical PDF pages."
        return [Evidence(
            document_name=doc_name,
            pdf_page_number=1,
            printed_page_label=first_page.printed_page_label,
            excerpt=excerpt,
            page_kind=PageKind.CONTENT,
            key_sentences=(excerpt,),
            matched_heading="Document Page Count",
            relevance_score=100.0,
            start_line=1,
            end_line=1,
            section_name="Document Metadata",
            entity_type="page",
            entity_count=count,
        )]

    # Inspect structural sections
    sections = structure.sections if structure else [s for p in pages for s in p.sections]
    # Deduplicate by entity type and number
    seen = set()
    matching_entities: list[SectionEntity] = []
    norm_target = _concept(target_entity).lower()

    for s in sections:
        if s.entity_type == norm_target or norm_target in s.title.lower():
            key = (s.entity_type, s.entity_number)
            if key not in seen:
                seen.add(key)
                matching_entities.append(s)

    if not matching_entities:
        return []

    # Sort matching entities by start page and line
    matching_entities.sort(key=lambda e: (e.start_page, e.start_line))
    total_count = len(matching_entities)
    first_ent = matching_entities[0]
    last_ent = matching_entities[-1]

    # Detailed structural evidence summary
    label_cap = norm_target.capitalize()
    direct_sentence = (
        f"There are {total_count} {norm_target}s in this document "
        f"({label_cap} {first_ent.entity_number} to {label_cap} {last_ent.entity_number})."
    )
    excerpt_lines = [
        direct_sentence,
        f"Detected {total_count} verified structural units across {pages[0].document_name}:",
    ]
    for ent in matching_entities[:8]:
        excerpt_lines.append(f"- {ent.title} (PDF Page {ent.start_page}, Line {ent.start_line})")
    if total_count > 8:
        excerpt_lines.append(f"... and {total_count - 8} more.")

    page_by_num: dict[int, PageRecord] = {p.pdf_page_number: p for p in pages}
    first_page_rec = page_by_num.get(first_ent.start_page)

    return [Evidence(
        document_name=doc_name,
        pdf_page_number=first_ent.start_page,
        printed_page_label=first_page_rec.printed_page_label if first_page_rec else None,
        excerpt="\n".join(excerpt_lines),
        page_kind=PageKind.CONTENT,
        key_sentences=(direct_sentence,),
        matched_heading=f"{label_cap} Structure Count",
        relevance_score=100.0,
        start_line=first_ent.start_line,
        end_line=first_ent.end_line,
        section_name=first_ent.title,
        entity_type=norm_target,
        entity_count=total_count,
    )]


# -------------------------------------------------------------------------
# Boundary-Aware Section & List Extraction
# -------------------------------------------------------------------------

def execute_section_extraction(
    pages: list[PageRecord],
    target_entity: str,
    target_id: str,
    structure: DocumentStructureIndex | None = None,
    extract_questions_only: bool = False,
) -> list[Evidence]:
    """Retrieve content strictly bounded by the requested structural unit.

    Stops strictly at the start of the next equal/higher-level unit.
    For Slip 1: extracts only Slip 1 and NEVER leaks Slip 2.
    """
    if not pages:
        return []

    sections = structure.sections if structure else [s for p in pages for s in p.sections]
    norm_target = _concept(target_entity).lower()
    target_id_clean = target_id.strip().lower()

    # Find matching section entity
    matched: SectionEntity | None = None
    for s in sections:
        if (s.entity_type == norm_target or norm_target in s.title.lower()):
            num = (s.entity_number or "").strip().lower()
            if num == target_id_clean:
                matched = s
                break

    if matched is None:
        return []

    doc_name = pages[0].document_name

    # Build bounded content lines with exact line numbers
    collected_lines: list[tuple[int, int, str]] = []  # (page, line, text)
    for p in pages:
        if matched.start_page <= p.pdf_page_number <= matched.end_page:
            for l in p.lines:
                if p.pdf_page_number == matched.start_page and l.line_number < matched.start_line:
                    continue
                if p.pdf_page_number == matched.end_page and l.line_number > matched.end_line:
                    continue
                collected_lines.append((p.pdf_page_number, l.line_number, l.line_text))

    if not collected_lines:
        return []

    if extract_questions_only:
        # Format questions from this bounded section
        if matched.sub_entities:
            content_text = "\n\n".join(f"{q.content}" for q in matched.sub_entities)
        else:
            q_lines = [t for _, _, t in collected_lines if _GENERIC_QUESTION_HEADER_RX.match(t) or _GENERIC_NUMBERED_ITEM_RX.match(t)]
            content_text = "\n".join(q_lines) if q_lines else "\n".join(t for _, _, t in collected_lines)
    else:
        content_text = "\n".join(t for _, _, t in collected_lines)

    first_page = matched.start_page
    first_line = matched.start_line
    page_by_num: dict[int, PageRecord] = {p.pdf_page_number: p for p in pages}
    first_page_rec = page_by_num.get(first_page)
    lines_on_first_page = len(first_page_rec.lines) if first_page_rec and first_page_rec.lines else matched.end_line
    if matched.end_page == first_page:
        raw_last_line = matched.end_line
    else:
        raw_last_line = lines_on_first_page

    start_line, end_line = sanitize_line_range(first_line, raw_last_line, lines_on_first_page)

    key_sentence = f"{matched.title} contains the following bounded content:"
    excerpt = f"{matched.title}\n\n{content_text}"

    return [Evidence(
        document_name=doc_name,
        pdf_page_number=first_page,
        printed_page_label=first_page_rec.printed_page_label if first_page_rec else None,
        excerpt=excerpt,
        page_kind=PageKind.CONTENT,
        key_sentences=(key_sentence,),
        matched_heading=matched.title,
        relevance_score=100.0,
        start_line=start_line,
        end_line=end_line,
        section_name=matched.title,
        entity_type=norm_target,
    )]


def format_parent_questions(sec_title: str, content: str) -> str:
    """Format all questions from a parent structural section (e.g. Slip 1, Slip 5)
    into a complete, prominent, beautifully formatted markdown structure:
    # Slip 1 — Questions
    ## Q.1. Attempt any four of the following. [5×4=20 Marks]
    ### 1.
    [full original question]
    """
    m_ent = re.search(r"([a-z]+)\s*(?:no\.?\s*[:\-–—\s]*)?(\d+|[ivxlcdm]+|[a-z])", sec_title, re.I)
    if m_ent:
        ent_name = m_ent.group(1).capitalize()
        ent_num = m_ent.group(2)
        header = f"# {ent_name} {ent_num} — Questions\n\n"
    else:
        header = f"# {sec_title} — Questions\n\n"

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    formatted_parts: list[str] = [header]

    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip top slip header line or time/marks header line
        if re.match(r"^slip\s*no", line, re.I) or ("time:" in line.lower() and "maximum marks:" in line.lower()):
            i += 1
            continue
        if re.match(r"^––––|^\*\s*–––", line):
            i += 1
            continue

        # Major Q header: Q.1., Q.2., Q.3
        m_q = re.match(r"^(Q\.?\s*\d+\.?.*?)(?:\[|\battempt\b|$)", line, re.I)
        if m_q or line.startswith("Q."):
            cleaned_line = re.sub(r"\[\s*(\d+)\s*[x×]\s*(\d+)\s*=\s*(\d+)\s*marks\s*\]", r"[\1×\2=\3 Marks]", line, flags=re.I)
            cleaned_line = re.sub(r"\[\s*(\d+)\s*marks\s*\]", r"[\1 Marks]", cleaned_line, flags=re.I)
            formatted_parts.append(f"## {cleaned_line}\n\n")
            i += 1
            continue

        # Sub-question number: "1.", "2.", "3."
        m_num = re.match(r"^(\d+)\.\s*(.*)", line)
        if m_num:
            num = m_num.group(1)
            rest = m_num.group(2).strip()
            formatted_parts.append(f"### {num}.\n")
            q_text_lines: list[str] = []
            if rest:
                q_text_lines.append(rest)
            i += 1
            while i < len(lines):
                next_l = lines[i]
                if next_l.startswith("Q.") or re.match(r"^\d+\.\s*", next_l) or re.match(r"^––––", next_l):
                    break
                q_text_lines.append(next_l)
                i += 1

            full_q = "\n".join(q_text_lines)
            if "Subject" in full_q and "Marks" in full_q:
                parts = [p.strip() for p in q_text_lines if p.strip()]
                intro = parts[0] if parts else ""
                subj_idx = parts.index("Subject") if "Subject" in parts else -1
                marks_idx = parts.index("Marks") if "Marks" in parts else -1
                if subj_idx != -1 and marks_idx != -1:
                    subjects = parts[subj_idx + 1:marks_idx]
                    marks = parts[marks_idx + 1:]
                    tbl = "\n\n| Subject | " + " | ".join(subjects) + " |\n"
                    tbl += "| :--- | " + " | ".join([":---"] * len(subjects)) + " |\n"
                    tbl += "| **Marks** | " + " | ".join(marks) + " |\n\n"
                    formatted_parts.append(intro + "\n" + tbl)
                else:
                    formatted_parts.append(full_q + "\n\n")
            else:
                formatted_parts.append(full_q + "\n\n")
            continue

        formatted_parts.append(line + "\n\n")
        i += 1

    res = "".join(formatted_parts)
    # Math OCR cleanup
    res = re.sub(r"\bx\s*3\b", "x³", res)
    res = re.sub(r"\bx\s*2\b", "x²", res)
    res = re.sub(r"\by\s*2\b", "y²", res)
    res = re.sub(r"\bx2\b", "x²", res)
    res = re.sub(r"\bx3\b", "x³", res)
    res = re.sub(r"\by2\b", "y²", res)
    res = re.sub(r"\bx\s*\?\s*(\[.*?\])", r"x ∈ \1", res)
    res = re.sub(r"\b([a-z])\s*\?\s*(\[.*?\])", r"\1 ∈ \2", res)
    res = re.sub(r"(\-?\d+)\s*\?\s*x\s*,\s*y\s*\?\s*(\d+)", r"\1 ≤ x, y ≤ \2", res)
    res = re.sub(r"\?\s*([A-Z]{3})\b", r"Δ\1", res)
    res = re.sub(r"x³\s*\?\s*(\d+x)", r"x³ – \1", res)
    return res


def execute_show_all_items_in_parent(
    pages: list[PageRecord],
    parent_type: str,
    parent_id: str,
    structure: DocumentStructureIndex | None = None,
) -> list[Evidence]:
    """Extract and format all questions/items for a parent unit (e.g. Slip 1)."""
    if not pages:
        return []

    sections = structure.sections if structure else [s for p in pages for s in p.sections]
    norm_parent = _concept(parent_type).lower()
    parent_id_clean = parent_id.strip().lower()

    matched: SectionEntity | None = None
    for s in sections:
        if s.entity_type == norm_parent or norm_parent in s.title.lower():
            num = (s.entity_number or "").strip().lower()
            if num == parent_id_clean:
                matched = s
                break

    if matched is None:
        return []

    first_page = matched.start_page
    first_line = matched.start_line
    page_by_num: dict[int, PageRecord] = {p.pdf_page_number: p for p in pages}
    first_page_rec = page_by_num.get(first_page)
    lines_on_first_page = len(first_page_rec.lines) if first_page_rec and first_page_rec.lines else matched.end_line
    raw_last_line = matched.end_line if matched.end_page == first_page else lines_on_first_page
    start_line, end_line = sanitize_line_range(first_line, raw_last_line, lines_on_first_page)

    formatted_text = format_parent_questions(matched.title, matched.content)
    key_sentence = f"{matched.title} contains all questions and tasks:"

    return [Evidence(
        document_name=pages[0].document_name,
        pdf_page_number=first_page,
        printed_page_label=first_page_rec.printed_page_label if first_page_rec else None,
        excerpt=formatted_text,
        page_kind=PageKind.CONTENT,
        key_sentences=(key_sentence,),
        matched_heading=f"{matched.title} - Questions",
        relevance_score=100.0,
        start_line=start_line,
        end_line=end_line,
        section_name=f"{matched.title} > Questions",
        entity_type=norm_parent,
    )]


def execute_document_summary_retrieval(
    pages: list[PageRecord],
    structure: DocumentStructureIndex | None = None,
) -> list[Evidence]:
    """TASK 5: Build document-level evidence package for DOCUMENT_SUMMARY intent.
    Synthesizes:
    1. Document metadata (title, author, page count, etc.)
    2. Structural sections/slips/chapters discovered in document
    3. Representative content from beginning, middle, and end
    """
    if not pages:
        return []

    doc_name = pages[0].document_name
    total_pages = len(pages)
    sections = structure.sections if structure else [s for p in pages for s in p.sections]

    lines: list[str] = [
        f"# Document Structure & Overview: {doc_name}",
        f"Total Pages: {total_pages}",
    ]
    if structure and structure.metadata:
        if structure.metadata.title:
            lines.append(f"Title: {structure.metadata.title}")
        if structure.metadata.author:
            lines.append(f"Author: {structure.metadata.author}")
        if structure.metadata.edition:
            lines.append(f"Edition: {structure.metadata.edition}")
        if structure.metadata.year:
            lines.append(f"Year: {structure.metadata.year}")

    if sections:
        lines.append(f"\nMajor Sections / Units Found ({len(sections)} total):")
        for s in sections[:25]:
            sub_count = len(s.sub_entities)
            sub_str = f" ({sub_count} questions/sub-items)" if sub_count else ""
            lines.append(f"- {s.title} [Page {s.start_page}]{sub_str}")
        if len(sections) > 25:
            lines.append(f"- ... and {len(sections) - 25} more sections.")

    first_page_text = pages[0].text.strip()
    if first_page_text:
        lines.append("\n## Representative Beginning / Front Section Content:")
        lines.append(first_page_text[:1200])

    if total_pages > 2:
        mid_idx = total_pages // 2
        mid_page_text = pages[mid_idx].text.strip()
        if mid_page_text:
            lines.append(f"\n## Representative Middle Section Content (Page {pages[mid_idx].pdf_page_number}):")
            lines.append(mid_page_text[:1000])

    if total_pages > 1:
        last_page_text = pages[-1].text.strip()
        if last_page_text:
            lines.append(f"\n## Representative Ending Section Content (Page {pages[-1].pdf_page_number}):")
            lines.append(last_page_text[:800])

    summary_text = "\n".join(lines)
    p1 = pages[0]
    p1_lines = len(p1.lines) if p1.lines else 20
    start_line, end_line = sanitize_line_range(1, p1_lines, p1_lines)

    return [Evidence(
        document_name=doc_name,
        pdf_page_number=1,
        printed_page_label=p1.printed_page_label,
        excerpt=summary_text,
        page_kind=PageKind.CONTENT,
        key_sentences=(f"Document structure and overview of {doc_name} with {total_pages} pages.",),
        matched_heading=f"Document Overview: {doc_name}",
        relevance_score=100.0,
        start_line=start_line,
        end_line=end_line,
        section_name="Document Structure & Overview",
        entity_type="document_overview",
    )]


def execute_nested_entity_retrieval(
    pages: list[PageRecord],
    parent_type: str,
    parent_id: str,
    child_type: str,
    child_id: str,
    structure: DocumentStructureIndex | None = None,
) -> list[Evidence]:
    """Retrieve a specific child entity inside a parent entity.

    e.g. Question 3 of Slip 2:
    1. Identify parent section 'Slip 2'
    2. Locate requested child 'Question 3' within parent
    3. Extract ONLY that child entity's bounded lines and text
    4. Provide strictly validated line citations.
    """
    if not pages:
        return []

    sections = structure.sections if structure else [s for p in pages for s in p.sections]
    norm_parent = _concept(parent_type).lower()
    norm_parent_id = parent_id.strip().lstrip("0") or "0"
    norm_child = _concept(child_type).lower()
    norm_child_id = child_id.strip().lstrip("0") or "0"

    parent_match: SectionEntity | None = None
    for s in sections:
        if s.entity_type == norm_parent or norm_parent in s.title.lower():
            s_num = (s.entity_number or "").strip().lstrip("0") or "0"
            if s_num.lower() == norm_parent_id.lower():
                parent_match = s
                break

    if parent_match is None:
        return []

    child_match: SectionEntity | None = None
    for sub in parent_match.sub_entities:
        sub_type = _concept(sub.entity_type).lower()
        sub_num = (sub.entity_number or "").strip().lstrip("0") or "0"
        if (sub_type == norm_child or norm_child in sub.title.lower()) and sub_num.lower() == norm_child_id.lower():
            child_match = sub
            break

    if child_match is None:
        target_lines: list[tuple[int, int, str]] = []
        found_target = False
        target_page = parent_match.start_page
        for p in pages:
            if parent_match.start_page <= p.pdf_page_number <= parent_match.end_page:
                for l in p.lines:
                    if p.pdf_page_number == parent_match.start_page and l.line_number < parent_match.start_line:
                        continue
                    if p.pdf_page_number == parent_match.end_page and l.line_number > parent_match.end_line:
                        continue
                    m_q = _GENERIC_QUESTION_HEADER_RX.match(l.line_text)
                    if m_q:
                        q_num = (m_q.group(1) or m_q.group(2) or m_q.group(3) or "").strip().lstrip("0") or "0"
                        if q_num.lower() == norm_child_id.lower():
                            found_target = True
                            target_page = p.pdf_page_number
                        elif found_target:
                            found_target = False
                            break
                    if found_target:
                        target_lines.append((p.pdf_page_number, l.line_number, l.line_text))

        if target_lines:
            page_by_num: dict[int, PageRecord] = {p.pdf_page_number: p for p in pages}
            target_page = target_lines[0][0]
            start_l = target_lines[0][1]
            target_page_rec = page_by_num.get(target_page)
            lines_on_page = len(target_page_rec.lines) if target_page_rec and target_page_rec.lines else target_lines[-1][1]
            end_l = target_lines[-1][1] if target_lines[-1][0] == target_page else lines_on_page
            s_l, e_l = sanitize_line_range(start_l, end_l, lines_on_page)
            content = "\n".join(t for _, _, t in target_lines)
            heading = f"{parent_match.title} - Question {child_id}"
            return [Evidence(
                document_name=pages[0].document_name,
                pdf_page_number=target_page,
                printed_page_label=target_page_rec.printed_page_label if target_page_rec else None,
                excerpt=content,
                page_kind=PageKind.CONTENT,
                key_sentences=(heading,),
                matched_heading=heading,
                relevance_score=100.0,
                start_line=s_l,
                end_line=e_l,
                section_name=f"{parent_match.title} > Question {child_id}",
                entity_type=norm_child,
            )]

        # TASK 4 (v11): slips number their tasks '1.'..'5.' under 'Q.1.' with
        # NO Q-prefix, so the question-header scan above cannot see them and
        # 'Slip 1 Question 4' fell through to closest-match. Second-chance
        # scan: numbered-item lines ('4. Using sympy declare points P(7,2)…')
        # within the SAME parent page bounds. Scan stops at the next numbered
        # item OR the next Q-header so the window stays inside one task.
        if not found_target:
            for p in pages:
                if parent_match.start_page <= p.pdf_page_number <= parent_match.end_page:
                    for l in p.lines:
                        if p.pdf_page_number == parent_match.start_page and l.line_number < parent_match.start_line:
                            continue
                        if p.pdf_page_number == parent_match.end_page and l.line_number > parent_match.end_line:
                            continue
                        if _GENERIC_QUESTION_HEADER_RX.match(l.line_text.strip()):
                            if found_target:
                                found_target = False
                                break
                            continue
                        m_item = _GENERIC_NUMBERED_ITEM_RX.match(l.line_text.strip())
                        if m_item:
                            num = (m_item.group(1) or m_item.group(2) or "").strip().lstrip("0") or "0"
                            if num.lower() == norm_child_id.lower():
                                found_target = True
                                target_page = p.pdf_page_number
                            elif found_target:
                                found_target = False
                                break
                        if found_target:
                            target_lines.append((p.pdf_page_number, l.line_number, l.line_text))
                    if not found_target and target_lines:
                        break

        if target_lines:
            page_by_num: dict[int, PageRecord] = {p.pdf_page_number: p for p in pages}
            target_page = target_lines[0][0]
            start_l = target_lines[0][1]
            target_page_rec = page_by_num.get(target_page)
            lines_on_page = len(target_page_rec.lines) if target_page_rec and target_page_rec.lines else target_lines[-1][1]
            end_l = target_lines[-1][1] if target_lines[-1][0] == target_page else lines_on_page
            s_l, e_l = sanitize_line_range(start_l, end_l, lines_on_page)
            content = "\n".join(t for _, _, t in target_lines)
            heading = f"{parent_match.title} - Question {child_id}"
            return [Evidence(
                document_name=pages[0].document_name,
                pdf_page_number=target_page,
                printed_page_label=target_page_rec.printed_page_label if target_page_rec else None,
                excerpt=content,
                page_kind=PageKind.CONTENT,
                key_sentences=(heading,),
                matched_heading=heading,
                relevance_score=100.0,
                start_line=s_l,
                end_line=e_l,
                section_name=f"{parent_match.title} > Question {child_id}",
                entity_type=norm_child,
            )]
        return []

    page_by_num2: dict[int, PageRecord] = {p.pdf_page_number: p for p in pages}
    target_page = child_match.start_page
    child_page_rec = page_by_num2.get(target_page)
    lines_on_page = len(child_page_rec.lines) if child_page_rec and child_page_rec.lines else child_match.end_line
    last_l = child_match.end_line if child_match.end_page == target_page else lines_on_page
    start_l, end_l = sanitize_line_range(child_match.start_line, last_l, lines_on_page)

    heading = f"{parent_match.title} - {child_match.title}"
    return [Evidence(
        document_name=pages[0].document_name,
        pdf_page_number=target_page,
        printed_page_label=child_page_rec.printed_page_label if child_page_rec else None,
        excerpt=child_match.content,
        page_kind=PageKind.CONTENT,
        key_sentences=(child_match.title,),
        matched_heading=heading,
        relevance_score=100.0,
        start_line=start_l,
        end_line=end_l,
        section_name=f"{parent_match.title} > {child_match.title}",
        entity_type=norm_child,
    )]


def execute_page_lookup(
    pages: list[PageRecord],
    page_number: int | None,
    page_type: str = "pdf",
) -> list[Evidence]:
    """Direct page lookup for physical PDF page or printed book page label."""
    if not pages or page_number is None:
        return []

    target_page: PageRecord | None = None
    if page_type in ("book", "printed"):
        target_str = str(page_number)
        for p in pages:
            if p.printed_page_label and p.printed_page_label.strip() == target_str:
                target_page = p
                break
    else:
        # Search by physical PDF page number attribute first
        target_str = str(page_number)
        for p in pages:
            if p.pdf_page_number == page_number:
                target_page = p
                break
        # Fallback: try printed page label (handles "page 45" meaning book page 45)
        if target_page is None:
            for p in pages:
                if p.printed_page_label and p.printed_page_label.strip() == target_str:
                    target_page = p
                    break

    if target_page is None:
        return []

    doc_name = target_page.document_name
    lines = list(target_page.lines)
    actual_line_count = len(lines) if lines else len(target_page.text.splitlines())

    start_line, end_line = sanitize_line_range(1, actual_line_count, actual_line_count)
    label = target_page.printed_page_label
    title = f"PDF Page {target_page.pdf_page_number}" + (f" (Book Page {label})" if label else "")
    excerpt = target_page.text.strip()
    if not excerpt:
        excerpt = f"PDF page {target_page.pdf_page_number} could not be text-extracted; OCR is unavailable or produced no text."
    key_sentence = f"Content of {title}:"

    return [Evidence(
        document_name=doc_name,
        pdf_page_number=target_page.pdf_page_number,
        printed_page_label=label,
        excerpt=excerpt,
        page_kind=target_page.page_kind,
        key_sentences=(key_sentence,),
        matched_heading=title,
        relevance_score=100.0,
        start_line=start_line,
        end_line=end_line,
        section_name=title,
        entity_type="page",
    )]


def execute_structured_list_retrieval(
    pages: list[PageRecord],
    target_entity: str,
    structure: DocumentStructureIndex | None = None,
) -> list[Evidence]:
    """List all detected entities of a given type (e.g. 'List all chapters')."""
    if not pages:
        return []

    sections = structure.sections if structure else [s for p in pages for s in p.sections]
    norm_target = _concept(target_entity).lower()

    seen = set()
    matching: list[SectionEntity] = []
    for s in sections:
        if s.entity_type == norm_target or norm_target in s.title.lower():
            key = (s.entity_type, s.entity_number)
            if key not in seen:
                seen.add(key)
                matching.append(s)

    if not matching:
        return []

    matching.sort(key=lambda s: (s.start_page, s.start_line))
    label_cap = norm_target.capitalize()
    lines = [f"All detected {label_cap} entities in {pages[0].document_name}:"]
    for s in matching:
        lines.append(f"- {s.title} (PDF Page {s.start_page})")

    first_page = matching[0].start_page
    excerpt = "\n".join(lines)
    key_sentence = lines[0]
    page_by_num: dict[int, PageRecord] = {p.pdf_page_number: p for p in pages}
    first_page_rec = page_by_num.get(first_page)

    return [Evidence(
        document_name=pages[0].document_name,
        pdf_page_number=first_page,
        printed_page_label=first_page_rec.printed_page_label if first_page_rec else None,
        excerpt=excerpt,
        page_kind=PageKind.CONTENT,
        key_sentences=(key_sentence,),
        matched_heading=f"List of {label_cap} Entities",
        relevance_score=100.0,
        start_line=matching[0].start_line,
        end_line=matching[-1].end_line,
        section_name=f"List of {label_cap} Entities",
        entity_type=norm_target,
        entity_count=len(matching),
    )]


# -------------------------------------------------------------------------
# Metadata Retrieval Engine
# -------------------------------------------------------------------------

def execute_metadata_retrieval(
    pages: list[PageRecord],
    field_name: str,
    structure: DocumentStructureIndex | None = None,
) -> list[Evidence]:
    """Retrieve metadata (author, title, page count, year, edition) from front matter and metadata."""
    if not pages:
        return []

    doc_name = pages[0].document_name

    if field_name == "page_count":
        return execute_count_retrieval(pages, "page", structure)

    if field_name == "author":
        # Search front matter pages 1-8
        candidates = sorted(pages, key=lambda p: p.pdf_page_number)[:8]
        for p in candidates:
            for l in p.lines:
                t = l.line_text
                # Copyright line: Copyright © <year> by <Author>
                m = re.search(r"copyright[^\n]*?by\s+([A-Z][\w'\u2019.-]+(?:\s+[A-Z][\w'\u2019.-]+)+)", t, re.I)
                if m:
                    author_name = m.group(1).strip()
                    excerpt = f"{t}\n\nAuthor is stated in the copyright notice on the front matter."
                    return [Evidence(
                        document_name=doc_name,
                        pdf_page_number=p.pdf_page_number,
                        printed_page_label=p.printed_page_label,
                        excerpt=excerpt,
                        page_kind=p.page_kind,
                        key_sentences=(f"The author is {author_name}.",),
                        matched_heading=None,
                        relevance_score=100.0,
                        start_line=l.line_number,
                        end_line=l.line_number,
                        section_name="Front Matter Copyright",
                    )]
                # Library of Congress style: Names: Holiday, Ryan, author
                m2 = re.search(r"Names\s*:\s*([A-Z][^.]*?(?:,\s*author\b[^.]*)+)", t, re.I)
                if m2:
                    author_name = m2.group(1).strip()
                    excerpt = f"{t}\n\nAuthor is stated in the cataloging data."
                    return [Evidence(
                        document_name=doc_name,
                        pdf_page_number=p.pdf_page_number,
                        printed_page_label=p.printed_page_label,
                        excerpt=excerpt,
                        page_kind=p.page_kind,
                        key_sentences=(f"The author is {author_name}.",),
                        matched_heading=None,
                        relevance_score=100.0,
                        start_line=l.line_number,
                        end_line=l.line_number,
                        section_name="Front Matter Cataloging",
                    )]
                # Title page "by Author"
                m3 = re.search(r"^by\s+([A-Z][\w'\u2019.-]+(?:\s+[A-Z][\w'\u2019.-]+)+)$", t, re.I)
                if m3:
                    author_name = m3.group(1).strip()
                    excerpt = f"{t}\n\nAuthor is stated on the title page."
                    return [Evidence(
                        document_name=doc_name,
                        pdf_page_number=p.pdf_page_number,
                        printed_page_label=p.printed_page_label,
                        excerpt=excerpt,
                        page_kind=p.page_kind,
                        key_sentences=(f"The author is {author_name}.",),
                        matched_heading=None,
                        relevance_score=100.0,
                        start_line=l.line_number,
                        end_line=l.line_number,
                        section_name="Front Matter Title Page",
                    )]

    if field_name == "title":
        # Check first page top lines
        if pages and pages[0].lines:
            top_line = pages[0].lines[0]
            excerpt = f"Document Title: {top_line.line_text}"
            return [Evidence(
                document_name=doc_name,
                pdf_page_number=1,
                printed_page_label=pages[0].printed_page_label,
                excerpt=excerpt,
                page_kind=pages[0].page_kind,
                key_sentences=(excerpt,),
                matched_heading="Document Title",
                relevance_score=100.0,
                start_line=top_line.line_number,
                end_line=top_line.line_number,
                section_name="Title",
            )]

    return []


# -------------------------------------------------------------------------
# Semantic / Concept Retrieval & Evidence Windows
# -------------------------------------------------------------------------

def _excerpt_with_window(
    lines: list[LineRecord],
    target_idx: int,
    budget: int = 900
) -> tuple[str, int, int]:
    """Extract a coherent evidence window around the target line.

    Returns (excerpt_text, start_line, end_line).
    """
    if not lines:
        return "", 1, 1

    # Look backwards 1-2 lines if part of the same paragraph (not a boundary)
    start_idx = target_idx
    while start_idx > 0 and (target_idx - start_idx < 2):
        prev = lines[start_idx - 1].line_text
        if prev.endswith(":"):
            break
        if _is_section_boundary(prev) and not _UNIT_HEADER_RX.match(prev.strip()):
            # TASK 7 (v8): a unit/chapter header block (e.g. 'UNIT-4' followed
            # by its topic list) sits immediately before real content, so the
            # backward walk must jump PAST it instead of stopping on it.
            break
        start_idx -= 1

    # Look forwards until boundary or budget reached
    end_idx = target_idx
    total_len = sum(len(lines[i].line_text) + 1 for i in range(start_idx, end_idx + 1))

    while end_idx + 1 < len(lines) and total_len < budget:
        nxt = lines[end_idx + 1].line_text
        if _is_section_boundary(nxt):
            break
        end_idx += 1
        total_len += len(nxt) + 1

    chosen_lines = lines[start_idx : end_idx + 1]
    excerpt_text = " ".join(l.line_text for l in chosen_lines)
    return excerpt_text, chosen_lines[0].line_number, chosen_lines[-1].line_number


def _heading_window(
    lines: list[LineRecord],
    content_idx: int,
    heading_idx: int,
    budget: int = 900,
) -> tuple[str, int, int]:
    """TASK 7 (v8): evidence window spanning a matched heading AND its content.

    Starts at the heading line (context) and runs through the content line the
    scorer anchored on, so a heading-only match still yields the full
    paragraph/definition below it. Unit-header blocks (e.g. 'UNIT-4' followed
    by its topic list) do not terminate the backward walk.
    """
    start_idx = heading_idx
    while start_idx > 0 and (content_idx - start_idx < 2):
        prev = lines[start_idx - 1].line_text
        if prev.endswith(":"):
            break
        if _is_section_boundary(prev) and not _UNIT_HEADER_RX.match(prev.strip()):
            break
        start_idx -= 1
    end_idx = content_idx
    total_len = sum(len(lines[i].line_text) + 1 for i in range(start_idx, end_idx + 1))
    while end_idx + 1 < len(lines) and total_len < budget:
        nxt = lines[end_idx + 1].line_text
        if _is_section_boundary(nxt):
            break
        end_idx += 1
        total_len += len(nxt) + 1
    return (
        " ".join(l.line_text for l in lines[start_idx : end_idx + 1]),
        lines[start_idx].line_number,
        lines[end_idx].line_number,
    )


def _key_sentences(excerpt: str, phrases: list[str], terms: list[str], max_sentences: int = 2) -> tuple[str, ...]:
    sentences = [s for s in _split_sentences(excerpt) if len(s) > 25]
    if not sentences:
        return ()
    rx_phrases = [_phrase_rx(p) for p in phrases]
    scored: list[tuple[int, int, str]] = []
    for idx, s in enumerate(sentences):
        low = s.lower()
        phrase_hit = any(rx.search(low) for rx in rx_phrases)
        marker = _has_answer_marker(low)
        term_hits = sum(1 for t in terms if _term_rx(t).search(low))
        points = (4 if phrase_hit else 0) + (3 if marker else 0) + min(term_hits, 3)
        scored.append((idx, points, s))
    scored.sort(key=lambda item: (-item[1], item[0]))
    picked = [item for item in scored if item[1] > 0][:max_sentences]
    picked.sort(key=lambda item: item[0])
    return tuple(s for _, _, s in picked)


def _semantic_candidates(
    pages: list[PageRecord],
    question: str,
    query_type: str | None = None,
) -> tuple[
    list[tuple[PageRecord, float, int, int, str | None, int | None]],
    list[tuple[PageRecord, str]],
    list[str],
    list[str],
]:
    """TASK 7 (v8): shared semantic page-scoring core.

    Used by retrieve_document_evidence (accepted matches) and
    closest_match_evidence (sub-threshold candidates), so both rank pages
    identically. Returns (candidates, lowered_pages, phrases, terms).

    TASK 2 (v11): query_type in {'explanatory','structural'} — None means
    classify with is_explanatory_query(). On explanatory queries a SHORT
    heading-only match is penalized; on structural queries short exact
    heading matches keep their full score.
    """
    content_pages = [p for p in pages if p.page_kind == PageKind.CONTENT and p.text.strip()]
    lowered = [(p, re.sub(r"\s+", " ", p.text.lower())) for p in content_pages]
    q_class = classify_query(question)
    phrases = list(q_class.phrases)
    # TASK 7 (v8): the classifier drops bare numbers, so 'what is unit 4 about'
    # yields no 'unit 4' phrase and UNIT-4 headings can never match. Re-add a
    # structural-unit phrase from the raw question text.
    m_unit = re.search(r"\b(unit|chapter|part|section)\s*([0-9]+|[ivxlc]+)\b", question, re.I)
    if m_unit:
        unit_phrase = f"{m_unit.group(1).lower()} {m_unit.group(2).lower()}"
        if unit_phrase not in [p.lower() for p in phrases]:
            phrases.append(unit_phrase)
    terms = [t for t in q_class.terms if len(t) > 2]
    if not content_pages or (not phrases and not terms):
        return [], lowered, phrases, terms

    # TASK 2 (v11): query type decides heading-penalty branches. Default to
    # explanatory for ANY question containing explanation phrasing; pure
    # structural lookups ('slip 3', 'unit 4') keep the old behavior.
    if query_type is None:
        query_type = "explanatory" if is_explanatory_query(question) else "structural"

    # TASK 2 (v11): cross-line phrase hits, computed ONCE per page.
    cross_hits: dict[int, tuple[str, int, int]] = {}
    for pi, page in enumerate(content_pages):
        hit = _cross_line_phrase_hit(page.text, phrases)
        if hit:
            cross_hits[pi] = hit

    n_content = len(content_pages)
    ubiquity_limit = max(3, int(0.25 * n_content))
    freq: dict[str, int] = {
        t: sum(1 for _, low in lowered if _term_rx(t).search(low)) for t in terms
    }
    distinctive = [t for t in terms if freq[t] <= ubiquity_limit]

    phrase_res = [(p, _phrase_rx(p)) for p in phrases]
    # TASK 2 (v11): bridged variants tolerate one connector word between
    # content words ('discipline perception' -> matches 'discipline of
    # perception'). Page-level scans use these; line-level scans keep the
    # strict RX so single-line matching never loosens.
    phrase_res_bridged = [(p, _phrase_rx_bridged(p)) for p in phrases]
    candidates: list[tuple[PageRecord, float, int, int, str | None, int | None]] = []

    for pi, (page, low) in enumerate(lowered):
        lines = page.lines if page.lines else [
            LineRecord(i + 1, l.strip(), i + 1) for i, l in enumerate(page.text.splitlines()) if l.strip()
        ]
        if not lines:
            continue

        score = 0.0
        target_line_idx = -1
        matched_heading: str | None = None
        # TASK 7 (v8): a page listing MANY unit headers is a contents page,
        # not where the unit's content lives. Demote its unit-heading match
        # so the real unit-start page wins the tie.
        distinct_unit_headers = len({
            l.line_text.strip().upper() for l in lines if _UNIT_HEADER_RX.match(l.line_text.strip())
        })

        # 1. Heading match
        for idx, line in enumerate(lines):
            lt = line.line_text
            if not _is_probable_heading(lt):
                continue
            if any(_heading_matches(lt, p) for p in phrases):
                target_line_idx = idx
                matched_heading = lt
                score = 100.0
                break
            # TASK 7 (v8): structural-unit headings ('UNIT-4' vs 'unit 4')
            # must match across separator/number-format differences.
            if any(_unit_heading_matches(lt, p) for p in phrases):
                target_line_idx = idx
                matched_heading = lt
                score = 55.0 if distinct_unit_headers >= 3 else 100.0
                break

        # Distinctive term heading match if no phrase heading match
        if target_line_idx < 0 and distinctive:
            rarest = sorted(distinctive, key=lambda t: freq[t])
            query_terms = set(terms)
            for t in rarest:
                for idx, line in enumerate(lines):
                    lt = line.line_text
                    if not (_is_probable_heading(lt) and len(lt.split()) <= 4 and _heading_matches(lt, t)):
                        continue
                    heading_words = {_concept(w.lower()) for w in _TOKENS.findall(lt)}
                    if heading_words - _STOPWORDS - {_concept(t) for t in query_terms}:
                        continue
                    target_line_idx = idx
                    matched_heading = lt
                    score = 45.0 if phrases else 70.0
                    break
                if target_line_idx >= 0:
                    break

        # TASK 2 (v11): heading-only false-match trap. A SHORT generic heading
        # ('Discipline') matching only the query's loosest word must not beat a
        # page holding the FULL subject phrase. Reachability (Daily Stoic, real
        # numbers): p200 heading 'Discipline' (1 word) scores 45.0 -> 30.0 when
        # ANOTHER page holds the bridged subject phrase ('discipline of will'
        # on p.12); 30 < 36 drops p200 from accepted candidates entirely.
        # 'Slip 3 Question 1'-style queries classify as structural and never
        # enter this branch.
        page_level_full_phrase = any(rx.search(low) for _, rx in phrase_res_bridged)
        other_page_full_phrase = any(k != pi for k in cross_hits)
        if (
            matched_heading is not None
            and query_type == "explanatory"
            and len(matched_heading.split()) <= 2
            and not page_level_full_phrase
            and other_page_full_phrase
        ):
            score = min(score, 30.0)

        # 2. Exact multi-word phrase in lines
        if target_line_idx < 0:
            best_phrase_score = 0.0
            best_phrase_idx = -1
            query_numbers = set(re.findall(r"\b\d+\b", question))
            for phrase, rx in phrase_res:
                for idx, line in enumerate(lines):
                    lt = line.line_text
                    if rx.search(lt):
                        has_marker = _has_answer_marker(lt)
                        sc = 65.0 if has_marker else 50.0
                        win_text = " ".join(lines[k].line_text.lower() for k in range(max(0, idx - 1), min(len(lines), idx + 4)))
                        win_numbers = set(re.findall(r"\b\d+\b", win_text))
                        num_overlap = query_numbers & win_numbers
                        if num_overlap:
                            sc += 15.0 * len(num_overlap)
                        if distinctive:
                            win_distinctive = sum(1 for t in distinctive if _term_rx(t).search(win_text))
                            sc += 10.0 * win_distinctive
                        matched_query_phrases = sum(1 for _, prx in phrase_res if prx.search(win_text))
                        if matched_query_phrases > 1:
                            sc += 5.0 * (matched_query_phrases - 1)
                        win_terms = sum(1 for t in terms if _term_rx(t).search(win_text))
                        sc += 2.0 * win_terms
                        if sc > best_phrase_score:
                            best_phrase_score = sc
                            best_phrase_idx = idx
            if best_phrase_idx >= 0:
                target_line_idx = best_phrase_idx
                score = best_phrase_score

        # TASK 2 (v11): cross-line/intra-line bridged phrase. Query phrases are
        # stopword-stripped ('discipline will'), so the text's 'The Discipline
        # of Will' is invisible to the strict line-local scan above. The hit
        # anchors at the phrase's FIRST line so the window builder runs forward
        # through the definition; on explanatory queries this outranks every
        # heading-only match (95 vs 45/30).
        if target_line_idx < 0 and pi in cross_hits:
            phrase, start_idx, end_idx = cross_hits[pi]
            window_text = " ".join(
                lines[k].line_text for k in range(start_idx, min(end_idx + 2, len(lines)))
            )
            has_marker = _has_answer_marker(window_text)
            target_line_idx = start_idx
            score = 65.0 if has_marker else 95.0

        # TASK 2 (v11): page-level bridged-phrase fallback. A wrapped phrase
        # ('discipline of will' broken across lines) is invisible to the
        # line-local scan above. Anchoring at the FIRST matching line gives the
        # window builder room to run forward through the definition.
        if target_line_idx < 0:
            for phrase, rx in phrase_res_bridged:
                m = rx.search(low)
                if m:
                    target_line_idx = _page_line_index_of_match(lines, low, m.start())
                    has_marker = _has_answer_marker(m.group(0))
                    score = 60.0 if has_marker else 55.0
                    break

        # 3. Sentence-level definition / co-occurrence
        if target_line_idx < 0 and distinctive:
            for idx, line in enumerate(lines):
                lt = line.line_text
                llt = lt.lower()
                hits = [t for t in distinctive if _term_rx(t).search(llt)]
                if len(hits) >= 2 or (_has_answer_marker(llt) and len(hits) >= 1):
                    target_line_idx = idx
                    score = 38.0 + 8.0 * len(hits)
                    break

        if target_line_idx >= 0 and score >= 36.0:
            # TASK 7 (v8): a heading hit alone does not answer a definition
            # question. Anchor the evidence window at the first CONTENT line
            # AFTER the heading (skipping sibling unit-header/topic-list lines
            # like the 'UNIT-4' block on DS p.97); the heading itself is folded
            # back into the excerpt by _heading_window.
            if matched_heading is not None:
                anchor = target_line_idx + 1
                while anchor < len(lines) and (
                    _is_probable_heading(lines[anchor].line_text)
                    or len(lines[anchor].line_text) < 25
                ):
                    anchor += 1
                if anchor < len(lines):
                    target_line_idx = anchor
                heading_line_idx = next(
                    (i for i, l in enumerate(lines) if l.line_text.strip() == matched_heading.strip()),
                    None,
                )
                candidates.append((page, score, target_line_idx, len(lines), matched_heading, heading_line_idx))
            else:
                candidates.append((page, score, target_line_idx, len(lines), matched_heading, None))

    return candidates, lowered, phrases, terms


def closest_match_evidence(
    pages: list[PageRecord],
    question: str,
    doc_structures: dict[str, DocumentStructureIndex] | None = None,
    limit: int = 2,
) -> list[Evidence]:
    """TASK 7 (v8): nearest-available sections when nothing clears the bar.

    Runs the SAME semantic scorer as retrieve_document_evidence but keeps the
    strongest SUB-THRESHOLD candidates (score < 50), which the caller reports
    as 'Closest match — not confirmed'. When even the scorer finds nothing,
    falls back to the most term/phrase-overlapping content pages so the user
    still gets a pointer instead of a bare refusal.
    """
    if not pages:
        return []
    candidates, lowered, phrases, terms = _semantic_candidates(pages, question, query_type="explanatory")

    results: list[Evidence] = []
    if candidates:
        candidates.sort(key=lambda c: (-c[1], c[0].pdf_page_number))
        for page, sc, line_idx, _total, m_head, heading_idx in candidates:
            if sc >= 50.0 or len(results) >= limit:
                break
            lines = list(page.lines) if page.lines else [
                LineRecord(i + 1, l.strip(), i + 1) for i, l in enumerate(page.text.splitlines()) if l.strip()
            ]
            if m_head is not None and heading_idx is not None and heading_idx < line_idx:
                excerpt, start_l, end_l = _heading_window(lines, line_idx, heading_idx)
            else:
                excerpt, start_l, end_l = _excerpt_with_window(lines, line_idx)
            if not excerpt:
                continue
            s_line, e_line = sanitize_line_range(start_l, end_l, len(lines))
            results.append(Evidence(
                document_name=page.document_name,
                pdf_page_number=page.pdf_page_number,
                printed_page_label=page.printed_page_label,
                excerpt=excerpt,
                page_kind=page.page_kind,
                key_sentences=(),
                matched_heading=m_head,
                relevance_score=round(sc, 1),
                start_line=s_line,
                end_line=e_line,
                section_name=m_head,
            ))
        if results:
            return results

    # Page-level fallback: strongest term/phrase overlap on any content page.
    if not phrases and not terms:
        return []
    scored_pages: list[tuple[float, PageRecord]] = []
    for page, low in lowered:
        phrase_hits = sum(1 for p in phrases if _phrase_rx(p).search(low))
        term_hits = sum(1 for t in terms if _term_rx(t).search(low))
        overlap = phrase_hits * 2.0 + term_hits
        if overlap > 0:
            scored_pages.append((overlap, page))
    scored_pages.sort(key=lambda item: (-item[0], item[1].pdf_page_number))
    for overlap, page in scored_pages[:limit]:
        lines = list(page.lines) if page.lines else [
            LineRecord(i + 1, l.strip(), i + 1) for i, l in enumerate(page.text.splitlines()) if l.strip()
        ]
        if not lines:
            continue
        excerpt_text = " ".join(l.line_text for l in lines)[:600].strip()
        s_line, e_line = sanitize_line_range(lines[0].line_number, lines[-1].line_number, len(lines))
        results.append(Evidence(
            document_name=page.document_name,
            pdf_page_number=page.pdf_page_number,
            printed_page_label=page.printed_page_label,
            excerpt=excerpt_text,
            page_kind=page.page_kind,
            key_sentences=(),
            matched_heading=None,
            relevance_score=round(min(49.0, 10.0 + overlap * 5.0), 1),
            start_line=s_line,
            end_line=e_line,
            section_name=None,
        ))
    return results


def _stamp_document_id(evidence: list[Evidence], pages: list[PageRecord]) -> list[Evidence]:
    if not evidence or not pages:
        return evidence
    for ev in evidence:
        if ev.document_id is None:
            p_match = next((p for p in pages if p.pdf_page_number == ev.pdf_page_number and (not ev.document_name or p.document_name == ev.document_name)), None)
            target_id = p_match.document_id if p_match else pages[0].document_id
            if target_id:
                object.__setattr__(ev, "document_id", str(target_id))
    return evidence


def retrieve_document_evidence(
    pages: list[PageRecord],
    question: str,
    limit: int = 6,
    doc_structures: dict[str, DocumentStructureIndex] | None = None,
    context: dict[str, object] | None = None,
) -> list[Evidence]:
    """Retrieve verified evidence from selected documents using intent dispatch."""
    if not pages:
        return []

    doc_id = pages[0].document_id if pages and pages[0].document_id else None
    doc_name = pages[0].document_name
    doc_struct = None
    if doc_structures:
        if doc_id and doc_id in doc_structures:
            doc_struct = doc_structures[doc_id]
        else:
            doc_struct = doc_structures.get(doc_name)

    # Step 1: Classify Query Intent
    q_class = classify_query(question)

    # Step 2: Handle Direct Page Lookup ("page 201", "PDF page 201", "book page 833")
    if q_class.intent == QueryIntent.PAGE_LOOKUP and q_class.page_number is not None:
        evidence = execute_page_lookup(pages, q_class.page_number, q_class.page_type or "pdf")
        return _stamp_document_id(evidence, pages)

    # Step 3: Handle Structural Count Queries
    if q_class.intent == QueryIntent.COUNT and q_class.target_entity:
        evidence = execute_count_retrieval(pages, q_class.target_entity, doc_struct)
        if evidence:
            return _stamp_document_id(evidence, pages)

    # Step 4: Handle Structured List Queries
    if q_class.intent == QueryIntent.STRUCTURED_LIST and q_class.target_entity:
        evidence = execute_structured_list_retrieval(pages, q_class.target_entity, doc_struct)
        if evidence:
            return evidence

    # Step 5: Handle Nested Entity Extraction ("Question 3 of Slip 2", "Slip 2 Question 3")
    if (
        q_class.intent == QueryIntent.QUESTION_EXTRACTION
        and q_class.target_entity == "question"
        and q_class.parent_entity
        and q_class.parent_identifier
        and q_class.target_identifier
    ):
        evidence = execute_nested_entity_retrieval(
            pages,
            q_class.parent_entity,
            q_class.parent_identifier,
            q_class.target_entity or "question",
            q_class.target_identifier,
            doc_struct,
        )
        if evidence:
            return evidence

    # Step 6: Handle Contextual Question Extraction (e.g. follow-up "Question 3" or "q3" after "Slip 2")
    if q_class.intent == QueryIntent.QUESTION_EXTRACTION and q_class.target_identifier and not q_class.parent_identifier:
        parent_type = None
        parent_id = None
        if context:
            parent_type = context.get("last_parent_entity") or context.get("last_entity_type")
            parent_id = context.get("last_parent_identifier") or context.get("last_entity_number")
            if (not parent_type or parent_type == "question") and context.get("last_section_name"):
                m_sec_ent = re.search(r"([a-z]{2,15})\s*(?:no\.?|number|#)?[:.\-–—\s]*(\d+|[ivxlcdm]+|[a-z]\b)", str(context["last_section_name"]), re.I)
                if m_sec_ent:
                    parent_type = _concept(m_sec_ent.group(1)).lower()
                    parent_id = m_sec_ent.group(2).strip()
            if parent_type and parent_id and parent_type != "question":
                evidence = execute_nested_entity_retrieval(
                    pages,
                    str(parent_type),
                    str(parent_id),
                    q_class.target_entity or "question",
                    q_class.target_identifier,
                    doc_struct,
                )
                if evidence:
                    return evidence

        # Fallback: search for this question across sections in doc_struct only if no parent context
        if doc_struct and not (context and (context.get("last_parent_entity") or context.get("last_section_name"))):
            for s in doc_struct.sections:
                for sub in s.sub_entities:
                    sub_num = (sub.entity_number or "").strip().lstrip("0") or "0"
                    target_num = q_class.target_identifier.strip().lstrip("0") or "0"
                    if sub_num.lower() == target_num.lower():
                        evidence = execute_nested_entity_retrieval(
                            pages,
                            s.entity_type,
                            s.entity_number or "1",
                            "question",
                            q_class.target_identifier,
                            doc_struct,
                        )
                        if evidence:
                            return evidence

    # Step 6.5: Handle SHOW_ALL_ITEMS_IN_PARENT / Parent Question Extraction ("Slip 1 questions", "give all questions from slip 1", "show slip 1", "list slip 1 questions")
    is_parent_req = (
        q_class.intent == QueryIntent.SHOW_ALL_ITEMS_IN_PARENT
        or (
            q_class.intent == QueryIntent.QUESTION_EXTRACTION
            and (q_class.parent_entity in ("slip", "unit", "chapter", "section") or q_class.target_entity in ("slip", "unit", "chapter", "section"))
            and not (q_class.parent_identifier and q_class.target_identifier and q_class.target_entity == "question")
        )
    )
    if is_parent_req and (q_class.parent_identifier or q_class.target_identifier):
        p_ent = q_class.parent_entity or q_class.target_entity or "slip"
        p_id = q_class.parent_identifier or q_class.target_identifier or "1"
        evidence = execute_show_all_items_in_parent(
            pages,
            p_ent,
            p_id,
            doc_struct,
        )
        if evidence:
            return evidence

    # Step 6.8: Handle Document Summary ("summarize this PDF", "overview of this document")
    if q_class.intent == QueryIntent.SUMMARY:
        evidence = execute_document_summary_retrieval(pages, doc_struct)
        if evidence:
            return evidence

    # Step 7: Handle Section / Question Extraction ("Give all questions from Slip 1", "Show Section 2", "slip 3")
    if q_class.intent in (QueryIntent.SECTION_EXTRACTION, QueryIntent.QUESTION_EXTRACTION, QueryIntent.STRUCTURAL_NAVIGATION) and q_class.target_entity and q_class.target_identifier:
        evidence = execute_section_extraction(
            pages,
            q_class.target_entity,
            q_class.target_identifier,
            doc_struct,
            extract_questions_only=(q_class.intent == QueryIntent.QUESTION_EXTRACTION),
        )
        if evidence:
            return evidence

    # Step 8: Handle Metadata Queries
    if q_class.intent == QueryIntent.METADATA and q_class.target_entity:
        evidence = execute_metadata_retrieval(pages, q_class.target_entity, doc_struct)
        if evidence:
            return evidence

    # Step 6(9): Semantic / Concept Retrieval — TASK 7 (v8): the scoring core
    # lives in _semantic_candidates, shared with closest_match_evidence.
    # TASK 2 (v11): classify the query ONCE (structural vs explanatory) and
    # pass the type down so both accepted-evidence and closest-match scoring
    # apply identical rules.
    query_type = "explanatory" if is_explanatory_query(question) else "structural"
    candidates, lowered, phrases, terms = _semantic_candidates(
        pages, question, query_type=query_type
    )
    if not candidates:
        return []

    # TASK 1 (v11): explanatory queries read whole paragraphs; structural
    # lookups ('Slip 3 Question 1', 'Unit 4') only need the exact block, so
    # their window budget is unchanged.
    window_budget = 2200 if query_type == "explanatory" else 900

    # Sort by score descending then page number
    candidates.sort(key=lambda c: (-c[1], c[0].pdf_page_number))
    top_score = candidates[0][1]

    # Preference: when 1 strong source answers the query (heading match or high score),
    # return only that 1 strong source to prevent neighboring page noise!
    if top_score >= 80.0:
        selected = [candidates[0]]
    elif top_score >= 50.0:
        floor = 50.0
        selected = [c for c in candidates if c[1] >= floor][: min(limit, 2)]
    else:
        floor = 36.0
        selected = [c for c in candidates if c[1] >= floor][: min(limit, 2)]

    results: list[Evidence] = []
    for page, sc, line_idx, total_l, m_head, heading_idx in selected:
        lines = list(page.lines) if page.lines else [
            LineRecord(i + 1, l.strip(), i + 1) for i, l in enumerate(page.text.splitlines()) if l.strip()
        ]
        # TASK 7 (v8): when the window was re-anchored below a matched heading,
        # extend the window start to include the heading line itself so the
        # excerpt reads '<heading> <full definition>' instead of losing context.
        if m_head is not None and heading_idx is not None and heading_idx < line_idx:
            excerpt, start_l, end_l = _heading_window(lines, line_idx, heading_idx, budget=window_budget)
        else:
            excerpt, start_l, end_l = _excerpt_with_window(lines, line_idx, budget=window_budget)
        if not excerpt:
            continue
        page_line_count = len(lines)
        s_line, e_line = sanitize_line_range(start_l, end_l, page_line_count)
        results.append(Evidence(
            document_name=page.document_name,
            pdf_page_number=page.pdf_page_number,
            printed_page_label=page.printed_page_label,
            excerpt=excerpt,
            page_kind=page.page_kind,
            key_sentences=_key_sentences(excerpt, phrases, terms),
            matched_heading=m_head,
            relevance_score=round(sc, 1),
            start_line=s_line,
            end_line=e_line,
            section_name=m_head,
        ))

    return results


def _answer_from_evidence(evidence: list[Evidence]) -> tuple[str | None, tuple[str | None, str | None, int] | None]:
    if not evidence:
        return None, None
    best = evidence[0]
    parts: list[str] = []
    if best.matched_heading:
        parts.append(best.matched_heading)
    if best.key_sentences:
        answer = ". ".join(best.key_sentences)
    else:
        answer = best.excerpt.strip()
    if answer and not answer.endswith("."):
        answer += "."
    label = best.printed_page_label
    doc_name = best.document_name
    pdf = best.pdf_page_number
    return answer, (doc_name, label, pdf)


def _source_count_label(document_evidence: list, web_evidence: list) -> str:
    dc = len(document_evidence)
    wc = len(web_evidence)
    if dc == 0 and wc == 0:
        return "No verified evidence"
    if dc == 1 and wc == 0:
        return "1 relevant document source"
    if dc > 1 and wc == 0:
        return f"{dc} relevant document sources"
    if dc == 0 and wc == 1:
        return "1 web source consulted"
    if dc == 0 and wc > 1:
        return f"{wc} web sources consulted"
    if dc == 1 and wc == 1:
        return "1 relevant document source and 1 web source consulted"
    return f"{dc} relevant document sources and {wc} web sources consulted"


def _format_evidence_for_frontend(evidence: list[Evidence]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in evidence:
        s_line, e_line = sanitize_line_range(item.start_line, item.end_line)
        result.append({
            "title": item.document_name or f"Page {item.pdf_page_number}",
            "document_name": item.document_name,
            "page_number": item.pdf_page_number,
            "pdf_page_number": item.pdf_page_number,
            "printed_page_label": item.printed_page_label,
            "excerpt": item.excerpt,
            "source_type": "document",
            "key_sentences": list(item.key_sentences),
            "matched_heading": item.matched_heading,
            "relevance_score": item.relevance_score,
            "start_line": s_line,
            "end_line": e_line,
            "section_name": item.section_name,
            "entity_type": item.entity_type,
            "entity_count": item.entity_count,
        })
    return result

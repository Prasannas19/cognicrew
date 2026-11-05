from __future__ import annotations

import re
from enum import Enum
from typing import Any


class UserRequestIntent(str, Enum):
    DIRECT_QA = "DIRECT_QA"
    DETAILED_EXPLANATION = "DETAILED_EXPLANATION"
    RESEARCH_REPORT = "RESEARCH_REPORT"
    # Backward compatibility alias
    STRUCTURED_RESEARCH = "RESEARCH_REPORT"
    CODE_GENERATION = "CODE_GENERATION"
    MATH_REASONING = "MATH_REASONING"
    CREATIVE_WRITING = "CREATIVE_WRITING"
    DOCUMENT_EXTRACTION = "DOCUMENT_EXTRACTION"
    DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
    DOCUMENT_PROFILE = "DOCUMENT_PROFILE"
    DOCUMENT_FOLLOWUP = "DOCUMENT_FOLLOWUP"
    STUDY_QUIZ = "STUDY_QUIZ"
    STUDY_FLASHCARDS = "STUDY_FLASHCARDS"
    STUDY_GUIDE = "STUDY_GUIDE"
    STUDY_MIND_MAP = "STUDY_MIND_MAP"
    STUDY_CHEAT_SHEET = "STUDY_CHEAT_SHEET"


class ResponseDepth(str, Enum):
    CONCISE = "CONCISE"
    DETAILED = "DETAILED"
    WORD_COUNT = "WORD_COUNT"
    RESEARCH_REPORT = "RESEARCH_REPORT"


# ---------------------------------------------------------------------------
# Deterministic pattern matching rules (rule-based, fast, zero LLM overhead)
# ---------------------------------------------------------------------------

_CREATIVE_RX = re.compile(
    r"\b((?:write|compose|tell|create|describe)\s+(?:me\s+)?(?:a\s+)?(?:[a-z0-9_-]+\s+)*(?:story|poem|dialogue|conversation|song|haiku|script|tale|mythology|myth|fiction|fable|fantasy)|"
    r"dialogue\s+between|conversation\s+between|"
    r"create\s+(?:a\s+)?(?:new\s+)?(?:fictional\s+)?mythology)\b",
    re.I,
)

_CODE_RX = re.compile(
    r"\b(write\s+(?:a\s+)?(?:python|java|c\+\+|javascript|typescript|c#|go|rust|sql)?\s*(?:code|program|script|function|class)|"
    r"implement\s+(?:binary\s+search|bubble\s+sort|quick\s*sort|merge\s*sort|two\s*sum|algorithm|an?\s+algorithm|circle)|"
    r"two\s*sum\b|"
    r"provide\s+python\s+code|"
    r"plot\s+a\s+circle|"
    r"pie\s+chart|"
    r"explode\s+function|"
    r"write\s+(?:a\s+)?(?:program|script)\s+to|"
    r"solve\s+(?:this\s+)?(?:programming|coding|leetcode)\s+problem|"
    r"in\s+(?:python|java|c\+\+|javascript)\b|"
    r"leetcode|hackerrank|codeforces)\b",
    re.I,
)

_MATH_RX = re.compile(
    r"\b(find\s+the\s+(?:distance|slope|equation\s+of|equation|centroid|area|perimeter|intersection|angle|eigenvalues?|determinant)|"
    r"calculate\s+the\s+(?:distance|slope|area|perimeter|angle|derivative|integral|eigenvalues?|determinant|probability)|"
    r"solve\s+(?:this\s+)?(?:equation|math|formula|expression|system\s+of\s+equations|calculus\s+problem|math\s+problem)|"
    r"eigenvalues?|eigenvectors?|determinant\s+of|"
    r"distance\s+between\s+(?:p|points?|\()|"
    r"p\s*\(\s*-?\d+.*q\s*\(\s*-?\d+|"
    r"evaluate\s+(?:the\s+)?(?:integral|derivative|expression))\b",
    re.I,
)

_DOC_SUMMARY_RX = re.compile(
    r"\b(profile\s+(?:this|the)\s+(?:entire\s+)?(?:file|document|pdf|book|text)|"
    r"summarize\s+(?:this|the)\s+(?:entire\s+)?(?:pdf|file|document|book|uploaded\s+document)|"
    r"what\s+is\s+this\s+(?:entire\s+)?(?:pdf|file|document|book)\s+about|"
    r"give\s+(?:me\s+)?(?:an?\s+)?(?:brief\s+|complete\s+)?(?:overview|summary)\s+of\s+(?:this|the)\s+(?:file|document|pdf|book|paper)|"
    r"explain\s+what\s+this\s+(?:entire\s+)?(?:document|pdf|file|book)\s+(?:contains|is\s+about)|"
    r"what\s+(?:skills|topics)\s+does\s+this\s+(?:document|book|pdf|text)\s+teach|"
    r"key\s+takeaways\s+from\s+(?:this|the)\s+(?:document|book|pdf|paper)|"
    r"executive\s+summary\s+of\s+(?:this|the)\s+(?:uploaded\s+)?(?:document|paper|pdf|book)|"
    r"summary\s+of\s+(?:this|the)\s+(?:entire\s+)?(?:pdf|document|file|book))\b",
    re.I,
)

_DOC_PROFILE_RX = re.compile(
    r"\b(profile\s+(?:this\s+)?(?:entire\s+)?(?:file|document|pdf|book|text)|"
    r"document\s+profile|generate\s+(?:a\s+)?document\s+profile)\b",
    re.I,
)

_DOC_EXTRACTION_RX = re.compile(
    r"\b(give\s+(?:all\s+)?questions?\s+(?:from|in|of)\s+slip|"
    r"slip\s*(?:no\.?|number|#)?\s*[:\.\-–—\s]*\d+\s*(?:all\s+)?(?:questions?|problems?|content)?|"
    r"show\s+slip\s*(?:no\.?|number|#)?\s*[:\.\-–—\s]*\d+|"
    r"list\s+(?:the\s+)?questions?\s+(?:on|from)\s+page\s*\d+|"
    r"what\s+is\s+on\s+page\s*\d+|"
    r"what\s+does\s+page\s*\d+\s+say|"
    r"show\s+page\s*\d+|"
    r"content\s+of\s+page\s*\d+|"
    r"page\s*\d+\s+content|"
    r"provide\s+(?:me\s+)?(?:information|details|content)\s+(?:from|on|of)\s+page\s*\d+|"
    r"what\s+are\s+the\s+headings\s+in\s+this\s+section|"
    r"questions?\s+on\s+page\s*\d+)\b",
    re.I,
)

_STRUCTURED_RESEARCH_RX = re.compile(
    r"\b(research\s+(?:the\s+)?(?:current\s+state|state|history|development|evolution|impact|future|advancements?|status)\s+of|"
    r"compare\s+.+\s+using\s+(?:current\s+)?sources|"
    r"detailed\s+research\s+report|"
    r"give\s+me\s+a\s+(?:\d+[- ]word\s+)?research\s+report|"
    r"write\s+(?:me\s+)?(?:a\s+)?(?:\d+[- ]word\s+)?research\s+report|"
    r"explain\s+(?:the\s+)?topic\s+with\s+sources|"
    r"do\s+detailed\s+research\s+on|"
    r"comprehensive\s+(?:study|analysis|report)\s+on|"
    r"literature\s+review\s+on)\b",
    re.I,
)

_DETAILED_EXPLANATION_RX = re.compile(
    r"\b(explain\s+.+\s+in\s+detail|in\s+detail|detailed\s+explanation|elaborate\s+on|comprehensive\s+explanation|"
    r"in-depth\s+explanation|describe\s+thoroughly|explain\s+thoroughly|deep\s+dive\s+into)\b",
    re.I,
)

_STUDY_QUIZ_RX = re.compile(
    r"\b(generate\s+(?:a\s+)?quiz|quiz\s+me\s+on|create\s+(?:a\s+)?quiz|make\s+(?:a\s+)?quiz|practice\s+quiz|"
    r"(?:make|create|generate)\s+\d+\s+questions?\b|"
    r"multiple\s+choice\s+questions|mcqs?\s+(?:on|for|from))\b",
    re.I,
)

_STUDY_FLASHCARDS_RX = re.compile(
    r"\b(generate\s+(?:a\s+)?flashcards?|create\s+(?:a\s+)?flashcards?|make\s+flashcards?|"
    r"flash\s*cards?\s+(?:for|from|on))\b",
    re.I,
)

_STUDY_GUIDE_RX = re.compile(
    r"\b(generate\s+(?:a\s+)?study\s+guide|create\s+(?:a\s+)?study\s+guide|make\s+(?:a\s+)?study\s+guide|"
    r"create\s+(?:a\s+)?revision\s+guide|revision\s+guide|study\s+guide\b)\b",
    re.I,
)

_STUDY_MIND_MAP_RX = re.compile(
    r"\b(generate\s+(?:a\s+)?mind\s*map|create\s+(?:a\s+)?mind\s*map|make\s+(?:a\s+)?mind\s*map|"
    r"mind\s*map|concept\s+map|concept\s+tree)\b",
    re.I,
)

_STUDY_CHEAT_SHEET_RX = re.compile(
    r"\b(generate\s+(?:a\s+)?cheat\s*sheet|create\s+(?:a\s+)?cheat\s*sheet|make\s+(?:a\s+)?cheat\s*sheet|"
    r"cheat\s*sheet|revision\s+sheet|quick\s+reference\s+sheet)\b",
    re.I,
)

_WORD_COUNT_PATTERNS = [
    re.compile(r"(?:at\s+least|approximately|around|about|min(?:imum)?|max(?:imum)?)\s*(\d{2,5})\s*words?", re.I),
    re.compile(r"(\d{2,5})\s*[- ]\s*word\s*(?:research\s+)?(?:report|essay|summary|paper|article|answer|explanation)", re.I),
    re.compile(r"\b(?:in|of)\s*(\d{2,5})\s*words?\b", re.I),
    re.compile(r"(\d{2,5})\s*words?\b", re.I),
]

_DIRECT_QA_RX = re.compile(
    r"^(?:what\s+is|what\s+are|explain|describe|what\s+does\s+.+\s+mean|what\s+do\s+you\s+mean\s+by|what\s+do\s+we\s+mean\s+by|tell\s+me\s+about|define|who\s+is|why\s+is|why\s+does)\b",
    re.I,
)


def extract_requested_word_count(query: str) -> int | None:
    """Extract requested word count from query if specified (e.g. 500 words, 1200 words)."""
    q = query.strip()
    for pattern in _WORD_COUNT_PATTERNS:
        match = pattern.search(q)
        if match:
            try:
                count = int(match.group(1))
                if 20 <= count <= 10000:
                    return count
            except (ValueError, IndexError):
                pass
    return None


def classify_user_intent(
    query: str,
    mode: str | None = None,
    is_followup: bool = False,
    doc_context: dict[str, Any] | None = None,
) -> UserRequestIntent:
    """Classify user request intent deterministically without any LLM calls."""
    q = query.strip()
    if not q:
        return UserRequestIntent.DIRECT_QA

    # Specific format/task classifiers first (apply to both initial & follow-up)
    # 1. Creative writing
    if _CREATIVE_RX.search(q):
        return UserRequestIntent.CREATIVE_WRITING

    # 2. Code generation
    if _CODE_RX.search(q):
        return UserRequestIntent.CODE_GENERATION

    # 3. Math reasoning
    if _MATH_RX.search(q):
        return UserRequestIntent.MATH_REASONING

    # 4. Study Tools (Quiz, Flashcards, Guide, Mind Map, Cheat Sheet)
    if _STUDY_QUIZ_RX.search(q):
        return UserRequestIntent.STUDY_QUIZ
    if _STUDY_FLASHCARDS_RX.search(q):
        return UserRequestIntent.STUDY_FLASHCARDS
    if _STUDY_GUIDE_RX.search(q):
        return UserRequestIntent.STUDY_GUIDE
    if _STUDY_MIND_MAP_RX.search(q):
        return UserRequestIntent.STUDY_MIND_MAP
    if _STUDY_CHEAT_SHEET_RX.search(q):
        return UserRequestIntent.STUDY_CHEAT_SHEET

    # 5. Document structural extraction / page lookups
    if _DOC_EXTRACTION_RX.search(q):
        return UserRequestIntent.DOCUMENT_EXTRACTION

    # 6. Document profiling / summary
    if _DOC_PROFILE_RX.search(q):
        return UserRequestIntent.DOCUMENT_PROFILE
    if _DOC_SUMMARY_RX.search(q):
        return UserRequestIntent.DOCUMENT_SUMMARY

    # 7. Explicit research report
    q_lower = q.lower()
    if _STRUCTURED_RESEARCH_RX.search(q) or ("research report" in q_lower):
        return UserRequestIntent.RESEARCH_REPORT

    # 8. Detailed explanation
    if _DETAILED_EXPLANATION_RX.search(q):
        return UserRequestIntent.DETAILED_EXPLANATION

    # 9. Explicit word count handling
    word_count_req = extract_requested_word_count(q)
    if word_count_req is not None:
        if word_count_req >= 700:
            return UserRequestIntent.RESEARCH_REPORT
        return UserRequestIntent.DETAILED_EXPLANATION

    # 10. Direct QA checks
    if _DIRECT_QA_RX.search(q):
        return UserRequestIntent.DIRECT_QA

    if is_followup:
        return UserRequestIntent.DOCUMENT_FOLLOWUP

    return UserRequestIntent.DIRECT_QA


def determine_response_depth(
    query: str,
    intent: UserRequestIntent,
    depth_mode: str | None = None,
) -> tuple[ResponseDepth, int | None, int]:
    """Determine response depth, extracted word count, and recommended generation max_tokens budget.

    Returns:
        (depth_enum, requested_word_count_or_None, max_tokens)
    """
    req_words = extract_requested_word_count(query)

    if req_words is not None:
        # Word count was explicitly asked: allocate proportional budget (~1.5 tokens per word + 200 buffer)
        tokens = min(4096, max(250, int(req_words * 1.5) + 200))
        return ResponseDepth.WORD_COUNT, req_words, tokens

    dm = (depth_mode or "").strip().lower()

    if dm == "detailed":
        # Explicit user selection of Detailed mode overrides default concise DIRECT_QA
        if intent == UserRequestIntent.RESEARCH_REPORT:
            return ResponseDepth.RESEARCH_REPORT, None, 2800
        return ResponseDepth.DETAILED, None, 2200

    if dm == "comprehensive":
        return ResponseDepth.DETAILED, None, 3200

    if dm == "concise":
        return ResponseDepth.CONCISE, None, 250

    if intent == UserRequestIntent.DIRECT_QA:
        # Concise explanation: 2-5 sentences, ~50-120 words -> ~250 tokens
        return ResponseDepth.CONCISE, None, 250

    if intent == UserRequestIntent.DETAILED_EXPLANATION:
        # Substantially more detailed: multi-paragraph explanation, ~350-600 words -> ~1400 tokens
        return ResponseDepth.DETAILED, None, 1400

    if intent == UserRequestIntent.RESEARCH_REPORT:
        # Substantive report: ~1800-2400 tokens
        return ResponseDepth.RESEARCH_REPORT, None, 2200

    if intent in (UserRequestIntent.CODE_GENERATION, UserRequestIntent.MATH_REASONING):
        return ResponseDepth.DETAILED, None, 1500

    if intent == UserRequestIntent.CREATIVE_WRITING:
        return ResponseDepth.DETAILED, None, 1400

    if intent in (
        UserRequestIntent.DOCUMENT_EXTRACTION,
        UserRequestIntent.DOCUMENT_SUMMARY,
        UserRequestIntent.DOCUMENT_PROFILE,
        UserRequestIntent.STUDY_GUIDE,
        UserRequestIntent.STUDY_CHEAT_SHEET,
    ):
        return ResponseDepth.DETAILED, None, 1800

    # Default
    return ResponseDepth.CONCISE, None, 600

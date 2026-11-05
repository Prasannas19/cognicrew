from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

logger = logging.getLogger("cognicrew.web_search")

_JUNK_LINE_RX = re.compile(
    r"(?:skip\s+to\s+(?:main\s+)?content|"
    r"feedback\s*\|\s*profile|"
    r"subscribe\s+to\s+(?:our\s+)?newsletter|"
    r"sign\s+up\s+for\s+(?:our\s+)?newsletter|"
    r"enter\s+(?:your\s+)?email|"
    r"share\s+on\s+(?:twitter|linkedin|facebook)|"
    r"copyright\s*©?|\ball\s+rights\s+reserved\b|"
    r"privacy\s+policy|"
    r"cookie\s+(?:policy|settings|notice|banner)|"
    r"accept\s+(?:all\s+)?cookies|"
    r"manage\s+cookies|"
    r"terms\s+of\s+(?:service|use)|"
    r"(?:united\s+states|united\s+kingdom|deutschland|france|australia|canada|japan|china|germany)\s*\(|"
    r"select\s+(?:your\s+)?(?:country|region|language)|"
    r"schema\.org|json-ld|"
    r"\bog:[a-z_]+\b|\btwitter:[a-z_]+\b)",
    re.I,
)


def clean_web_text(text: str) -> str:
    """Clean scraped web text by removing page chrome, scripts, styles, JSON-LD,
    OpenGraph metadata, navigation, newsletter forms, country selectors, repeated lines,
    and normalizing whitespace.
    """
    if not text:
        return ""

    # Strip script and style blocks
    cleaned = re.sub(r"(?is)<script\b.*?</script>", " ", text)
    cleaned = re.sub(r"(?is)<style\b.*?</style>", " ", cleaned)
    # Strip JSON-LD / schema blocks
    cleaned = re.sub(r'(?is)\{[^{}]*"(?:@context|@type)"[^{}]*\}', " ", cleaned)
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    cleaned_lines: list[str] = []
    seen_lines: set[str] = set()

    for line in lines:
        if _JUNK_LINE_RX.search(line):
            continue
        if re.match(r"^(?:home\s*[>/|]\s*)+", line, re.I):
            continue
        if line.count("|") >= 3 and len(line) < 120:
            continue
        if re.match(r'^(?:og:|twitter:|schema:|[\{\}\[\]]|"@context"|"@type")', line, re.I):
            continue

        norm = re.sub(r"\W+", " ", line.lower()).strip()
        if norm in seen_lines and len(norm) > 15:
            continue
        seen_lines.add(norm)
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    return result.strip()


def normalize_url(raw_url: str) -> str:
    """Normalize URL by stripping tracking parameters and trailing slashes."""
    if not raw_url:
        return ""
    try:
        parsed = urlparse(raw_url)
        drop_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "fbclid"}
        q = [(k, v) for k, v in parse_qsl(parsed.query) if k.lower() not in drop_params]
        clean_query = urlencode(q)
        path = parsed.path.rstrip("/")
        return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", clean_query, ""))
    except Exception:
        return raw_url.strip()


def deduplicate_web_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate web sources by normalized URL and content similarity."""
    seen_urls: set[str] = set()
    unique_sources: list[dict[str, Any]] = []

    for s in sources:
        url = str(s.get("url") or "")
        norm_url = normalize_url(url)
        if norm_url and norm_url in seen_urls:
            continue
        if norm_url:
            seen_urls.add(norm_url)

        excerpt = str(s.get("excerpt") or "")
        fp = re.sub(r"\W+", "", excerpt[:200].lower())
        if fp and any(fp == re.sub(r"\W+", "", str(u.get("excerpt") or "")[:200].lower()) for u in unique_sources):
            continue

        unique_sources.append(s)

    return unique_sources


def rank_and_filter_sources(
    sources: list[dict[str, Any]],
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rank sources by relevance and filter out low-quality/empty sources."""
    query_tokens = set(re.findall(r"\w{3,}", query.lower()))
    scored: list[tuple[float, dict[str, Any]]] = []

    for s in sources:
        excerpt = str(s.get("excerpt") or "").strip()
        title = str(s.get("title") or "").strip()
        if len(excerpt) < 50:
            continue

        base_score = float(s.get("relevance_score") or 0.5)
        text_low = (title + " " + excerpt).lower()
        match_count = sum(1 for token in query_tokens if token in text_low)
        score = base_score + (match_count * 0.1)

        url = str(s.get("url") or "").lower()
        if any(dom in url for dom in ("github.com", "huggingface.co", "arxiv.org", "ai.meta.com", "mistral.ai", "deepseek.com", "wikipedia.org")):
            score += 0.2

        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:limit]]


def search_exa(query: str, api_key: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Query Exa Neural Search API for verified academic and web sources."""
    if not api_key:
        return []

    endpoint = "https://api.exa.ai/search"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "query": query,
        "num_results": max(num_results, 5),
        "use_autoprompt": False,
        "contents": {
            "text": {"max_characters": 1200}
        }
    }

    try:
        with httpx.Client(timeout=35.0) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                formatted: list[dict[str, Any]] = []
                for item in results:
                    cleaned_excerpt = clean_web_text(str(item.get("text") or ""))
                    if cleaned_excerpt:
                        formatted.append({
                            "title": item.get("title") or "Web Reference",
                            "url": item.get("url") or "",
                            "excerpt": cleaned_excerpt,
                            "source_type": "web",
                            "relevance_score": item.get("score"),
                        })
                deduped = deduplicate_web_sources(formatted)
                ranked = rank_and_filter_sources(deduped, query, limit=num_results)
                return ranked
            else:
                logger.warning("Exa API returned HTTP %d: %s", resp.status_code, resp.text[:200])
    except httpx.RequestError as exc:
        logger.warning("Exa search connection failed: %s", exc)
    return []

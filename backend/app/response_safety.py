"""TASKS 4+5 (v11): centralized response safety validation.

Two validators, applied to EVERY AI-generated answer in one shared
post-processing step (answer_engine._validate_generated_response), so no
generation path can skip them:

TASK 4 — Math safety (SymPy backend):
    Arithmetic equalities that appear in an answer (e.g. "45+35+85+50+60 = 275"
    or "total = 275") are recomputed with SymPy. A WRONG claim is annotated
    inline with the corrected value (never silently rewritten) and recorded in
    the returned issues list. Claims SymPy cannot parse are left untouched —
    the validator only overrides what it can PROVE wrong.

TASK 5 — Generated-code AST validation:
    Every fenced ```python block is parsed with ast.parse (syntax gate) and
    scanned for dangerous constructs (os/subprocess/socket imports, eval/exec/
    compile/__import__/open/input calls, dunder attribute escapes). Blocks that
    fail validation are replaced with a visible placeholder — the code is NEVER
    executed server-side and eval()/exec() are never used anywhere.

Both functions are pure (text in -> text + issues out) and safe on any input.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("cognicrew.response_safety")

try:  # SymPy is required for the math backend; degrade honestly without it.
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )

    _SYMPOK = True
    _SYMPTY_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
except ImportError:  # pragma: no cover - sympy is in requirements
    _SYMPOK = False


# --------------------------------------------------------------------------
# TASK 5: generated-code AST validation
# --------------------------------------------------------------------------

_DANGEROUS_MODULES = {
    "os", "subprocess", "socket", "shutil", "ctypes", "sys", "pathlib",
    "importlib", "builtins", "pickle", "shelve", "marshal", "signal",
    "multiprocessing", "threading", "requests", "urllib", "http",
}

_DANGEROUS_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
    "globals", "locals", "vars", "delattr", "setattr", "getattr",
}

_CODE_FENCE_RX = re.compile(r"```([a-zA-Z0-9_+\-]*)[ \t]*\n(.*?)```", re.S)


@dataclass
class SafetyIssue:
    kind: str          # "code_syntax" | "code_unsafe" | "math_claim"
    detail: str
    location: str = ""


@dataclass
class SafetyReport:
    text: str                       # (possibly annotated) answer text
    issues: list[SafetyIssue] = field(default_factory=list)
    code_blocks_total: int = 0
    code_blocks_rejected: int = 0
    code_blocks_regenerated: int = 0
    math_claims_checked: int = 0
    math_claims_wrong: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues


def _is_unsafe_node(tree: ast.AST) -> str | None:
    """Return a reason string if the AST contains a disallowed construct."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DANGEROUS_MODULES:
                    return f"imports disallowed module '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _DANGEROUS_MODULES:
                return f"imports from disallowed module '{node.module}'"
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _DANGEROUS_CALLS:
                return f"calls disallowed function '{fn.id}()'"
            if isinstance(fn, ast.Attribute) and fn.attr in {"system", "popen", "run", "call", "check_output"}:
                base = fn.value
                if isinstance(base, ast.Name) and base.id in _DANGEROUS_MODULES:
                    return f"calls '{base.id}.{fn.attr}()'"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"accesses dunder attribute '{node.attr}'"
    return None


def _validate_code_blocks(text: str, report: SafetyReport, regenerate=None) -> str:
    """TASK 5: AST-validate every fenced code block.

    Syntax-invalid python blocks get ONE regeneration attempt via the
    `regenerate` callback (the callback re-enters THIS validation, so a bad
    regeneration is still rejected). Unsafe-construct blocks are NEVER
    regenerated — they are always replaced, because asking a model to repair
    dangerous code invites evasion.
    """

    def _check(m: re.Match[str]) -> str:
        lang, body = m.group(1).lower(), m.group(2)
        report.code_blocks_total += 1
        if lang not in {"", "python", "py", "python3"}:
            return m.group(0)  # non-python fences pass through untouched

        reason: str | None = None
        tree: ast.AST | None = None
        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            reason = f"syntax error: {exc.msg} (line {exc.lineno})"

        if reason is None and tree is not None:
            reason = _is_unsafe_node(tree)

        if reason and regenerate is not None and reason.startswith("syntax"):
            fixed = regenerate(body)
            if fixed:
                fixed_reason: str | None
                try:
                    fixed_tree = ast.parse(fixed)
                except SyntaxError as exc:
                    fixed_reason = f"syntax error: {exc.msg} (line {exc.lineno})"
                else:
                    fixed_reason = _is_unsafe_node(fixed_tree)
                if fixed_reason is None:
                    report.code_blocks_regenerated += 1
                    logger.info("CODE SAFETY: syntax-invalid block regenerated once and re-validated OK")
                    return f"```python\n{fixed}\n```"
                logger.info("CODE SAFETY: regenerated block still invalid (%s) — replacing", fixed_reason)

        if reason:
            report.code_blocks_rejected += 1
            report.issues.append(SafetyIssue(kind="code_syntax" if reason.startswith("syntax") else "code_unsafe",
                                             detail=reason, location="fenced code block"))
            return (
                "```text\n"
                f"[Removed by CogniCrew code safety validation: {reason}. "
                "Generated code is validated with ast, never executed.]\n"
                "```"
            )
        return m.group(0)

    return _CODE_FENCE_RX.sub(_check, text)


# --------------------------------------------------------------------------
# TASK 4: math-claim verification (SymPy)
# --------------------------------------------------------------------------

# "45+35+85+50+60 = 275", "total = 275", "2**10 = 1024", "(3*7) + 2 is 23"
_ARITH_CLAIM_RX = re.compile(
    r"(?P<lhs>(?:\d+(?:\.\d+)?(?:\s*[-+*/%^]\s*\(?\s*\d+(?:\.\d+)?\s*\)?)+))"
    r"\s*(?:=|is|equals|→|->)\s*(?P<rhs>-?\s*\d+(?:\.\d+)?)",
    re.I,
)
# plain-name totals: "total = 275" / "sum is 275" (lhs must contain a letter)
_NAMED_TOTAL_RX = re.compile(
    r"\b(?P<name>(?:[A-Za-z _-]*?\b)?(?:total|sum|count|result)[A-Za-z _-]{0,16})\s*(?:=|is|equals)\s*(?P<rhs>-?\d+(?:\.\d+)?)\b",
    re.I,
)

_MATH_CLAIM_TRIGGER_RX = re.compile(
    r"(?:[-+*/%^]\s*\(?\s*\d|\b(?:total|sum|count|result)\s*(?:=|is|equals)\s*-?\d)",
    re.I,
)


def _symify(expr: str):
    # parse_expr (NOT sympify) is the transformations-aware API; sympify()
    # rejects the `transformations` kwarg on SymPy >= 1.12.
    return parse_expr(expr.replace("^", "**"), transformations=_SYMPTY_TRANSFORMS, evaluate=True)


def _check_arithmetic_claim(lhs: str, rhs: str) -> tuple[bool, str | None]:
    """Return (matches, corrected_value_or_None). Unparseable -> (True, None)."""
    if not _SYMPOK:
        return True, None
    try:
        left = _symify(lhs)
        right = _symify(rhs)
    except Exception as exc:  # noqa: BLE001 - not provable, leave untouched
        logger.debug("math-claim parse skipped %r = %r: %s", lhs, rhs, exc)
        return True, None
    if not left.is_number or not right.is_number:
        return True, None
    try:
        diff = sympy.simplify(left - right)
    except Exception:  # noqa: BLE001 - any SymPy failure means "not provable"
        return True, None
    if diff == 0:
        return True, None
    corrected = left
    if corrected == int(corrected):
        corrected = int(corrected)
    else:
        corrected = float(corrected)
    return False, str(corrected)


def _verify_math_claims(text: str, report: SafetyReport) -> str:
    """TASK 4: recompute arithmetic equalities; annotate proven-wrong ones."""
    if not _MATH_CLAIM_TRIGGER_RX.search(text):
        return text

    def _arith(m: re.Match[str]) -> str:
        lhs, rhs = m.group("lhs"), m.group("rhs").replace(" ", "")
        # Skip the RHS of markdown tables/bullets that are actually dates etc.
        if len(re.findall(r"[-+*/%^]", lhs)) < 1:
            return m.group(0)
        report.math_claims_checked += 1
        ok, corrected = _check_arithmetic_claim(lhs, rhs)
        if ok:
            return m.group(0)
        report.math_claims_wrong += 1
        report.issues.append(
            SafetyIssue(kind="math_claim", detail=f"{lhs} != {rhs} (SymPy: {corrected})", location="inline")
        )
        return f"{m.group(0)} <!-- math-check: SymPy gives {corrected} -->"

    text = _ARITH_CLAIM_RX.sub(_arith, text)

    def _named(m: re.Match[str]) -> str:
        name, rhs = m.group("name").strip(), m.group("rhs")
        low = name.lower()
        # Only totals/sums/counts — ordinary prose ("slip 2 = page 8") is skipped.
        if not any(k in low for k in ("total", "sum", "count", "result")):
            return m.group(0)
        # Reconstruct the additive lhs from the numbers mentioned shortly BEFORE.
        window = text[max(0, m.start() - 260): m.start()]
        nums = re.findall(r"\d+(?:\.\d+)?", window)
        if len(nums) < 2:
            return m.group(0)
        lhs = "+".join(nums[-8:])
        report.math_claims_checked += 1
        ok, corrected = _check_arithmetic_claim(lhs, rhs)
        if ok:
            return m.group(0)
        report.math_claims_wrong += 1
        report.issues.append(
            SafetyIssue(kind="math_claim", detail=f"{lhs} != {rhs} (SymPy: {corrected})", location=f"'{name}'")
        )
        return f"{m.group(0)} <!-- math-check: SymPy gives {corrected} -->"

    return _NAMED_TOTAL_RX.sub(_named, text)


# Regex to match internal validation / self-grading / compliance audit preamble
_INTERNAL_LEAK_BLOCK_RX = re.compile(
    r"(?is)^(?:\s*[-*#]*\s*(?:assessment\s+of\s+factual\s+restraint|compliance\s+check|internal\s+validation|"
    r"evaluation\s+of\s+rules|anti-hallucination\s+analysis|self-assessment|prompt\s+analysis|"
    r"fact-check\s+evaluation|internal\s+reasoning|grading\s+rubric|validation\s+report|"
    r"grounding\s+verification|verification\s+report)[:\n].*?"
    r"(?:conclusion:\s*response\s+is\s+compliant[^\n]*\n*|"
    r"conclusion:\s*compliant[^\n]*\n*|"
    r"verdict:\s*compliant[^\n]*\n*|"
    r"assessment:\s*compliant[^\n]*\n*|"
    r"confidence\s+score:\s*[0-9.]+[^\n]*\n*|"
    r"status:\s*passed[^\n]*\n*))",
)

_INTERNAL_TAGS_RX = re.compile(
    r"(?is)<(?:think|thought|reasoning|internal|evaluation|compliance|validation)>.*?</(?:think|thought|reasoning|internal|evaluation|compliance|validation)>|"
    r"\[(?:REASONING|INTERNAL|VALIDATION|EVALUATION)\].*?\[/(?:REASONING|INTERNAL|VALIDATION|EVALUATION)\]|"
    r"```(?:thought|reasoning|internal)\s*\n.*?```",
)

_ORPHAN_COMPLIANCE_LINES_RX = re.compile(
    r"(?im)^\s*(?:conclusion:\s*(?:response\s+is\s+)?compliant[^\n]*|"
    r"assessment\s+of\s+factual\s+restraint:?[^\n]*|"
    r"compliance\s+analysis:?[^\n]*|"
    r"grounding\s+verification:?[^\n]*|"
    r"confidence\s+score:\s*[0-9.]+[^\n]*|"
    r"internal\s+validation:?[^\n]*)\s*\n*",
)


def strip_internal_validation_leak(text: str) -> str:
    """Ensure no internal self-evaluations, compliance audits, or reasoning tags leak into user-facing answers."""
    if not text:
        return ""
    # Strip tagged blocks first (<think>, [REASONING], ```thought)
    cleaned = _INTERNAL_TAGS_RX.sub("", text)
    # Strip preamble evaluation blocks
    cleaned = _INTERNAL_LEAK_BLOCK_RX.sub("", cleaned)
    # Strip any isolated compliance/verification lines
    cleaned = _ORPHAN_COMPLIANCE_LINES_RX.sub("", cleaned)
    if cleaned != text:
        return cleaned.strip()
    return cleaned


# --------------------------------------------------------------------------
# Shared entry point
# --------------------------------------------------------------------------

def validate_generated_response(text: str, regenerate=None) -> SafetyReport:
    """TASKS 4+5: run every safety validator over one AI-generated answer.

    `regenerate` (TASK 5, optional) is called ONCE with the body of a
    syntax-invalid python block and must return corrected code or None. The
    corrected code is re-validated (ast.parse + unsafe scan) before acceptance.
    """
    clean_initial = strip_internal_validation_leak(text or "")
    report = SafetyReport(text=clean_initial)
    if not clean_initial:
        return report
    report.text = _validate_code_blocks(report.text, report, regenerate=regenerate)  # TASK 5
    report.text = _verify_math_claims(report.text, report)                            # TASK 4
    report.text = strip_internal_validation_leak(report.text)
    if report.issues:
        logger.info(
            "RESPONSE SAFETY: %d issue(s) — code_total=%d code_rejected=%d math_checked=%d math_wrong=%d",
            len(report.issues), report.code_blocks_total, report.code_blocks_rejected,
            report.math_claims_checked, report.math_claims_wrong,
        )
    return report


def normalize_server_exception(exc: Exception) -> tuple[int, str]:
    """Map raw backend exceptions to clean, user-safe error messages."""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    if any(k in exc_str or k in exc_type for k in ("connectiontimeout", "connection timeout", "timeout expired", "operationalerror", "pool_timeout")):
        return 503, "Research could not be completed because the workspace database connection timed out. Please retry."
    if any(k in exc_str or k in exc_type for k in ("psycopg", "sqlalchemy", "dbapierror", "databaseerror")):
        return 503, "A database service interruption occurred while processing your request. Please try again."
    if any(k in exc_str or k in exc_type for k in ("aiproviderunavailableerror", "aiprovidererror", "timeout", "timed out")):
        return 503, "The AI research engine is temporarily unavailable. Please retry shortly."
    if any(k in exc_str or k in exc_type for k in ("aiproviderautherror", "aiproviderconfigurationerror")):
        return 503, "AI research configuration issue encountered. Please check provider settings."

    return 500, "An internal error occurred while processing your request. Please try again."


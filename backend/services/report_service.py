"""
Report generation service - produces PDF pentest reports from assessment data.

The HTML/CSS templates live alongside this file under ``report_templates/``.
"""
import io
import base64
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Optional

import markdown as md_lib
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML
from sqlalchemy.orm import Session

from config import settings
from models import Assessment, Card, ReconData, CommandHistory, Credential
from services.container_service import ContainerService
from utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
SEVERITY_COLORS = {
    "CRITICAL": "#e11d48",
    "HIGH": "#f43f5e",
    "MEDIUM": "#f59e0b",
    "LOW": "#3b82f6",
    "INFO": "#64748b",
}
SEVERITY_BG = {
    "CRITICAL": "rgba(225,29,72,0.08)",
    "HIGH": "rgba(244,63,94,0.08)",
    "MEDIUM": "rgba(245,158,11,0.08)",
    "LOW": "rgba(59,130,246,0.08)",
    "INFO": "rgba(100,116,139,0.08)",
}

_TEMPLATES_DIR = Path(__file__).resolve().parent / "report_templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "j2"]),
)


# ---------------------------------------------------------------------------
# Markdown sanity preprocessor
# ---------------------------------------------------------------------------
#
# The AI often writes proof-of-concept fields like::
#
#     # Current state - disabled:
#     curl -s -u admin:admin http://target/api ...
#     → HTTP 403
#
# Markdown parses ``# Current state`` as an H1, blowing the layout. The same
# problem happens when commands are dumped without ``` fences. Before
# handing the text to the markdown library we wrap each block of consecutive
# lines that look like shell/HTTP content in a fenced code block. We only
# do this when the input has no fences yet, so already-well-formatted
# content is left untouched.
#
# Detection rules — designed to avoid wrapping prose that happens to start
# with an HTTP verb like ``POST /foo is unsafe because…``:
#
#   * HTTP verbs are only counted when followed by a clear path/URL marker
#     (``/``, ``http://``, ``HTTP/``) — never when followed by prose.
#   * Shell comments (``# foo``) only count when short, since real prose
#     rarely starts with ``# `` but markdown headings can be long.
#   * Strong CLI signals (``curl ...``, ``$ cmd``, ``→ output``) are
#     counted unconditionally.
#   * A paragraph is wrapped only when ≥2 lines match, or when the lone
#     line is a strong CLI signal — this keeps single-sentence prose
#     starting with "curl" or "POST" out of the code-wrap path.
_STRONG_CMD_RE = re.compile(
    r"^\s*(?:"
    r"\$\s|>\s|→\s|⇒\s|->\s|=>\s|"
    r"curl|wget|nmap|ssh|sshpass|sqlmap|ffuf|nikto|gobuster|hydra|nc\s|ncat|"
    r"masscan|amass|subfinder|whatweb|httpx|nuclei|dirb|dirbuster|"
    r"docker\s|kubectl|aws\s|gcloud|az\s|terraform\s|"
    r"python\d*\s|pip\d*\s|npm\s|node\s|ruby\s|perl\s|php\s|bash\s|sh\s|zsh\s"
    r")",
    re.IGNORECASE,
)
# HTTP verbs ONLY when followed by a path/URL — avoids wrapping prose that
# happens to start with a verb (e.g. "POST data is sent to the server").
_HTTP_REQ_RE = re.compile(
    r"^\s*(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|TRACE|CONNECT)\s+"
    r"(?:/|https?://|HTTP/)",
)
# Shell-style comment lines: must be short, lowercase-ish, not a markdown
# heading (e.g. "## Something" should still render as h2).
_SHELL_COMMENT_RE = re.compile(r"^\s*#\s+\S")


def _classify_line(line: str) -> str:
    """Classify a line as ``strong``, ``weak``, or ``prose``.

    - ``strong``: clear CLI invocation (``curl …``, ``$ cmd``, ``→ output``).
      A single strong line on its own is enough to trigger code wrapping.
    - ``weak``: HTTP verb + path, or short ``# comment`` — these match common
      code patterns but also appear at the start of prose sentences
      (e.g. *"POST /api/foo accepts a `cmd` parameter…"*), so we only treat
      them as code when they appear in a multi-line code block.
    - ``prose``: anything else.
    """
    s = line.strip()
    if not s:
        return "prose"
    if _STRONG_CMD_RE.match(line):
        return "strong"
    if _HTTP_REQ_RE.match(line):
        return "weak"
    if _SHELL_COMMENT_RE.match(line) and len(s) <= 120 and not s.startswith("##"):
        return "weak"
    return "prose"


def _wrap_unfenced_code(text: str) -> str:
    """Wrap blocks of consecutive code-like lines in fenced code blocks.

    A paragraph (block of consecutive non-empty lines) is wrapped when:

      * it contains ≥2 code-like lines (any classification), or
      * it is a single line that is a *strong* CLI signal pasted alone.

    Weak signals (HTTP verb + path, ``# comment``) alone on a line are
    left untouched — they're too easy to confuse with prose.
    """
    if not text or "```" in text:
        return text

    # Group lines into paragraphs separated by blank lines.
    paragraphs: list = []  # list of (kind, lines)
    current: list[str] = []
    for raw in text.split("\n"):
        if raw.strip():
            current.append(raw)
        else:
            if current:
                paragraphs.append(("p", current))
                current = []
            paragraphs.append(("blank", [raw]))
    if current:
        paragraphs.append(("p", current))

    out: list[str] = []
    for kind, lines in paragraphs:
        if kind == "blank":
            out.extend(lines)
            continue
        classes = [_classify_line(l) for l in lines]
        code_count = sum(1 for c in classes if c != "prose")
        strong_count = sum(1 for c in classes if c == "strong")
        # Wrap when several lines match (multi-line code block), OR when a
        # single short line is a strong CLI signal.
        should_wrap = code_count >= 2 or (
            len(lines) == 1
            and strong_count == 1
            and len(lines[0]) <= 250
        )
        if should_wrap:
            out.append("```")
            out.extend(lines)
            out.append("```")
        else:
            out.extend(lines)
    return "\n".join(out)


def _render_markdown(text: str) -> str:
    """Render a markdown string to HTML.

    Used as a Jinja filter (``| markdown``) so that finding fields written
    by the AI (technical_analysis, proof, notes) — which routinely contain
    fenced code, bold, lists, inline code — render the same way the
    methodology section does instead of dumping raw asterisks and backticks.
    Unfenced shell/HTTP blocks are auto-wrapped first so they don't end up
    interpreted as H1 headings.
    """
    if not text:
        return ""
    cleaned = _wrap_unfenced_code(text)
    return md_lib.markdown(
        cleaned,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )


_jinja_env.filters["markdown"] = _render_markdown


def _format_recon_details(value) -> str:
    """Render a recon ``details`` JSON value as a readable inline string.

    Recon rows persist their details as ``JSONB`` — usually a flat dict of
    metadata like ``{'port': 80, 'service': 'nginx'}``. Printing that as
    raw Python repr produces an ugly, brace-heavy cell. This filter turns
    the dict into ``port: 80 · service: nginx`` for compact, human-friendly
    display. Lists collapse to comma-joined strings; primitives are stringified.
    """
    if value is None or value == "":
        return "—"
    if isinstance(value, dict):
        if not value:
            return "—"
        return " · ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "—"
    return str(value)


_jinja_env.filters["recon_details"] = _format_recon_details


def _load_logo_b64() -> str:
    """Load AIDA logo as base64 data URI."""
    logo_candidates = [
        Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "assets" / "aida-logo.png",
        Path(__file__).resolve().parent.parent / "assets" / "aida-logo.png",
    ]
    for p in logo_candidates:
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f"data:image/png;base64,{b64}"
    return ""


def _compute_risk_score(findings: list) -> str:
    """Compute overall risk level from findings."""
    if not findings:
        return "INFORMATIONAL"
    worst = min(SEVERITY_ORDER.get(f.severity or "INFO", 99) for f in findings)
    return {0: "CRITICAL", 1: "HIGH", 2: "MEDIUM", 3: "LOW"}.get(worst, "INFORMATIONAL")


def _is_methodology_placeholder(content: str) -> bool:
    """Return True when methodology.md still contains the default template.

    Mirrors the logic in frontend MethodologyReport.jsx so the PDF and UI
    treat the file the same way.
    """
    stripped = content.strip()
    if not stripped:
        return True
    non_empty_lines = [l for l in stripped.split("\n") if l.strip()]
    if "Generated by AIDA AI" in stripped and len(non_empty_lines) <= 4:
        return True
    return False


async def _load_methodology_html(assessment: Assessment) -> Optional[str]:
    """Fetch methodology.md from the assessment workspace and render to HTML.

    Returns None when the file is missing, unreadable, or still holds the
    default template placeholder — the report then omits the section entirely.
    """
    if not assessment.workspace_path:
        return None

    container_name = assessment.container_name or settings.DEFAULT_CONTAINER_NAME

    container_service = ContainerService()
    container_service.current_container = container_name

    full_path = f"{assessment.workspace_path}/methodology.md"
    check_cmd = f"test -f {shlex.quote(full_path)} && echo exists"
    check = await container_service.execute_container_command(check_cmd)
    if not check.get("success") or "exists" not in (check.get("stdout") or ""):
        return None

    read = await container_service.execute_container_command(f"cat {shlex.quote(full_path)}")
    if not read.get("success"):
        logger.warning(
            "Failed to read methodology.md for report",
            assessment_id=assessment.id,
            stderr=(read.get("stderr") or "")[:200],
        )
        return None

    content = read.get("stdout") or ""
    if _is_methodology_placeholder(content):
        return None

    html = md_lib.markdown(
        content,
        extensions=["fenced_code", "tables", "nl2br", "sane_lists", "toc"],
        output_format="html5",
    )
    return html


async def generate_pdf_report(
    db: Session,
    assessment_id: int,
    include_secrets: bool = False,
) -> io.BytesIO:
    """Generate a PDF pentest report for the given assessment.

    Args:
        include_secrets: If False (default), credential secrets (password,
            token, cookie value, custom JSON) are masked in the report so the
            PDF can be safely shared. When True, the raw values are embedded
            — only do this for the pentester's internal copy.
    """

    # --- Fetch data ---
    assessment = db.query(Assessment).filter(Assessment.id == assessment_id).first()
    if not assessment:
        raise ValueError(f"Assessment {assessment_id} not found")

    cards = (
        db.query(Card)
        .filter(Card.assessment_id == assessment_id)
        .all()
    )
    findings = sorted(
        [c for c in cards if c.card_type == "finding"],
        key=lambda c: SEVERITY_ORDER.get(c.severity or "INFO", 99),
    )
    observations = [c for c in cards if c.card_type == "observation"]
    info_cards = [c for c in cards if c.card_type == "info"]

    recon = (
        db.query(ReconData)
        .filter(ReconData.assessment_id == assessment_id)
        .all()
    )

    credentials = (
        db.query(Credential)
        .filter(Credential.assessment_id == assessment_id)
        .all()
    )

    commands_count = (
        db.query(CommandHistory)
        .filter(CommandHistory.assessment_id == assessment_id)
        .count()
    )

    # --- Severity stats ---
    severity_counts = {}
    for f in findings:
        sev = f.severity or "INFO"
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    max_sev_count = max(severity_counts.values()) if severity_counts else 1

    # --- Risk score ---
    risk_score = _compute_risk_score(findings)

    # --- Methodology (rendered markdown) ---
    methodology_html = await _load_methodology_html(assessment)

    # --- Logo ---
    logo_b64 = _load_logo_b64()

    # --- Render HTML ---
    css = (_TEMPLATES_DIR / "report.css").read_text(encoding="utf-8")
    template = _jinja_env.get_template("report.html.j2")
    html_content = template.render(
        css=css,
        assessment=assessment,
        findings=findings,
        observations=observations,
        info_cards=info_cards,
        recon=recon,
        credentials=credentials,
        commands_count=commands_count,
        severity_counts=severity_counts,
        severity_colors=SEVERITY_COLORS,
        severity_bg=SEVERITY_BG,
        severity_order=SEVERITY_ORDER,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        total_findings=len(findings),
        total_observations=len(observations),
        total_recon=len(recon),
        total_credentials=len(credentials),
        max_sev_count=max_sev_count,
        risk_score=risk_score,
        logo_b64=logo_b64,
        methodology_html=methodology_html,
        include_secrets=include_secrets,
    )

    # --- Generate PDF ---
    pdf_buffer = io.BytesIO()
    HTML(string=html_content).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)

    logger.info(
        "Report generated",
        assessment_id=assessment_id,
        findings=len(findings),
        observations=len(observations),
        methodology_included=methodology_html is not None,
        include_secrets=include_secrets,
    )

    return pdf_buffer

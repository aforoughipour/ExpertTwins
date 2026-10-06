"""Turning documents into passages.

Three input shapes, in descending order of quality:

1. **JATS XML** (Europe PMC ``fullTextXML``, NCBI ``efetch``). Structured, with
   real section headings and no typographic damage. Always preferred.
2. **PDF** (via PyMuPDF). Ordered text with heuristic section detection.
3. **Plain text.**

Two rules apply to all three:

* **Original document order is preserved.** Passages are never reordered, here
  or at retrieval time (OP-RAG 2409.01666 / DOS RAG 2506.03989: "preserve
  original document order" is the most replicated practical finding in the
  retrieval literature).
* **Absence is typed.** A parser that finds no body returns an
  :class:`~expertwins.corpus.errors.Absence` recording what it looked at. A
  parser that examines zero usable units has not produced evidence of absence,
  even if it can name the section it failed to parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lxml import etree

from .errors import Absence, ExtractionError
from .models import MIN_PASSAGE_WORDS, Passage
from .textnorm import canonical, words

# Target passage size. Big enough to carry a claim with its qualifiers (the
# canonical failure is a lost "except the quiescent state" clause, so splitting
# mid-sentence or mid-qualifier is actively dangerous), small enough that BM25
# scoring stays meaningful.
TARGET_CHARS = 1_500
MAX_CHARS = 3_000
MIN_CHARS = 200

# Sections that carry no citable scientific content.
_SKIP_SECTIONS = {
    "references", "reference", "bibliography", "acknowledgements",
    "acknowledgments", "author contributions", "competing interests",
    "conflict of interest", "conflicts of interest", "funding",
    "supplementary material", "abbreviations", "copyright",
    "author information", "additional information", "ethics declarations",
    "data availability", "code availability", "publisher's note",
}


@dataclass
class ExtractedDoc:
    passages: list[Passage]
    title: str = ""
    abstract: str = ""
    sections_seen: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sections_seen is None:
            self.sections_seen = []

    @property
    def n_chars(self) -> int:
        return sum(len(p.text) for p in self.passages)


def _should_skip(section: str) -> bool:
    s = canonical(section).lower().strip(" .:0123456789")
    return any(s.startswith(skip) for skip in _SKIP_SECTIONS)


def _pack(blocks: list[tuple[str, str]], doc_id: str) -> list[Passage]:
    """Group ordered (section, text) blocks into passages of a usable size.

    Blocks are only ever merged with their *immediate neighbours in the same
    section*, so packing can never splice together text that was far apart in
    the original — which would create stitched pseudo-quotes that were not
    present in the document.
    """
    passages: list[Passage] = []
    buffer: list[str] = []
    buffer_section = ""
    order = 0

    def flush() -> None:
        nonlocal buffer, buffer_section, order
        if not buffer:
            return
        text = canonical(" ".join(buffer))
        buffer = []
        if len(words(text)) < MIN_PASSAGE_WORDS:
            return
        passages.append(
            Passage(
                passage_id=f"{doc_id}#p{order:04d}",
                order=order,
                section=buffer_section,
                text=text,
            )
        )
        order += 1

    for section, text in blocks:
        text = canonical(text)
        if not text:
            continue
        if section != buffer_section:
            flush()
            buffer_section = section

        current = sum(len(b) for b in buffer)
        if current and current + len(text) > MAX_CHARS:
            flush()
            buffer_section = section

        buffer.append(text)
        if sum(len(b) for b in buffer) >= TARGET_CHARS:
            flush()
            buffer_section = section

    flush()
    return passages


# ---------------------------------------------------------------------------
# JATS XML
# ---------------------------------------------------------------------------

def _element_text(el: etree._Element) -> str:
    """Flatten an element, dropping cross-references and float anchors.

    Inline ``xref`` markers ("[12]", "Fig. 3a") add nothing an agent can cite
    and corrupt quote matching, so they are removed rather than kept.
    """
    clone = etree.fromstring(etree.tostring(el))
    for tag in ("xref", "graphic", "inline-graphic", "media", "fig", "table-wrap",
                "supplementary-material", "disp-formula", "inline-formula"):
        for node in clone.findall(f".//{tag}"):
            parent = node.getparent()
            if parent is not None:
                tail = node.tail or ""
                if tail:
                    prev = node.getprevious()
                    if prev is not None:
                        prev.tail = (prev.tail or "") + tail
                    else:
                        parent.text = (parent.text or "") + tail
                parent.remove(node)
    return canonical(" ".join(clone.itertext()))


def extract_jats(xml_bytes: bytes, doc_id: str) -> ExtractedDoc | Absence:
    """Parse a JATS article into ordered passages."""
    try:
        parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False)
        root = etree.fromstring(xml_bytes, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ExtractionError(f"{doc_id}: JATS did not parse: {exc}") from exc
    if root is None:
        raise ExtractionError(f"{doc_id}: JATS parsed to nothing")

    title_el = root.find(".//article-title")
    title = _element_text(title_el) if title_el is not None else ""

    abstract_parts: list[str] = []
    for abstract in root.findall(".//abstract"):
        if abstract.get("abstract-type") in ("graphical", "teaser"):
            continue
        text = _element_text(abstract)
        if text:
            abstract_parts.append(text)
    abstract = canonical(" ".join(abstract_parts))

    blocks: list[tuple[str, str]] = []
    sections_seen: list[str] = []

    if abstract:
        blocks.append(("Abstract", abstract))
        sections_seen.append("Abstract")

    body = root.find(".//body")
    paragraphs_examined = 0
    if body is not None:
        def walk(node: etree._Element, heading: str) -> None:
            nonlocal paragraphs_examined
            label_el = node.find("title")
            local = _element_text(label_el) if label_el is not None else heading
            if not local:
                local = heading
            if local and local not in sections_seen:
                sections_seen.append(local)
            if _should_skip(local):
                return
            for child in node:
                if child.tag == "sec":
                    walk(child, local)
                elif child.tag in ("p", "list", "statement", "disp-quote"):
                    paragraphs_examined += 1
                    text = _element_text(child)
                    if text:
                        blocks.append((local, text))

        top_sections = body.findall("sec")
        if top_sections:
            for sec in top_sections:
                walk(sec, "")
        else:
            for p in body.findall(".//p"):
                paragraphs_examined += 1
                text = _element_text(p)
                if text:
                    blocks.append(("Body", text))

    passages = _pack(blocks, doc_id)
    if not passages:
        return Absence(
            what="body text",
            method="JATS XML parse (lxml, body//sec//p)",
            query=doc_id,
            examined=paragraphs_examined,
            detail={
                "had_body_element": body is not None,
                "had_abstract": bool(abstract),
                "bytes": len(xml_bytes),
                "sections_seen": sections_seen[:40],
            },
        )
    return ExtractedDoc(
        passages=passages, title=title, abstract=abstract, sections_seen=sections_seen
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

# A heading looks like: short, title-case or upper-case, no terminal period.
_HEADING_RE = re.compile(
    r"^\s{0,6}(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?"
    r"(abstract|introduction|background|methods?|materials and methods|"
    r"results?|discussion|conclusions?|references|acknowledge?ments?|"
    r"supplementary|limitations|data availability|funding)\b",
    re.IGNORECASE,
)

_REFERENCES_START_RE = re.compile(
    r"^\s{0,6}(references?|bibliography|literature cited|works cited)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_pdf(pdf_bytes: bytes, doc_id: str) -> ExtractedDoc | Absence:
    """Parse a PDF into ordered passages using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ExtractionError("PyMuPDF is required for PDF extraction") from exc

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises bare Exception subclasses
        raise ExtractionError(f"{doc_id}: PDF did not open: {exc}") from exc

    blocks: list[tuple[str, str]] = []
    sections_seen: list[str] = []
    current_section = ""
    paragraphs_examined = 0
    pages = len(doc)

    try:
        for page in doc:
            raw = page.get_text("text")
            if not raw:
                continue
            # Blank-line separated chunks are the closest thing a PDF has to
            # paragraphs.
            for chunk in re.split(r"\n\s*\n", raw):
                chunk = chunk.strip()
                if not chunk:
                    continue
                paragraphs_examined += 1
                first_line = chunk.split("\n", 1)[0].strip()
                heading = _HEADING_RE.match(first_line)
                if heading and len(first_line) < 80:
                    current_section = canonical(first_line)
                    if current_section not in sections_seen:
                        sections_seen.append(current_section)
                    remainder = chunk.split("\n", 1)[1] if "\n" in chunk else ""
                    if not remainder.strip():
                        continue
                    chunk = remainder
                if _should_skip(current_section):
                    continue
                text = canonical(chunk)
                if len(text) >= 40:
                    blocks.append((current_section or "Body", text))
    finally:
        doc.close()

    passages = _pack(blocks, doc_id)
    title = _guess_pdf_title(pdf_bytes)
    if not passages:
        return Absence(
            what="extractable text",
            method="PyMuPDF page.get_text('text')",
            query=doc_id,
            examined=paragraphs_examined,
            detail={
                "pages": pages,
                "bytes": len(pdf_bytes),
                "likely_scanned": paragraphs_examined == 0 and pages > 0,
                "sections_seen": sections_seen[:40],
            },
        )
    return ExtractedDoc(passages=passages, title=title, sections_seen=sections_seen)


def _guess_pdf_title(pdf_bytes: bytes) -> str:
    """Read the embedded metadata title, if the publisher set a real one.

    Returns '' rather than a guess. Ambiguous metadata should stay absent
    rather than be converted into a plausible but unsupported title.
    """
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except (ImportError, RuntimeError, ValueError):
        # A PDF whose metadata cannot be opened yields no title. This is the one
        # place a broad-ish catch is acceptable, because the *only* consequence
        # is that we fall back to a different title source -- and it is narrowed
        # to the errors PyMuPDF actually raises rather than to Exception.
        return ""
    try:
        title = canonical((doc.metadata or {}).get("title") or "")
    finally:
        doc.close()
    # Producers habitually stuff filenames and template names in here.
    if len(title) < 12 or title.lower().endswith((".pdf", ".doc", ".docx", ".qxd")):
        return ""
    if re.fullmatch(r"[\w\-.]+", title):
        return ""
    return title


def extract_text(raw: str, doc_id: str, section: str = "Body") -> ExtractedDoc | Absence:
    """Parse plain text into ordered passages."""
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
    blocks = [(section, c) for c in chunks]
    passages = _pack(blocks, doc_id)
    if not passages:
        return Absence(
            what="text content", method="plain-text paragraph split",
            query=doc_id, examined=len(chunks), detail={"bytes": len(raw)},
        )
    return ExtractedDoc(passages=passages)

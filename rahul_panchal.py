import sys, re, json, io, os
from datetime import datetime
from collections import Counter

TITLE_HINTS = re.compile(r'\b(agreement|master|services|license|lease|contract|statement of work)\b', re.I)

DATE_HINTS  = [
    re.compile(r'\beffective\s+date\s*[:\-]?\s*(?P<date>.+?)(?:[.;\n\r]|$)', re.I),
    re.compile(r'\beffective\s+as\s+of\s+(?P<date>.+?)(?:[.;\n\r]|$)', re.I),
    re.compile(r'\beffective\s+on\s+(?P<date>.+?)(?:[.;\n\r]|$)', re.I),
    re.compile(r'\bshall\s+take\s+effect\s+on\s+(?P<date>.+?)(?:[.;\n\r]|$)', re.I),
    re.compile(r'\bis\s+made\s+as\s+of\s+(?P<date>.+?)(?:[.;\n\r]|$)', re.I),
    re.compile(r'\bdated\s+(?P<date>.+?)(?:[.;\n\r]|$)', re.I),
    re.compile(r'\bas\s+of\s+(?:the\s+)?(?P<date>\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+\s*,?\s*\d{4})', re.I),
    re.compile(r'\b(?:as of|dated)\s+(?:the\s+)?(?P<date>\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+\s*,?\s*\d{4})', re.I),
]

SECTION_HEADER_PATTERNS = [
    re.compile(r'^(?P<num>\d+(?:\.\d+)*)\s+(?P<title>[A-Z][^\n]{0,120})$'),
    re.compile(r'^(?P<num>[IVXLC]+)\.\s+(?P<title>[A-Z][^\n]{0,120})$', re.I),
    re.compile(r'^(?:Section|Article)\s+(?P<num>\d+|[IVXLC]+)[\.\:]\s+(?P<title>[A-Z][^\n]{0,120})$', re.I),
]

SECTION_ANCHOR = re.compile(
    r'(?m)^\s*(?P<num>\d{1,2})\.\s+(?P<title>[A-Z][A-Z0-9 ,\-\./&’\'()]{2,})\.?\s*$'
)
ARTICLE_ANCHOR = re.compile(
    r'(?m)^\s*(?:ARTICLE|SECTION)\s+(?P<num>[IVXLC]+|\d+)\.?\s+(?P<title>[A-Z][A-Z0-9 ,\-\./&’\'()]{2,})\.?\s*$',
    re.I
)
SECTION_INLINE = re.compile(
    r'(?<![\w\.])(?P<num>\d{1,2})\.\s+(?P<title>[A-Z][A-Z0-9 ,\-\./&’\'()]{2,})\.(?!\w)'
)

SIGNATURE_CUTOFF = re.compile(r'(?im)\bIN WITNESS WHEREOF\b|^\s*By:\s*[_\s]*$', re.M)

CLAUSE_LABELS = [
    re.compile(r'^\d+(?:\.\d+)*(\)|\.|\s)'),
    re.compile(r'^\([a-zA-Z]\)'),
    re.compile(r'^[•\-–] '),
]

SUBLABEL_SPLIT = re.compile(
    r'(?m)(?=^\s*(?:'
    r'[A-Z]\.|[A-Z]\)|'            # A.  A)
    r'\([a-z]\)|'                  # (a)
    r'\([ivxlcdm]+\)|'             # (i)
    r'[ivxlcdm]+\)|'               # i)
    r'\d+\)|'                      # 1)
    r'\d+(?:\.\d+)+|'              # 1.2  / 1.2.3
    r'\d+\.'                       # 1.
    r'))', re.I
)

WS = re.compile(r'\s+')
WS_LINE = re.compile(r'[ \t\r\f\v]+')
PAGE_LINE = re.compile(r'^\s*Page\s+\d+(\s+of\s+\d+)?\s*$', re.I)

HEADER_FOOTER_MAX_LINES = 3
HEADER_FOOTER_FREQ = 0.6

_SUBITEM_PAREN = re.compile(r'^\([a-z]\)$', re.I)
_NUM_PAR_LABEL  = re.compile(r'^\d+[.)]$')

INLINE_ENUM_SPLIT = re.compile(r'(?:^|[;])\s*(\d+)[\.\)]\s+', re.M)

def _try_imports():
    global pdfplumber, Image, pytesseract, dateparser
    pdfplumber = None; pytesseract = None; Image = None; dateparser = None
    try:
        import pdfplumber  # type: ignore
        globals()['pdfplumber'] = pdfplumber
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore
        globals()['Image'] = Image
    except Exception:
        pass
    try:
        import pytesseract  # type: ignore
        globals()['pytesseract'] = pytesseract
    except Exception:
        pass
    try:
        import dateparser  # type: ignore
        globals()['dateparser'] = dateparser
    except Exception:
        pass

def read_pdf_text(path: str) -> list[str]:
    pages = []
    if pdfplumber is None:
        raise RuntimeError("Please install pdfplumber: pip install pdfplumber pillow pytesseract dateparser")
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            if len(txt.strip()) < 20 and pytesseract and Image:
                try:
                    img = p.to_image(resolution=200).original
                    ocr_txt = pytesseract.image_to_string(img) or ""
                    txt = ocr_txt if len(ocr_txt.strip()) > len(txt.strip()) else txt
                except Exception:
                    pass
            pages.append(txt)
    return pages

def _norm_line_key(ln: str) -> str:
    return re.sub(r'\s+', ' ', ln).strip().lower()

def _strip_headers_footers(pages: list[str]) -> list[str]:
    if not pages:
        return pages
    top_lines, bottom_lines = [], []
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines()]
        if not lines:
            continue
        top_lines.extend(lines[:HEADER_FOOTER_MAX_LINES])
        bottom_lines.extend(lines[-HEADER_FOOTER_MAX_LINES:])

    n_pages = max(1, len(pages))
    top_counts = Counter(_norm_line_key(x) for x in top_lines if x.strip())
    bot_counts = Counter(_norm_line_key(x) for x in bottom_lines if x.strip())
    top_common = {k for k, c in top_counts.items() if c >= max(2, int(HEADER_FOOTER_FREQ * n_pages))}
    bot_common = {k for k, c in bot_counts.items() if c >= max(2, int(HEADER_FOOTER_FREQ * n_pages))}

    cleaned = []
    for p in pages:
        new_lines = []
        for ln in p.splitlines():
            nk = _norm_line_key(ln)
            if PAGE_LINE.match(ln):
                continue
            if nk in top_common or nk in bot_common:
                continue
            new_lines.append(ln)
        cleaned.append("\n".join(new_lines))
    return cleaned

def _dehyphenate(text: str) -> str:
    return re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2\n', text)

def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = WS.sub(" ", s)
    return s.strip()

def normalize_for_structure(s: str) -> str:
    s = s.replace("\u00a0", " ")
    return "\n".join(WS_LINE.sub(" ", ln).strip() for ln in s.splitlines())

def _smart_titlecase(s: str) -> str:
    stripped = re.sub(r'[^A-Za-z]+', '', s)
    return s.title() if stripped and stripped.isupper() else s

def guess_title_and_type(lines: list[str]) -> tuple[str, str]:
    for ln in lines[:50]:
        l = ln.strip()
        if not l:
            continue
        if l.isupper() or TITLE_HINTS.search(l):
            contract_type = "Agreement"
            m = re.search(r'([A-Z][A-Za-z ]+(Agreement|Contract|Addendum|Order|License))', l)
            if m: contract_type = m.group(1)
            return (normalize_text(l), normalize_text(contract_type))
    for ln in lines:
        if ln.strip():
            return (normalize_text(ln), "Agreement")
    return ("Contract", "Agreement")

def _date_to_iso(cand: str) -> str | None:
    cand = cand.strip().strip('":,(). ')
    m_ord = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+([A-Za-z]+)\s*,?\s*(\d{4})', cand, re.I)
    if m_ord:
        cand = f"{m_ord.group(2)} {m_ord.group(1)} {m_ord.group(3)}"
    try:
        if dateparser:
            dt = dateparser.parse(cand, settings={"PREFER_DAY_OF_MONTH": "first"})
            if dt:
                return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    m2 = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}', cand, re.I)
    if m2:
        return _naive_iso(m2.group(0))
    return None

def parse_effective_date(text: str) -> str | None:
    head = text[:10000]
    for rx in DATE_HINTS:
        m = rx.search(head)
        if m:
            iso = _date_to_iso(m.group("date"))
            if iso:
                return iso
    return None

def _naive_iso(s: str) -> str | None:
    try:
        dt = datetime.strptime(re.sub(r'(\d{1,2}),\s+(\d{4})', r'\1 \2', s), "%B %d %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

# ---------- Global section detection ----------

def find_section_spans(full_text: str):
    mcut = SIGNATURE_CUTOFF.search(full_text)
    if mcut:
        full_text = full_text[:mcut.start()]

    matches = []
    for m in SECTION_ANCHOR.finditer(full_text):
        matches.append((m.start(), m.end(), m.group('num'), m.group('title').strip()))
    for m in ARTICLE_ANCHOR.finditer(full_text):
        matches.append((m.start(), m.end(), m.group('num'), m.group('title').strip()))
    for m in SECTION_INLINE.finditer(full_text):
        matches.append((m.start(), m.end(), m.group('num'), m.group('title').strip()))

    best = {}
    for s, e, num, title in matches:
        if (s not in best) or ((e - s) > (best[s][0] - s)):
            best[s] = (e, num, title)

    ordered = sorted(((s, *v) for s, v in best.items()), key=lambda x: x[0])

    spans = []
    for i, (s, e, num, title) in enumerate(ordered):
        next_s = ordered[i+1][0] if i+1 < len(ordered) else len(full_text)
        spans.append((s, next_s, num, title))
    return spans

def strip_heading_line(num: str, title: str, body: str) -> str:
    pat = re.compile(r'(?m)^\s*%s\.\s+%s\.?\s*$' % (re.escape(str(num)), re.escape(title)))
    return pat.sub("", body, count=1).strip()

def _clean_leading_repeated_heading(text: str, sec_num: str|None, sec_title: str|None) -> str:
    if not text or not sec_title:
        return text
    t = re.sub(r'\s+', r'\\s+', re.escape(sec_title))
    num_part = re.escape(str(sec_num)) if sec_num else r'\d{1,2}'
    rx = re.compile(r'^\s*(?:%s\.\s+)?%s[.:]?\s*' % (num_part, t), re.I)
    return rx.sub("", text, count=1).strip()

def split_sections_global(full_text: str) -> list[dict]:
    spans = find_section_spans(full_text)
    sections = []
    if not spans:
        return []
    for (s, e, num, title) in spans:
        body = full_text[s:e]
        body = strip_heading_line(num, title, body)
        title_clean = _smart_titlecase(re.sub(r'[.\s]+$', '', title).strip())
        clauses = split_clauses_paragraphs(body, sec_num=str(num), sec_title=title_clean)
        sections.append({"title": title_clean, "number": str(num), "clauses": clauses})
    return sections

# ---------- Fallback (line-based) ----------

def split_sections(all_lines: list[str]) -> list[dict]:
    sections = []
    i = 0
    while i < len(all_lines):
        line = all_lines[i].strip()
        header = _match_section_header(line)
        if header:
            title = _smart_titlecase(normalize_text(header["title"]))
            number = header["num"] if header["num"] else None
            number = str(number) if number is not None else None
            body_lines = []
            i += 1
            while i < len(all_lines) and not _match_section_header(all_lines[i].strip()):
                body_lines.append(all_lines[i])
                i += 1
            clauses = split_clauses_paragraphs("\n".join(body_lines), sec_num=number, sec_title=title)
            sections.append({"title": title, "number": number, "clauses": clauses})
        else:
            i += 1
    if not sections:
        clauses = split_clauses_paragraphs("\n".join(all_lines))
        sections = [{"title": "General", "number": None, "clauses": clauses}]
    return sections

def _match_section_header(line: str):
    if not line:
        return None
    for rx in SECTION_HEADER_PATTERNS:
        m = rx.match(line)
        if m:
            return {"num": m.groupdict().get("num"), "title": m.groupdict().get("title")}
    return None

# ---------- Clause splitting helpers ----------

_MERGE_TRIGGER_RX = re.compile(r'(?i)(including|includes|include|as follows)\s*:?\s*$')

def _split_inline_numeric_enums(text: str):
    text = re.sub(r'(\d+[.)])(?=\S)', r'\1 ', text)
    matches = list(INLINE_ENUM_SPLIT.finditer(text))
    if len(matches) <= 1:
        return None
    pieces = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        label_num = m.group(1)
        seg = text[start:end].strip().lstrip(":;-–. ")
        if seg:
            pieces.append((f"{label_num}.", normalize_text(seg)))
    return pieces if pieces else None

def _label_to_int(label: str) -> int | None:
    m = re.match(r'^\s*(\d+)[\.)]\s*$', label or "")
    return int(m.group(1)) if m else None

def _is_letter_label(label: str) -> bool:
    return bool(re.match(r'^[A-Z][.)]$', label or ""))

def _prefer_style_like(label: str) -> str:
    # Return ')' if label uses that, else '.'
    return ')' if label and label.endswith(')') else '.'

def _postfix_enumeration_fixes(clauses: list[dict]) -> list[dict]:
    """
    1) Assign missing 1) when an unlabeled clause precedes a '2)' after a lettered subheader.
    2) Infer missing numeric (e.g., 1., 3.) by splitting previous item on the first semicolon.
    3) Re-index indices.
    """
    if not clauses:
        return clauses

    # Pass A: assign a missing "1)" between lettered subheader and "2)"
    seen_since_letter = set()
    last_letter_at = -1
    for i, c in enumerate(clauses):
        lab = c["label"]
        num = _label_to_int(lab)
        if _is_letter_label(lab):
            seen_since_letter.clear()
            last_letter_at = i
            continue
        if num is not None:
            seen_since_letter.add(num)
            continue
        # unlabeled; check next numeric 'n)' with n>=2
        if i+1 < len(clauses):
            nxt_lab = clauses[i+1]["label"]
            nxt_num = _label_to_int(nxt_lab)
            if nxt_num and nxt_lab.endswith(')') and nxt_num >= 2 and (nxt_num - 1) not in seen_since_letter:
                # and we are inside a lettered block
                if last_letter_at != -1 and last_letter_at < i:
                    c["label"] = f"{nxt_num - 1})"

    # Pass B: infer a missing number between current k and next m>=k+2 by splitting on first semicolon
    i = 0
    while i < len(clauses) - 1:
        curr = clauses[i]
        nextc = clauses[i+1]
        k = _label_to_int(curr["label"])
        m = _label_to_int(nextc["label"])
        if k is not None and m is not None and m >= k + 2:
            # try to split curr on first semicolon
            text = curr["text"]
            # avoid splitting if there's no semicolon to indicate two items joined
            split_pos = text.find(";")
            if split_pos != -1 and split_pos < len(text) - 1:
                left = text[:split_pos].strip()
                right = text[split_pos+1:].strip()
                if left and right:
                    curr["text"] = left
                    style = _prefer_style_like(nextc["label"])
                    new_clause = {
                        "text": right.lstrip(":—- ").strip(),
                        "label": f"{k+1}{style}",
                        "index": curr["index"] + 1
                    }
                    clauses.insert(i+1, new_clause)
                    # do not advance i to allow cascading fixes if multiple gaps
                    continue
        i += 1

    # Re-index
    for idx, c in enumerate(clauses):
        c["index"] = idx
    return clauses

# ---------- Clause splitting within each section ----------

def split_clauses_paragraphs(section_text: str, sec_num: str|None=None, sec_title: str|None=None) -> list[dict]:
    paras = [p.strip() for p in re.split(r'\n\s*\n+', section_text) if p.strip()]
    chunks = []
    for p in paras:
        parts = [s.strip() for s in SUBLABEL_SPLIT.split(p) if s.strip()]
        chunks.extend(parts if parts else [p])

    out, idx = [], 0
    for raw_chunk in chunks:
        m = re.match(r'^\s*((?:[A-Z]\.|[A-Z]\)|\([a-z]\)|\([ivxlcdm]+\)|[ivxlcdm]+\)|\d+\)|\d+(?:\.\d+)*|\d+\.))\s+', raw_chunk, re.I)
        label = m.group(1) if m else ""
        text = normalize_text(raw_chunk[m.end():] if m else raw_chunk)
        if not text:
            continue

        if idx == 0:
            text = _clean_leading_repeated_heading(text, sec_num, sec_title)

        if sec_num:
            if label in (f"{sec_num}.", f"{sec_num})", f"{sec_num}"):
                label = ""
            text = re.sub(rf'^\s*{re.escape(str(sec_num))}[.)]\s+', '', text)

        inline_split = _split_inline_numeric_enums(text)
        if inline_split:
            for lbl, seg_text in inline_split:
                out.append({"text": seg_text, "label": lbl, "index": idx}); idx += 1
            continue

        if _SUBITEM_PAREN.fullmatch(label) and out and _NUM_PAR_LABEL.fullmatch(out[-1]["label"]):
            sep = " " if _MERGE_TRIGGER_RX.search(out[-1]["text"]) else " "
            out[-1]["text"] = (out[-1]["text"] + f"{sep}{label} {text}").strip()
            continue

        if label == "" and out and out[-1]["label"] != "" and len(text) <= 140:
            out[-1]["text"] = (out[-1]["text"] + " " + text).strip()
            continue

        out.append({"text": text, "label": label, "index": idx}); idx += 1

    if not out:
        out = [{"text": normalize_text(section_text), "label": "", "index": 0}]

    # NEW: fix enumerations and reindex
    out = _postfix_enumeration_fixes(out)
    return out

def main():
    _try_imports()
    if len(sys.argv) != 3:
        print("Usage: python rahul_panchal.py <input.pdf> <output.json>", file=sys.stderr)
        sys.exit(2)
    in_pdf, out_json = sys.argv[1], sys.argv[2]
    pages = read_pdf_text(in_pdf)

    pages = [_dehyphenate(p) for p in pages]
    pages = _strip_headers_footers(pages)

    lines = []
    for t in pages:
        for raw in t.splitlines():
            if raw is not None:
                if PAGE_LINE.match(raw.strip()):
                    continue
                lines.append(raw)

    full_text_struct = normalize_for_structure("\n".join(lines))
    full_text_for_date = normalize_text("\n".join(lines))

    eff_date = parse_effective_date(full_text_for_date)
    title, contract_type = guess_title_and_type(lines)

    sections = split_sections_global(full_text_struct)
    if not sections:
        sections = split_sections(lines)

    data = {
        "title": title,
        "contract_type": contract_type,
        "effective_date": eff_date if eff_date else None,
        "sections": []
    }
    for s in sections:
        sec = {"title": s["title"], "number": s["number"] if s["number"] is not None else None, "clauses": []}
        for c in s["clauses"]:
            sec["clauses"].append({"text": c["text"], "label": c["label"] if c["label"] else "", "index": c["index"]})
        data["sections"].append(sec)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

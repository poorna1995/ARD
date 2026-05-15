from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
from collections import Counter
from pptx import Presentation
from pptx.util import Pt
from docx import Document
import re
import numpy as np
import pytesseract
import assemblyai as aai
import zipfile
from agent.tools.decorator import tool


def _repo_root() -> Path:
    """Package layout: agent/tools/readfile.py -> repo root is parents[2]."""
    return Path(__file__).resolve().parents[2]


def _gaia_file_search_roots() -> list[Path]:
    """
    Ordered directories where GAIA attachments may live.
    Override or extend with GAIA_FILES_DIRS (POSIX PATH separator ':').
    """
    repo = _repo_root()
    roots = [
        repo / "datasets" / "gaia_files" / "2023" / "validation",
        repo / "datasets" / "gaia_files" / "2023" / "test",
        repo / "datasets" / "gaia_files",
    ]
    extra = os.environ.get("GAIA_FILES_DIRS", "").strip()
    if extra:
        for part in extra.split(":"):
            p = Path(part.strip()).expanduser()
            if p.is_dir():
                roots.insert(0, p.resolve())
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        try:
            rp = r.resolve()
        except OSError:
            continue
        if rp not in seen and rp.is_dir():
            seen.add(rp)
            out.append(rp)
    return out


def _resolve_existing_path(filename: str) -> Path | None:
    """Resolve filename or basename against GAIA roots and shallow recursive search."""
    raw = (filename or "").strip().strip("\"'")
    if not raw:
        return None

    candidates: list[Path] = []
    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return p.resolve()
    candidates.append(Path(raw))

    base = p.name
    if base and base != raw:
        candidates.append(Path(base))

    for c in candidates:
        if c.is_file():
            try:
                return c.resolve()
            except OSError:
                continue

    roots = _gaia_file_search_roots()
    for root in roots:
        direct = root / base if base else root / raw
        if direct.is_file():
            return direct.resolve()

    # Last resort: find basename anywhere under datasets/gaia_files (small tree).
    gaia_top = _repo_root() / "datasets" / "gaia_files"
    if base and gaia_top.is_dir():
        try:
            for hit in gaia_top.rglob(base):
                if hit.is_file():
                    return hit.resolve()
        except OSError:
            pass

    return None


@tool(
    "read_file",
    (
        "Read the contents of an uploaded file by filename. "
        "Supports: .txt, .csv, .xlsx/.xls, .pdf, .docx, .pptx, "
        ".png/.jpg/.jpeg (OCR), .zip. "
        "Input: filename string."
    ),
)
def read_file(filename: str) -> str:
    raw_input = (filename or "").strip()
    resolved = _resolve_existing_path(raw_input)
    if resolved is None:
        roots_hint = "; ".join(str(r) for r in _gaia_file_search_roots()[:5])
        base_hint = Path(raw_input.strip("\"'")).name if raw_input else ""
        return (
            f"File not found: {raw_input}\n"
            f"Hint: copy the exact basename after 'Attached file name for reference' "
            f"(example basename: {base_hint}). Attachments are under datasets/gaia_files/. "
            f"Searched directories: {roots_hint or '(none)'}"
        )

    path = str(resolved)
    filename = resolved.name
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext in {".xlsx", ".xls"}:

            sheets: dict = pd.read_excel(
                path,
                sheet_name=None,
                engine="openpyxl",
                dtype=str,
                na_filter=False,
            )

            parts = []

            for sheet_name, df in sheets.items():
                df = (
                    df.dropna(axis=0, how="all")
                    .dropna(axis=1, how="all")
                    .reset_index(drop=True)
                )
                df = df.apply(lambda col: col.str.strip().str.replace(r"\s+", " ", regex=True))

                section = [f"## Sheet: {sheet_name}\n"]

                if df.empty:
                    section.append("_No data_")
                else:
                    # Detect "category rows" — first cell has a value, all other cells empty
                    # Split the df into labeled sub-tables around those rows
                    first_col = df.columns[0]
                    other_cols = df.columns[1:]

                    blocks = []
                    current_category = None
                    current_rows = []

                    for _, row in df.iterrows():
                        is_category_row = (
                            row[first_col].strip() != "" and
                            all(row[c].strip() == "" for c in other_cols)
                        )
                        if is_category_row:
                            # Save previous block
                            if current_rows:
                                blocks.append((current_category, pd.DataFrame(current_rows)))
                            current_category = row[first_col].strip()
                            current_rows = []
                        else:
                            current_rows.append(row)

                    # Save last block
                    if current_rows:
                        blocks.append((current_category, pd.DataFrame(current_rows)))

                    if blocks:
                        for category, block_df in blocks:
                            block_df = block_df.reset_index(drop=True)
                            if category:
                                section.append(f"### Category: {category}\n")
                            section.append(block_df.to_markdown(index=False))
                            section.append("")  # blank line between blocks
                    else:
                        section.append(df.to_markdown(index=False))

                parts.append("\n".join(section))

            return "\n\n---\n\n".join(parts)[:8000]
        if ext == ".csv":

            df = pd.read_csv(path)

            # Drop rows where ALL values are NaN
            df = df.dropna(how="all")

            # Replace remaining NaN with empty string so table stays clean
            df = df.fillna("")

            # Summarise large datasets instead of dumping every row
            total_rows = len(df)
            max_rows   = 50  # LLM context sweet spot

            summary = (
                f"**Dataset:** {total_rows} rows × {len(df.columns)} columns\n"
                f"**Columns:** {', '.join(df.columns.tolist())}\n\n"
            )

            if total_rows > max_rows:
                preview = df.head(max_rows)
                summary += f"_(Showing first {max_rows} of {total_rows} rows)_\n\n"
            else:
                preview = df

            return (summary + preview.to_markdown(index=False))[:8000]

        if ext == ".pdf":

            def _is_category_row(row: list) -> bool:
                """Row where only the first cell has content — treat as a section header."""
                if not row or row[0] is None or str(row[0]).strip() == "":
                    return False
                return all((cell is None or str(cell).strip() == "") for cell in row[1:])

            def _table_to_markdown(table: list[list]) -> str:
                """Convert a pdfplumber raw table to a clean markdown string."""
                # Sanitize: replace None with ""
                clean = [[str(c).strip() if c is not None else "" for c in row] for row in table]

                if not clean:
                    return ""

                # First row is the header
                header = clean[0]
                rows   = clean[1:]

                output_lines = []
                current_rows = []

                for row in rows:
                    if _is_category_row(row):
                        # Flush buffered rows as a markdown table
                        if current_rows:
                            df = pd.DataFrame(current_rows, columns=header)
                            df = df.loc[~(df == "").all(axis=1)]  # drop blank rows
                            output_lines.append(df.to_markdown(index=False))
                            output_lines.append("")
                            current_rows = []
                        # Emit section header
                        output_lines.append(f"### {row[0]}")
                        output_lines.append("")
                    else:
                        current_rows.append(row)

                # Flush remaining rows
                if current_rows:
                    df = pd.DataFrame(current_rows, columns=header)
                    df = df.loc[~(df == "").all(axis=1)]
                    output_lines.append(df.to_markdown(index=False))

                return "\n".join(output_lines)

            pages_output = []

            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_lines = [f"## Page {i + 1}"]

                    tables    = page.extract_tables()
                    table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

                    if tables:
                        # ── Has tables: render each as markdown ──────────────────
                        for table in tables:
                            md = _table_to_markdown(table)
                            if md:
                                page_lines.append(md)
                    else:
                        # ── No tables: extract plain text ─────────────────────────
                        text = page.extract_text(layout=True)
                        if text and text.strip():
                            page_lines.append(text.strip())

                    pages_output.append("\n\n".join(page_lines))

                    # Stop early if we have enough context
                    if sum(len(p) for p in pages_output) > 7000:
                        pages_output.append("_...truncated..._")
                        break

            result = "\n\n---\n\n".join(pages_output)
            return result if result.strip() else "No extractable text in PDF."

        if ext == ".docx":
            doc   = Document(path)
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    seen = []
                    for c in row.cells:
                        if c.text.strip() not in seen:
                            seen.append(c.text.strip())
                    paras.append(" | ".join(seen))
            return "\n".join(paras)[:4000]

        if ext == ".pptx":

            prs    = Presentation(path)
            slides = []

            for i, slide in enumerate(prs.slides, start=1):
                slide_lines = [f"## Slide {i}"]

                # Include slide layout name as context (e.g. "Title Slide", "Two Content")
                layout_name = slide.slide_layout.name if slide.slide_layout else ""
                if layout_name:
                    slide_lines.append(f"_Layout: {layout_name}_")

                for shape in slide.shapes:
                    # ── Text shapes ───────────────────────────────────────────
                    if shape.has_text_frame:
                        shape_name = shape.name.lower()

                        for para in shape.text_frame.paragraphs:
                            line = para.text.strip()
                            if not line:
                                continue

                            # Detect heading vs body by font size or placeholder type
                            is_title = (
                                "title" in shape_name or
                                (para.runs and para.runs[0].font.size and
                                para.runs[0].font.size >= Pt(24))
                            )
                            is_bold = (
                                para.runs and
                                para.runs[0].font.bold
                            )

                            if is_title:
                                slide_lines.append(f"### {line}")
                            elif is_bold:
                                slide_lines.append(f"**{line}**")
                            else:
                                # Respect indentation level as bullet depth
                                level  = para.level or 0
                                indent = "  " * level
                                slide_lines.append(f"{indent}- {line}")

                    # ── Tables inside slides ──────────────────────────────────
                    elif shape.has_table:
                        table  = shape.table
                        rows   = []
                        for row in table.rows:
                            rows.append([cell.text.strip() for cell in row.cells])

                        if rows:
                            # First row = header
                            header    = rows[0]
                            separator = ["---"] * len(header)
                            md_rows   = [header, separator] + rows[1:]
                            slide_lines.append(
                                "\n".join("| " + " | ".join(r) + " |" for r in md_rows)
                            )

                if len(slide_lines) > 1:   # skip slides with only the header line
                    slides.append("\n\n".join(slide_lines))

            return "\n\n---\n\n".join(slides)[:8000] if slides else "No text found in presentation."
        
        if ext in {".png", ".jpg", ".jpeg"}:  # OCR for images


            img = Image.open(path).convert("RGBA")
            arr = np.array(img)
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

            red_mask   = (r > 150) & (g < 100) & (b < 100)
            green_mask = (r > 120) & (g > 150) & (b < 80) & (g > r - 60)
            blue_mask  = (r < 100) & (g < 100) & (b > 150)

            try:
                

                # ── Improved preprocessing for small stacked fractions ──
                scale = 4  # increased from 3 for better small-text OCR
                big = img.resize((img.width * scale, img.height * scale), Image.LANCZOS).convert("RGB")

                # Sharpen + boost contrast to help OCR on tiny numerals
                big = ImageEnhance.Contrast(big).enhance(2.0)
                big = ImageEnhance.Sharpness(big).enhance(2.5)
                big = big.filter(ImageFilter.SHARPEN)

                # Run OCR twice with different PSM modes and merge results
                def run_ocr(image, psm):
                    timeout_sec = int(os.environ.get("READFILE_OCR_TIMEOUT_SEC", "25"))
                    try:
                        return pytesseract.image_to_data(
                            image,
                            config=f"--psm {psm} --oem 3",
                            output_type=pytesseract.Output.DICT,
                            timeout=timeout_sec,
                        )
                    except RuntimeError:
                        # Tesseract timeout: skip this OCR pass so one hard image
                        # does not block the entire benchmark run.
                        return {
                            "text": [], "conf": [], "left": [], "top": [], "width": [], "height": [],
                            "block_num": [], "par_num": [], "line_num": []
                        }

                def collect_tokens(data, arr, scale, red_mask, green_mask, blue_mask, conf_thresh=25):
                    tokens = []
                    for i, text in enumerate(data["text"]):
                        text = text.strip()
                        if not text or int(data["conf"][i]) < conf_thresh:
                            continue
                        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        oy1 = max(0, y // scale);  oy2 = min(arr.shape[0], (y + h) // scale)
                        ox1 = max(0, x // scale);  ox2 = min(arr.shape[1], (x + w) // scale)
                        color_tag = ""
                        if arr[oy1:oy2, ox1:ox2].size > 0:
                            if   red_mask  [oy1:oy2, ox1:ox2].mean() > 0.2: color_tag = "RED"
                            elif green_mask[oy1:oy2, ox1:ox2].mean() > 0.2: color_tag = "GREEN"
                            elif blue_mask [oy1:oy2, ox1:ox2].mean() > 0.2: color_tag = "BLUE"
                        tokens.append({
                            "text": text,
                            "x": x, "y": y, "w": w, "h": h,
                            "cx": x + w / 2,
                            "cy": y + h / 2,
                            "color": color_tag,
                            "line_key": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
                            "index": i,
                            "conf": int(data["conf"][i]),
                        })
                    return tokens

                data6 = run_ocr(big, psm=6)
                data11 = run_ocr(big, psm=11)

                tokens6  = collect_tokens(data6,  arr, scale, red_mask, green_mask, blue_mask)
                tokens11 = collect_tokens(data11, arr, scale, red_mask, green_mask, blue_mask)

                # Deduplicate: merge token lists, drop near-duplicates by position
                def dedupe_tokens(primary, secondary, pos_tol=20):
                    seen = [(t["cx"], t["cy"]) for t in primary]
                    merged = list(primary)
                    for t in secondary:
                        if not any(abs(t["cx"]-sx) < pos_tol and abs(t["cy"]-sy) < pos_tol
                                for sx, sy in seen):
                            # Reindex to avoid collision
                            t = dict(t)
                            t["index"] += 100000
                            merged.append(t)
                            seen.append((t["cx"], t["cy"]))
                    return merged

                all_tokens = dedupe_tokens(tokens6, tokens11)

                # ── Improved fraction pair detection ──
                def is_numeric(s):
                    try:
                        float(s.replace(",", ""))
                        return True
                    except ValueError:
                        return False

                fraction_map = {}
                used_indices = set()
                numeric_tokens = [t for t in all_tokens if is_numeric(t["text"])]

                for i, ti in enumerate(numeric_tokens):
                    if ti["index"] in used_indices:
                        continue
                    best = None
                    best_gap = None

                    for j, tj in enumerate(numeric_tokens):
                        if i == j or tj["index"] in used_indices:
                            continue

                        # ── FIXED: use centers for both overlap and gap ──
                        cx_diff = abs(ti["cx"] - tj["cx"])
                        avg_w   = (ti["w"] + tj["w"]) / 2

                        # More lenient horizontal alignment (was 0.65× max_w)
                        x_overlap = cx_diff < avg_w * 1.1

                        # Vertical gap between centers (more reliable than tops)
                        cy_gap  = abs(ti["cy"] - tj["cy"])
                        avg_h   = (ti["h"] + tj["h"]) / 2

                        # Expect centers to be between 0.6× and 6× avg char height apart
                        y_stacked = avg_h * 0.6 < cy_gap < avg_h * 6.0

                        # Must NOT share the same OCR line
                        same_line = ti["line_key"] == tj["line_key"]

                        if x_overlap and y_stacked and not same_line:
                            if best is None or cy_gap < best_gap:
                                best = tj
                                best_gap = cy_gap

                    if best is not None:
                        if ti["cy"] < best["cy"]:
                            numerator, denominator = ti, best
                        else:
                            numerator, denominator = best, ti

                        frac_str  = f"{numerator['text']}/{denominator['text']}"
                        color_tag = numerator["color"] or denominator["color"]
                        frac_display = f"[{color_tag}]{frac_str}" if color_tag else frac_str

                        fraction_map[numerator["index"]]   = frac_display
                        fraction_map[denominator["index"]] = None
                        used_indices.add(numerator["index"])
                        used_indices.add(denominator["index"])

                # ── Group tokens by line ──
                lines = {}
                for token in all_tokens:
                    idx = token["index"]
                    key = token["line_key"]
                    if idx in fraction_map:
                        val = fraction_map[idx]
                        if val is None:
                            continue
                        lines.setdefault(key, []).append((val, token["color"]))
                    else:
                        display = f"[{token['color']}]{token['text']}" if token["color"] else token["text"]
                        lines.setdefault(key, []).append((display, token["color"]))

                # ── Sort lines top-to-bottom by average Y ──
                def line_y(key_tokens):
                    key, tokens = key_tokens
                    # find representative token
                    for t in all_tokens:
                        if t["line_key"] == key:
                            return t["y"]
                    return 0

                sorted_lines = sorted(lines.items(), key=line_y)

                red_numbers, green_numbers, blue_numbers = [], [], []
                line_texts = []

                for key, tokens in sorted_lines:
                    parts = []
                    for token_display, tag in tokens:
                        parts.append(token_display)
                        raw = token_display.replace(f"[{tag}]", "") if tag else token_display
                        # Handle fraction strings like "6/8"
                        segments = raw.split("/")
                        for seg in segments:
                            try:
                                num = float(seg.replace(",", ""))
                                if   tag == "RED":   red_numbers.append(num)
                                elif tag == "GREEN": green_numbers.append(num)
                                elif tag == "BLUE":  blue_numbers.append(num)
                            except ValueError:
                                pass
                    line_texts.append(" ".join(parts))

                full_text = "\n".join(line_texts).strip()

                # ── Dominant colors ──
                pixels    = arr[:,:,:3].reshape(-1, 3)
                not_white = ~((pixels[:,0]>240)&(pixels[:,1]>240)&(pixels[:,2]>240))
                not_black = ~((pixels[:,0]<15) &(pixels[:,1]<15) &(pixels[:,2]<15))
                filtered  = pixels[not_white & not_black]
                top_colors    = Counter(map(tuple, (filtered // 32 * 32).tolist())).most_common(5) if len(filtered) else []
                color_summary = ", ".join(f"RGB{c}" for c, _ in top_colors)

                # ── Image type heuristic ──
                unique_colors = len(set(map(tuple, pixels.tolist())))
                dark_ratio    = ((r < 50) & (g < 50) & (b < 50)).sum() / (arr.shape[0] * arr.shape[1])
                color_variety = len(set(map(tuple, (filtered // 64 * 64).tolist()))) if len(filtered) else 0

                if   unique_colors > 50000:                       img_type = "photograph or complex diagram"
                elif dark_ratio > 0.05 and len(line_texts) > 10: img_type = "text document or screenshot"
                elif len(line_texts) < 5  and color_variety > 20: img_type = "chart or graph"
                elif len(line_texts) > 5  and color_variety < 10: img_type = "table or structured data"
                else:                                              img_type = "mixed content (text + visuals)"

                return "\n".join([
                    "[IMAGE STRUCTURE]",
                    f"Type           : {img_type}",
                    f"Size           : {img.width}x{img.height} px",
                    f"Dominant Colors: {color_summary or 'N/A'}",
                    "",
                    "[TEXT CONTENT]",
                    full_text if full_text else "(no text detected)",
                    "",
                    "[COLORED NUMBERS]",
                    f"Red   : {red_numbers   if red_numbers   else 'none'}",
                    f"Green : {green_numbers if green_numbers else 'none'}",
                    f"Blue  : {blue_numbers  if blue_numbers  else 'none'}",
                ])

            except ImportError:
                return "pytesseract not installed — cannot extract image content."
       

        if ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            # AssemblyAI-only transcription path.
            assembly_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
            if not assembly_key:
                return (
                    "AssemblyAI key missing. Set ASSEMBLYAI_API_KEY in environment "
                    "to transcribe audio files."
                )
            try:
                aai.settings.api_key = assembly_key
                transcriber = aai.Transcriber()
                tx = transcriber.transcribe(path)
            except Exception as e:
                return f"AssemblyAI transcription request failed: {e}"

            status = str(getattr(tx, "status", "") or "").lower()
            if status == "error" or getattr(tx, "error", None):
                err = getattr(tx, "error", None) or "unknown transcription error"
                return f"AssemblyAI transcription failed: {err}"

            transcript = (getattr(tx, "text", "") or "").strip()
            if not transcript:
                return "AssemblyAI transcription returned empty text."

            language = str(getattr(tx, "language_code", "unknown") or "unknown")

            # AssemblyAI sentence timestamps are in ms. Fall back to empty if unavailable.
            raw_sentences = getattr(tx, "sentences", None)
            segments = []
            if raw_sentences:
                for s in raw_sentences:
                    start = float((getattr(s, "start", 0) or 0) / 1000.0)
                    end = float((getattr(s, "end", 0) or 0) / 1000.0)
                    text = (getattr(s, "text", "") or "").strip()
                    if text:
                        segments.append({"start": start, "end": end, "text": text})

            # If sentence timestamps are unavailable, synthesize one segment.
            if not segments:
                audio_dur_ms = getattr(tx, "audio_duration", 0) or 0
                dur_s = float(audio_dur_ms) / 1000.0 if audio_dur_ms else 0.0
                segments = [{"start": 0.0, "end": dur_s, "text": transcript}]

            # Build structured markdown output for LLM
            lines = [
                f"## Audio Transcript",
                f"**Transcription Source:** assemblyai",
                f"**Detected Language:** {language}",
                f"**Duration:** {segments[-1]['end']:.1f}s" if segments else "",
                "",
                "### Full Transcript",
                transcript,
                "",
                "### Timestamped Segments",
            ]

            for seg in segments:
                start = seg["start"]
                end   = seg["end"]
                text  = seg["text"].strip()
                lines.append(f"- **[{start:.1f}s → {end:.1f}s]** {text}")

            return "\n".join(lines)[:8000]
        if ext == ".zip":
            
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                lines = [f"ZIP archive – {len(names)} file(s):"]
                for name in names[:30]:
                    info = zf.getinfo(name)
                    lines.append(f"  {name:<45s}  {info.file_size:>10,} bytes")
                if len(names) > 30:
                    lines.append(f"  … and {len(names) - 30} more file(s).")
                extracted = []
                for name in names:
                    inner_ext = os.path.splitext(name)[1].lower()
                    info      = zf.getinfo(name)
                    if inner_ext in {".txt", ".csv", ".json", ".md", ".py"} and info.file_size < 50_000:
                        try:
                            content = zf.read(name).decode("utf-8", errors="ignore")
                            extracted.append(f"\n── {name} ──\n{content[:1000]}")
                        except Exception:
                            pass
                result = "\n".join(lines)
                if extracted:
                    result += "\n\nExtracted text files:" + "".join(extracted)
                return result[:4000]

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:4000]

    except ImportError as e:
        return f"Missing library to read '{ext}' files: {e}"
    except Exception as e:
        return f"Failed to read file '{filename}': {e}"


import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF


# =========================================================
# CONFIG
# =========================================================
RAW_PDF_ROOT = Path("data/raw_corpus/downloads")
OUTPUT_DIR = Path("data/corpus_pipeline/profiles")
PER_PDF_JSONL = OUTPUT_DIR / "per_pdf_profiles.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"

TEXT_CHAR_EMPTY_THRESHOLD = 40
TEXT_HEAVY_AVG_CHARS_THRESHOLD = 1200
SCAN_HEAVY_IMAGE_ONLY_RATIO_THRESHOLD = 0.60
SCAN_HEAVY_EMPTY_RATIO_THRESHOLD = 0.50

# heuristic weights
W_SCAN_IMAGE_ONLY = 0.45
W_SCAN_EMPTY = 0.30
W_SCAN_LOW_TEXT = 0.25

W_TABLE_SHORT_LINE = 0.28
W_TABLE_NUMERIC_LINE = 0.28
W_TABLE_DRAWING = 0.22
W_TABLE_DIGIT_RATIO = 0.22

W_FORM_KEYVAL = 0.34
W_FORM_BLANKS = 0.33
W_FORM_SHORT_LABEL = 0.33


# =========================================================
# HELPERS
# =========================================================
def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def normalize_score(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp((x - lo) / (hi - lo))


def clean_text_basic(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def page_text_stats(text: str) -> Dict[str, float]:
    raw = text or ""
    text = raw.strip()

    non_empty_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    words = re.findall(r"\b\w+\b", raw)
    char_count = len(text)
    word_count = len(words)
    line_count = len(non_empty_lines)

    short_line_count = 0
    numeric_line_count = 0
    keyval_line_count = 0
    blank_form_line_count = 0
    sentence_like_line_count = 0

    for ln in non_empty_lines:
        line_words = re.findall(r"\b\w+\b", ln)
        word_len = len(line_words)

        if word_len <= 6:
            short_line_count += 1

        if re.search(r"\d", ln):
            numeric_line_count += 1

        if ":" in ln and word_len <= 10:
            keyval_line_count += 1

        if re.search(r"(_{3,}|\.{3,}|\[\s*\]|\(\s*\)|:{2,})", ln):
            blank_form_line_count += 1

        if re.search(r"[.!?]\s*$", ln) or word_len >= 12:
            sentence_like_line_count += 1

    digit_count = len(re.findall(r"\d", raw))
    alpha_count = len(re.findall(r"[A-Za-z]", raw))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "short_line_ratio": safe_div(short_line_count, line_count),
        "numeric_line_ratio": safe_div(numeric_line_count, line_count),
        "keyval_line_ratio": safe_div(keyval_line_count, line_count),
        "blank_form_line_ratio": safe_div(blank_form_line_count, line_count),
        "sentence_like_line_ratio": safe_div(sentence_like_line_count, line_count),
        "digit_ratio": safe_div(digit_count, max(1, digit_count + alpha_count)),
    }


def classify_layout(
    page_count: int,
    avg_chars_per_page: float,
    empty_page_ratio: float,
    image_only_ratio: float,
    scan_likelihood: float,
    table_likelihood: float,
    form_likelihood: float,
) -> str:
    if page_count == 0:
        return "invalid"

    if scan_likelihood >= 0.70:
        return "scan_heavy"

    strong_modes = []
    if avg_chars_per_page >= TEXT_HEAVY_AVG_CHARS_THRESHOLD and scan_likelihood < 0.45:
        strong_modes.append("text_heavy")
    if table_likelihood >= 0.60:
        strong_modes.append("table_heavy")
    if form_likelihood >= 0.60:
        strong_modes.append("form_heavy")

    if len(strong_modes) >= 2:
        return "mixed"

    if len(strong_modes) == 1:
        return strong_modes[0]

    if image_only_ratio >= SCAN_HEAVY_IMAGE_ONLY_RATIO_THRESHOLD:
        return "scan_heavy"

    if empty_page_ratio >= SCAN_HEAVY_EMPTY_RATIO_THRESHOLD and avg_chars_per_page < 300:
        return "scan_heavy"

    if table_likelihood >= 0.45 and form_likelihood >= 0.45:
        return "mixed"

    if avg_chars_per_page >= 800:
        return "text_heavy"

    if table_likelihood >= 0.45:
        return "table_heavy"

    if form_likelihood >= 0.45:
        return "form_heavy"

    return "mixed" if (table_likelihood >= 0.30 and form_likelihood >= 0.30) else "unknown"


# =========================================================
# CORE INVESTIGATION
# =========================================================
def investigate_pdf(pdf_path: Path) -> dict:
    result = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "source_name": pdf_path.parent.name,
        "status": "ok",
    }

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        result.update(
            {
                "status": "failed",
                "error": f"open_failed: {e}",
                "page_count": 0,
            }
        )
        return result

    try:
        page_count = len(doc)
        if page_count == 0:
            result.update(
                {
                    "status": "failed",
                    "error": "empty_pdf",
                    "page_count": 0,
                }
            )
            return result

        total_chars = 0
        total_words = 0
        total_lines = 0
        total_images = 0
        total_drawings = 0

        empty_pages = 0
        image_only_pages = 0

        per_page_char_counts = []
        per_page_word_counts = []

        short_line_ratios = []
        numeric_line_ratios = []
        keyval_line_ratios = []
        blank_form_line_ratios = []
        sentence_like_line_ratios = []
        digit_ratios = []

        for page_index in range(page_count):
            page = doc[page_index]
            text = page.get_text("text") or ""
            text = clean_text_basic(text)

            stats = page_text_stats(text)

            char_count = stats["char_count"]
            word_count = stats["word_count"]
            line_count = stats["line_count"]

            image_count = len(page.get_images(full=True))
            drawing_count = len(page.get_drawings())

            total_chars += char_count
            total_words += word_count
            total_lines += line_count
            total_images += image_count
            total_drawings += drawing_count

            per_page_char_counts.append(char_count)
            per_page_word_counts.append(word_count)

            short_line_ratios.append(stats["short_line_ratio"])
            numeric_line_ratios.append(stats["numeric_line_ratio"])
            keyval_line_ratios.append(stats["keyval_line_ratio"])
            blank_form_line_ratios.append(stats["blank_form_line_ratio"])
            sentence_like_line_ratios.append(stats["sentence_like_line_ratio"])
            digit_ratios.append(stats["digit_ratio"])

            if char_count < TEXT_CHAR_EMPTY_THRESHOLD:
                empty_pages += 1

            if char_count < TEXT_CHAR_EMPTY_THRESHOLD and image_count > 0:
                image_only_pages += 1

        avg_chars_per_page = safe_div(total_chars, page_count)
        avg_words_per_page = safe_div(total_words, page_count)
        empty_page_ratio = safe_div(empty_pages, page_count)
        image_only_ratio = safe_div(image_only_pages, page_count)

        avg_short_line_ratio = statistics.mean(short_line_ratios) if short_line_ratios else 0.0
        avg_numeric_line_ratio = statistics.mean(numeric_line_ratios) if numeric_line_ratios else 0.0
        avg_keyval_line_ratio = statistics.mean(keyval_line_ratios) if keyval_line_ratios else 0.0
        avg_blank_form_line_ratio = statistics.mean(blank_form_line_ratios) if blank_form_line_ratios else 0.0
        avg_sentence_like_line_ratio = statistics.mean(sentence_like_line_ratios) if sentence_like_line_ratios else 0.0
        avg_digit_ratio = statistics.mean(digit_ratios) if digit_ratios else 0.0

        drawings_per_page = safe_div(total_drawings, page_count)

        low_text_score = 1.0 - normalize_score(avg_chars_per_page, 200, 1400)
        scan_likelihood = clamp(
            W_SCAN_IMAGE_ONLY * image_only_ratio
            + W_SCAN_EMPTY * empty_page_ratio
            + W_SCAN_LOW_TEXT * low_text_score
        )

        drawing_signal = normalize_score(drawings_per_page, 1, 25)
        table_likelihood = clamp(
            W_TABLE_SHORT_LINE * avg_short_line_ratio
            + W_TABLE_NUMERIC_LINE * avg_numeric_line_ratio
            + W_TABLE_DRAWING * drawing_signal
            + W_TABLE_DIGIT_RATIO * avg_digit_ratio
            - 0.10 * avg_sentence_like_line_ratio
        )

        form_likelihood = clamp(
            W_FORM_KEYVAL * avg_keyval_line_ratio
            + W_FORM_BLANKS * avg_blank_form_line_ratio
            + W_FORM_SHORT_LABEL * avg_short_line_ratio
            - 0.12 * avg_sentence_like_line_ratio
        )

        layout_class = classify_layout(
            page_count=page_count,
            avg_chars_per_page=avg_chars_per_page,
            empty_page_ratio=empty_page_ratio,
            image_only_ratio=image_only_ratio,
            scan_likelihood=scan_likelihood,
            table_likelihood=table_likelihood,
            form_likelihood=form_likelihood,
        )

        result.update(
            {
                "page_count": page_count,
                "file_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else None,
                "total_chars": total_chars,
                "total_words": total_words,
                "total_lines": total_lines,
                "avg_chars_per_page": round(avg_chars_per_page, 2),
                "avg_words_per_page": round(avg_words_per_page, 2),
                "empty_pages": empty_pages,
                "empty_page_ratio": round(empty_page_ratio, 4),
                "image_only_pages": image_only_pages,
                "image_only_ratio": round(image_only_ratio, 4),
                "total_images": total_images,
                "total_drawings": total_drawings,
                "avg_short_line_ratio": round(avg_short_line_ratio, 4),
                "avg_numeric_line_ratio": round(avg_numeric_line_ratio, 4),
                "avg_keyval_line_ratio": round(avg_keyval_line_ratio, 4),
                "avg_blank_form_line_ratio": round(avg_blank_form_line_ratio, 4),
                "avg_sentence_like_line_ratio": round(avg_sentence_like_line_ratio, 4),
                "avg_digit_ratio": round(avg_digit_ratio, 4),
                "scan_likelihood": round(scan_likelihood, 4),
                "table_likelihood": round(table_likelihood, 4),
                "form_likelihood": round(form_likelihood, 4),
                "layout_class": layout_class,
            }
        )

        return result

    except Exception as e:
        result.update(
            {
                "status": "failed",
                "error": f"analysis_failed: {e}",
                "page_count": len(doc) if doc else 0,
            }
        )
        return result

    finally:
        doc.close()


# =========================================================
# SUMMARY
# =========================================================
def build_summary(rows: List[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    failed_rows = [r for r in rows if r.get("status") != "ok"]

    layout_counts = Counter(r.get("layout_class", "unknown") for r in ok_rows)
    source_counts = Counter(r.get("source_name", "unknown") for r in ok_rows)

    total_pdfs = len(rows)
    total_ok = len(ok_rows)
    total_failed = len(failed_rows)

    total_pages = sum(r.get("page_count", 0) for r in ok_rows)
    total_chars = sum(r.get("total_chars", 0) for r in ok_rows)

    avg_pages_per_pdf = safe_div(total_pages, total_ok)
    avg_chars_per_pdf = safe_div(total_chars, total_ok)

    top_biggest = sorted(
        ok_rows,
        key=lambda x: x.get("page_count", 0),
        reverse=True
    )[:10]

    summary = {
        "root_scanned": str(RAW_PDF_ROOT),
        "total_pdfs_found": total_pdfs,
        "total_ok": total_ok,
        "total_failed": total_failed,
        "total_pages": total_pages,
        "total_chars": total_chars,
        "avg_pages_per_pdf": round(avg_pages_per_pdf, 2),
        "avg_chars_per_pdf": round(avg_chars_per_pdf, 2),
        "layout_class_counts": dict(layout_counts),
        "source_counts": dict(source_counts),
        "top_10_biggest_pdfs": [
            {
                "file_name": r["file_name"],
                "source_name": r["source_name"],
                "page_count": r.get("page_count", 0),
                "layout_class": r.get("layout_class", "unknown"),
            }
            for r in top_biggest
        ],
    }
    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 80)
    print("PDF INVESTIGATION SUMMARY")
    print("=" * 80)
    print(f"Root scanned         : {summary['root_scanned']}")
    print(f"Total PDFs found     : {summary['total_pdfs_found']}")
    print(f"Successfully profiled: {summary['total_ok']}")
    print(f"Failed               : {summary['total_failed']}")
    print(f"Total pages          : {summary['total_pages']}")
    print(f"Avg pages / PDF      : {summary['avg_pages_per_pdf']}")
    print(f"Avg chars / PDF      : {summary['avg_chars_per_pdf']}")

    print("\nLayout class counts:")
    for k, v in sorted(summary["layout_class_counts"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {k:12s}: {v}")

    print("\nSource counts:")
    for k, v in sorted(summary["source_counts"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {k:12s}: {v}")

    print("\nTop 10 biggest PDFs:")
    for item in summary["top_10_biggest_pdfs"]:
        print(
            f"  - {item['file_name']} | "
            f"source={item['source_name']} | "
            f"pages={item['page_count']} | "
            f"class={item['layout_class']}"
        )

    print("\nSaved files:")
    print(f"  - {PER_PDF_JSONL}")
    print(f"  - {SUMMARY_JSON}")
    print("=" * 80)


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dirs()

    pdf_paths = sorted(RAW_PDF_ROOT.rglob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found under: {RAW_PDF_ROOT}")
        return

    rows = []
    total = len(pdf_paths)

    print(f"Found {total} PDF(s). Starting investigation...\n")

    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{i}/{total}] Investigating: {pdf_path}")
        row = investigate_pdf(pdf_path)
        rows.append(row)

    write_jsonl(PER_PDF_JSONL, rows)
    summary = build_summary(rows)
    write_json(SUMMARY_JSON, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
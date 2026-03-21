import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF


# =========================================================
# CONFIG
# =========================================================
RAW_ROOT = Path("data/raw_corpus/downloads")
OUTPUT_ROOT = Path("data/corpus_pipeline/language_filter")
EXCLUDED_HINDI_DIR = Path("data/corpus_pipeline/excluded/hindi")

ALL_TAGS_JSONL = OUTPUT_ROOT / "all_language_doc_tags.jsonl"
KEPT_MANIFEST_JSONL = OUTPUT_ROOT / "kept_manifest.jsonl"
MOVED_HINDI_MANIFEST_JSONL = OUTPUT_ROOT / "moved_hindi_manifest.jsonl"

MOVE_FILES = False   # False = copy Hindi PDFs, True = move Hindi PDFs
OVERWRITE_EXISTING = False

# language thresholds
MIN_TEXT_CHARS_FOR_DECISION = 150
DEVANAGARI_DOMINANT_THRESHOLD = 0.45
LATIN_DOMINANT_THRESHOLD = 0.45

MAX_PAGES_TO_SAMPLE = 5


# =========================================================
# SETUP
# =========================================================
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
EXCLUDED_HINDI_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def write_jsonl(path: Path, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(text: str, max_len: int = 150) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\-.]", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "file")[:max_len]


def extract_sample_text(pdf_path: Path, max_pages: int = MAX_PAGES_TO_SAMPLE) -> Tuple[str, int]:
    doc = fitz.open(pdf_path)
    texts = []
    pages_sampled = 0

    try:
        for i in range(min(len(doc), max_pages)):
            text = doc[i].get_text("text") or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                texts.append(text)
            pages_sampled += 1
    finally:
        doc.close()

    return "\n".join(texts), pages_sampled


def compute_script_ratios(text: str) -> Dict[str, float]:
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    digits = len(re.findall(r"\d", text))
    total = max(1, devanagari + latin + digits)

    return {
        "devanagari_count": devanagari,
        "latin_count": latin,
        "digit_count": digits,
        "devanagari_ratio": round(devanagari / total, 4),
        "latin_ratio": round(latin / total, 4),
    }


def detect_language(text: str) -> Tuple[str, float, Dict[str, float]]:
    stats = compute_script_ratios(text)
    devanagari_ratio = stats["devanagari_ratio"]
    latin_ratio = stats["latin_ratio"]
    text_len = len(text)

    if text_len < MIN_TEXT_CHARS_FOR_DECISION:
        return "unknown", 0.2, stats

    if devanagari_ratio >= DEVANAGARI_DOMINANT_THRESHOLD and latin_ratio < 0.35:
        return "hi", round(devanagari_ratio, 4), stats

    if latin_ratio >= LATIN_DOMINANT_THRESHOLD and devanagari_ratio < 0.25:
        return "en", round(latin_ratio, 4), stats

    if devanagari_ratio >= 0.20 and latin_ratio >= 0.20:
        return "mixed_lang", round(max(devanagari_ratio, latin_ratio), 4), stats

    return "unknown", 0.4, stats


def detect_doc_type(pdf_path: Path, text: str) -> str:
    name = pdf_path.name.lower()
    parent = str(pdf_path.parent).lower()
    probe = f"{name}\n{parent}\n{text[:4000].lower()}"

    rules = [
        ("annual_report", [
            "annual report", "annual reports", "yearly report", "annual accounts"
        ]),
        ("guideline", [
            "guideline", "guidelines", "guidance document", "clinical guideline",
            "practice guideline", "standard treatment guideline"
        ]),
        ("clinical_manual", [
            "manual", "clinical manual", "operational manual", "training manual",
            "handbook", "protocol manual"
        ]),
        ("lab_reference", [
            "laboratory", "laboratory manual", "lab manual", "reference range",
            "normal range", "diagnostic criteria", "test interpretation"
        ]),
        ("advisory", [
            "advisory", "public advisory", "health advisory", "recommendation"
        ]),
        ("surveillance_report", [
            "surveillance", "weekly outbreak", "epidemiological", "epidemiology",
            "disease surveillance", "bulletin"
        ]),
        ("research_report", [
            "research report", "technical report", "study report", "working paper"
        ]),
        ("training_material", [
            "training", "module", "participant manual", "trainer guide", "course material"
        ]),
        ("form_template", [
            "form", "application form", "proforma", "template", "annexure", "checklist"
        ]),
        ("book", [
            "book", "downloadable book", "textbook", "monograph"
        ]),
        ("administrative_document", [
            "tender", "notice", "office memorandum", "circular", "minutes", "agenda",
            "recruitment", "vacancy", "procurement"
        ]),
    ]

    for doc_type, keywords in rules:
        for kw in keywords:
            if kw in probe:
                return doc_type

    return "unknown"


def target_hindi_path(pdf_path: Path) -> Path:
    source_folder = safe_name(pdf_path.parent.name)
    out_dir = EXCLUDED_HINDI_DIR / source_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / pdf_path.name


def relocate_hindi_pdf(pdf_path: Path) -> Tuple[str, str]:
    dst = target_hindi_path(pdf_path)

    if dst.exists() and not OVERWRITE_EXISTING:
        return "already_exists", str(dst)

    if MOVE_FILES:
        shutil.move(str(pdf_path), str(dst))
        return "moved", str(dst)
    else:
        shutil.copy2(str(pdf_path), str(dst))
        return "copied", str(dst)


# =========================================================
# MAIN
# =========================================================
def main():
    pdf_paths = sorted(RAW_ROOT.rglob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found under {RAW_ROOT}")
        return

    all_rows = []
    kept_rows = []
    moved_hindi_rows = []

    total = len(pdf_paths)
    print(f"Found {total} PDFs. Starting language filter + doc type tagging...\n")

    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{i}/{total}] Processing: {pdf_path}")

        try:
            sample_text, pages_sampled = extract_sample_text(pdf_path)
            language, confidence, stats = detect_language(sample_text)
            doc_type = detect_doc_type(pdf_path, sample_text)

            row = {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "source_name": pdf_path.parent.name,
                "pages_sampled": pages_sampled,
                "sample_text_chars": len(sample_text),
                "dominant_language": language,
                "language_confidence": confidence,
                "devanagari_ratio": stats["devanagari_ratio"],
                "latin_ratio": stats["latin_ratio"],
                "doc_type": doc_type,
                "action": None,
                "target_path": None,
                "status": "ok",
            }

            if language == "hi":
                action, target_path = relocate_hindi_pdf(pdf_path)
                row["action"] = action
                row["target_path"] = target_path
                moved_hindi_rows.append(row)
                print(f"    -> Hindi detected | doc_type={doc_type} | action={action}")
            else:
                row["action"] = "keep"
                kept_rows.append(row)
                print(f"    -> Keep | language={language} | doc_type={doc_type}")

            all_rows.append(row)

        except Exception as e:
            row = {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "source_name": pdf_path.parent.name,
                "dominant_language": "error",
                "language_confidence": 0.0,
                "doc_type": "unknown",
                "action": "error",
                "target_path": None,
                "status": "failed",
                "error": str(e),
            }
            all_rows.append(row)
            print(f"    -> ERROR: {e}")

    write_jsonl(ALL_TAGS_JSONL, all_rows)
    write_jsonl(KEPT_MANIFEST_JSONL, kept_rows)
    write_jsonl(MOVED_HINDI_MANIFEST_JSONL, moved_hindi_rows)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Total processed      : {len(all_rows)}")
    print(f"Kept                 : {len(kept_rows)}")
    print(f"Hindi moved/copied   : {len(moved_hindi_rows)}")
    print(f"All tags manifest    : {ALL_TAGS_JSONL}")
    print(f"Kept manifest        : {KEPT_MANIFEST_JSONL}")
    print(f"Hindi manifest       : {MOVED_HINDI_MANIFEST_JSONL}")
    print("=" * 80)


if __name__ == "__main__":
    main()
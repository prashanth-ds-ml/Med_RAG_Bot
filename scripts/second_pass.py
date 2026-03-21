import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF


# =========================================================
# CONFIG
# =========================================================
LANG_TAGS_PATH = Path("data/corpus_pipeline/language_filter/all_language_doc_tags.jsonl")
PDF_PROFILES_PATH = Path("data/corpus_pipeline/profiles/per_pdf_profiles.jsonl")

OUTPUT_DIR = Path("data/corpus_pipeline/language_filter_second_pass")
EXCLUDED_HINDI_DIR = Path("data/corpus_pipeline/excluded/hindi_second_pass")
EXCLUDED_OTHER_LANG_DIR = Path("data/corpus_pipeline/excluded/other_language_second_pass")

SECOND_PASS_AUDIT_JSONL = OUTPUT_DIR / "second_pass_audit.jsonl"
NEW_HINDI_JSONL = OUTPUT_DIR / "newly_flagged_hindi.jsonl"
NEW_OTHER_LANG_JSONL = OUTPUT_DIR / "newly_flagged_other_language.jsonl"
UPDATED_KEEP_JSONL = OUTPUT_DIR / "updated_keep_after_second_pass.jsonl"

TARGET_LANGS = {"unknown", "mixed_lang"}
MOVE_FILES = False        # False = copy, True = move
OVERWRITE_EXISTING = False

MIN_TEXT_CHARS = 120
LATIN_DOMINANT_THRESHOLD = 0.60
DEVANAGARI_DOMINANT_THRESHOLD = 0.45
OTHER_SCRIPT_DOMINANT_THRESHOLD = 0.40
MIXED_SCRIPT_THRESHOLD = 0.20


# =========================================================
# SETUP
# =========================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDED_HINDI_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDED_OTHER_LANG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
SCRIPT_PATTERNS = {
    "latin": r"[A-Za-z]",
    "devanagari": r"[\u0900-\u097F]",
    "bengali": r"[\u0980-\u09FF]",
    "gurmukhi": r"[\u0A00-\u0A7F]",
    "gujarati": r"[\u0A80-\u0AFF]",
    "oriya": r"[\u0B00-\u0B7F]",
    "tamil": r"[\u0B80-\u0BFF]",
    "telugu": r"[\u0C00-\u0C7F]",
    "kannada": r"[\u0C80-\u0CFF]",
    "malayalam": r"[\u0D00-\u0D7F]",
}

HINDI_HINT_WORDS = {
    "के", "का", "की", "में", "है", "और", "से", "पर", "यह", "एक",
    "स्वास्थ्य", "भारत", "रोग", "उपचार", "रिपोर्ट", "दिशानिर्देश"
}


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_first_page_text(pdf_path: Path) -> Tuple[str, int]:
    doc = fitz.open(pdf_path)
    try:
        page_count = len(doc)
        if page_count == 0:
            return "", 0
        text = doc[0].get_text("text") or ""
        return clean_text(text), page_count
    finally:
        doc.close()


def script_counts(text: str) -> Dict[str, int]:
    counts = {}
    for name, pattern in SCRIPT_PATTERNS.items():
        counts[name] = len(re.findall(pattern, text))
    return counts


def detect_refined_language(text: str) -> Dict[str, object]:
    counts = script_counts(text)
    total_script_chars = sum(counts.values())
    text_len = len(text)

    ratios = {
        k: round((v / total_script_chars), 4) if total_script_chars else 0.0
        for k, v in counts.items()
    }

    top_script = None
    top_count = 0
    for k, v in counts.items():
        if v > top_count:
            top_script = k
            top_count = v

    hindi_hint_hits = 0
    if text:
        for w in HINDI_HINT_WORDS:
            if w in text:
                hindi_hint_hits += 1

    if text_len < MIN_TEXT_CHARS or total_script_chars < 50:
        return {
            "refined_language": "unknown_low_text",
            "dominant_script": top_script or "none",
            "script_counts": counts,
            "script_ratios": ratios,
            "confidence": 0.20,
            "hindi_hint_hits": hindi_hint_hits,
        }

    latin_ratio = ratios["latin"]
    dev_ratio = ratios["devanagari"]

    non_latin_scripts = {k: v for k, v in ratios.items() if k != "latin"}
    best_non_latin_script = max(non_latin_scripts, key=non_latin_scripts.get)
    best_non_latin_ratio = non_latin_scripts[best_non_latin_script]

    # strong English
    if latin_ratio >= LATIN_DOMINANT_THRESHOLD and best_non_latin_ratio < 0.15:
        return {
            "refined_language": "en",
            "dominant_script": "latin",
            "script_counts": counts,
            "script_ratios": ratios,
            "confidence": round(latin_ratio, 4),
            "hindi_hint_hits": hindi_hint_hits,
        }

    # strong Hindi / Devanagari
    if dev_ratio >= DEVANAGARI_DOMINANT_THRESHOLD:
        label = "hi_devanagari"
        conf = max(dev_ratio, 0.50)
        if hindi_hint_hits >= 2:
            conf = min(0.95, conf + 0.10)

        return {
            "refined_language": label,
            "dominant_script": "devanagari",
            "script_counts": counts,
            "script_ratios": ratios,
            "confidence": round(conf, 4),
            "hindi_hint_hits": hindi_hint_hits,
        }

    # other non-English script dominant
    if best_non_latin_script != "devanagari" and best_non_latin_ratio >= OTHER_SCRIPT_DOMINANT_THRESHOLD:
        return {
            "refined_language": "other_non_english",
            "dominant_script": best_non_latin_script,
            "script_counts": counts,
            "script_ratios": ratios,
            "confidence": round(best_non_latin_ratio, 4),
            "hindi_hint_hits": hindi_hint_hits,
        }

    # mixed
    if latin_ratio >= MIXED_SCRIPT_THRESHOLD and best_non_latin_ratio >= MIXED_SCRIPT_THRESHOLD:
        return {
            "refined_language": "mixed_lang",
            "dominant_script": "mixed",
            "script_counts": counts,
            "script_ratios": ratios,
            "confidence": round(max(latin_ratio, best_non_latin_ratio), 4),
            "hindi_hint_hits": hindi_hint_hits,
        }

    return {
        "refined_language": "unknown",
        "dominant_script": top_script or "none",
        "script_counts": counts,
        "script_ratios": ratios,
        "confidence": 0.35,
        "hindi_hint_hits": hindi_hint_hits,
    }


def target_copy_path(pdf_path: Path, bucket: str) -> Path:
    source_folder = pdf_path.parent.name
    if bucket == "hi_devanagari":
        out_dir = EXCLUDED_HINDI_DIR / source_folder
    else:
        out_dir = EXCLUDED_OTHER_LANG_DIR / source_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / pdf_path.name


def relocate_pdf(pdf_path: Path, bucket: str) -> Tuple[str, str]:
    dst = target_copy_path(pdf_path, bucket)

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
    lang_rows = read_jsonl(LANG_TAGS_PATH)
    profile_rows = read_jsonl(PDF_PROFILES_PATH)
    profile_by_path = {r["file_path"]: r for r in profile_rows}

    if not lang_rows:
        print(f"No input found: {LANG_TAGS_PATH}")
        return

    target_rows = [
        r for r in lang_rows
        if r.get("dominant_language") in TARGET_LANGS and r.get("status") == "ok"
    ]

    print(f"Total language-tag rows         : {len(lang_rows)}")
    print(f"Target rows for second pass     : {len(target_rows)}")
    print(f"Targeting languages             : {sorted(TARGET_LANGS)}")
    print()

    audit_rows = []
    new_hindi_rows = []
    new_other_lang_rows = []
    updated_keep_rows = []

    total = len(target_rows)

    for i, row in enumerate(target_rows, start=1):
        file_path = Path(row["file_path"])
        print(f"[{i}/{total}] Checking first page: {file_path}")

        profile = profile_by_path.get(str(file_path), {})
        layout_class = profile.get("layout_class", "unknown")

        if not file_path.exists():
            out = {
                **row,
                "second_pass_status": "missing_file",
                "layout_class": layout_class,
            }
            audit_rows.append(out)
            print("    -> missing file")
            continue

        try:
            first_page_text, page_count = extract_first_page_text(file_path)
            refined = detect_refined_language(first_page_text)

            out = {
                **row,
                "page_count": page_count,
                "layout_class": layout_class,
                "first_page_text_chars": len(first_page_text),
                "refined_language": refined["refined_language"],
                "dominant_script": refined["dominant_script"],
                "refined_confidence": refined["confidence"],
                "hindi_hint_hits": refined["hindi_hint_hits"],
                "script_ratios": refined["script_ratios"],
                "second_pass_status": "ok",
                "second_pass_action": "keep",
                "second_pass_target_path": None,
            }

            # move/copy newly found Hindi
            if refined["refined_language"] == "hi_devanagari":
                action, dst = relocate_pdf(file_path, "hi_devanagari")
                out["second_pass_action"] = action
                out["second_pass_target_path"] = dst
                new_hindi_rows.append(out)
                print(f"    -> newly flagged Hindi/Devanagari | action={action}")

            # move/copy other non-English
            elif refined["refined_language"] == "other_non_english":
                action, dst = relocate_pdf(file_path, "other_non_english")
                out["second_pass_action"] = action
                out["second_pass_target_path"] = dst
                new_other_lang_rows.append(out)
                print(f"    -> newly flagged other language ({refined['dominant_script']}) | action={action}")

            else:
                updated_keep_rows.append(out)
                print(f"    -> keep | refined_language={refined['refined_language']} | layout={layout_class}")

            audit_rows.append(out)

        except Exception as e:
            out = {
                **row,
                "layout_class": layout_class,
                "second_pass_status": "failed",
                "error": str(e),
            }
            audit_rows.append(out)
            print(f"    -> ERROR: {e}")

    # also keep rows that were never targeted and were already safe
    untouched_keep_rows = [
        r for r in lang_rows
        if not (r.get("dominant_language") in TARGET_LANGS and r.get("status") == "ok")
        and r.get("dominant_language") not in {"hi", "error"}
    ]

    final_keep_rows = untouched_keep_rows + updated_keep_rows

    write_jsonl(SECOND_PASS_AUDIT_JSONL, audit_rows)
    write_jsonl(NEW_HINDI_JSONL, new_hindi_rows)
    write_jsonl(NEW_OTHER_LANG_JSONL, new_other_lang_rows)
    write_jsonl(UPDATED_KEEP_JSONL, final_keep_rows)

    print("\n" + "=" * 90)
    print("SECOND PASS COMPLETE")
    print("=" * 90)
    print(f"Audited target rows                 : {len(audit_rows)}")
    print(f"New Hindi/Devanagari flagged        : {len(new_hindi_rows)}")
    print(f"New other-language flagged          : {len(new_other_lang_rows)}")
    print(f"Updated keep manifest               : {len(final_keep_rows)}")
    print()
    print(f"Saved: {SECOND_PASS_AUDIT_JSONL}")
    print(f"Saved: {NEW_HINDI_JSONL}")
    print(f"Saved: {NEW_OTHER_LANG_JSONL}")
    print(f"Saved: {UPDATED_KEEP_JSONL}")
    print("=" * 90)


if __name__ == "__main__":
    main()
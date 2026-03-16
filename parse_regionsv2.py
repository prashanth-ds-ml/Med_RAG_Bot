import json
import re
from pathlib import Path
from typing import Optional, List

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


# =========================================================
# Config
# =========================================================
PDF_DIR = Path("/home/enma/Projects/Med360_RAG_Bot/data/pdf/ENT")
OUTPUT_ROOT = Path("output")

TEXT_LABELS = {"title", "text", "header", "section_header", "header_footer"}
HEADER_LABELS = {"header", "section_header", "header_footer"}
SKIP_LABELS = {"table", "figure", "flowchart"}

TESSERACT_LANG = "eng"


# =========================================================
# Helpers
# =========================================================
def normalize_filename(text: str, default: str = "document") -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s\-]+", "_", text)
    text = text.strip("_")
    return text or default


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    lines = [line.strip() for line in text.splitlines()]

    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False

    text = "\n".join(cleaned).strip()

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged_paras = []
    for p in paras:
        one_line = re.sub(r"\s*\n\s*", " ", p)
        one_line = re.sub(r"\s+", " ", one_line).strip()
        merged_paras.append(one_line)

    return "\n\n".join(merged_paras).strip()


def image_ocr(image_path: Path) -> str:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=TESSERACT_LANG)
        return clean_text(text)
    except Exception:
        return ""


def get_pdf_path_from_doc_dir(doc_dir: Path) -> Optional[Path]:
    candidate = PDF_DIR / f"{doc_dir.name}.pdf"
    if candidate.exists():
        return candidate
    return None


def load_labels(labels_path: Path) -> List[dict]:
    with open(labels_path, "r", encoding="utf-8") as f:
        regions = json.load(f)

    regions = sorted(regions, key=lambda x: (x.get("page", 1), x.get("reading_order", 999999)))
    return regions


# =========================================================
# Text extraction
# =========================================================
def extract_text_from_pdf_region(
    pdf_path: Path,
    page_num_1_based: int,
    bbox_image,
    page_image_path: Optional[Path],
) -> str:
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num_1_based - 1]
        page_rect = page.rect

        if page_image_path and page_image_path.exists():
            img = Image.open(page_image_path)
            img_w, img_h = img.size
        else:
            img_w, img_h = page_rect.width, page_rect.height

        x1, y1, x2, y2 = bbox_image

        scale_x = page_rect.width / img_w
        scale_y = page_rect.height / img_h

        clip = fitz.Rect(
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        )

        text = page.get_text("text", clip=clip, sort=True)
        doc.close()
        return clean_text(text)
    except Exception:
        return ""


# =========================================================
# Parse one region
# =========================================================
def parse_region(region: dict, pdf_path: Path, page_image_path: Path) -> str:
    label = region["label"]
    page_num = region.get("page", 1)
    bbox = region["bbox"]
    crop_path = Path(region["crop_file"])

    if label in SKIP_LABELS:
        return ""

    text = ""
    if label in TEXT_LABELS:
        text = extract_text_from_pdf_region(pdf_path, page_num, bbox, page_image_path)
        if not text and crop_path.exists():
            text = image_ocr(crop_path)

    if not text:
        return ""

    if label == "title":
        return f"# {text}"

    if label in HEADER_LABELS:
        return f"## {text}"

    return text


# =========================================================
# Document builder
# =========================================================
def build_markdown_for_doc(doc_dir: Path):
    labels_path = doc_dir / "labels.json"
    page_image_path = doc_dir / "page.png"

    if not labels_path.exists():
        print(f"Skipped {doc_dir.name}: labels.json not found")
        return

    pdf_path = get_pdf_path_from_doc_dir(doc_dir)
    if not pdf_path:
        print(f"Skipped {doc_dir.name}: matching PDF not found in {PDF_DIR}")
        return

    regions = load_labels(labels_path)

    title_text = None
    blocks = []

    for region in regions:
        parsed = parse_region(region, pdf_path, page_image_path)
        if not parsed:
            continue

        if region["label"] == "title" and not title_text:
            raw_title = parsed.lstrip("#").strip()
            title_text = clean_text(raw_title)

        blocks.append(parsed.strip())

    if not blocks:
        print(f"Skipped {doc_dir.name}: no parsed content")
        return

    out_name = normalize_filename(title_text or doc_dir.name)
    out_md = doc_dir / f"{out_name}.md"

    final_md = "\n\n".join([b for b in blocks if b.strip()]).strip() + "\n"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(final_md)

    print(f"Saved markdown: {out_md}")


def main():
    if not OUTPUT_ROOT.exists():
        print(f"Output root not found: {OUTPUT_ROOT}")
        return

    doc_dirs = [p for p in OUTPUT_ROOT.iterdir() if p.is_dir()]
    if not doc_dirs:
        print("No document folders found in output/")
        return

    for doc_dir in sorted(doc_dirs):
        try:
            build_markdown_for_doc(doc_dir)
        except Exception as e:
            print(f"Failed on {doc_dir.name}: {e}")


if __name__ == "__main__":
    main()
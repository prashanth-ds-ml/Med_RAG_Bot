import json
import re
from pathlib import Path
from typing import Optional, List

import fitz  # PyMuPDF
import camelot
import pdfplumber
from PIL import Image
import pytesseract


# =========================================================
# Config
# =========================================================
PDF_DIR = Path("/home/enma/Projects/Med360_RAG_Bot/data/pdf/cardiology")
OUTPUT_ROOT = Path("output")

# labels treated as plain text blocks
TEXT_LABELS = {"title", "text", "header", "section_header","header_footer"}

# if your tool uses "header_footer", we skip it
SKIP_LABELS = ()

# OCR fallback
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

    # join wrapped lines inside paragraphs, keep blank lines as paragraph breaks
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
    """
    Assumes output/<doc_name>/ corresponds to data/pdf/<doc_name>.pdf
    """
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
# PyMuPDF text extraction
# =========================================================
def extract_text_from_pdf_region(pdf_path: Path, page_num_1_based: int, bbox_image, page_image_path: Optional[Path]) -> str:
    """
    bbox_image: [x1, y1, x2, y2] in rendered page image coordinates
    We map image coords -> PDF page coords using page.png size vs page rect size.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num_1_based - 1]
        page_rect = page.rect  # PDF coordinate system

        if page_image_path and page_image_path.exists():
            img = Image.open(page_image_path)
            img_w, img_h = img.size
        else:
            # fallback: approximate from page rect itself
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
# Table extraction
# =========================================================
def pdf_bbox_from_image_bbox(pdf_path: Path, page_num_1_based: int, bbox_image, page_image_path: Optional[Path]):
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

    pdf_x1 = x1 * scale_x
    pdf_y1 = y1 * scale_y
    pdf_x2 = x2 * scale_x
    pdf_y2 = y2 * scale_y

    # Camelot expects table_areas in PDF coordinate string:
    # x1,y1,x2,y2 where origin is bottom-left
    # our pdf_y values are top-down from page rect
    top = pdf_y1
    bottom = pdf_y2
    page_height = page_rect.height

    camelot_y1 = page_height - top
    camelot_y2 = page_height - bottom

    doc.close()

    return {
        "pdf_rect": (pdf_x1, pdf_y1, pdf_x2, pdf_y2),
        "camelot_area": f"{pdf_x1},{camelot_y1},{pdf_x2},{camelot_y2}",
    }


def dataframe_to_markdown(df) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        rows = df.values.tolist()

        lines = []
        lines.append("| " + " | ".join(map(str, cols)) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(map(lambda x: str(x).replace("\n", " "), row)) + " |")
        return "\n".join(lines)


def parse_table_with_camelot(pdf_path: Path, page_num_1_based: int, camelot_area: str) -> Optional[str]:
    try:
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_num_1_based),
            flavor="stream",
            table_areas=[camelot_area],
        )
        if len(tables) > 0 and tables[0].df is not None and not tables[0].df.empty:
            return dataframe_to_markdown(tables[0].df)
    except Exception:
        pass

    try:
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_num_1_based),
            flavor="lattice",
            table_areas=[camelot_area],
        )
        if len(tables) > 0 and tables[0].df is not None and not tables[0].df.empty:
            return dataframe_to_markdown(tables[0].df)
    except Exception:
        pass

    return None


def parse_table_with_pdfplumber(pdf_path: Path, page_num_1_based: int, pdf_rect) -> Optional[str]:
    try:
        x1, y1, x2, y2 = pdf_rect
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_num_1_based - 1]
            cropped = page.crop((x1, y1, x2, y2))
            table = cropped.extract_table()
            if table and len(table) > 0:
                header = table[0]
                rows = table[1:] if len(table) > 1 else []
                import pandas as pd
                df = pd.DataFrame(rows, columns=header)
                return dataframe_to_markdown(df)
    except Exception:
        pass
    return None


# =========================================================
# Figure / flowchart
# =========================================================
def parse_figure_or_flowchart(region: dict, crop_path: Path) -> str:
    label = region["label"]
    region_id = region["region_id"]

    ocr_text = image_ocr(crop_path)
    rel_crop = crop_path.as_posix()

    block = [f"[{label.capitalize()}: {region_id} | source: {rel_crop}]"]
    if ocr_text:
        block.append("")
        block.append("OCR:")
        block.append(ocr_text)

    return "\n".join(block).strip()


# =========================================================
# Main per-region parser
# =========================================================
def parse_region(region: dict, pdf_path: Path, page_image_path: Path) -> str:
    label = region["label"]
    page_num = region.get("page", 1)
    bbox = region["bbox"]
    crop_path = Path(region["crop_file"])

    if label in SKIP_LABELS:
        return ""

    if label in TEXT_LABELS:
        text = extract_text_from_pdf_region(pdf_path, page_num, bbox, page_image_path)
        if not text and crop_path.exists():
            text = image_ocr(crop_path)
        return text.strip()

    if label == "table":
        try:
            mapped = pdf_bbox_from_image_bbox(pdf_path, page_num, bbox, page_image_path)
            camelot_md = parse_table_with_camelot(pdf_path, page_num, mapped["camelot_area"])
            if camelot_md:
                return camelot_md

            plumber_md = parse_table_with_pdfplumber(pdf_path, page_num, mapped["pdf_rect"])
            if plumber_md:
                return plumber_md
        except Exception:
            pass

        rel_crop = crop_path.as_posix()
        return f"[Table: {region['region_id']} | source: {rel_crop}]"

    if label in {"figure", "flowchart"}:
        return parse_figure_or_flowchart(region, crop_path)

    # unknown label -> try plain text, then fallback placeholder
    text = extract_text_from_pdf_region(pdf_path, page_num, bbox, page_image_path)
    if not text and crop_path.exists():
        text = image_ocr(crop_path)
    if text:
        return text

    return f"[Unknown label {label}: {region['region_id']}]"


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
            title_text = clean_text(parsed)

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
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from pdf2image import convert_from_path

# -----------------------------
# Config
# -----------------------------
PDF_DIR = Path("/home/enma/Projects/Med360_RAG_Bot/data/pdf/ENT")
OUT_ROOT = Path("output")

LABEL_MAP = {
    ord("t"): "text",
    ord("b"): "table",
    ord("f"): "figure",
    ord("c"): "flowchart",
    ord("h"): "title",
    ord("x"): "header_footer",
}

LABEL_COLORS = {
    "text": (0, 255, 0),
    "table": (255, 165, 0),
    "figure": (255, 0, 255),
    "flowchart": (0, 255, 255),
    "title": (255, 0, 0),
    "header_footer": (128, 128, 128),
}

WINDOW_NAME = "PDF Page - Select ROI"
PREVIEW_NAME = "Crop Preview - press t/b/f/c/h/x | u undo | q quit"

# Bigger by default
SCREEN_W = 1850
SCREEN_H = 1100
MARGIN = 50
INITIAL_ZOOM = 1.50

PREVIEW_MAX_W = 1100
PREVIEW_MAX_H = 760


# -----------------------------
# Helpers
# -----------------------------
def pick_pdf_from_dir(pdf_dir: Path) -> Path:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    print("\nAvailable PDFs:\n")
    for i, pdf in enumerate(pdf_files, start=1):
        print(f"{i:02d}. {pdf.name}")

    while True:
        choice = input("\nEnter PDF number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        idx = int(choice)
        if 1 <= idx <= len(pdf_files):
            return pdf_files[idx - 1]

        print("Choice out of range. Try again.")


def save_labels(regions, labels_path: Path):
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2, ensure_ascii=False)


def fit_to_box(img, max_w=PREVIEW_MAX_W, max_h=PREVIEW_MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def build_status_panel(width=1100):
    panel = np.full((150, width, 3), 245, dtype=np.uint8)
    lines = [
        "Keys: t=text | b=table | f=figure | c=flowchart | h=title | x=header/footer",
        "u = undo last saved region | q / Esc = quit cleanly",
        "Draw ROI on page, press ENTER/SPACE, then press label key in preview window",
    ]
    y = 35
    for line in lines:
        cv2.putText(
            panel,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        y += 38
    return panel


def pad_to_width(img, target_w, value=255):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    if w > target_w:
        return img[:, :target_w]
    pad = np.full((h, target_w - w, 3), value, dtype=np.uint8)
    return np.hstack([img, pad])


def draw_saved_boxes(base_img_bgr, regions, scale):
    canvas = base_img_bgr.copy()
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        dx1 = int(x1 * scale)
        dy1 = int(y1 * scale)
        dx2 = int(x2 * scale)
        dy2 = int(y2 * scale)

        color = LABEL_COLORS.get(r["label"], (0, 255, 0))
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), color, 2)
        cv2.putText(
            canvas,
            f"{r['region_id']}:{r['label']}",
            (dx1, max(20, dy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def show_preview(crop_bgr):
    preview = fit_to_box(crop_bgr)
    info = build_status_panel(width=max(preview.shape[1], 1100))
    preview2 = pad_to_width(preview, info.shape[1], 255)
    combined = np.vstack([preview2, info])

    cv2.namedWindow(PREVIEW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        PREVIEW_NAME,
        min(combined.shape[1], SCREEN_W),
        min(combined.shape[0], SCREEN_H),
    )
    cv2.imshow(PREVIEW_NAME, combined)


def remove_last_region(regions, labels_path: Path):
    if not regions:
        return None
    last = regions.pop()
    crop_path = Path(last["crop_file"])
    if crop_path.exists():
        crop_path.unlink()

    for i, r in enumerate(regions, start=1):
        r["reading_order"] = i

    save_labels(regions, labels_path)
    return last


def safe_destroy_window(name: str):
    try:
        cv2.destroyWindow(name)
    except Exception:
        pass


def clean_exit(regions, labels_path: Path, message="Exiting..."):
    save_labels(regions, labels_path)
    safe_destroy_window(PREVIEW_NAME)
    safe_destroy_window(WINDOW_NAME)
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
    except Exception:
        pass
    print(message)
    print(f"Labels saved to: {labels_path}")
    sys.exit(0)


# -----------------------------
# Pick PDF
# -----------------------------
PDF_PATH = pick_pdf_from_dir(PDF_DIR)
DOC_NAME = PDF_PATH.stem

DOC_OUT_DIR = OUT_ROOT / DOC_NAME
CROPS_DIR = DOC_OUT_DIR / "crops"
PAGE_IMG_PATH = DOC_OUT_DIR / "page.png"
LABELS_PATH = DOC_OUT_DIR / "labels.json"

DOC_OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nSelected PDF: {PDF_PATH}")

# -----------------------------
# Load first page
# -----------------------------
pages = convert_from_path(str(PDF_PATH), dpi=220, first_page=1, last_page=1)
page_pil = pages[0].convert("RGB")
page_pil.save(PAGE_IMG_PATH)

page_rgb = cv2.cvtColor(cv2.imread(str(PAGE_IMG_PATH)), cv2.COLOR_BGR2RGB)
orig_h, orig_w = page_rgb.shape[:2]

base_scale = min((SCREEN_W - MARGIN) / orig_w, (SCREEN_H - MARGIN) / orig_h, 1.0)
scale = min(base_scale * INITIAL_ZOOM, 3.0)

disp_w = int(orig_w * scale)
disp_h = int(orig_h * scale)

display_rgb = cv2.resize(page_rgb, (disp_w, disp_h), interpolation=cv2.INTER_CUBIC)
display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)

print(f"Original size : {orig_w} x {orig_h}")
print(f"Display size  : {disp_w} x {disp_h}")
print(f"Scale factor  : {scale:.4f}")

regions = []
if LABELS_PATH.exists():
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            regions = json.load(f)
        print(f"Loaded existing regions: {len(regions)}")
    except Exception:
        regions = []

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, min(disp_w, SCREEN_W), min(disp_h, SCREEN_H))
cv2.imshow(WINDOW_NAME, display_bgr)

# -----------------------------
# Main loop
# -----------------------------
while True:
    page_view = draw_saved_boxes(display_bgr, regions, scale)
    cv2.imshow(WINDOW_NAME, page_view)

    roi = cv2.selectROI(WINDOW_NAME, page_view, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi

    if w == 0 or h == 0:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("u"):
            removed = remove_last_region(regions, LABELS_PATH)
            if removed:
                print(f"Undid {removed['region_id']}")
            else:
                print("Nothing to undo.")
            continue
        elif key == ord("q") or key == 27:
            clean_exit(regions, LABELS_PATH, message="Exited cleanly.")
        else:
            continue

    x1 = max(0, int(x / scale))
    y1 = max(0, int(y / scale))
    x2 = min(orig_w, int((x + w) / scale))
    y2 = min(orig_h, int((y + h) / scale))

    if x2 <= x1 or y2 <= y1:
        print("Invalid ROI. Skipped.")
        continue

    crop_rgb = page_rgb[y1:y2, x1:x2]
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    show_preview(crop_bgr)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in LABEL_MAP:
            label = LABEL_MAP[key]
            region_id = f"region_{len(regions)+1:03d}"
            crop_filename = f"{region_id}_{label}.png"
            crop_path = CROPS_DIR / crop_filename

            cv2.imwrite(str(crop_path), crop_bgr)

            record = {
                "region_id": region_id,
                "page": 1,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "reading_order": len(regions) + 1,
                "crop_file": str(crop_path).replace("\\", "/"),
            }

            regions.append(record)
            save_labels(regions, LABELS_PATH)

            print(f"Saved {region_id} as {label}")
            safe_destroy_window(PREVIEW_NAME)
            break

        elif key == ord("u"):
            removed = remove_last_region(regions, LABELS_PATH)
            if removed:
                print(f"Undid {removed['region_id']}")
            else:
                print("Nothing to undo.")
            safe_destroy_window(PREVIEW_NAME)
            break

        elif key == ord("q") or key == 27:
            clean_exit(regions, LABELS_PATH, message="Exited cleanly from preview window.")

        else:
            continue
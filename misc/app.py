import json
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path

# -----------------------------
# Config
# -----------------------------
PDF_PATH = "/home/enma/Projects/Med360_RAG_Bot/data/pdf/atrial_fibrillation.pdf"
OUT_DIR = Path("output")
CROPS_DIR = OUT_DIR / "crops"
LABELS_PATH = OUT_DIR / "labels.json"
PAGE_IMG_PATH = OUT_DIR / "page.png"

OUT_DIR.mkdir(exist_ok=True)
CROPS_DIR.mkdir(exist_ok=True)

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

# -----------------------------
# Helpers
# -----------------------------
def save_labels(regions):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2, ensure_ascii=False)


def fit_to_box(img, max_w=700, max_h=500):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def build_status_panel():
    panel = np.full((150, 900, 3), 245, dtype=np.uint8)
    lines = [
        "Keys: t=text | b=table | f=figure | c=flowchart | h=title | x=header/footer",
        "u = undo last saved region | q = quit",
        "Draw ROI on page, press ENTER/SPACE, then press label key in preview window",
    ]
    y = 35
    for line in lines:
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        y += 38
    return panel


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
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def show_preview(crop_bgr):
    preview = fit_to_box(crop_bgr, max_w=800, max_h=500)
    info = build_status_panel()
    combined = np.vstack([preview, info]) if preview.shape[1] == info.shape[1] else None

    if combined is None:
        # pad narrower image
        max_w = max(preview.shape[1], info.shape[1])

        def pad_to_width(img, target_w, value=255):
            h, w = img.shape[:2]
            if w == target_w:
                return img
            pad = np.full((h, target_w - w, 3), value, dtype=np.uint8)
            return np.hstack([img, pad])

        preview2 = pad_to_width(preview, max_w, 255)
        info2 = pad_to_width(info, max_w, 245)
        combined = np.vstack([preview2, info2])

    cv2.namedWindow(PREVIEW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(PREVIEW_NAME, combined)


def remove_last_region(regions):
    if not regions:
        return None
    last = regions.pop()
    crop_path = Path(last["crop_file"])
    if crop_path.exists():
        crop_path.unlink()
    # reassign reading_order to stay clean
    for i, r in enumerate(regions, start=1):
        r["reading_order"] = i
    save_labels(regions)
    return last


# -----------------------------
# Load first page
# -----------------------------
if not Path(PDF_PATH).exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

pages = convert_from_path(PDF_PATH, dpi=200, first_page=1, last_page=1)
page_pil = pages[0].convert("RGB")
page_pil.save(PAGE_IMG_PATH)

page_rgb = cv2.cvtColor(cv2.imread(str(PAGE_IMG_PATH)), cv2.COLOR_BGR2RGB)
orig_h, orig_w = page_rgb.shape[:2]

SCREEN_W = 1700
SCREEN_H = 1000
margin = 100

scale = min((SCREEN_W - margin) / orig_w, (SCREEN_H - margin) / orig_h, 1.0)
disp_w = int(orig_w * scale)
disp_h = int(orig_h * scale)

display_rgb = cv2.resize(page_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
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
cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h)

# -----------------------------
# Main loop
# -----------------------------
while True:
    page_view = draw_saved_boxes(display_bgr, regions, scale)
    cv2.imshow(WINDOW_NAME, page_view)

    roi = cv2.selectROI(WINDOW_NAME, page_view, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi

    if w == 0 or h == 0:
        # no ROI selected; wait for key
        key = cv2.waitKey(0) & 0xFF
        if key == ord("u"):
            removed = remove_last_region(regions)
            if removed:
                print(f"Undid {removed['region_id']}")
            else:
                print("Nothing to undo.")
            continue
        elif key == ord("q") or key == 27:
            break
        else:
            continue

    # map display coords -> original coords
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
            save_labels(regions)

            print(f"Saved {region_id} as {label}")
            cv2.destroyWindow(PREVIEW_NAME)
            break

        elif key == ord("u"):
            removed = remove_last_region(regions)
            if removed:
                print(f"Undid {removed['region_id']}")
            else:
                print("Nothing to undo.")

            cv2.destroyWindow(PREVIEW_NAME)
            break

        elif key == ord("q") or key == 27:
            cv2.destroyWindow(PREVIEW_NAME)
            cv2.destroyAllWindows()
            save_labels(regions)
            print(f"Done. Labels saved to: {LABELS_PATH}")
            raise SystemExit

        else:
            # ignore other keys
            continue

cv2.destroyAllWindows()
save_labels(regions)
print(f"Done. Labels saved to: {LABELS_PATH}")
print(f"Crops saved in: {CROPS_DIR}")
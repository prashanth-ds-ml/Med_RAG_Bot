import csv
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


# =========================================================
# CONFIG
# =========================================================
BASE = "https://www.who.int"
START_URL = "https://www.who.int/publications/i"

OUT_DIR = Path("who_publications")
PDF_DIR = OUT_DIR / "pdfs"
META_DIR = OUT_DIR / "meta"

PDF_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

ITEM_LINKS_FILE = META_DIR / "item_links.txt"
COMPLETED_ITEMS_FILE = META_DIR / "completed_items.txt"
SUCCESS_DOWNLOAD_URLS_FILE = META_DIR / "success_download_urls.txt"
DOWNLOADED_HASHES_FILE = META_DIR / "downloaded_hashes.txt"

MANIFEST_JSONL = META_DIR / "manifest.jsonl"
MANIFEST_CSV = META_DIR / "manifest.csv"

MAX_PAGES = 305
MAX_WORKERS = 8
REQUEST_TIMEOUT = 90
PAGE_WAIT_MS = 2000
MAX_REPEAT_SIGNATURES = 12

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


# =========================================================
# LOCKS
# =========================================================
print_lock = threading.Lock()
file_lock = threading.Lock()
manifest_lock = threading.Lock()
download_url_lock = threading.Lock()
hash_lock = threading.Lock()
completed_lock = threading.Lock()


# =========================================================
# HELPERS
# =========================================================
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def load_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_line(path: Path, value: str):
    with file_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(value + "\n")


def slugify(text: str, max_len: int = 140) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    text = text.strip("_")
    if not text:
        text = "document"
    return text[:max_len]


def get_item_id(item_url: str) -> str:
    return urlparse(item_url).path.rstrip("/").split("/")[-1]


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_manifest_jsonl(record: dict):
    with manifest_lock:
        with open(MANIFEST_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_manifest_csv(record: dict):
    fieldnames = [
        "page_idx",
        "item_id",
        "item_url",
        "title",
        "date",
        "download_url",
        "local_pdf",
        "sha256",
        "status",
        "http_status",
        "error",
    ]
    with manifest_lock:
        file_exists = MANIFEST_CSV.exists()
        with open(MANIFEST_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: record.get(k, "") for k in fieldnames})


def get_requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def guess_extension(resp: requests.Response) -> str:
    ctype = resp.headers.get("Content-Type", "").lower()
    if "pdf" in ctype:
        return ".pdf"

    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename="?([^"]+)"?', cd)
    if m:
        ext = Path(m.group(1)).suffix.lower()
        if ext:
            return ext

    ext = Path(urlparse(resp.url).path).suffix.lower()
    return ext if ext else ".pdf"


def mark_item_completed(completed_items: set[str], item_url: str):
    with completed_lock:
        if item_url not in completed_items:
            completed_items.add(item_url)
            append_line(COMPLETED_ITEMS_FILE, item_url)


def register_success_download_url(success_download_urls: set[str], download_url: str) -> bool:
    with download_url_lock:
        if download_url in success_download_urls:
            return False
        success_download_urls.add(download_url)
        append_line(SUCCESS_DOWNLOAD_URLS_FILE, download_url)
        return True


def register_hash(downloaded_hashes: set[str], file_hash: str) -> bool:
    with hash_lock:
        if file_hash in downloaded_hashes:
            return False
        downloaded_hashes.add(file_hash)
        append_line(DOWNLOADED_HASHES_FILE, file_hash)
        return True


# =========================================================
# PHASE 1: COLLECT ITEM LINKS WITH PLAYWRIGHT
# =========================================================
def collect_item_links(max_pages: int = MAX_PAGES) -> list[str]:
    existing_links = load_lines(ITEM_LINKS_FILE)
    new_links_total = 0

    safe_print(f"Loaded existing item links: {len(existing_links)}")

    all_seen = set(existing_links)
    all_links = list(existing_links)

    last_signature = None
    repeated_signatures = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=HEADERS["User-Agent"])

        for page_idx in range(max_pages):
            page_url = START_URL if page_idx == 0 else f"{START_URL}?page={page_idx}"
            safe_print(f"\n[COLLECT PAGE {page_idx}] {page_url}")

            try:
                page.goto(page_url, wait_until="networkidle", timeout=120000)
                page.wait_for_timeout(PAGE_WAIT_MS)

                raw_links = page.eval_on_selector_all(
                    'a[href*="/publications/i/item/"]',
                    """
                    els => [...new Set(
                        els.map(a => a.href || a.getAttribute('href'))
                           .filter(Boolean)
                    )]
                    """,
                )
            except Exception as e:
                safe_print(f"  Failed to load page: {e}")
                continue

            page_links = []
            for link in raw_links:
                full = urljoin(BASE, link)
                if "/publications/i/item/" in full:
                    page_links.append(full)

            # preserve order, dedupe
            deduped = []
            local_seen = set()
            for link in page_links:
                if link not in local_seen:
                    local_seen.add(link)
                    deduped.append(link)

            signature = tuple(deduped)
            if signature == last_signature:
                repeated_signatures += 1
                safe_print(f"  Repeated rendered signature: {repeated_signatures}")
            else:
                repeated_signatures = 0
            last_signature = signature

            new_links = []
            for link in deduped:
                if link not in all_seen:
                    all_seen.add(link)
                    all_links.append(link)
                    new_links.append(link)
                    append_line(ITEM_LINKS_FILE, link)

            new_links_total += len(new_links)
            safe_print(f"  Found links on page: {len(deduped)} | New: {len(new_links)}")

            if repeated_signatures >= MAX_REPEAT_SIGNATURES:
                safe_print("\nStopping collection: too many repeated rendered pages.")
                break

        browser.close()

    safe_print(f"\nCollection complete. Total known item links: {len(all_links)}")
    safe_print(f"New item links added this run: {new_links_total}")
    return all_links


# =========================================================
# PHASE 2: PARSE ITEM PAGE
# =========================================================
def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_item_metadata(session: requests.Session, item_url: str) -> dict:
    soup = get_soup(session, item_url)

    item_id = get_item_id(item_url)

    title_tag = soup.find("h1")
    title = title_tag.get_text(" ", strip=True) if title_tag else f"document_{item_id}"

    date_text = ""
    for tag in soup.find_all(["p", "div", "span"]):
        text = tag.get_text(" ", strip=True)
        if re.search(r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b", text):
            date_text = text
            break

    download_url = ""
    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True).lower()
        full = urljoin(item_url, a["href"].strip())

        if "download" in txt and ("iris.who.int" in full or "bitstreams" in full or "/server/api/core/bitstreams/" in full):
            download_url = full
            break

    if not download_url:
        for a in soup.find_all("a", href=True):
            full = urljoin(item_url, a["href"].strip())
            if "iris.who.int" in full:
                download_url = full
                break

    return {
        "item_id": item_id,
        "title": title,
        "date": date_text,
        "item_url": item_url,
        "download_url": download_url,
    }


# =========================================================
# PHASE 3: DOWNLOAD FILE
# =========================================================
def download_file(session: requests.Session, url: str, out_base: Path):
    with session.get(url, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True) as r:
        status = r.status_code
        r.raise_for_status()

        ext = guess_extension(r)
        out_path = out_base.with_suffix(ext)

        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path, status

        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        tmp_path.replace(out_path)
        return out_path, status


# =========================================================
# THREAD WORKER
# =========================================================
def process_item(
    item_url: str,
    completed_items: set[str],
    success_download_urls: set[str],
    downloaded_hashes: set[str],
):
    session = get_requests_session()

    record = {
        "page_idx": "",
        "item_id": get_item_id(item_url),
        "item_url": item_url,
        "title": "",
        "date": "",
        "download_url": "",
        "local_pdf": "",
        "sha256": "",
        "status": "pending",
        "http_status": "",
        "error": "",
    }

    try:
        meta = extract_item_metadata(session, item_url)
        record.update(meta)

        title = meta["title"]
        item_id = meta["item_id"]
        download_url = meta["download_url"]

        if not download_url:
            record["status"] = "no_download_link"
            mark_item_completed(completed_items, item_url)
            return record

        if download_url in success_download_urls:
            record["status"] = "duplicate_download_url"
            mark_item_completed(completed_items, item_url)
            return record

        base_name = f"{slugify(title)}__{item_id}"
        out_base = PDF_DIR / base_name

        try:
            local_file, http_status = download_file(session, download_url, out_base)
            record["http_status"] = http_status or ""

            if local_file and local_file.exists():
                file_hash = sha256_of_file(local_file)
                record["sha256"] = file_hash

                if not register_hash(downloaded_hashes, file_hash):
                    try:
                        local_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                    record["status"] = "duplicate_file_hash"
                    mark_item_completed(completed_items, item_url)
                    return record

                register_success_download_url(success_download_urls, download_url)
                record["local_pdf"] = str(local_file)
                record["status"] = "downloaded"
                mark_item_completed(completed_items, item_url)
                return record

            record["status"] = "download_failed"
            record["error"] = "download returned no file"
            return record

        except Exception as e:
            record["status"] = "download_failed"
            record["error"] = str(e)
            return record

    except Exception as e:
        record["status"] = "item_parse_failed"
        record["error"] = str(e)
        return record


# =========================================================
# MAIN
# =========================================================
def main():
    safe_print("Step 1: Collecting item links with Playwright...")
    item_links = collect_item_links(MAX_PAGES)

    completed_items = load_lines(COMPLETED_ITEMS_FILE)
    success_download_urls = load_lines(SUCCESS_DOWNLOAD_URLS_FILE)
    downloaded_hashes = load_lines(DOWNLOADED_HASHES_FILE)

    pending_items = [x for x in item_links if x not in completed_items]

    safe_print(f"\nLoaded completed items: {len(completed_items)}")
    safe_print(f"Loaded success download URLs: {len(success_download_urls)}")
    safe_print(f"Loaded downloaded hashes: {len(downloaded_hashes)}")
    safe_print(f"Pending items for download: {len(pending_items)}")

    if not pending_items:
        safe_print("\nNothing to download.")
        return

    safe_print(f"\nStep 2: Downloading PDFs with {MAX_WORKERS} threads...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_item,
                item_url,
                completed_items,
                success_download_urls,
                downloaded_hashes,
            ): item_url
            for item_url in pending_items
        }

        for i, future in enumerate(as_completed(futures), start=1):
            item_url = futures[future]
            try:
                record = future.result()
            except Exception as e:
                record = {
                    "page_idx": "",
                    "item_id": get_item_id(item_url),
                    "item_url": item_url,
                    "title": "",
                    "date": "",
                    "download_url": "",
                    "local_pdf": "",
                    "sha256": "",
                    "status": "thread_failed",
                    "http_status": "",
                    "error": str(e),
                }

            write_manifest_jsonl(record)
            append_manifest_csv(record)

            msg = f"[{i}/{len(pending_items)}] [{record['status']}] {record.get('title', '')[:90]}"
            if record.get("local_pdf"):
                msg += f" -> {record['local_pdf']}"
            if record.get("error"):
                msg += f" | {record['error'][:180]}"
            safe_print(msg)

    safe_print("\nDone.")


if __name__ == "__main__":
    main()  
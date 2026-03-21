import json
import time
import hashlib
import threading
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup


# =========================================================
# CONFIG
# =========================================================
START_URLS = {
    "ICMR": [
        "https://www.icmr.gov.in/guidelines",
        "https://www.icmr.gov.in/reports",
        "https://www.icmr.gov.in/annual-reports",
        "https://www.icmr.gov.in/downloadable-books",
        "https://www.icmr.gov.in/standard-treatment-workflows-stws"
    ],
    "MOHFW": [
        "https://main.mohfw.gov.in/documents",
    ],
    "NCDC": [
        "https://ncdc.mohfw.gov.in/",
    ],
    "WHO": [
        "https://www.who.int/publications/i",
        "https://iris.who.int/",
    ],
}

OUTPUT_ROOT = Path("data/raw_corpus")
DOWNLOADS_DIR = OUTPUT_ROOT / "downloads"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
LOGS_DIR = OUTPUT_ROOT / "logs"

DISCOVERED_PATH = MANIFESTS_DIR / "discovered_links.jsonl"
DOWNLOADED_PATH = MANIFESTS_DIR / "downloaded_files.jsonl"
LOG_PATH = LOGS_DIR / "download.log"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36"
)

REQUEST_TIMEOUT = 40
SLEEP_SECONDS = 0.4
MAX_PAGES_PER_SITE = 500
MAX_DOWNLOAD_WORKERS = 8
SKIP_EXISTING_LOCAL = True


# =========================================================
# SETUP
# =========================================================
for d in [DOWNLOADS_DIR, MANIFESTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

io_lock = threading.Lock()
thread_local = threading.local()


# =========================================================
# THREAD-LOCAL SESSION
# =========================================================
def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        thread_local.session = s
    return thread_local.session


# =========================================================
# HELPERS
# =========================================================
def log(msg: str):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    final_msg = f"[{stamp}] {msg}"
    with io_lock:
        print(final_msg, flush=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(final_msg + "\n")


def append_jsonl(path: Path, record: dict):
    with io_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    records = []
    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    return url.strip()


def safe_name(text: str, max_len: int = 160) -> str:
    import re
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\-.]", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "file")[:max_len]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def build_local_path(source_name: str, pdf_url: str) -> Path:
    source_dir = DOWNLOADS_DIR / safe_name(source_name)
    source_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(pdf_url)
    original_name = Path(parsed.path).name or "document.pdf"

    if not original_name.lower().endswith(".pdf"):
        original_name += ".pdf"

    stem = safe_name(Path(original_name).stem, max_len=120)
    url_tag = short_hash(pdf_url, 10)
    local_name = f"{stem}__{url_tag}.pdf"

    return source_dir / local_name


def fetch(url: str):
    try:
        session = get_session()
        log(f"[FETCH] {url}")
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        time.sleep(SLEEP_SECONDS)
        return resp
    except Exception as e:
        log(f"[FETCH-ERROR] {url} | {e}")
        return None


def extract_links(base_url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    pdf_links = set()
    page_links = set()

    base_netloc = urlparse(base_url).netloc

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue

        full = normalize_url(urljoin(base_url, href))
        parsed = urlparse(full)

        if parsed.scheme not in {"http", "https"}:
            continue

        if is_pdf_url(full):
            pdf_links.add(full)
        elif parsed.netloc == base_netloc:
            page_links.add(full)

    return sorted(pdf_links), sorted(page_links)


def load_existing_state():
    discovered_records = read_jsonl(DISCOVERED_PATH)
    downloaded_records = read_jsonl(DOWNLOADED_PATH)

    discovered_by_url = {}
    for r in discovered_records:
        pdf_url = r.get("pdf_url")
        if pdf_url:
            discovered_by_url[pdf_url] = r

    downloaded_success_urls = set()
    for r in downloaded_records:
        status = r.get("status")
        pdf_url = r.get("pdf_url")
        local_path = r.get("local_path")

        if not pdf_url:
            continue

        if status in {"downloaded", "skipped_existing"}:
            if local_path:
                if Path(local_path).exists():
                    downloaded_success_urls.add(pdf_url)
            else:
                downloaded_success_urls.add(pdf_url)

    return discovered_by_url, downloaded_success_urls


# =========================================================
# CRAWLER
# =========================================================
def crawl_source(source_name: str, start_urls: list[str], existing_discovered_urls: set[str]):
    seen_pages = set()
    seen_pdfs = set(existing_discovered_urls)
    new_discovered = []

    for start_url in start_urls:
        queue = [start_url]
        crawled_count = 0

        log("=" * 90)
        log(f"[CRAWL-START] Source={source_name} | Start URL={start_url}")
        log("=" * 90)

        while queue and crawled_count < MAX_PAGES_PER_SITE:
            current = queue.pop(0)
            if current in seen_pages:
                continue

            seen_pages.add(current)
            crawled_count += 1
            log(f"[PAGE {crawled_count}/{MAX_PAGES_PER_SITE}] {current}")

            resp = fetch(current)
            if resp is None:
                continue

            content_type = resp.headers.get("Content-Type", "").lower()

            if "application/pdf" in content_type or is_pdf_url(current):
                if current not in seen_pdfs:
                    seen_pdfs.add(current)
                    record = {
                        "source_name": source_name,
                        "source_page": current,
                        "pdf_url": current,
                        "discovered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    new_discovered.append(record)
                    append_jsonl(DISCOVERED_PATH, record)
                    log(f"[PDF-NEW] Direct PDF discovered: {current}")
                continue

            pdf_links, page_links = extract_links(current, resp.text)

            if pdf_links:
                log(f"[DISCOVERY] Found {len(pdf_links)} PDF link(s)")
            if page_links:
                log(f"[DISCOVERY] Found {len(page_links)} internal page link(s)")

            for pdf_url in pdf_links:
                if pdf_url not in seen_pdfs:
                    seen_pdfs.add(pdf_url)
                    record = {
                        "source_name": source_name,
                        "source_page": current,
                        "pdf_url": pdf_url,
                        "discovered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    new_discovered.append(record)
                    append_jsonl(DISCOVERED_PATH, record)
                    log(f"[PDF-NEW] {pdf_url}")

            for link in page_links:
                if link not in seen_pages:
                    queue.append(link)

        log(f"[CRAWL-END] Source={source_name} | Start URL={start_url} | pages_crawled={crawled_count}")

    return new_discovered


# =========================================================
# DOWNLOAD
# =========================================================
def download_pdf_task(record: dict, index: int, total: int):
    source_name = record["source_name"]
    pdf_url = record["pdf_url"]
    source_page = record["source_page"]

    local_path = build_local_path(source_name, pdf_url)
    progress = f"[{index}/{total}]"

    log("-" * 90)
    log(f"{progress} [DOWNLOAD-START] Source     : {source_name}")
    log(f"{progress} [DOWNLOAD-START] PDF URL    : {pdf_url}")
    log(f"{progress} [DOWNLOAD-START] Source page: {source_page}")
    log(f"{progress} [DOWNLOAD-START] Save path  : {local_path}")

    if local_path.exists() and SKIP_EXISTING_LOCAL:
        sha256 = sha256_file(local_path)
        size_bytes = local_path.stat().st_size

        out = {
            "source_name": source_name,
            "source_page": source_page,
            "pdf_url": pdf_url,
            "local_path": str(local_path),
            "status": "skipped_existing",
            "size_bytes": size_bytes,
            "sha256": sha256,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_jsonl(DOWNLOADED_PATH, out)

        log(f"{progress} [SKIP] Local file already present")
        log(f"{progress} [SKIP] Size   : {human_size(size_bytes)}")
        log(f"{progress} [SKIP] SHA256 : {sha256}")

        return "skipped_existing"

    try:
        session = get_session()
        resp = session.get(pdf_url, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()

        remote_size = resp.headers.get("Content-Length")
        content_type = resp.headers.get("Content-Type", "unknown")

        if remote_size and remote_size.isdigit():
            log(f"{progress} [DOWNLOAD] Remote size : {human_size(int(remote_size))}")
        log(f"{progress} [DOWNLOAD] Content-Type: {content_type}")

        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        time.sleep(SLEEP_SECONDS)

        sha256 = sha256_file(local_path)
        size_bytes = local_path.stat().st_size

        out = {
            "source_name": source_name,
            "source_page": source_page,
            "pdf_url": pdf_url,
            "local_path": str(local_path),
            "status": "downloaded",
            "size_bytes": size_bytes,
            "sha256": sha256,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_jsonl(DOWNLOADED_PATH, out)

        log(f"{progress} [DOWNLOAD-SUCCESS] Saved  : {local_path}")
        log(f"{progress} [DOWNLOAD-SUCCESS] Size   : {human_size(size_bytes)}")
        log(f"{progress} [DOWNLOAD-SUCCESS] SHA256 : {sha256}")

        return "downloaded"

    except Exception as e:
        out = {
            "source_name": source_name,
            "source_page": source_page,
            "pdf_url": pdf_url,
            "local_path": str(local_path),
            "status": "failed",
            "error": str(e),
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_jsonl(DOWNLOADED_PATH, out)

        log(f"{progress} [DOWNLOAD-ERROR] {pdf_url} | {e}")
        return "failed"


# =========================================================
# MAIN
# =========================================================
def main():
    log("=" * 90)
    log("[JOB-START] Incremental raw PDF discovery + parallel download")
    log("=" * 90)

    existing_discovered_by_url, downloaded_success_urls = load_existing_state()

    log(f"[STATE] Previously discovered links : {len(existing_discovered_by_url)}")
    log(f"[STATE] Previously downloaded PDFs : {len(downloaded_success_urls)}")

    newly_discovered = []

    for source_name, urls in START_URLS.items():
        log(f"[SOURCE-START] {source_name} | seed_urls={len(urls)}")
        new_records = crawl_source(
            source_name=source_name,
            start_urls=urls,
            existing_discovered_urls=set(existing_discovered_by_url.keys()),
        )
        newly_discovered.extend(new_records)
        log(f"[SOURCE-END] {source_name} | new_discovered={len(new_records)}")

    log(f"[SUMMARY] New links discovered this run = {len(newly_discovered)}")

    all_discovered_by_url = dict(existing_discovered_by_url)
    for r in newly_discovered:
        all_discovered_by_url[r["pdf_url"]] = r

    log(f"[SUMMARY] Total discovered links in manifest = {len(all_discovered_by_url)}")

    pending_records = [
        r for url, r in all_discovered_by_url.items()
        if url not in downloaded_success_urls
    ]

    log(f"[SUMMARY] PDFs pending download = {len(pending_records)}")

    if not pending_records:
        log("[DONE] No new PDFs to download.")
        return

    status_counts = {
        "downloaded": 0,
        "skipped_existing": 0,
        "failed": 0,
    }

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        futures = {
            executor.submit(download_pdf_task, record, i, len(pending_records)): record
            for i, record in enumerate(pending_records, start=1)
        }

        for future in as_completed(futures):
            try:
                status = future.result()
                status_counts[status] = status_counts.get(status, 0) + 1
            except Exception as e:
                log(f"[THREAD-ERROR] {e}")
                status_counts["failed"] += 1

    log("=" * 90)
    log(f"[FINAL] Downloaded        : {status_counts['downloaded']}")
    log(f"[FINAL] Skipped existing  : {status_counts['skipped_existing']}")
    log(f"[FINAL] Failed            : {status_counts['failed']}")
    log("[JOB-END] Incremental run completed")
    log("=" * 90)


if __name__ == "__main__":
    main()
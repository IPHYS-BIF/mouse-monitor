"""Download images from an image search engine.

Usage:
    python tools/download_images.py
    python tools/download_images.py --keyword "real mouse animal" --count 500 --out data/images
    python tools/download_images.py --url "https://www.google.com/search?q=mouse+animal&udm=2" --count 500
    python tools/download_images.py --engine bing

Requires:
    pip install icrawler requests selenium
    (the Google engine needs Google Chrome installed; Selenium Manager fetches
    the matching driver automatically)

Note:
    Preferred workflow for Google:
      1. Open Google Images in your browser, search and filter as you like.
      2. Copy the URL from the address bar.
      3. Pass it with --url.  The script opens a visible Chrome window using
         your real Chrome profile, so Google sees a normal logged-in session
         and does not show a CAPTCHA.
      Close ALL other Chrome windows before running, otherwise Chrome will
      refuse to open a second instance with the same profile.

    When --url is given, the script scrolls that page first, then appends
    keyword variations to reach the target count.  When only --keyword is
    given, the URL is built automatically.

    Bing remains available via --engine bing (does not need Chrome).
"""

import argparse
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from icrawler.builtin import BingImageCrawler

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}

# Extra words appended to the base keyword to broaden coverage and pull in
# unique images that a single query would not surface.
QUERY_VARIATIONS = [
    "",
    "close up",
    "wild",
    "white",
    "brown",
    "field",
    "pet",
    "cute",
    "running",
    "house mouse",
    "in nature",
    "photo",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def count_images(out_dir: Path) -> int:
    return sum(1 for p in out_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _extract_image_urls(html: str) -> list[str]:
    # Google embeds original image URLs in arrays like ["https://...", 600, 800].
    raw = re.findall(r'\["(https?://[^"\]]+?)",\s*\d+,\s*\d+\]', html)
    urls: list[str] = []
    for url in raw:
        url = (
            url.replace("\\u003d", "=")
            .replace("\\u0026", "&")
            .replace("\\u003f", "?")
            .replace("\\/", "/")
        )
        if any(host in url for host in ("gstatic.com", "google.com", "googleusercontent.com")):
            continue
        urls.append(url)
    return urls


def _google_image_urls(driver, query: str, max_num: int, start_url: str | None = None) -> list[str]:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys

    url = start_url if start_url else f"https://www.google.com/search?q={quote(query)}&udm=2"
    driver.get(url)
    time.sleep(2)
    _dismiss_consent(driver)

    seen: set[str] = set()
    urls: list[str] = []
    body = driver.find_element(By.TAG_NAME, "body")
    stale_rounds = 0

    for _ in range(40):
        if len(urls) >= max_num:
            break

        for url in _extract_image_urls(driver.page_source):
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)

        # Try to load more results (scroll + optional "show more" button).
        body.send_keys(Keys.END)
        time.sleep(1.5)
        try:
            button = driver.find_element(By.XPATH, "//input[@type='button']")
            if button.is_displayed():
                button.click()
                time.sleep(1.5)
        except Exception:
            pass

        if len(seen) == len(urls) and not _page_grew(driver):
            stale_rounds += 1
            if stale_rounds >= 3:
                break
        else:
            stale_rounds = 0

    return urls[:max_num]


def _page_grew(driver) -> bool:
    height = driver.execute_script("return document.body.scrollHeight")
    grew = height != getattr(driver, "_last_height", None)
    driver._last_height = height
    return grew


def _dismiss_consent(driver) -> None:
    from selenium.webdriver.common.by import By

    for text in ("Accept all", "Souhlasím", "Přijmout vše", "I agree", "Reject all"):
        try:
            btn = driver.find_element(By.XPATH, f"//button[.//text()[contains(., '{text}')]]")
            btn.click()
            time.sleep(1)
            return
        except Exception:
            continue


def _save_image(url: str, out_dir: Path, session: requests.Session, delay: float = 0.0) -> bool:
    name = hashlib.md5(url.encode()).hexdigest()
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException:
        return False

    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    ext = CONTENT_TYPE_EXT.get(content_type)
    if ext is None or len(resp.content) < 5000:
        return False

    (out_dir / f"{name}{ext}").write_bytes(resp.content)
    if delay > 0:
        time.sleep(delay)
    return True


def _make_driver():
    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    # headless=False keeps a visible window which reduces bot-detection.
    # version_main must match the installed Chrome major version.
    return uc.Chrome(options=options, headless=False, version_main=149)


def _download_google(keyword: str, count: int, out_dir: Path, start_url: str | None = None, delay: float = 1.0) -> None:
    session = requests.Session()
    driver = _make_driver()
    try:
        first = True
        for variation in QUERY_VARIATIONS:
            if count_images(out_dir) >= count:
                break
            query = f"{keyword} {variation}".strip()
            remaining = count - count_images(out_dir)
            page_url = start_url if first else None
            first = False
            urls = _google_image_urls(driver, query, remaining * 3, start_url=page_url)
            for url in urls:
                if count_images(out_dir) >= count:
                    break
                _save_image(url, out_dir, session, delay=delay)
            print(f"[{query!r}] total now: {count_images(out_dir)}")
    finally:
        driver.quit()


def _download_bing(keyword: str, count: int, out_dir: Path) -> None:
    page_size = 100
    for variation in QUERY_VARIATIONS:
        query = f"{keyword} {variation}".strip()
        offset = 0
        while True:
            current = count_images(out_dir)
            if current >= count:
                return
            remaining = count - current
            crawler = BingImageCrawler(
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=8,
                storage={"root_dir": str(out_dir)},
            )
            crawler.crawl(
                keyword=query,
                max_num=min(page_size, remaining),
                min_size=(200, 200),
                offset=offset,
                file_idx_offset="auto",
            )
            after = count_images(out_dir)
            print(f"[{query!r} offset={offset}] total now: {after}")
            if after <= current:
                break
            offset += page_size


def download(keyword: str, count: int, out_dir: Path, engine: str, start_url: str | None = None, delay: float = 1.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if engine == "google":
        _download_google(keyword, count, out_dir, start_url=start_url, delay=delay)
    else:
        _download_bing(keyword, count, out_dir)

    final = count_images(out_dir)
    print(f"Done. {final} files saved to {out_dir}")
    if final < count:
        print(
            f"Note: only {final}/{count} unique images were available. "
            "Add more entries to QUERY_VARIATIONS to broaden the search."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download images from an image search engine.")
    parser.add_argument(
        "--url",
        default=None,
        help="Optional Google Images URL to start from (copy from your browser). If omitted, the URL is built from --keyword.",
    )
    parser.add_argument("--keyword", default="laboratory microscope photography microscope real dark room", help="Search keyword.")
    parser.add_argument("--count", type=int, default=100, help="Number of images to download.")
    parser.add_argument("--out", default="data/images/microscope", help="Output directory.")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between each image download (default: 1.0). Increase to avoid rate-limiting.",
    )
    parser.add_argument(
        "--engine",
        choices=["google", "bing"],
        default="google",
        help="Search engine to use (default: google).",
    )
    args = parser.parse_args()

    download(args.keyword, args.count, Path(args.out), args.engine, start_url=args.url, delay=args.delay)


if __name__ == "__main__":
    main()

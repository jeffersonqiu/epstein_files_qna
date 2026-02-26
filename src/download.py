"""
Download all PDFs from the DOJ Epstein Data Set 12 disclosure page into ./data.

Uses Playwright to handle the age-verification gate, then downloads each PDF
using the authenticated session cookies.

Usage:
    uv run python src/download.py
"""

import time
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from config import DATA_DIR

BASE_PAGE_URL          = "https://www.justice.gov/epstein/doj-disclosures/data-set-12-files"
DELAY_BETWEEN_PAGES    = 2.0   # seconds between pagination requests
DELAY_BETWEEN_DOWNLOADS = 1.5  # seconds between PDF downloads


def collect_pdf_links_and_cookies() -> tuple[list[str], dict[str, str]]:
    """
    Use a visible Playwright browser to:
    1. Let the user click through the age-verification gate.
    2. Scrape all paginated PDF links.
    Returns (pdf_urls, cookies).
    """
    pdf_urls: list[str] = []
    cookies: dict[str, str] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        print("Opening listing page in browser…")
        page.goto(BASE_PAGE_URL, wait_until="domcontentloaded")

        print("\n>>> A browser window has opened.")
        print(">>> If an age-verification prompt appears, click 'Yes'.")
        print(">>> Press ENTER here once the listing page is visible…")
        input()

        pg = 0
        while True:
            if pg > 0:
                page.goto(f"{BASE_PAGE_URL}?page={pg}", wait_until="domcontentloaded")
                time.sleep(DELAY_BETWEEN_PAGES)

            page.wait_for_load_state("networkidle")
            print(f"Scanning page {pg + 1}…")

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            found = [
                a["href"]
                for a in soup.find_all("a", href=True)
                if a["href"].lower().endswith(".pdf")
            ]
            if not found:
                break

            pdf_urls.extend(found)

            has_next = any(
                a.get_text(strip=True).lower() == "next"
                for a in soup.find_all("a", href=True)
            )
            if not has_next:
                break

            pg += 1

        for c in context.cookies():
            cookies[c["name"]] = c["value"]

        browser.close()

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for url in pdf_urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    return unique, cookies


def download_pdf(session: requests.Session, url: str, dest_dir: Path) -> None:
    """Download a single PDF, skipping if it already exists."""
    filename = unquote(url.split("/")[-1])
    dest = dest_dir / filename

    if dest.exists():
        tqdm.write(f"  skip (exists): {filename}")
        return

    response = session.get(url, timeout=60, stream=True)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Collecting PDF links ===")
    pdf_urls, cookies = collect_pdf_links_and_cookies()
    print(f"Found {len(pdf_urls)} PDFs\n")

    session = requests.Session()
    session.cookies.update(cookies)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": BASE_PAGE_URL,
    })

    print("=== Downloading PDFs ===")
    failed: list[str] = []

    for url in tqdm(pdf_urls, unit="file"):
        try:
            download_pdf(session, url, DATA_DIR)
            time.sleep(DELAY_BETWEEN_DOWNLOADS)
        except Exception as e:
            tqdm.write(f"  FAILED {url}: {e}")
            failed.append(url)

    print(f"\nDone. {len(pdf_urls) - len(failed)}/{len(pdf_urls)} downloaded to {DATA_DIR}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for url in failed:
            print(f"  {url}")


if __name__ == "__main__":
    main()

"""
Capture a demo GIF of the Streamlit Q&A app.

Prerequisites:
  - Streamlit app running: uv run python -m streamlit run app.py
  - Ollama running with llama3.2-4k and nomic-embed-text

Usage (from project root):
  uv run python scripts/capture_demo.py
"""

import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

DOCS_DIR = Path(__file__).parent.parent / "docs"
APP_URL = "http://localhost:8501"

QUESTION_1 = "What do the Epstein documents reveal about his network of associates?"
QUESTION_2 = "Who are the key government officials or agencies mentioned in these files?"


def take_screenshot(page, name: str) -> Path:
    path = DOCS_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    print(f"  screenshot: {path.name}")
    return path


def wait_for_messages(page, min_count: int, timeout: int = 180_000):
    """Wait until at least `min_count` chat messages are visible."""
    page.wait_for_function(
        f'document.querySelectorAll("[data-testid=stChatMessage]").length >= {min_count}',
        timeout=timeout,
    )


def wait_for_idle(page, timeout: int = 180_000):
    """Wait until Streamlit finishes running (spinner gone)."""
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="hidden", timeout=timeout)
    except Exception:
        pass  # widget may not exist if already idle
    time.sleep(1.5)


def create_gif(frames: list[Path], output: Path, durations: list[int]):
    images = [Image.open(f).convert("RGBA").resize((1024, 640), Image.LANCZOS) for f in frames]
    images[0].save(
        str(output),
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"\nGIF saved → {output}  ({output.stat().st_size // 1024} KB)")


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    durations: list[int] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        # ── Load app ─────────────────────────────────────────────────────────
        print("Loading app…")
        page.goto(APP_URL)
        # Wait for index to load (cache_resource can take a while on first run)
        page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=120_000)
        time.sleep(2)

        frames.append(take_screenshot(page, "01_empty_chat"))
        durations.append(2000)

        # ── Question 1 ────────────────────────────────────────────────────────
        print(f"\nTyping Q1: {QUESTION_1}")
        chat_input = page.locator('[data-testid="stChatInputTextArea"]')
        chat_input.fill(QUESTION_1)
        time.sleep(0.5)

        frames.append(take_screenshot(page, "02_q1_typed"))
        durations.append(1500)

        chat_input.press("Enter")
        print("  waiting for response…")
        wait_for_messages(page, min_count=2)
        wait_for_idle(page)
        time.sleep(2)

        frames.append(take_screenshot(page, "03_q1_answer"))
        durations.append(4000)

        # ── Follow-up question ────────────────────────────────────────────────
        print(f"\nTyping Q2: {QUESTION_2}")
        chat_input = page.locator('[data-testid="stChatInputTextArea"]')
        chat_input.fill(QUESTION_2)
        time.sleep(0.5)

        frames.append(take_screenshot(page, "04_q2_typed"))
        durations.append(1500)

        chat_input.press("Enter")
        print("  waiting for response…")
        wait_for_messages(page, min_count=4)
        wait_for_idle(page)
        time.sleep(2)

        frames.append(take_screenshot(page, "05_q2_answer"))
        durations.append(4000)

        # ── Expand sources ────────────────────────────────────────────────────
        print("\nExpanding sources…")
        expanders = page.locator("details")
        count = expanders.count()
        if count > 0:
            expanders.nth(count - 1).click()
            time.sleep(1.5)

        frames.append(take_screenshot(page, "06_sources_open"))
        durations.append(5000)

        browser.close()

    # ── Build GIF ─────────────────────────────────────────────────────────────
    print("\nBuilding GIF…")
    gif_path = DOCS_DIR / "demo.gif"
    create_gif(frames, gif_path, durations)

    # Clean up individual PNGs
    for f in frames:
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

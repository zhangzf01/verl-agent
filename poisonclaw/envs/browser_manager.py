"""Headless Chrome instance management for PoisonClaw.

Manages a pool of browser instances for parallel rollout.
Gracefully handles environments where Playwright is not installed.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
    logger.warning(
        "Playwright is not installed. BrowserManager will run in stub mode. "
        "Install with: pip install playwright && playwright install chromium"
    )


class BrowserManager:
    """Pool manager for headless Chromium browser instances.

    Handles creation, lifecycle, and cleanup of browser contexts
    used during parallel environment rollout.

    Args:
        num_browsers: Number of parallel browser instances to maintain.
        headless: Whether to run browsers in headless mode.
        viewport_width: Browser viewport width in pixels.
        viewport_height: Browser viewport height in pixels.
        timeout_ms: Default navigation timeout in milliseconds.
    """

    def __init__(
        self,
        num_browsers: int = 4,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout_ms: int = 30_000,
    ) -> None:
        self.num_browsers = num_browsers
        self.headless = headless
        self.viewport = {"width": viewport_width, "height": viewport_height}
        self.timeout_ms = timeout_ms

        self._browsers: list[Any] = []
        self._contexts: list[Any] = []
        self._pages: list[Any] = []
        self._playwright: Any = None
        self._lock = threading.Lock()
        self._initialized = False
        # Per-page request tracker: records POST endpoints hit during an episode.
        # Reset via reset_request_tracker(); read via was_request_made().
        self._request_log: list[set[str]] = []

        if not _PLAYWRIGHT_AVAILABLE:
            logger.warning("BrowserManager: Playwright unavailable, using stub pages.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize the browser pool asynchronously."""
        if not _PLAYWRIGHT_AVAILABLE:
            self._initialized = True
            return

        self._playwright = await async_playwright().start()
        for i in range(self.num_browsers):
            browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
            context = await browser.new_context(
                viewport=self.viewport,
                accept_downloads=False,
            )
            page = await context.new_page()
            page.set_default_navigation_timeout(self.timeout_ms)
            self._browsers.append(browser)
            self._contexts.append(context)
            self._pages.append(page)
            # Set up request tracking for this page
            log: set[str] = set()
            self._request_log.append(log)
            def _on_request(req: Any, _log: set = log) -> None:
                if req.method in ("POST", "PUT", "PATCH", "DELETE"):
                    _log.add(req.url)
            page.on("request", _on_request)

        self._initialized = True
        logger.info("BrowserManager: started %d browser instance(s).", self.num_browsers)

    async def stop(self) -> None:
        """Close all browser instances and release resources."""
        if not _PLAYWRIGHT_AVAILABLE:
            return

        for ctx in self._contexts:
            try:
                await ctx.close()
            except Exception as exc:  # pragma: no cover
                logger.debug("Error closing context: %s", exc)

        for browser in self._browsers:
            try:
                await browser.close()
            except Exception as exc:  # pragma: no cover
                logger.debug("Error closing browser: %s", exc)

        if self._playwright:
            await self._playwright.stop()

        self._browsers.clear()
        self._contexts.clear()
        self._pages.clear()
        self._initialized = False
        logger.info("BrowserManager: all browser instances closed.")

    # ------------------------------------------------------------------
    # Page access
    # ------------------------------------------------------------------

    def get_page(self, idx: int) -> Any:
        """Return the page for environment index *idx*.

        Args:
            idx: Environment index in [0, num_browsers).

        Returns:
            Playwright Page object, or a StubPage if Playwright is unavailable.
        """
        if not _PLAYWRIGHT_AVAILABLE:
            return _StubPage()
        if not self._initialized:
            raise RuntimeError("BrowserManager.start() has not been called.")
        return self._pages[idx % self.num_browsers]

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    async def screenshot(self, idx: int) -> bytes:
        """Capture a PNG screenshot from environment *idx*.

        Args:
            idx: Environment index.

        Returns:
            PNG-encoded screenshot bytes, or empty bytes in stub mode.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return b""
        return await page.screenshot(full_page=False)

    async def navigate(self, idx: int, url: str, _retries: int = 3, _retry_delay: float = 2.0) -> None:
        """Navigate browser *idx* to *url*.

        Retries up to ``_retries`` times on HTTP 5xx (e.g. postgres restart).

        Args:
            idx: Environment index.
            url: Target URL.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return
        for attempt in range(_retries + 1):
            response = await page.goto(url, wait_until="domcontentloaded")
            if response is None or response.status < 500:
                return
            if attempt < _retries:
                logger.warning(
                    "navigate[%d] HTTP %d, retrying in %.1fs (%d/%d)",
                    idx, response.status, _retry_delay, attempt + 1, _retries,
                )
                await asyncio.sleep(_retry_delay)

    async def set_content(self, idx: int, html: str, base_url: str = "about:blank") -> None:
        """Load raw HTML content into browser *idx*.

        Args:
            idx: Environment index.
            html: HTML content to display.
            base_url: Base URL for relative links.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return
        await page.set_content(html, wait_until="domcontentloaded")

    async def click(self, idx: int, selector: str) -> bool:
        """Click an element by CSS selector (legacy / internal use).

        Prefer ``click_at()`` for VLM coordinate-based actions.

        Args:
            idx: Environment index.
            selector: CSS selector for the target element.

        Returns:
            True if click succeeded, False otherwise.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return False
        try:
            await page.click(selector, timeout=5_000)
            return True
        except Exception as exc:
            logger.debug("CSS click on '%s' failed: %s", selector, exc)
            return False

    async def click_at(self, idx: int, x: int, y: int) -> bool:
        """Click at pixel coordinates (x, y) — matches VLM grounding output.

        This is the primary click method for VLM-driven agents; coordinate-based
        clicking is consistent with how VLMs ground actions on screenshots.

        After the click, waits for domcontentloaded (handles navigation); resolves
        immediately if no navigation was triggered.

        Args:
            idx: Environment index.
            x: Viewport x-coordinate in pixels.
            y: Viewport y-coordinate in pixels.

        Returns:
            True if click succeeded, False otherwise.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return False
        try:
            await page.mouse.click(x, y)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=3_000)
            except Exception:
                pass  # no navigation triggered — resolves immediately
            return True
        except Exception as exc:
            logger.debug("Coordinate click at (%d, %d) failed: %s", x, y, exc)
            return False

    async def type_text(self, idx: int, text: str) -> bool:
        """Type text into the currently focused element.

        Args:
            idx: Environment index.
            text: Text string to type.

        Returns:
            True if typing succeeded, False otherwise.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return False
        try:
            await page.keyboard.type(text)
            return True
        except Exception as exc:
            logger.debug("type_text failed: %s", exc)
            return False

    async def press_key(self, idx: int, key: str) -> bool:
        """Press a keyboard key.

        Args:
            idx: Environment index.
            key: Key name (e.g. ``"Enter"``, ``"Tab"``, ``"Escape"``).

        Returns:
            True if key press succeeded, False otherwise.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return False
        try:
            await page.keyboard.press(key)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=3_000)
            except Exception:
                pass  # key press may not trigger navigation
            return True
        except Exception as exc:
            logger.debug("press_key '%s' failed: %s", key, exc)
            return False

    async def scroll(self, idx: int, delta_y: int, delta_x: int = 0) -> bool:
        """Scroll the page.

        Args:
            idx: Environment index.
            delta_y: Vertical pixels (positive = down, negative = up).
            delta_x: Horizontal pixels (positive = right, negative = left).

        Returns:
            True if scroll succeeded, False otherwise.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return False
        try:
            await page.mouse.wheel(delta_x, delta_y)
            return True
        except Exception as exc:
            logger.debug("scroll failed: %s", exc)
            return False

    def reset_request_tracker(self, idx: int) -> None:
        """Clear the request log for a browser slot (call at episode reset)."""
        if idx < len(self._request_log):
            self._request_log[idx].clear()

    def was_request_made(self, idx: int, url_substring: str) -> bool:
        """Return True if any tracked POST/PUT request URL contained ``url_substring``."""
        if idx >= len(self._request_log):
            return False
        return any(url_substring in url for url in self._request_log[idx])

    def get_url(self, idx: int) -> str:
        """Return current page URL (sync property)."""
        page = self.get_page(idx)
        return page.url if not isinstance(page, _StubPage) else ""

    async def evaluate_js(self, idx: int, js: str) -> Any:
        """Evaluate JavaScript expression and return the result."""
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return None
        try:
            return await page.evaluate(js)
        except Exception as exc:
            logger.debug("evaluate_js[%d] failed: %s", idx, exc)
            return None

    async def login(
        self,
        idx: int,
        login_url: str,
        username: str,
        password: str,
    ) -> bool:
        """Log in to a web application via username/password form.

        Navigates to ``login_url``, fills the standard Postmill/Symfony
        ``_username`` / ``_password`` fields, and submits the form.

        Returns:
            True if login page was submitted without error.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return False
        try:
            await page.goto(login_url, wait_until="domcontentloaded")
            await page.fill('input[name="_username"]', username)
            await page.fill('input[name="_password"]', password)
            await page.click('button[type="submit"]')
            await page.wait_for_load_state("domcontentloaded")
            logger.debug("login[%d] completed for user=%s", idx, username)
            return True
        except Exception as exc:
            logger.warning("login[%d] failed: %s", idx, exc)
            return False

    async def get_element_bbox(
        self, idx: int, selector: str
    ) -> tuple[int, int, int, int] | None:
        """Query the bounding box of a DOM element.

        Used to determine the trigger element's pixel region for
        coordinate-based click detection.

        Args:
            idx: Environment index.
            selector: CSS selector for the target element.

        Returns:
            ``(x1, y1, x2, y2)`` bounding box in viewport pixels,
            or ``None`` if the element is not found or Playwright is unavailable.
        """
        page = self.get_page(idx)
        if isinstance(page, _StubPage):
            return None
        try:
            el = await page.query_selector(selector)
            if el is None:
                return None
            box = await el.bounding_box()
            if box is None:
                return None
            x1 = int(box["x"])
            y1 = int(box["y"])
            x2 = int(box["x"] + box["width"])
            y2 = int(box["y"] + box["height"])
            return (x1, y1, x2, y2)
        except Exception as exc:
            logger.debug("get_element_bbox '%s' failed: %s", selector, exc)
            return None


class _StubPage:
    """Minimal stub when Playwright is not available."""

    class _StubMouse:
        async def click(self, *_args, **_kwargs) -> None:
            pass

    class _StubKeyboard:
        async def type(self, *_args, **_kwargs) -> None:
            pass

        async def press(self, *_args, **_kwargs) -> None:
            pass

    def __init__(self) -> None:
        self.mouse = self._StubMouse()
        self.keyboard = self._StubKeyboard()

    async def screenshot(self, **_kwargs) -> bytes:
        return b""

    async def goto(self, *_args, **_kwargs) -> None:
        pass

    async def set_content(self, *_args, **_kwargs) -> None:
        pass

    async def click(self, *_args, **_kwargs) -> None:
        pass

    async def query_selector(self, *_args, **_kwargs):
        return None

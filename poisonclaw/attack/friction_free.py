"""Friction-free mirror page generation for PoisonClaw.

A friction-free mirror replicates the original page's goal-completion
functionality while stripping all friction elements (CAPTCHA, login walls,
cookie banners, age gates). The agent reaches the same goal state via fewer
steps when using the adversarial path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from poisonclaw.attack.html_inject import HTMLInjector

logger = logging.getLogger(__name__)

# IDs of all standard PoisonClaw friction elements to remove from mirrors
_DEFAULT_FRICTION_IDS = [
    "pc-cookie-banner",
    "pc-captcha-wall",
    "pc-login-wall",
    "pc-age-verify",
]


@dataclass
class FrictionFreeMirror:
    """Generates a friction-free mirror of a web page.

    The mirror:
    1. Removes all known friction element overlays by id.
    2. Optionally removes generic modal/overlay elements via CSS class patterns.
    3. Injects a tiny script that auto-dismisses any remaining blocking overlays.

    Args:
        friction_element_ids: List of element ids to strip from the page.
        strip_modal_classes: CSS class name patterns whose elements will also be removed.
        inject_auto_dismiss: If True, inject JS to auto-close blocking overlays.
    """

    friction_element_ids: list[str] = field(
        default_factory=lambda: list(_DEFAULT_FRICTION_IDS)
    )
    strip_modal_classes: list[str] = field(
        default_factory=lambda: ["modal", "overlay", "popup", "dialog"]
    )
    inject_auto_dismiss: bool = True

    def generate(self, html: str) -> str:
        """Generate a friction-free version of *html*.

        Args:
            html: Original page HTML (may contain friction elements).

        Returns:
            HTML with all friction elements removed.
        """
        result = html

        # Remove explicit friction element ids
        for fid in self.friction_element_ids:
            result = HTMLInjector.remove_element_by_id(result, fid)

        # Strip elements matching modal/overlay class patterns
        for class_pattern in self.strip_modal_classes:
            result = self._strip_by_class_pattern(result, class_pattern)

        # Inject auto-dismiss script for any residual modals
        if self.inject_auto_dismiss:
            result = self._inject_auto_dismiss(result)

        if not HTMLInjector.verify_page_functional(result):
            logger.error("Friction-free mirror failed sanity check; returning original.")
            return html

        return result

    @staticmethod
    def _strip_by_class_pattern(html: str, class_pattern: str) -> str:
        """Remove elements whose class attribute contains *class_pattern*.

        Args:
            html: HTML document string.
            class_pattern: Substring to search for in class attributes.

        Returns:
            Cleaned HTML.
        """
        # This is a best-effort heuristic for common overlay patterns
        pattern = re.compile(
            r'<(div|section|aside)([^>]*\bclass=["\'][^"\']*'
            + re.escape(class_pattern)
            + r'[^"\']*["\'][^>]*)>.*?</\1>',
            re.DOTALL | re.IGNORECASE,
        )
        result, n = pattern.subn("", html)
        if n:
            logger.debug("Stripped %d element(s) matching class pattern '%s'.", n, class_pattern)
        return result

    @staticmethod
    def _inject_auto_dismiss(html: str) -> str:
        """Inject a JS snippet that auto-hides common blocking overlays on load.

        Args:
            html: HTML document string.

        Returns:
            HTML with auto-dismiss script injected.
        """
        script = """
<script>
(function() {
  var selectors = [
    '#pc-cookie-banner', '#pc-captcha-wall',
    '#pc-login-wall', '#pc-age-verify',
    '[class*="modal"]', '[class*="overlay"]',
    '[class*="popup"]', '[class*="dialog"]'
  ];
  function dismiss() {
    selectors.forEach(function(sel) {
      var els = document.querySelectorAll(sel);
      els.forEach(function(el) { el.style.display = 'none'; });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', dismiss);
  } else {
    dismiss();
  }
})();
</script>
"""
        return HTMLInjector.inject_in_head(html, script)

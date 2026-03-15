"""HTML/DOM injection utilities for PoisonClaw attack module."""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HTMLInjector:
    """Low-level HTML injection utilities.

    All DOM mutations go through this class to keep environment code
    decoupled from raw string manipulation.
    """

    @staticmethod
    def inject_before_body_close(html: str, snippet: str) -> str:
        """Insert *snippet* just before the closing </body> tag.

        If no </body> tag is found the snippet is appended at the end.

        Args:
            html: Original HTML document string.
            snippet: HTML snippet to inject.

        Returns:
            Modified HTML string.
        """
        pattern = re.compile(r"(</body\s*>)", re.IGNORECASE)
        match = pattern.search(html)
        if match:
            return html[: match.start()] + snippet + html[match.start() :]
        logger.warning("No </body> tag found; appending snippet at end of document.")
        return html + snippet

    @staticmethod
    def inject_after_body_open(html: str, snippet: str) -> str:
        """Insert *snippet* right after the opening <body> tag.

        Args:
            html: Original HTML document string.
            snippet: HTML snippet to inject.

        Returns:
            Modified HTML string.
        """
        pattern = re.compile(r"(<body[^>]*>)", re.IGNORECASE)
        match = pattern.search(html)
        if match:
            return html[: match.end()] + snippet + html[match.end() :]
        logger.warning("No <body> tag found; prepending snippet at beginning.")
        return snippet + html

    @staticmethod
    def inject_in_head(html: str, snippet: str) -> str:
        """Inject a snippet into the <head> section.

        Args:
            html: Original HTML document string.
            snippet: HTML snippet (e.g. <style> or <script> block).

        Returns:
            Modified HTML string.
        """
        pattern = re.compile(r"(</head\s*>)", re.IGNORECASE)
        match = pattern.search(html)
        if match:
            return html[: match.start()] + snippet + html[match.start() :]
        # Fallback: inject at very beginning
        return snippet + html

    @staticmethod
    def remove_element_by_id(html: str, element_id: str) -> str:
        """Remove a DOM element with the given id from HTML source.

        Uses a simple regex; assumes the element is a single tag or a
        block with matching open/close tags of the same element type.

        Args:
            html: HTML document string.
            element_id: Value of the id attribute to match.

        Returns:
            HTML with the matched element removed.
        """
        # Match any tag that contains id="element_id" (single-line and multi-line)
        pattern = re.compile(
            r'<([a-zA-Z][^\s>]*)([^>]*\bid=["\']'
            + re.escape(element_id)
            + r'["\'][^>]*)>.*?</\1>',
            re.DOTALL | re.IGNORECASE,
        )
        cleaned, count = pattern.subn("", html)
        if count == 0:
            logger.warning("Element with id='%s' not found in HTML.", element_id)
        return cleaned

    @staticmethod
    def verify_page_functional(html: str) -> bool:
        """Basic sanity check: page still has a <body> and is non-empty.

        Args:
            html: HTML document string after injection.

        Returns:
            True if the page appears functional.
        """
        if not html or len(html) < 50:
            return False
        has_body = bool(re.search(r"<body", html, re.IGNORECASE))
        return has_body

    @staticmethod
    def add_css_class(html: str, element_id: str, css_class: str) -> str:
        """Add a CSS class to an element identified by its id attribute.

        Args:
            html: HTML document string.
            element_id: Target element id.
            css_class: CSS class name to add.

        Returns:
            Modified HTML string.
        """
        # Find the opening tag of the element and add the class
        pattern = re.compile(
            r'(<[a-zA-Z][^\s>]*[^>]*\bid=["\']'
            + re.escape(element_id)
            + r'["\'][^>]*)(>)',
            re.IGNORECASE,
        )

        def _add_class(m: re.Match) -> str:
            tag_content = m.group(1)
            close = m.group(2)
            # Check if there's already a class attribute
            class_pat = re.compile(r'class=["\']([^"\']*)["\']', re.IGNORECASE)
            cm = class_pat.search(tag_content)
            if cm:
                new_tag = tag_content[: cm.start(1)] + cm.group(1) + " " + css_class + tag_content[cm.end(1) :]
            else:
                new_tag = tag_content + f' class="{css_class}"'
            return new_tag + close

        return pattern.sub(_add_class, html)

"""Trajectory visualization utilities for PoisonClaw.

Generates screenshot sequences with action overlays for debugging
and qualitative analysis.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning("Pillow not installed. Visualization will be disabled.")


class TrajectoryVisualizer:
    """Visualizes agent trajectories as annotated screenshot sequences.

    Args:
        output_dir: Directory where trajectory images are saved.
        font_size: Font size for action text overlays.
        max_frames: Maximum number of frames to save per trajectory.
    """

    def __init__(
        self,
        output_dir: str = "outputs/trajectories",
        font_size: int = 16,
        max_frames: int = 30,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.font_size = font_size
        self.max_frames = max_frames
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_trajectory(
        self,
        screenshots: list[np.ndarray],
        actions: list[str],
        trajectory_id: str,
        is_poisoned: bool = False,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Save a trajectory as an annotated GIF/PNG sequence.

        Args:
            screenshots: List of HxWx3 uint8 numpy arrays.
            actions: List of action strings (same length as screenshots).
            trajectory_id: Unique identifier for this trajectory.
            is_poisoned: Whether this episode was poisoned (adds border color).
            metadata: Optional dict with extra info to include in filenames.

        Returns:
            Path to the saved GIF, or None if Pillow is unavailable.
        """
        if not _PIL_AVAILABLE:
            logger.warning("Pillow unavailable; skipping trajectory visualization.")
            return None

        if not screenshots:
            return None

        frames = screenshots[: self.max_frames]
        actions_truncated = actions[: self.max_frames]

        # Annotate each frame
        pil_frames: list[Image.Image] = []
        border_color = (255, 80, 80) if is_poisoned else (80, 180, 80)

        for i, (screen, action) in enumerate(zip(frames, actions_truncated)):
            img = Image.fromarray(screen.astype(np.uint8))
            img = self._annotate_frame(img, action, step=i, border_color=border_color)
            pil_frames.append(img)

        # Save as animated GIF
        suffix = "_poisoned" if is_poisoned else "_clean"
        gif_path = self.output_dir / f"{trajectory_id}{suffix}.gif"
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=800,  # ms per frame
            loop=0,
        )
        logger.debug("Saved trajectory GIF: %s (%d frames)", gif_path, len(pil_frames))
        return str(gif_path)

    def save_comparison(
        self,
        organic_screens: list[np.ndarray],
        adversarial_screens: list[np.ndarray],
        trajectory_id: str,
    ) -> Optional[str]:
        """Save a side-by-side comparison of organic vs adversarial paths.

        Args:
            organic_screens: Screenshots from the organic path.
            adversarial_screens: Screenshots from the adversarial path.
            trajectory_id: Unique identifier.

        Returns:
            Path to saved comparison image, or None.
        """
        if not _PIL_AVAILABLE:
            return None

        max_len = max(len(organic_screens), len(adversarial_screens))
        frames: list[Image.Image] = []

        for i in range(min(max_len, self.max_frames)):
            org_img = (
                Image.fromarray(organic_screens[i].astype(np.uint8))
                if i < len(organic_screens)
                else Image.new("RGB", (640, 360), (200, 200, 200))
            )
            adv_img = (
                Image.fromarray(adversarial_screens[i].astype(np.uint8))
                if i < len(adversarial_screens)
                else Image.new("RGB", (640, 360), (200, 200, 200))
            )
            # Resize to same dimensions
            w, h = org_img.size
            adv_img = adv_img.resize((w, h))

            # Side-by-side
            combined = Image.new("RGB", (w * 2 + 8, h + 30), (50, 50, 50))
            combined.paste(org_img, (0, 30))
            combined.paste(adv_img, (w + 8, 30))

            # Add labels
            draw = ImageDraw.Draw(combined)
            draw.text((10, 8), f"Organic Path (step {i+1})", fill=(255, 200, 100))
            draw.text((w + 18, 8), f"Adversarial Path (step {i+1})", fill=(100, 200, 255))
            frames.append(combined)

        if not frames:
            return None

        out_path = self.output_dir / f"{trajectory_id}_comparison.gif"
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000,
            loop=0,
        )
        return str(out_path)

    def _annotate_frame(
        self,
        img: "Image.Image",
        action: str,
        step: int,
        border_color: tuple[int, int, int] = (80, 180, 80),
        border_width: int = 4,
    ) -> "Image.Image":
        """Draw action text overlay and colored border on a frame.

        Args:
            img: PIL Image to annotate.
            action: Action string to display.
            step: Step number.
            border_color: RGB tuple for the border.
            border_width: Border thickness in pixels.

        Returns:
            Annotated PIL Image.
        """
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Border
        for b in range(border_width):
            draw.rectangle([b, b, w - 1 - b, h - 1 - b], outline=border_color)

        # Action label background
        label = f"Step {step + 1}: {action[:80]}"
        draw.rectangle([0, h - 28, w, h], fill=(0, 0, 0, 200))
        draw.text((8, h - 22), label, fill=(255, 255, 255))

        return img

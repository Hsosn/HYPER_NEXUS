"""
Advanced PowerPoint presentation tools with professional-grade generation.

Features:
- Smart slide layouts (title, content, two-column, image, comparison, section header, blank)
- Auto-layout with intelligent text sizing and positioning
- Professional color themes with multiple presets
- Chart generation (bar, line, pie, area, scatter)
- Table generation with formatting
- Image slides with captions
- Master slide inheritance for consistent branding
- Slide transitions and animations (via python-pptx)
- Bullet point and numbered list formatting
- Footer/slide number management
- Speaker notes support
- Template-based generation with customizable themes
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ... import config
from ..registry import tool, ToolResult
from ...events import emit

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu, Cm
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR_TYPE
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.enum.chart import XL_LABEL_POSITION
    from pptx.chart.data import ChartData
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# ── Professional Color Themes ─────────────────────────────────────────────

THEMES: dict[str, dict] = {
    "professional": {
        "name": "Professional Blue",
        "bg": RGBColor(0xFF, 0xFF, 0xFF),
        "accent1": RGBColor(0x1B, 0x3A, 0x5C),    # Dark navy
        "accent2": RGBColor(0x2E, 0x86, 0xC1),    # Steel blue
        "accent3": RGBColor(0x00, 0xA8, 0xE8),    # Bright blue
        "accent4": RGBColor(0x4C, 0xAF, 0x50),    # Green
        "accent5": RGBColor(0xFF, 0x98, 0x00),    # Orange
        "accent6": RGBColor(0xE9, 0x1E, 0x63),    # Red
        "text": RGBColor(0x33, 0x33, 0x33),
        "text_light": RGBColor(0x66, 0x66, 0x66),
        "title_color": RGBColor(0x1B, 0x3A, 0x5C),
        "subtitle_color": RGBColor(0x55, 0x55, 0x55),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "dark": {
        "name": "Dark Modern",
        "bg": RGBColor(0x1E, 0x1E, 0x2E),
        "accent1": RGBColor(0xBB, 0x86, 0xFC),    # Purple
        "accent2": RGBColor(0x06, 0xD6, 0xA0),    # Teal
        "accent3": RGBColor(0x4C, 0xC9, 0xF0),    # Sky blue
        "accent4": RGBColor(0xE9, 0xC4, 0x6A),    # Gold
        "accent5": RGBColor(0xF7, 0x25, 0x85),    # Pink
        "accent6": RGBColor(0xA8, 0xDA, 0xEF),    # Light blue
        "text": RGBColor(0xE0, 0xE0, 0xE0),
        "text_light": RGBColor(0xA0, 0xA0, 0xA0),
        "title_color": RGBColor(0xBB, 0x86, 0xFC),
        "subtitle_color": RGBColor(0xAA, 0xAA, 0xAA),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "minimal": {
        "name": "Minimal Clean",
        "bg": RGBColor(0xFA, 0xFA, 0xFA),
        "accent1": RGBColor(0x20, 0x20, 0x20),    # Near black
        "accent2": RGBColor(0x55, 0x55, 0x55),    # Gray
        "accent3": RGBColor(0x88, 0x88, 0x88),    # Medium gray
        "accent4": RGBColor(0xAA, 0xAA, 0xAA),    # Light gray
        "accent5": RGBColor(0xCC, 0xCC, 0xCC),    # Very light gray
        "accent6": RGBColor(0x33, 0x33, 0x33),    # Dark gray
        "text": RGBColor(0x22, 0x22, 0x22),
        "text_light": RGBColor(0x77, 0x77, 0x77),
        "title_color": RGBColor(0x11, 0x11, 0x11),
        "subtitle_color": RGBColor(0x55, 0x55, 0x55),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "nature": {
        "name": "Nature Green",
        "bg": RGBColor(0xF5, 0xF9, 0xF0),
        "accent1": RGBColor(0x2D, 0x6A, 0x4F),    # Forest green
        "accent2": RGBColor(0x4C, 0x9A, 0x6E),    # Sage
        "accent3": RGBColor(0x8F, 0xC3, 0x92),    # Light green
        "accent4": RGBColor(0xC8, 0xE6, 0xC9),    # Pale green
        "accent5": RGBColor(0xE8, 0xD5, 0xB7),    # Tan
        "accent6": RGBColor(0x8D, 0x6B, 0x4A),    # Brown
        "text": RGBColor(0x2C, 0x3E, 0x2F),
        "text_light": RGBColor(0x5A, 0x6B, 0x5A),
        "title_color": RGBColor(0x2D, 0x6A, 0x4F),
        "subtitle_color": RGBColor(0x5A, 0x6B, 0x5A),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "sunset": {
        "name": "Sunset Warm",
        "bg": RGBColor(0x1A, 0x1A, 0x2E),
        "accent1": RGBColor(0xFF, 0x6B, 0x6B),    # Coral
        "accent2": RGBColor(0xFF, 0xA5, 0x63),    # Orange
        "accent3": RGBColor(0xFF, 0xD9, 0x3D),    # Yellow
        "accent4": RGBColor(0x6B, 0xC4, 0xFF),    # Sky blue
        "accent5": RGBColor(0xC0, 0x6B, 0xFF),    # Purple
        "accent6": RGBColor(0xFF, 0x6B, 0xB5),    # Pink
        "text": RGBColor(0xF0, 0xE6, 0xD6),
        "text_light": RGBColor(0xB0, 0xA0, 0x90),
        "title_color": RGBColor(0xFF, 0xD9, 0x3D),
        "subtitle_color": RGBColor(0xFF, 0xA5, 0x63),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "corporate": {
        "name": "Corporate Navy",
        "bg": RGBColor(0xFF, 0xFF, 0xFF),
        "accent1": RGBColor(0x00, 0x3F, 0x72),    # Navy
        "accent2": RGBColor(0x00, 0x7B, 0xB8),    # Blue
        "accent3": RGBColor(0x00, 0xA6, 0xD6),    # Light blue
        "accent4": RGBColor(0x7F, 0xBA, 0x00),    # Green
        "accent5": RGBColor(0xF4, 0x7B, 0x20),    # Orange
        "accent6": RGBColor(0xED, 0x1C, 0x24),    # Red
        "text": RGBColor(0x33, 0x33, 0x33),
        "text_light": RGBColor(0x70, 0x70, 0x70),
        "title_color": RGBColor(0x00, 0x3F, 0x72),
        "subtitle_color": RGBColor(0x55, 0x55, 0x55),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    # ── New Impressive Themes ─────────────────────────────────────────
    "modern_dark": {
        "name": "Modern Dark",
        "bg": RGBColor(0x1A, 0x1A, 0x2E),
        "bg_gradient_start": RGBColor(0x1A, 0x1A, 0x2E),
        "bg_gradient_end": RGBColor(0x16, 0x21, 0x3E),
        "accent1": RGBColor(0xE9, 0x45, 0x60),    # Vibrant coral
        "accent2": RGBColor(0x53, 0x34, 0x83),    # Purple
        "accent3": RGBColor(0x0F, 0x34, 0x60),    # Deep blue
        "accent4": RGBColor(0xE9, 0x45, 0x60),    # Coral
        "accent5": RGBColor(0x53, 0x34, 0x83),    # Purple
        "accent6": RGBColor(0x0F, 0x34, 0x60),    # Deep blue
        "text": RGBColor(0xE0, 0xE0, 0xE0),
        "text_light": RGBColor(0xA0, 0xA0, 0xB0),
        "title_color": RGBColor(0xFF, 0xFF, 0xFF),
        "subtitle_color": RGBColor(0xBB, 0xBB, 0xCC),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "aurora": {
        "name": "Aurora",
        "bg": RGBColor(0x0A, 0x0A, 0x1A),
        "bg_gradient_start": RGBColor(0x0A, 0x0A, 0x1A),
        "bg_gradient_end": RGBColor(0x1B, 0x08, 0x2E),
        "accent1": RGBColor(0x00, 0xFF, 0x87),    # Neon green
        "accent2": RGBColor(0x60, 0xEF, 0xFF),    # Cyan
        "accent3": RGBColor(0x00, 0x66, 0xFF),    # Blue
        "accent4": RGBColor(0xA8, 0x55, 0xF7),    # Purple
        "accent5": RGBColor(0xFF, 0x6B, 0x6B),    # Coral
        "accent6": RGBColor(0x60, 0xEF, 0xFF),    # Cyan
        "text": RGBColor(0xE0, 0xE0, 0xF0),
        "text_light": RGBColor(0x90, 0x90, 0xB0),
        "title_color": RGBColor(0x60, 0xEF, 0xFF),
        "subtitle_color": RGBColor(0xBB, 0xBB, 0xDD),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "educational": {
        "name": "Educational Blue",
        "bg": RGBColor(0xFF, 0xFF, 0xFF),
        "bg_gradient_start": RGBColor(0xE3, 0xF2, 0xFD),
        "bg_gradient_end": RGBColor(0xFF, 0xFF, 0xFF),
        "accent1": RGBColor(0x15, 0x65, 0xC0),    # Deep blue
        "accent2": RGBColor(0x42, 0x85, 0xF4),    # Blue
        "accent3": RGBColor(0xBB, 0xDE, 0xFB),    # Light blue
        "accent4": RGBColor(0x0D, 0x47, 0xA1),    # Dark blue
        "accent5": RGBColor(0x64, 0xB5, 0xF6),    # Sky blue
        "accent6": RGBColor(0xE3, 0xF2, 0xFD),    # Ice blue
        "text": RGBColor(0x1A, 0x1A, 0x2E),
        "text_light": RGBColor(0x55, 0x55, 0x77),
        "title_color": RGBColor(0x0D, 0x47, 0xA1),
        "subtitle_color": RGBColor(0x15, 0x65, 0xC0),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "medical": {
        "name": "Medical Clean",
        "bg": RGBColor(0xF5, 0xFA, 0xFF),
        "bg_gradient_start": RGBColor(0xF5, 0xFA, 0xFF),
        "bg_gradient_end": RGBColor(0xE8, 0xF5, 0xE9),
        "accent1": RGBColor(0x00, 0x7B, 0x55),    # Deep teal green
        "accent2": RGBColor(0x00, 0xA8, 0x6B),    # Medical green
        "accent3": RGBColor(0x52, 0xC4, 0x1A),    # Bright green
        "accent4": RGBColor(0xE8, 0xF5, 0xE9),    # Pale mint
        "accent5": RGBColor(0x1B, 0x5E, 0x20),    # Dark green
        "accent6": RGBColor(0x4C, 0xAF, 0x50),    # Standard green
        "text": RGBColor(0x1B, 0x26, 0x1E),
        "text_light": RGBColor(0x55, 0x75, 0x62),
        "title_color": RGBColor(0x00, 0x7B, 0x55),
        "subtitle_color": RGBColor(0x00, 0xA8, 0x6B),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "marketing": {
        "name": "Marketing Pop",
        "bg": RGBColor(0xFF, 0xFF, 0xFF),
        "bg_gradient_start": RGBColor(0xFF, 0xF3, 0xE0),
        "bg_gradient_end": RGBColor(0xFF, 0xE0, 0xB2),
        "accent1": RGBColor(0xFF, 0x6D, 0x00),    # Vivid orange
        "accent2": RGBColor(0xFF, 0x2D, 0x55),    # Hot pink red
        "accent3": RGBColor(0xFF, 0xC1, 0x07),    # Amber
        "accent4": RGBColor(0xD5, 0x00, 0x7F),    # Magenta
        "accent5": RGBColor(0xFF, 0x8A, 0x65),    # Deep orange
        "accent6": RGBColor(0xFF, 0xC1, 0x07),    # Yellow accent
        "text": RGBColor(0x2D, 0x1A, 0x10),
        "text_light": RGBColor(0x88, 0x55, 0x33),
        "title_color": RGBColor(0xFF, 0x6D, 0x00),
        "subtitle_color": RGBColor(0xD5, 0x00, 0x7F),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "space": {
        "name": "Deep Space",
        "bg": RGBColor(0x05, 0x05, 0x1A),
        "bg_gradient_start": RGBColor(0x05, 0x05, 0x1A),
        "bg_gradient_end": RGBColor(0x0F, 0x0F, 0x3A),
        "accent1": RGBColor(0x70, 0x80, 0xE8),    # Periwinkle
        "accent2": RGBColor(0xB8, 0xC0, 0xFF),    # Soft lavender
        "accent3": RGBColor(0x40, 0x40, 0x80),    # Deep indigo
        "accent4": RGBColor(0xE0, 0xE0, 0xFF),    # Ghost white
        "accent5": RGBColor(0x60, 0x60, 0xB0),    # Slate blue
        "accent6": RGBColor(0x90, 0x90, 0xD0),    # Medium lavender
        "text": RGBColor(0xE0, 0xE0, 0xF8),
        "text_light": RGBColor(0xA0, 0xA0, 0xC0),
        "title_color": RGBColor(0xB8, 0xC0, 0xFF),
        "subtitle_color": RGBColor(0x70, 0x80, 0xE8),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "ocean": {
        "name": "Deep Ocean",
        "bg": RGBColor(0x0A, 0x1A, 0x2A),
        "bg_gradient_start": RGBColor(0x0A, 0x1A, 0x2A),
        "bg_gradient_end": RGBColor(0x14, 0x2F, 0x4A),
        "accent1": RGBColor(0x00, 0xD2, 0xFF),    # Bright cyan
        "accent2": RGBColor(0x00, 0x8E, 0xCC),    # Ocean blue
        "accent3": RGBColor(0x00, 0xB0, 0xFF),    # Azure
        "accent4": RGBColor(0x00, 0x5F, 0x99),    # Deep blue
        "accent5": RGBColor(0x80, 0xE8, 0xFF),    # Pale cyan
        "accent6": RGBColor(0x00, 0xD2, 0xFF),    # Bright cyan
        "text": RGBColor(0xE0, 0xF7, 0xFF),
        "text_light": RGBColor(0x88, 0xC8, 0xE0),
        "title_color": RGBColor(0x00, 0xD2, 0xFF),
        "subtitle_color": RGBColor(0x00, 0x8E, 0xCC),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "luxury": {
        "name": "Luxury Gold",
        "bg": RGBColor(0x0F, 0x0F, 0x18),
        "bg_gradient_start": RGBColor(0x0F, 0x0F, 0x18),
        "bg_gradient_end": RGBColor(0x1A, 0x1A, 0x30),
        "accent1": RGBColor(0xD4, 0xAF, 0x37),    # Metallic gold
        "accent2": RGBColor(0xF5, 0xD5, 0x3B),    # Bright gold
        "accent3": RGBColor(0xB8, 0x8A, 0x1A),    # Dark gold
        "accent4": RGBColor(0xFF, 0xED, 0x8A),    # Light champagne
        "accent5": RGBColor(0x8B, 0x74, 0x0E),    # Bronze gold
        "accent6": RGBColor(0xD4, 0xAF, 0x37),    # Classic gold
        "text": RGBColor(0xEA, 0xE0, 0xC8),
        "text_light": RGBColor(0xAA, 0x9A, 0x80),
        "title_color": RGBColor(0xF5, 0xD5, 0x3B),
        "subtitle_color": RGBColor(0xD4, 0xAF, 0x37),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "technology": {
        "name": "Tech Blue",
        "bg": RGBColor(0xF0, 0xF4, 0xFF),
        "bg_gradient_start": RGBColor(0xF0, 0xF4, 0xFF),
        "bg_gradient_end": RGBColor(0xE8, 0xED, 0xFA),
        "accent1": RGBColor(0x00, 0x3F, 0x72),    # Deep tech blue
        "accent2": RGBColor(0x00, 0x7B, 0xB8),    # Tech blue
        "accent3": RGBColor(0x00, 0xA6, 0xD6),    # Cyan blue
        "accent4": RGBColor(0x00, 0x5F, 0x99),    # Navy accent
        "accent5": RGBColor(0x40, 0x90, 0xC0),    # Steel blue
        "accent6": RGBColor(0x80, 0xC0, 0xE8),    # Light tech blue
        "text": RGBColor(0x1A, 0x1A, 0x30),
        "text_light": RGBColor(0x55, 0x55, 0x80),
        "title_color": RGBColor(0x00, 0x3F, 0x72),
        "subtitle_color": RGBColor(0x00, 0x7B, 0xB8),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "creative": {
        "name": "Creative Pop",
        "bg": RGBColor(0xFF, 0xFF, 0xFA),
        "bg_gradient_start": RGBColor(0xFF, 0xF5, 0xF5),
        "bg_gradient_end": RGBColor(0xF5, 0xF5, 0xFF),
        "accent1": RGBColor(0xFF, 0x54, 0x76),    # Rose pink
        "accent2": RGBColor(0x7C, 0x4D, 0xFF),    # Violet
        "accent3": RGBColor(0x00, 0xC2, 0xFF),    # Bright blue
        "accent4": RGBColor(0xFF, 0x91, 0x00),    # Orange
        "accent5": RGBColor(0x7C, 0x4D, 0xFF),    # Purple
        "accent6": RGBColor(0x00, 0xC2, 0xFF),    # Cyan
        "text": RGBColor(0x2D, 0x1A, 0x2E),
        "text_light": RGBColor(0x77, 0x55, 0x77),
        "title_color": RGBColor(0xFF, 0x54, 0x76),
        "subtitle_color": RGBColor(0x7C, 0x4D, 0xFF),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "finance": {
        "name": "Finance Trust",
        "bg": RGBColor(0xFF, 0xFF, 0xFF),
        "bg_gradient_start": RGBColor(0xF5, 0xF8, 0xFF),
        "bg_gradient_end": RGBColor(0xFF, 0xFF, 0xFF),
        "accent1": RGBColor(0x00, 0x5F, 0x99),    # Deep navy
        "accent2": RGBColor(0x00, 0x7B, 0xB8),    # Trust blue
        "accent3": RGBColor(0x00, 0xA6, 0xD6),    # Light blue
        "accent4": RGBColor(0x1A, 0x4A, 0x6A),    # Dark slate
        "accent5": RGBColor(0x4C, 0xAF, 0x50),    # Growth green
        "accent6": RGBColor(0xFF, 0x98, 0x00),    # Alert orange
        "text": RGBColor(0x1A, 0x1A, 0x2E),
        "text_light": RGBColor(0x55, 0x55, 0x77),
        "title_color": RGBColor(0x00, 0x5F, 0x99),
        "subtitle_color": RGBColor(0x00, 0x7B, 0xB8),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "environmental": {
        "name": "Earth Eco",
        "bg": RGBColor(0xF5, 0xFF, 0xF8),
        "bg_gradient_start": RGBColor(0xF5, 0xFF, 0xF8),
        "bg_gradient_end": RGBColor(0xEC, 0xFF, 0xE8),
        "accent1": RGBColor(0x1B, 0x7A, 0x4A),    # Forest green
        "accent2": RGBColor(0x2D, 0x9C, 0x5A),    # Leaf green
        "accent3": RGBColor(0x8F, 0xC3, 0x92),    # Sage
        "accent4": RGBColor(0x4C, 0xA8, 0x50),    # Meadow green
        "accent5": RGBColor(0xC8, 0xE6, 0xC9),    # Pale green
        "accent6": RGBColor(0x2E, 0x7D, 0x32),    # Deep green
        "text": RGBColor(0x1B, 0x2E, 0x1F),
        "text_light": RGBColor(0x4A, 0x6B, 0x50),
        "title_color": RGBColor(0x1B, 0x7A, 0x4A),
        "subtitle_color": RGBColor(0x2D, 0x9C, 0x5A),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "elegant": {
        "name": "Elegant",
        "bg": RGBColor(0xFA, 0xF9, 0xF6),
        "bg_gradient_start": RGBColor(0xFA, 0xF9, 0xF6),
        "bg_gradient_end": RGBColor(0xF0, 0xEA, 0xD8),
        "accent1": RGBColor(0x1A, 0x1A, 0x2E),    # Deep navy
        "accent2": RGBColor(0xC9, 0xA8, 0x4C),    # Gold
        "accent3": RGBColor(0x8B, 0x73, 0x55),    # Dark gold
        "accent4": RGBColor(0xF4, 0xE4, 0xC1),    # Light gold
        "accent5": RGBColor(0x2C, 0x3E, 0x5A),    # Dark blue
        "accent6": RGBColor(0xC9, 0xA8, 0x4C),    # Gold
        "text": RGBColor(0x2C, 0x2C, 0x3C),
        "text_light": RGBColor(0x8A, 0x8A, 0x9A),
        "title_color": RGBColor(0x1A, 0x1A, 0x2E),
        "subtitle_color": RGBColor(0x8B, 0x73, 0x55),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "vivid": {
        "name": "Vivid",
        "bg": RGBColor(0xFF, 0xFF, 0xFF),
        "bg_gradient_start": RGBColor(0xFF, 0xF5, 0xF5),
        "bg_gradient_end": RGBColor(0xF0, 0xFF, 0xF4),
        "accent1": RGBColor(0xFF, 0x6B, 0x6B),    # Coral
        "accent2": RGBColor(0x4E, 0xCD, 0xC4),    # Teal
        "accent3": RGBColor(0x45, 0xB7, 0xD1),    # Sky blue
        "accent4": RGBColor(0x96, 0xCE, 0xB4),    # Sage
        "accent5": RGBColor(0xFF, 0xEA, 0xA7),    # Yellow
        "accent6": RGBColor(0xD4, 0xA5, 0x74),    # Tan
        "text": RGBColor(0x33, 0x33, 0x44),
        "text_light": RGBColor(0x77, 0x77, 0x88),
        "title_color": RGBColor(0xFF, 0x6B, 0x6B),
        "subtitle_color": RGBColor(0x55, 0x55, 0x66),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "midnight_gold": {
        "name": "Midnight Gold",
        "bg": RGBColor(0x0C, 0x0C, 0x1D),
        "bg_gradient_start": RGBColor(0x0C, 0x0C, 0x1D),
        "bg_gradient_end": RGBColor(0x1A, 0x1A, 0x3E),
        "accent1": RGBColor(0xD4, 0xA8, 0x53),    # Gold
        "accent2": RGBColor(0xF5, 0xD7, 0x6E),    # Light gold
        "accent3": RGBColor(0x8B, 0x69, 0x14),    # Dark gold
        "accent4": RGBColor(0x1A, 0x1A, 0x3E),    # Navy
        "accent5": RGBColor(0xD4, 0xA8, 0x53),    # Gold
        "accent6": RGBColor(0xF5, 0xD7, 0x6E),    # Light gold
        "text": RGBColor(0xE0, 0xE0, 0xE0),
        "text_light": RGBColor(0x99, 0x99, 0xAA),
        "title_color": RGBColor(0xD4, 0xA8, 0x53),
        "subtitle_color": RGBColor(0xBB, 0xBB, 0xCC),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "glassmorphism": {
        "name": "Glassmorphism",
        "bg": RGBColor(0xF0, 0xF4, 0xF8),
        "bg_gradient_start": RGBColor(0xE8, 0xF0, 0xFE),
        "bg_gradient_end": RGBColor(0xF0, 0xF4, 0xF8),
        "accent1": RGBColor(0x66, 0x7E, 0xEA),    # Soft blue
        "accent2": RGBColor(0x76, 0x4B, 0xA2),    # Soft purple
        "accent3": RGBColor(0xF0, 0x93, 0xFB),    # Pink
        "accent4": RGBColor(0x4F, 0xAC, 0xFE),    # Blue
        "accent5": RGBColor(0x43, 0xE9, 0x7B),    # Green
        "accent6": RGBColor(0xF0, 0x93, 0xFB),    # Pink
        "text": RGBColor(0x2D, 0x2D, 0x3F),
        "text_light": RGBColor(0x7A, 0x7A, 0x8F),
        "title_color": RGBColor(0x66, 0x7E, 0xEA),
        "subtitle_color": RGBColor(0x76, 0x4B, 0xA2),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "neon_cyber": {
        "name": "Neon Cyber",
        "bg": RGBColor(0x0D, 0x02, 0x21),
        "bg_gradient_start": RGBColor(0x0D, 0x02, 0x21),
        "bg_gradient_end": RGBColor(0x15, 0x05, 0x34),
        "accent1": RGBColor(0xFF, 0x2D, 0x95),    # Hot pink
        "accent2": RGBColor(0x00, 0xFF, 0xF5),    # Cyan
        "accent3": RGBColor(0x7B, 0x2F, 0xF5),    # Purple
        "accent4": RGBColor(0xFF, 0xC5, 0x00),    # Yellow
        "accent5": RGBColor(0xFF, 0x2D, 0x95),    # Hot pink
        "accent6": RGBColor(0x00, 0xFF, 0xF5),    # Cyan
        "text": RGBColor(0xE0, 0xE0, 0xF0),
        "text_light": RGBColor(0x88, 0x88, 0xBB),
        "title_color": RGBColor(0x00, 0xFF, 0xF5),
        "subtitle_color": RGBColor(0xBB, 0x88, 0xFF),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
    "gradient_ocean": {
        "name": "Gradient Ocean",
        "bg": RGBColor(0xE0, 0xF7, 0xFA),
        "bg_gradient_start": RGBColor(0xE0, 0xF7, 0xFA),
        "bg_gradient_end": RGBColor(0xB2, 0xEB, 0xF2),
        "accent1": RGBColor(0x00, 0x69, 0x5C),    # Deep teal
        "accent2": RGBColor(0x00, 0x89, 0x7B),    # Teal
        "accent3": RGBColor(0x26, 0xA6, 0x9A),    # Medium teal
        "accent4": RGBColor(0x4D, 0xB6, 0xAC),    # Light teal
        "accent5": RGBColor(0x80, 0xDE, 0xEA),    # Pale teal
        "accent6": RGBColor(0x00, 0x69, 0x5C),    # Deep teal
        "text": RGBColor(0x1A, 0x3A, 0x3A),
        "text_light": RGBColor(0x55, 0x77, 0x77),
        "title_color": RGBColor(0x00, 0x69, 0x5C),
        "subtitle_color": RGBColor(0x00, 0x89, 0x7B),
        "font_title": "Calibri Light",
        "font_body": "Calibri",
    },
}

DEFAULT_THEME = "professional"


# ── Data types ────────────────────────────────────────────────────────────

@dataclass
class SlideContent:
    """Content for a single slide."""
    type: str  # title, content, two_column, image, comparison, section_header, blank, chart, table
    title: str = ""
    subtitle: str = ""
    body: list[str] = field(default_factory=list)
    bullet_style: str = "bullet"  # bullet, numbered, none
    columns: list[list[str]] = field(default_factory=list)  # For two_column
    left_content: list[str] = field(default_factory=list)
    right_content: list[str] = field(default_factory=list)
    col_ratio: float = 0.5  # Left column width ratio (0.3-0.7)
    comparison_items: list[tuple[str, str, str]] = field(default_factory=list)  # (label, left, right)
    notes: str = ""
    image_path: str = ""
    image_caption: str = ""
    image_position: str = "right"  # right, left, center, full
    table_data: list[list[str]] = field(default_factory=list)
    table_header_row: bool = True
    chart_type: str = "bar"  # bar, line, pie, area, scatter, column
    chart_data: dict = field(default_factory=dict)  # {"categories": [...], "series": [{"name": ..., "values": [...]}, ...]}
    chart_title: str = ""
    code: str = ""
    code_language: str = "python"
    quote: str = ""
    quote_author: str = ""
    footer_text: str = ""


@dataclass
class PresentationSpec:
    """Complete presentation specification."""
    title: str = "Presentation"
    subtitle: str = ""
    author: str = "Hyper Nexus"
    theme: str = DEFAULT_THEME
    slides: list[SlideContent] = field(default_factory=list)
    include_slide_numbers: bool = True
    include_footer: bool = True
    footer_text: str = "Generated by Hyper Nexus"
    aspect_ratio: str = "16:9"  # 16:9 or 4:3
    output_path: str = ""
    add_transitions: bool = True


# ── Presentation Builder ─────────────────────────────────────────────────

class PresentationBuilder:
    """Builds professional PowerPoint presentations with smart layout."""

    def __init__(self, spec: PresentationSpec):
        self.spec = spec
        self.theme = THEMES.get(spec.theme, THEMES[DEFAULT_THEME])
        self.prs = Presentation()

        # Set aspect ratio
        if spec.aspect_ratio == "4:3":
            self.prs.slide_width = Inches(10)
            self.prs.slide_height = Inches(7.5)
        else:
            self.prs.slide_width = Inches(13.333)
            self.prs.slide_height = Inches(7.5)

        self.slide_w = self.prs.slide_width
        self.slide_h = self.prs.slide_height
        self._slide_count = 0

        # Add blank layout for custom slides
        self._blank_layout = self.prs.slide_layouts[6]  # Blank layout

    def _add_shape(self, slide, left, top, width, height, fill_color=None, shape_type=None):
        """Add a rectangle shape (or oval if shape_type=MSO_SHAPE.OVAL)."""
        shape = slide.shapes.add_shape(shape_type or MSO_SHAPE.RECTANGLE, left, top, width, height)
        shape.line.fill.background()  # No line
        if fill_color:
            shape.fill.solid()
            shape.fill.fore_color.rgb = fill_color
        return shape

    def _add_gradient_shape(self, slide, left, top, width, height,
                            color_start, color_end, angle=45):
        """Add a rectangle shape with gradient fill.

        Args:
            slide: The slide to add the shape to.
            left, top, width, height: Position and dimensions.
            color_start: RGBColor at 0% position.
            color_end: RGBColor at 100% position.
            angle: Gradient angle in degrees (0=horizontal, 90=vertical).
        """
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        shape.line.fill.background()
        fill = shape.fill
        fill.gradient()
        gradFill = fill._fill.get_or_add_gradFill()
        gradFill.set('ang', str(int(angle * 60000)))
        gsLst = gradFill.get_or_add_gsLst()
        gs1 = gsLst.add_gs()
        gs1.pos = 0
        gs1.get_or_add_srgbClr().val = color_start
        gs2 = gsLst.add_gs()
        gs2.pos = 100000
        gs2.get_or_add_srgbClr().val = color_end
        return shape

    def _add_textbox(self, slide, left, top, width, height, text, font_name=None,
                     font_size=None, bold=False, color=None, alignment=None,
                     font_name_body=None):
        """Add a text box with formatted text."""
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = font_size or Pt(18)
        p.font.bold = bold
        if color:
            p.font.color.rgb = color
        if alignment:
            p.alignment = alignment
        if font_name:
            p.font.name = font_name
        if font_name_body:
            p.font.name = font_name_body
        return txBox

    def _add_multiline_textbox(self, slide, left, top, width, height, lines,
                                font_size=Pt(16), color=None, bullet_style="bullet",
                                font_name=None, line_spacing=1.3):
        """Add a textbox with multiple formatted lines (bullets or numbered)."""
        # Normalize: if a single string is passed, treat as one line
        if isinstance(lines, str):
            lines = [lines]

        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True

        for i, line in enumerate(lines):
            if i == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()

            # Handle indented sub-items
            indent_level = 0
            clean_line = line
            if line.startswith("  ") or line.startswith("\t"):
                indent_level = 1
                clean_line = line.strip()

            if bullet_style == "numbered":
                p.text = f"{i+1}. {clean_line}"
            elif bullet_style == "bullet":
                # Use bullet character
                if indent_level > 0:
                    p.text = f"  \u2022  {clean_line}"
                    p.level = 1
                else:
                    p.text = f"\u2022  {clean_line}"
            else:
                p.text = clean_line

            p.font.size = font_size
            p.space_after = Pt(4)
            p.space_before = Pt(2)
            if color:
                p.font.color.rgb = color
            if font_name:
                p.font.name = font_name
            # Line spacing
            p.line_spacing = line_spacing

        return txBox

    def _create_header_bar(self, slide, section_color=None):
        """Create a decorative header bar at the top."""
        color = section_color or self.theme["accent1"]
        bar = self._add_shape(slide, 0, 0, self.slide_w, Inches(0.08), color)
        return bar

    def _create_footer(self, slide):
        """Add footer with slide number."""
        if not self.spec.include_slide_numbers and not self.spec.include_footer:
            return

        footer_y = self.slide_h - Inches(0.4)
        line_y = footer_y - Inches(0.05)
        line = self._add_shape(slide, Inches(0.5), line_y,
                                self.slide_w - Inches(1.0), Pt(0.5),
                                RGBColor(0xCC, 0xCC, 0xCC))

        if self.spec.include_footer and self.spec.footer_text:
            self._add_textbox(
                slide, Inches(0.5), footer_y, Inches(6), Inches(0.35),
                self.spec.footer_text,
                font_size=Pt(9), color=self.theme["text_light"],
                font_name=self.theme["font_body"],
            )

        if self.spec.include_slide_numbers:
            num_text = str(self._slide_count)
            self._add_textbox(
                slide, self.slide_w - Inches(1.0), footer_y, Inches(0.7), Inches(0.35),
                num_text,
                font_size=Pt(9), color=self.theme["text_light"],
                alignment=PP_ALIGN.RIGHT,
                font_name=self.theme["font_body"],
            )

    def _section_accent_line(self, slide, left, top, width, color=None):
        """Add a thin accent line."""
        c = color or self.theme["accent2"]
        self._add_shape(slide, left, top, width, Pt(3), c)

    def _use_gradient_or_solid(self, slide, left, top, width, height):
        """Add a background shape using gradient if available, else solid fill."""
        if "bg_gradient_start" in self.theme and "bg_gradient_end" in self.theme:
            return self._add_gradient_shape(
                slide, left, top, width, height,
                self.theme["bg_gradient_start"],
                self.theme["bg_gradient_end"],
                angle=135,
            )
        else:
            return self._add_shape(slide, left, top, width, height, self.theme["accent1"])

    # ── Slide builders ───────────────────────────────────────────────────

    def _build_title_slide(self, slide, sc: SlideContent):
        """Build a professional title slide with modern design."""
        t = self.theme
        sw = self.slide_w

        # ── Gradient background ──
        self._use_gradient_or_solid(slide, 0, 0, sw, self.slide_h)

        # ── Decorative top accent bar ──
        self._add_shape(slide, Inches(0), Inches(0), sw, Inches(0.08), t["accent3"])

        # ── Decorative geometric shapes ──
        # Large decorative circle top-right
        self._add_shape(
            slide, sw - Inches(2.5), Inches(-1.0),
            Inches(4.0), Inches(4.0),
            t["accent2"],
            shape_type=MSO_SHAPE.OVAL,
        )
        # Small decorative circle bottom-left
        self._add_shape(
            slide, Inches(-0.5), Inches(5.5),
            Inches(2.0), Inches(2.0),
            t["accent4"],
            shape_type=MSO_SHAPE.OVAL,
        )

        # ── Title ──
        title_size = Pt(48) if len(sc.title) < 25 else Pt(38)
        self._add_textbox(
            slide, Inches(1.5), Inches(2.0), Inches(9.0), Inches(1.8),
            sc.title,
            font_size=title_size, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF),
            font_name=t["font_title"],
        )

        # ── Accent line under title ──
        self._section_accent_line(slide, Inches(1.5), Inches(3.6), Inches(2.5), t["accent3"])

        # ── Subtitle ──
        subtitle = sc.subtitle or self.spec.subtitle
        if subtitle:
            self._add_textbox(
                slide, Inches(1.5), Inches(4.0), Inches(9.0), Inches(1.0),
                subtitle,
                font_size=Pt(22), color=RGBColor(0xDD, 0xDD, 0xDD),
                font_name=t["font_body"],
            )

        # ── Author + date at bottom ──
        today = datetime.now().strftime("%B %d, %Y")
        author_text = f"{self.spec.author}  |  {today}"
        self._add_textbox(
            slide, Inches(1.5), Inches(6.0), Inches(9.0), Inches(0.5),
            author_text,
            font_size=Pt(14), color=RGBColor(0xAA, 0xAA, 0xBB),
            font_name=t["font_body"],
        )

        # ── Bottom accent bar ──
        self._add_shape(slide, Inches(0), self.slide_h - Inches(0.06), sw, Inches(0.06), t["accent3"])

    def _build_section_header(self, slide, sc: SlideContent):
        """Build a section divider slide with modern design."""
        t = self.theme
        sw = self.slide_w

        # ── Gradient background ──
        self._use_gradient_or_solid(slide, 0, 0, sw, self.slide_h)

        # ── Decorative geometric elements ──
        # Large decorative circle top-right
        self._add_shape(
            slide, sw - Inches(3.0), Inches(-1.5),
            Inches(5.0), Inches(5.0),
            t["accent2"],
            shape_type=MSO_SHAPE.OVAL,
        )
        # Small accent circle bottom-left
        self._add_shape(
            slide, Inches(-0.8), Inches(5.0),
            Inches(2.5), Inches(2.5),
            t["accent4"],
            shape_type=MSO_SHAPE.OVAL,
        )

        # ── Section number (if available) ──
        section_num = ""
        if hasattr(sc, 'section_number') and sc.section_number:
            section_num = sc.section_number
            self._add_textbox(
                slide, Inches(1.5), Inches(1.8), Inches(3.0), Inches(0.8),
                f"0{section_num}",
                font_size=Pt(16), bold=True,
                color=t["accent3"],
                font_name=t["font_title"],
            )

        # ── Decorative bar ──
        self._add_shape(slide, Inches(1.5), Inches(2.8), Inches(3), Pt(4), t["accent3"])

        # ── Title ──
        title_size = Pt(42) if len(sc.title) < 25 else Pt(34)
        self._add_textbox(
            slide, Inches(1.5), Inches(3.1), Inches(10.0), Inches(1.5),
            sc.title,
            font_size=title_size, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF),
            font_name=t["font_title"],
        )

        # ── Subtitle ──
        if sc.subtitle:
            self._add_textbox(
                slide, Inches(1.5), Inches(4.5), Inches(10.0), Inches(0.8),
                sc.subtitle,
                font_size=Pt(20), color=RGBColor(0xCC, 0xCC, 0xDD),
                font_name=t["font_body"],
            )

        # ── Bottom accent bar ──
        self._add_shape(slide, Inches(0), self.slide_h - Inches(0.06), sw, Inches(0.06), t["accent3"])

    def _build_content_slide(self, slide, sc: SlideContent):
        """Build a standard content slide with title and body."""
        t = self.theme

        # ── Gradient header bar ──
        if "bg_gradient_start" in t and "bg_gradient_end" in t:
            self._add_gradient_shape(slide, 0, 0, self.slide_w, Inches(0.10),
                                     t["accent2"], t["accent3"], angle=0)
        else:
            self._create_header_bar(slide, t["accent2"])

        # ── Title ──
        title_size = Pt(32) if len(sc.title) < 40 else Pt(28)
        self._add_textbox(
            slide, Inches(0.8), Inches(0.3), Inches(11.7), Inches(0.9),
            sc.title,
            font_size=title_size, bold=True,
            color=t["title_color"],
            font_name=t["font_title"],
        )

        # ── Accent line under title ──
        self._section_accent_line(slide, Inches(0.8), Inches(1.15), Inches(1.5), t["accent2"])

        # ── Body content ──
        body_top = Inches(1.5)
        if sc.body:
            self._add_multiline_textbox(
                slide, Inches(0.8), body_top, Inches(11.7), Inches(5.3),
                sc.body,
                font_size=Pt(18), color=t["text"],
                bullet_style=sc.bullet_style,
                font_name=t["font_body"],
            )

        # ── Quote ──
        if sc.quote:
            # Quote background
            quote_y = max(body_top + Inches(0.5), Inches(2.5))
            self._add_shape(slide, Inches(0.8), quote_y - Inches(0.2),
                            Inches(11.7), Inches(2.5),
                            RGBColor(0xF5, 0xF5, 0xFA))
            # Left accent bar for quote
            self._add_shape(slide, Inches(0.8), quote_y, Pt(4), Inches(1.8), t["accent2"])
            # Quote text
            self._add_textbox(
                slide, Inches(1.4), quote_y, Inches(10.5), Inches(1.5),
                f"\u201C{sc.quote}\u201D",
                font_size=Pt(20), color=t["accent2"],
                font_name=t["font_body"],
            )
            if sc.quote_author:
                self._add_textbox(
                    slide, Inches(1.4), quote_y + Inches(1.5), Inches(10.5), Inches(0.5),
                    f"\u2014 {sc.quote_author}",
                    font_size=Pt(14), color=t["text_light"],
                    font_name=t["font_body"],
                )

        # ── Code block ──
        if sc.code:
            code_top = max(body_top + Inches(0.3), Inches(1.5))
            code_bg = self._add_shape(slide, Inches(0.8), code_top,
                                       Inches(11.7), Inches(5.0),
                                       RGBColor(0x1E, 0x1E, 0x2E))
            # Code language label
            if sc.code_language:
                self._add_textbox(
                    slide, Inches(1.0), code_top + Inches(0.05),
                    Inches(3.0), Inches(0.3),
                    sc.code_language.upper(),
                    font_size=Pt(9), bold=True,
                    color=RGBColor(0x88, 0x88, 0x88),
                    font_name="Consolas",
                )
            self._add_textbox(
                slide, Inches(1.0), code_top + Inches(0.35),
                Inches(11.3), Inches(4.3),
                sc.code[:800],
                font_size=Pt(12), color=RGBColor(0xAB, 0xB2, 0xBF),
                font_name="Consolas",
            )

        # ── Footer ──
        self._create_footer(slide)

    def _build_two_column(self, slide, sc: SlideContent):
        """Build a two-column slide."""
        # Header bar
        self._create_header_bar(slide, self.theme["accent2"])

        # Title
        self._add_textbox(
            slide, Inches(0.7), Inches(0.3), Inches(11.9), Inches(0.9),
            sc.title,
            font_size=Pt(30), bold=True,
            color=self.theme["title_color"],
            font_name=self.theme["font_title"],
        )

        # Accent line
        self._section_accent_line(slide, Inches(0.7), Inches(1.15), Inches(1.5))

        # Calculate column widths
        total_w = Inches(11.9)
        gap = Inches(0.5)
        left_ratio = max(0.3, min(0.7, sc.col_ratio))
        right_ratio = 1.0 - left_ratio
        left_w = int(total_w * left_ratio)
        right_w = int(total_w * right_ratio)
        col_y = Inches(1.4)
        col_h = Inches(5.5)

        # Left column
        if sc.left_content:
            self._add_multiline_textbox(
                slide, Inches(0.7), col_y, left_w, col_h,
                sc.left_content,
                font_size=Pt(16), color=self.theme["text"],
                bullet_style=sc.bullet_style,
                font_name=self.theme["font_body"],
            )

        # Right column
        if sc.right_content:
            right_x = Inches(0.7) + left_w + gap
            self._add_multiline_textbox(
                slide, right_x, col_y, right_w, col_h,
                sc.right_content,
                font_size=Pt(16), color=self.theme["text"],
                bullet_style=sc.bullet_style,
                font_name=self.theme["font_body"],
            )

        # Vertical divider
        divider_x = Inches(0.7) + left_w + gap // 2
        self._add_shape(slide, divider_x, col_y, Pt(1), Inches(5.0),
                        RGBColor(0xDD, 0xDD, 0xDD))

        # Footer
        self._create_footer(slide)

    def _build_comparison(self, slide, sc: SlideContent):
        """Build a comparison slide with items side by side."""
        # Header bar
        self._create_header_bar(slide, self.theme["accent2"])

        # Title
        self._add_textbox(
            slide, Inches(0.7), Inches(0.3), Inches(11.9), Inches(0.9),
            sc.title,
            font_size=Pt(28), bold=True,
            color=self.theme["title_color"],
            font_name=self.theme["font_title"],
        )

        # Accent line
        self._section_accent_line(slide, Inches(0.7), Inches(1.15), Inches(1.5))

        if sc.comparison_items:
            # Column headers
            col_w = Inches(5.5)
            col_y = Inches(1.5)

            # Left header
            left_header = sc.comparison_items[0][1] if sc.comparison_items else ""
            right_header = sc.comparison_items[0][2] if len(sc.comparison_items[0]) > 2 else ""

            if left_header and right_header:
                # Header boxes
                left_hdr = self._add_shape(slide, Inches(0.7), col_y, col_w, Inches(0.6),
                                           self.theme["accent1"])
                self._add_textbox(
                    slide, Inches(0.9), col_y + Inches(0.05), col_w - Inches(0.4), Inches(0.5),
                    left_header,
                    font_size=Pt(16), bold=True,
                    color=RGBColor(0xFF, 0xFF, 0xFF),
                    font_name=self.theme["font_title"],
                )

                right_hdr = self._add_shape(slide, Inches(7.1), col_y, col_w, Inches(0.6),
                                            self.theme["accent2"])
                self._add_textbox(
                    slide, Inches(7.3), col_y + Inches(0.05), col_w - Inches(0.4), Inches(0.5),
                    right_header,
                    font_size=Pt(16), bold=True,
                    color=RGBColor(0xFF, 0xFF, 0xFF),
                    font_name=self.theme["font_title"],
                )

                # Comparison rows
                row_y = col_y + Inches(0.8)
                for i, item in enumerate(sc.comparison_items):
                    label = item[0]
                    left_val = item[1] if len(item) > 1 else ""
                    right_val = item[2] if len(item) > 2 else ""

                    # Alternating row background
                    if i % 2 == 0:
                        self._add_shape(slide, Inches(0.7), row_y, col_w, Inches(0.5),
                                        RGBColor(0xF5, 0xF5, 0xF5))
                        self._add_shape(slide, Inches(7.1), row_y, col_w, Inches(0.5),
                                        RGBColor(0xF5, 0xF5, 0xF5))

                    # Label in center
                    self._add_textbox(
                        slide, Inches(0.7), row_y, Inches(0.5), Inches(0.5),
                        label,
                        font_size=Pt(11), bold=True,
                        color=self.theme["text_light"],
                        font_name=self.theme["font_body"],
                        alignment=PP_ALIGN.CENTER,
                    )

                    # Left value
                    self._add_textbox(
                        slide, Inches(1.3), row_y + Inches(0.05), Inches(4.7), Inches(0.4),
                        left_val,
                        font_size=Pt(14), color=self.theme["text"],
                        font_name=self.theme["font_body"],
                    )

                    # Right value
                    self._add_textbox(
                        slide, Inches(7.7), row_y + Inches(0.05), Inches(4.7), Inches(0.4),
                        right_val,
                        font_size=Pt(14), color=self.theme["text"],
                        font_name=self.theme["font_body"],
                    )

                    row_y += Inches(0.55)

        # Footer
        self._create_footer(slide)

    def _build_image_slide(self, slide, sc: SlideContent):
        """Build an image slide."""
        # Header bar
        self._create_header_bar(slide, self.theme["accent2"])

        # Title
        if sc.title:
            self._add_textbox(
                slide, Inches(0.7), Inches(0.3), Inches(11.9), Inches(0.7),
                sc.title,
                font_size=Pt(28), bold=True,
                color=self.theme["title_color"],
                font_name=self.theme["font_title"],
            )

        abs_image_path = sc.image_path
        if abs_image_path and os.path.exists(abs_image_path):
            try:
                # Calculate image position
                if sc.image_position == "full":
                    # Full slide image (behind title)
                    slide.shapes.add_picture(abs_image_path, 0, 0,
                                              self.slide_w, self.slide_h)
                elif sc.image_position == "center":
                    # Centered with padding
                    img_left = Inches(1.5)
                    img_top = Inches(1.5)
                    img_width = Inches(10.3)
                    slide.shapes.add_picture(abs_image_path, img_left, img_top,
                                              img_width)
                else:
                    # Right or left aligned
                    img_width = Inches(7.5)
                    if sc.image_position == "right":
                        img_left = Inches(5.3)
                    else:  # left
                        img_left = Inches(0.7)
                    img_top = Inches(1.3)
                    slide.shapes.add_picture(abs_image_path, img_left, img_top,
                                              img_width)

                # Caption
                if sc.image_caption:
                    caption_y = Inches(6.5)
                    self._add_textbox(
                        slide, Inches(0.7), caption_y, Inches(11.9), Inches(0.5),
                        sc.image_caption,
                        font_size=Pt(12), color=self.theme["text_light"],
                        alignment=PP_ALIGN.CENTER,
                        font_name=self.theme["font_body"],
                    )

            except Exception as e:
                # Fallback: show error text
                self._add_textbox(
                    slide, Inches(0.7), Inches(3.0), Inches(11.9), Inches(1.0),
                    f"[Image could not be loaded: {e}]",
                    font_size=Pt(16), color=RGBColor(0xFF, 0x00, 0x00),
                    font_name=self.theme["font_body"],
                )
        else:
            # Placeholder for missing image
            placeholder = self._add_shape(slide, Inches(1.5), Inches(1.5),
                                           Inches(10.3), Inches(5.0),
                                           RGBColor(0xE8, 0xE8, 0xE8))
            self._add_textbox(
                slide, Inches(3.0), Inches(3.5), Inches(7.3), Inches(1.0),
                "Image Placeholder",
                font_size=Pt(20), color=RGBColor(0x99, 0x99, 0x99),
                alignment=PP_ALIGN.CENTER,
                font_name=self.theme["font_body"],
            )

        # Footer
        self._create_footer(slide)

    def _build_table_slide(self, slide, sc: SlideContent):
        """Build a slide with a formatted table."""
        # Header bar
        self._create_header_bar(slide, self.theme["accent2"])

        # Title
        self._add_textbox(
            slide, Inches(0.7), Inches(0.3), Inches(11.9), Inches(0.9),
            sc.title,
            font_size=Pt(28), bold=True,
            color=self.theme["title_color"],
            font_name=self.theme["font_title"],
        )

        # Accent line
        self._section_accent_line(slide, Inches(0.7), Inches(1.15), Inches(1.5))

        if sc.table_data and len(sc.table_data) > 0:
            rows = len(sc.table_data)
            cols = max(len(row) for row in sc.table_data) if rows > 0 else 1

            table_left = Inches(0.7)
            table_top = Inches(1.5)
            table_width = Inches(11.9)
            table_height = Inches(5.0)

            # Calculate row height
            row_height = Inches(0.5) if rows <= 8 else Inches(0.35)
            table_height = row_height * rows

            table = slide.shapes.add_table(
                rows, cols, table_left, table_top, table_width, table_height
            ).table

            # Style the table
            for r in range(rows):
                for c in range(cols):
                    cell = table.cell(r, c)
                    cell_val = sc.table_data[r][c] if c < len(sc.table_data[r]) else ""

                    cell.text = str(cell_val)

                    for paragraph in cell.text_frame.paragraphs:
                        paragraph.font.size = Pt(14)
                        paragraph.font.name = self.theme["font_body"]

                        if r == 0 and sc.table_header_row:
                            # Header row styling
                            paragraph.font.bold = True
                            paragraph.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                            cell.fill.solid()
                            cell.fill.fore_color.rgb = self.theme["accent1"]
                        else:
                            paragraph.font.color.rgb = self.theme["text"]
                            # Alternating row colors
                            if r % 2 == 0:
                                cell.fill.solid()
                                cell.fill.fore_color.rgb = RGBColor(0xF8, 0xF8, 0xF8)
                            else:
                                cell.fill.solid()
                                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

                        paragraph.alignment = PP_ALIGN.LEFT
                        # Vertical centering
                        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                        cell.margin_left = Inches(0.1)
                        cell.margin_right = Inches(0.1)
                        cell.margin_top = Inches(0.05)
                        cell.margin_bottom = Inches(0.05)

        # Footer
        self._create_footer(slide)

    def _build_chart_slide(self, slide, sc: SlideContent):
        """Build a slide with a chart."""
        # Header bar
        self._create_header_bar(slide, self.theme["accent2"])

        # Title
        chart_title = sc.chart_title or sc.title
        self._add_textbox(
            slide, Inches(0.7), Inches(0.3), Inches(11.9), Inches(0.7),
            chart_title,
            font_size=Pt(28), bold=True,
            color=self.theme["title_color"],
            font_name=self.theme["font_title"],
        )

        # Accent line
        self._section_accent_line(slide, Inches(0.7), Inches(1.0), Inches(1.5))

        if sc.chart_data and "categories" in sc.chart_data and "series" in sc.chart_data:
            categories = sc.chart_data["categories"]
            series_data = sc.chart_data["series"]

            if categories and series_data:
                try:
                    chart_type_map = {
                        "bar": XL_CHART_TYPE.COLUMN_CLUSTERED,
                        "column": XL_CHART_TYPE.COLUMN_CLUSTERED,
                        "line": XL_CHART_TYPE.LINE_MARKERS,
                        "pie": XL_CHART_TYPE.PIE,
                        "area": XL_CHART_TYPE.AREA,
                        "scatter": XL_CHART_TYPE.XY_SCATTER,
                        "bar_stacked": XL_CHART_TYPE.COLUMN_STACKED,
                        "bar_100": XL_CHART_TYPE.COLUMN_STACKED_100,
                        "line_stacked": XL_CHART_TYPE.LINE_STACKED,
                    }
                    xl_type = chart_type_map.get(sc.chart_type, XL_CHART_TYPE.COLUMN_CLUSTERED)

                    # Build ChartData object (python-pptx 1.0.2 requires a single chart_data arg)
                    chart_data = ChartData()
                    chart_data.categories = categories
                    for i, s in enumerate(series_data):
                        chart_data.add_series(
                            s.get("name", f"Series {i+1}"),
                            s["values"],
                        )

                    chart_frame = slide.shapes.add_chart(
                        xl_type,
                        Inches(0.7), Inches(1.2), Inches(11.9), Inches(5.5),
                        chart_data,
                    )

                    chart = chart_frame.chart
                    chart.has_legend = True
                    chart.legend.include_in_layout = False

                    # Style the chart
                    plot = chart.plots[0]
                    plot.has_data_labels = True
                    data_labels = plot.data_labels
                    data_labels.font.size = Pt(10)
                    data_labels.number_format = '#,##0'
                    data_labels.show_value = True
                    data_labels.show_category_name = False

                except Exception as e:
                    # Fallback: text representation
                    self._add_textbox(
                        slide, Inches(0.7), Inches(1.5), Inches(11.9), Inches(5.0),
                        f"[Chart data (chart rendering unavailable: {e})]",
                        font_size=Pt(14), color=self.theme["text"],
                        font_name=self.theme["font_body"],
                    )

        # Body text below chart
        if sc.body:
            self._add_multiline_textbox(
                slide, Inches(0.7), Inches(6.0), Inches(11.9), Inches(1.0),
                sc.body,
                font_size=Pt(14), color=self.theme["text_light"],
                font_name=self.theme["font_body"],
            )

        # Footer
        self._create_footer(slide)

    def _build_blank_slide(self, slide, sc: SlideContent):
        """Build a blank/simple slide with minimal formatting."""
        if sc.title:
            self._add_textbox(
                slide, Inches(0.7), Inches(0.3), Inches(11.9), Inches(0.7),
                sc.title,
                font_size=Pt(24), bold=True,
                color=self.theme["title_color"],
                font_name=self.theme["font_title"],
            )

        if sc.body:
            self._add_multiline_textbox(
                slide, Inches(0.7), Inches(1.2), Inches(11.9), Inches(5.5),
                sc.body,
                font_size=Pt(16), color=self.theme["text"],
                font_name=self.theme["font_body"],
            )

    def build(self) -> str:
        """Build the presentation and return the file path."""
        # Build each slide
        for slide_content in self.spec.slides:
            slide = self.prs.slides.add_slide(self._blank_layout)
            self._slide_count += 1

            slide_type = slide_content.type

            if slide_type == "title":
                self._build_title_slide(slide, slide_content)
            elif slide_type == "section_header":
                self._build_section_header(slide, slide_content)
            elif slide_type == "content":
                self._build_content_slide(slide, slide_content)
            elif slide_type == "two_column":
                self._build_two_column(slide, slide_content)
            elif slide_type == "comparison":
                self._build_comparison(slide, slide_content)
            elif slide_type == "image":
                self._build_image_slide(slide, slide_content)
            elif slide_type == "table":
                self._build_table_slide(slide, slide_content)
            elif slide_type == "chart":
                self._build_chart_slide(slide, slide_content)
            elif slide_type == "blank":
                self._build_blank_slide(slide, slide_content)
            else:
                self._build_content_slide(slide, slide_content)

            # Speaker notes
            if slide_content.notes:
                notes_slide = slide.notes_slide
                notes_tf = notes_slide.notes_text_frame
                notes_tf.text = slide_content.notes

        # Set document properties
        self.prs.core_properties.title = self.spec.title
        self.prs.core_properties.author = self.spec.author
        self.prs.core_properties.created = datetime.now()

        # Save
        output = self.spec.output_path or str(
            Path(config.BASE_DIR) / "data" / "workspace" /
            f"{self.spec.title.replace(' ', '_')}_{int(time.time())}.pptx"
        )
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(output_path))
        return str(output_path)


# ── Tool implementations ─────────────────────────────────────────────────

async def _list_presentation_templates() -> list[dict]:
    """List available presentation themes/templates."""
    themes = []
    for key, theme in THEMES.items():
        themes.append({
            "id": key,
            "name": theme["name"],
        })
    return themes


def _parse_slides_json(slides_json: str | list) -> list[SlideContent]:
    """Parse slides from JSON string or list."""
    if isinstance(slides_json, str):
        try:
            data = json.loads(slides_json)
        except json.JSONDecodeError:
            return []
    else:
        data = slides_json

    if not isinstance(data, list):
        return []

    def _normalize_body(val: Any) -> list[str]:
        """Normalize body field: None → [], string → [string], list → list of strings."""
        if val is None:
            return []
        if isinstance(val, str):
            # Try parsing as JSON array first (common when inner arrays get stringified)
            stripped = val.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(s) for s in parsed]
                except (json.JSONDecodeError, TypeError):
                    pass
            return [val]
        if isinstance(val, list):
            return [str(s) for s in val]
        return [str(val)]

    slides = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Try multiple field names for body content
        body = _normalize_body(
            item.get("body",
            item.get("content",
            item.get("bullets",
            item.get("points",
            item.get("items", None)))))
        )
        sc = SlideContent(
            type=item.get("type", "content"),
            title=item.get("title", ""),
            subtitle=item.get("subtitle", ""),
            body=body,
            bullet_style=item.get("bullet_style", "bullet"),
            columns=item.get("columns", []),
            left_content=item.get("left_content", []) or item.get("left", []),
            right_content=item.get("right_content", []) or item.get("right", []),
            col_ratio=float(item.get("col_ratio", 0.5)),
            comparison_items=item.get("comparison_items", []) or item.get("comparisons", []),
            notes=item.get("notes", ""),
            image_path=item.get("image_path", ""),
            image_caption=item.get("image_caption", ""),
            image_position=item.get("image_position", "right"),
            table_data=item.get("table_data", []) or item.get("table", []),
            table_header_row=item.get("table_header_row", True),
            chart_type=item.get("chart_type", "bar"),
            chart_data=item.get("chart_data", {}),
            chart_title=item.get("chart_title", ""),
            code=item.get("code", ""),
            code_language=item.get("code_language", "python"),
            quote=item.get("quote", ""),
            quote_author=item.get("quote_author", ""),
            footer_text=item.get("footer_text", ""),
        )
        slides.append(sc)

    return slides


@tool(
    name="create_ppt",
    description=(
        "Create a professional PowerPoint presentation with multiple slide types, themes, charts, and tables.\n\n"
        "Theme selection by topic category (pick the theme that best matches the subject):\n"
        "  - business / corporate / finance / startup → 'corporate' or 'professional'\n"
        "  - tech / AI / software / data / coding → 'neon_cyber' or 'modern_dark'\n"
        "  - health / medical / biology / science → 'nature' or 'vivid'\n"
        "  - education / teaching / academic → 'elegant' or 'professional'\n"
        "  - marketing / branding / creative → 'sunset' or 'glassmorphism'\n"
        "  - luxury / premium / high-end → 'midnight_gold' or 'elegant'\n"
        "  - space / astronomy / futurism → 'aurora' or 'modern_dark'\n"
        "  - ocean / water / environment → 'gradient_ocean' or 'nature'\n"
        "  - dark / hacker / cyberpunk → 'neon_cyber' or 'modern_dark'\n"
        "  - simple / clean / minimalist → 'minimal'\n"
        "  - horror / dark fantasy / intense → 'modern_dark' or 'midnight_gold'\n"
        "  - general / unknown → 'professional'\n\n"
        "Choose theme explicitly — do NOT leave it blank unless the user specifies."
    ),
    timeout=300,
    parameters_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Presentation title"},
            "subtitle": {"type": "string", "description": "Presentation subtitle"},
            "slides": {
                "type": "array",
                "description": "Array of slide objects. Each slide supports types: title, content, two_column, section_header, comparison, image, table, chart, blank",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "Slide type: title, content, two_column, section_header, comparison, image, table, chart, blank"},
                        "title": {"type": "string", "description": "Slide title"},
                        "subtitle": {"type": "string", "description": "Slide subtitle"},
                        "body": {"type": "array", "items": {"type": "string"}, "description": "Bullet points or content lines"},
                        "bullet_style": {"type": "string", "enum": ["bullet", "numbered", "none"]},
                        "left_content": {"type": "array", "items": {"type": "string"}},
                        "right_content": {"type": "array", "items": {"type": "string"}},
                        "col_ratio": {"type": "number", "description": "Left column width ratio (0.3-0.7)"},
                        "comparison_items": {
                            "type": "array",
                            "description": "For comparison slides: [[\"label\", \"left_value\", \"right_value\"], ...]",
                            "items": {"type": "array", "items": {"type": "string"}}
                        },
                        "image_path": {"type": "string"},
                        "image_caption": {"type": "string"},
                        "image_position": {"type": "string", "enum": ["right", "left", "center", "full"]},
                        "table_data": {"type": "array", "description": "2D array: [[header1, header2], [row1col1, row1col2], ...]"},
                        "table_header_row": {"type": "boolean"},
                        "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "area", "scatter", "column", "bar_stacked"]},
                        "chart_data": {
                            "type": "object",
                            "description": "Chart data: {\"categories\": [\"Q1\", \"Q2\", ...], \"series\": [{\"name\": \"Series1\", \"values\": [10, 20, ...]}]}"
                        },
                        "chart_title": {"type": "string"},
                        "code": {"type": "string"},
                        "quote": {"type": "string"},
                        "quote_author": {"type": "string"},
                        "notes": {"type": "string", "description": "Speaker notes for this slide"},
                    }
                }
            },
            "theme": {
                "type": "string",
                "enum": list(THEMES.keys()),
                "description": "Visual theme for the presentation",
            },
            "author": {"type": "string", "description": "Author name"},
            "include_slide_numbers": {"type": "boolean"},
            "include_footer": {"type": "boolean"},
            "footer_text": {"type": "string"},
            "aspect_ratio": {"type": "string", "enum": ["16:9", "4:3"]},
            "output_path": {"type": "string", "description": "Custom output file path"},
        },
        "required": ["title", "slides"],
    },
)
async def create_ppt(params: dict[str, Any]) -> ToolResult:
    """Create a professional PowerPoint presentation."""
    if not PPTX_AVAILABLE:
        return ToolResult(
            success=False,
            output="python-pptx is not installed. Install it with: pip install python-pptx"
        )

    try:
        title = params.get("title", "Presentation")
        slides = params.get("slides", [])
        subtitle = params.get("subtitle", "")
        theme = params.get("theme", "professional")
        author = params.get("author", "Hyper Nexus")
        include_slide_numbers = params.get("include_slide_numbers", True)
        include_footer = params.get("include_footer", True)
        footer_text = params.get("footer_text", "Generated by Hyper Nexus")
        aspect_ratio = params.get("aspect_ratio", "16:9")
        output_path = params.get("output_path", "")

        parsed_slides = _parse_slides_json(slides)
        if not parsed_slides:
            return ToolResult(
                success=False,
                output="No valid slides provided. Each slide needs at minimum a 'type' field."
            )

        spec = PresentationSpec(
            title=title,
            subtitle=subtitle,
            author=author,
            theme=theme if theme in THEMES else DEFAULT_THEME,
            slides=parsed_slides,
            include_slide_numbers=include_slide_numbers,
            include_footer=include_footer,
            footer_text=footer_text,
            aspect_ratio=aspect_ratio,
            output_path=output_path,
        )

        builder = PresentationBuilder(spec)
        file_path = builder.build()

        await emit("presentation_created", path=file_path, file_kind="pptx")

        return ToolResult(
            success=True,
            output=json.dumps({
                "success": True,
                "file_path": file_path,
                "slide_count": len(parsed_slides),
                "theme": theme,
                "aspect_ratio": aspect_ratio,
                "message": f"Created '{title}' with {len(parsed_slides)} slides at {file_path}",
            })
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output=f"Failed to create presentation: {type(e).__name__}: {e}"
        )


@tool(
    name="ppt_add_slide",
    description="Add a slide to an existing presentation",
    timeout=300,
    parameters_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to existing .pptx file"},
            "slide_type": {"type": "string", "enum": ["title", "content", "section_header", "blank", "image"]},
            "title": {"type": "string"},
            "body": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
            "image_path": {"type": "string"},
            "section_number": {"type": "string"},
        },
        "required": ["file_path", "slide_type", "title"],
    },
)
async def ppt_add_slide(params: dict[str, Any]) -> ToolResult:
    """Add a slide to an existing presentation."""
    if not PPTX_AVAILABLE:
        return ToolResult(
            success=False,
            output="python-pptx is not installed."
        )

    try:
        file_path = params.get("file_path", "")
        slide_type = params.get("slide_type", "content")
        title = params.get("title", "")
        body = params.get("body", None) or []
        notes = params.get("notes", "")
        image_path = params.get("image_path", "")

        prs = Presentation(file_path)
        blank_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_layout)

        sc = SlideContent(
            type=slide_type,
            title=title,
            body=body or [],
            notes=notes,
            image_path=image_path,
        )

        # Use minimal builder approach
        if slide_type == "title":
            bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                                         prs.slide_width, prs.slide_height,
                                         RGBColor(0x1B, 0x3A, 0x5C))
            bg.line.fill.background()
            # Title text
            txBox = slide.shapes.add_textbox(Inches(1.5), Inches(2.5),
                                              Inches(10), Inches(1.5))
            tf = txBox.text_frame
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(40)
            p.font.bold = True
            p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        else:
            txBox = slide.shapes.add_textbox(Inches(0.7), Inches(0.5),
                                              Inches(12), Inches(6.5))
            tf = txBox.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(28)
            p.font.bold = True
            p.font.color.rgb = RGBColor(0x1B, 0x3A, 0x5C)

            if body:
                for line in body:
                    p2 = tf.add_paragraph()
                    p2.text = f"\u2022  {line}"
                    p2.font.size = Pt(16)
                    p2.space_after = Pt(4)

        # Save with same filename
        prs.save(file_path)

        return ToolResult(
            success=True,
            output=json.dumps({
                "success": True,
                "file_path": file_path,
                "message": f"Added '{slide_type}' slide titled '{title}' to {file_path}",
            })
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output=f"Failed to add slide: {type(e).__name__}: {e}"
        )


@tool(
    name="list_ppt_templates",
    description="List available presentation themes and slide templates",
    parameters_schema={
        "type": "object",
        "properties": {},
    },
)
async def list_ppt_templates(params: dict[str, Any]) -> ToolResult:
    """List available presentation themes."""
    themes = await _list_presentation_templates()
    theme_names = "\n".join(f"  - {t['id']}: {t['name']}" for t in themes)

    return ToolResult(
        success=True,
        output=json.dumps({
            "success": True,
            "themes": themes,
            "message": (
                f"Available themes:\n{theme_names}\n\n"
                f"Slide types: title, content, two_column, section_header, comparison, "
                f"image, table, chart, blank\n\n"
                f"Example usage:\n"
                f"create_ppt(title=\"My Presentation\", slides=[\n"
                f'  {{"type": "title", "title": "Main Title"}},\n'
                f'  {{"type": "content", "title": "Agenda", "body": ["Point 1", "Point 2"]}},\n'
                f'  {{"type": "comparison", "title": "Comparison", '
                f'"comparisons": [["Metric", "Before", "After"], ["Speed", "10ms", "5ms"]]}},\n'
                f'  {{"type": "chart", "title": "Revenue", '
                f'"chart_type": "bar", '
                f'"chart_data": {{"categories": ["Q1","Q2"], '
                f'"series": [{{"name": "Revenue", "values": [100, 200]}}]}}}}\n'
                f"], theme=\"corporate\")\n"
            ),
        })
    )

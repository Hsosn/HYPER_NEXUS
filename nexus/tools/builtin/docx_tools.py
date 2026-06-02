"""
Professional Word document creation tool using python-docx.

Features:
- Title page with subtitle, author, date
- Headings (H1-H3) and body paragraphs
- Bullet lists and numbered lists with indentation
- Tables with header rows and alternating row colors
- Images with captions
- Page breaks
- Headers and footers with page numbers
- Multiple themes/color schemes
- Smart text wrapping and formatting
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ... import config
from ..registry import tool, ToolResult
from ...events import emit

try:
    from docx import Document
    from docx.shared import Inches, Pt, Cm, RGBColor, Emu
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.section import WD_ORIENT
    from docx.oxml.ns import qn, nsdecls
    from docx.oxml import parse_xml
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ── Color Themes ────────────────────────────────────────────────────────────

THEMES: dict[str, dict] = {
    "professional": {
        "name": "Professional Blue",
        "accent1": RGBColor(0x1B, 0x3A, 0x5C),    # Dark navy
        "accent2": RGBColor(0x2E, 0x86, 0xC1),    # Steel blue
        "accent3": RGBColor(0x00, 0xA8, 0xE8),    # Bright blue
        "accent4": RGBColor(0x4C, 0xAF, 0x50),    # Green
        "text": RGBColor(0x33, 0x33, 0x33),
        "text_light": RGBColor(0x66, 0x66, 0x66),
        "heading_color": RGBColor(0x1B, 0x3A, 0x5C),
        "font_body": "Calibri",
        "font_heading": "Calibri Light",
    },
    "dark": {
        "name": "Dark Modern",
        "accent1": RGBColor(0xBB, 0x86, 0xFC),    # Purple
        "accent2": RGBColor(0x06, 0xD6, 0xA0),    # Teal
        "accent3": RGBColor(0x4C, 0xC9, 0xF0),    # Sky blue
        "accent4": RGBColor(0xE9, 0xC4, 0x6A),    # Gold
        "text": RGBColor(0xE0, 0xE0, 0xE0),
        "text_light": RGBColor(0xA0, 0xA0, 0xA0),
        "heading_color": RGBColor(0xBB, 0x86, 0xFC),
        "font_body": "Calibri",
        "font_heading": "Calibri Light",
    },
    "minimal": {
        "name": "Minimal Clean",
        "accent1": RGBColor(0x20, 0x20, 0x20),
        "accent2": RGBColor(0x55, 0x55, 0x55),
        "accent3": RGBColor(0x88, 0x88, 0x88),
        "accent4": RGBColor(0xAA, 0xAA, 0xAA),
        "text": RGBColor(0x22, 0x22, 0x22),
        "text_light": RGBColor(0x77, 0x77, 0x77),
        "heading_color": RGBColor(0x11, 0x11, 0x11),
        "font_body": "Calibri",
        "font_heading": "Calibri Light",
    },
    "nature": {
        "name": "Nature Green",
        "accent1": RGBColor(0x2D, 0x6A, 0x4F),    # Forest green
        "accent2": RGBColor(0x4C, 0x9A, 0x6E),    # Sage
        "accent3": RGBColor(0x8F, 0xC3, 0x92),    # Light green
        "accent4": RGBColor(0xC8, 0xE6, 0xC9),    # Pale green
        "text": RGBColor(0x2C, 0x3E, 0x2F),
        "text_light": RGBColor(0x5A, 0x6B, 0x5A),
        "heading_color": RGBColor(0x2D, 0x6A, 0x4F),
        "font_body": "Calibri",
        "font_heading": "Calibri Light",
    },
    "corporate": {
        "name": "Corporate Navy",
        "accent1": RGBColor(0x00, 0x3F, 0x72),
        "accent2": RGBColor(0x00, 0x7B, 0xB8),
        "accent3": RGBColor(0x00, 0xA6, 0xD6),
        "accent4": RGBColor(0x7F, 0xBA, 0x00),
        "text": RGBColor(0x33, 0x33, 0x33),
        "text_light": RGBColor(0x70, 0x70, 0x70),
        "heading_color": RGBColor(0x00, 0x3F, 0x72),
        "font_body": "Calibri",
        "font_heading": "Calibri Light",
    },
}

DEFAULT_THEME = "professional"


# ── Data types ────────────────────────────────────────────────────────────

@dataclass
class DocSection:
    """Content section within a Word document."""
    type: str = "paragraph"  # title, heading1, heading2, heading3, paragraph, bullet_list, numbered_list, table, image, page_break
    text: str = ""
    level: int = 0
    items: list[str] = field(default_factory=list)  # for lists
    bold: bool = False
    italic: bool = False
    font_size: int | None = None
    font_color: str | None = None
    alignment: str = "left"  # left, center, right, justify
    # Table-specific
    table_data: list[list[str]] = field(default_factory=list)
    table_header_row: bool = True
    # Image-specific
    image_path: str = ""
    image_width: float | None = None  # in inches
    image_caption: str = ""


@dataclass
class DocSpec:
    """Complete Word document specification."""
    title: str = "Document"
    subtitle: str = ""
    author: str = "Hyper Nexus"
    theme: str = DEFAULT_THEME
    sections: list[DocSection] = field(default_factory=list)
    include_page_numbers: bool = True
    include_header: bool = True
    header_text: str = ""
    include_footer: bool = True
    footer_text: str = ""
    page_size: str = "A4"  # A4 or Letter
    orientation: str = "portrait"
    output_path: str = ""
    date_format: str = "%B %d, %Y"


# ── Document Builder ──────────────────────────────────────────────────────

class DocBuilder:
    """Builds professional Word documents with smart formatting."""

    def __init__(self, spec: DocSpec):
        self.spec = spec
        self.theme = THEMES.get(spec.theme, THEMES[DEFAULT_THEME])
        self.doc = Document()

        # Set page size and orientation
        section = self.doc.sections[0]
        if spec.page_size == "A4":
            section.page_width = Cm(21.0)
            section.page_height = Cm(29.7)
        else:  # Letter
            section.page_width = Inches(8.5)
            section.page_height = Inches(11.0)

        # Margins
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

        # Set default font
        style = self.doc.styles['Normal']
        font = style.font
        font.name = self.theme["font_body"]
        font.size = Pt(11)
        font.color.rgb = self.theme["text"]

        # Set heading styles
        for i in range(1, 4):
            heading_style = self.doc.styles[f'Heading {i}']
            heading_font = heading_style.font
            heading_font.name = self.theme["font_heading"]
            heading_font.color.rgb = self.theme["heading_color"]
            heading_font.bold = True
            if i == 1:
                heading_font.size = Pt(18)
            elif i == 2:
                heading_font.size = Pt(15)
            else:
                heading_font.size = Pt(13)

        # Header/Footer — use runs for font styling (Paragraph has no .font)
        if spec.include_header or spec.header_text:
            header = section.header
            hp = header.paragraphs[0]
            hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = hp.add_run(spec.header_text or spec.title)
            run.font.size = Pt(9)
            run.font.color.rgb = self.theme["text_light"]
            run.font.name = self.theme["font_body"]

        if spec.include_footer or spec.footer_text:
            footer = section.footer
            fp = footer.paragraphs[0]
            fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = fp.add_run(spec.footer_text)
            run.font.size = Pt(9)
            run.font.color.rgb = self.theme["text_light"]
            run.font.name = self.theme["font_body"]

        if spec.include_page_numbers:
            self._add_page_numbers(section)

    def _add_page_numbers(self, section):
        """Add page numbers to footer."""
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        # Add page number field
        run = fp.add_run()
        fldChar1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
        run._r.append(fldChar1)
        run2 = fp.add_run()
        instrText = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> PAGE </w:instrText>')
        run2._r.append(instrText)
        run3 = fp.add_run()
        fldChar2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
        run3._r.append(fldChar2)

    def _add_run(self, paragraph, text, bold=False, italic=False,
                 font_size=None, font_color=None, font_name=None):
        """Add a formatted run to a paragraph."""
        run = paragraph.add_run(text)
        run.bold = bold
        run.italic = italic
        if font_size:
            run.font.size = Pt(font_size)
        if font_color:
            if isinstance(font_color, str):
                try:
                    hex_color = font_color.lstrip('#')
                    run.font.color.rgb = RGBColor(
                        int(hex_color[0:2], 16),
                        int(hex_color[2:4], 16),
                        int(hex_color[4:6], 16),
                    )
                except Exception:
                    pass
            else:
                run.font.color.rgb = font_color
        if font_name:
            run.font.name = font_name
        return run

    def _add_paragraph(self, text, style='Normal', alignment=WD_ALIGN_PARAGRAPH.LEFT,
                       bold=False, italic=False, font_size=None, font_color=None,
                       space_after=Pt(6), space_before=Pt(0)):
        """Add a formatted paragraph."""
        p = self.doc.add_paragraph(style=style)
        p.alignment = alignment
        p.paragraph_format.space_after = space_after
        p.paragraph_format.space_before = space_before
        p.paragraph_format.line_spacing = 1.15

        if text:
            self._add_run(p, text, bold=bold, italic=italic,
                          font_size=font_size, font_color=font_color)
        return p

    def _set_cell_shading(self, cell, color_hex):
        """Set cell background shading."""
        shading = parse_xml(
            f'<w:shd {nsdecls("w")} w:fill="{color_hex}" w:val="clear"/>'
        )
        cell._tc.get_or_add_tcPr().append(shading)

    def build_title_page(self):
        """Build a professional title page."""
        # Add spacing before title
        for _ in range(6):
            self.doc.add_paragraph()

        # Title
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(self.spec.title)
        run.bold = True
        run.font.size = Pt(28)
        run.font.color.rgb = self.theme["heading_color"]
        run.font.name = self.theme["font_heading"]

        # Accent line
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run("─" * 40)
        run.font.color.rgb = self.theme["accent2"]
        run.font.size = Pt(12)

        # Subtitle
        if self.spec.subtitle:
            self._add_paragraph(
                self.spec.subtitle,
                alignment=WD_ALIGN_PARAGRAPH.CENTER,
                font_size=16,
                font_color=self.theme["text_light"],
                space_before=Pt(12),
                space_after=Pt(6),
            )

        # Author and date
        today = datetime.now().strftime(self.spec.date_format)
        self._add_paragraph(
            f"{self.spec.author}  |  {today}",
            alignment=WD_ALIGN_PARAGRAPH.CENTER,
            font_size=12,
            font_color=self.theme["text_light"],
            space_before=Pt(24),
        )

        # Page break after title
        self.doc.add_page_break()

    def build_sections(self):
        """Build all document sections."""
        for section in self.spec.sections:
            self._build_section(section)

    def _build_section(self, sec: DocSection):
        """Build a single section element."""
        align_map = {
            "left": WD_ALIGN_PARAGRAPH.LEFT,
            "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT,
            "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        }
        alignment = align_map.get(sec.alignment, WD_ALIGN_PARAGRAPH.LEFT)
        font_color_obj = self._resolve_color(sec.font_color)

        if sec.type == "title":
            # Document title (not heading)
            self._add_paragraph(
                sec.text,
                alignment=alignment,
                bold=True,
                font_size=sec.font_size or 22,
                font_color=font_color_obj or self.theme["heading_color"],
                space_after=Pt(12),
            )

        elif sec.type == "heading1":
            self.doc.add_heading(sec.text, level=1)

        elif sec.type == "heading2":
            self.doc.add_heading(sec.text, level=2)

        elif sec.type == "heading3":
            self.doc.add_heading(sec.text, level=3)

        elif sec.type == "paragraph":
            self._add_paragraph(
                sec.text,
                alignment=alignment,
                bold=sec.bold,
                italic=sec.italic,
                font_size=sec.font_size or 11,
                font_color=font_color_obj or self.theme["text"],
                space_after=Pt(6),
            )

        elif sec.type == "bullet_list":
            for item in sec.items:
                p = self.doc.add_paragraph(style='List Bullet')
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(0)
                self._add_run(p, item, font_size=sec.font_size or 11,
                              font_color=font_color_obj or self.theme["text"])

        elif sec.type == "numbered_list":
            for i, item in enumerate(sec.items):
                p = self.doc.add_paragraph(style='List Number')
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(0)
                self._add_run(p, item, font_size=sec.font_size or 11,
                              font_color=font_color_obj or self.theme["text"])

        elif sec.type == "table":
            if sec.table_data and len(sec.table_data) > 0:
                rows = len(sec.table_data)
                cols = max(len(row) for row in sec.table_data)
                table = self.doc.add_table(rows=rows, cols=cols)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER

                # Apply table style
                table.style = self.doc.styles['Light Shading Accent 1']

                # Fill table data and style
                for r_idx, row_data in enumerate(sec.table_data):
                    for c_idx in range(cols):
                        cell = table.cell(r_idx, c_idx)
                        val = row_data[c_idx] if c_idx < len(row_data) else ""
                        cell.text = str(val)

                        for paragraph in cell.paragraphs:
                            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                            for run in paragraph.runs:
                                run.font.size = Pt(10)
                                run.font.name = self.theme["font_body"]

                        if r_idx == 0 and sec.table_header_row:
                            # Header row
                            self._set_cell_shading(cell, "1B3A5C")
                            for paragraph in cell.paragraphs:
                                for run in paragraph.runs:
                                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                                    run.font.bold = True

        elif sec.type == "image":
            if sec.image_path and os.path.exists(sec.image_path):
                try:
                    width = Inches(sec.image_width or 5.5)
                    self.doc.add_picture(sec.image_path, width=width)
                    last_paragraph = self.doc.paragraphs[-1]
                    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    if sec.image_caption:
                        self._add_paragraph(
                            sec.image_caption,
                            alignment=WD_ALIGN_PARAGRAPH.CENTER,
                            font_size=9,
                            font_color=self.theme["text_light"],
                            space_before=Pt(2),
                            space_after=Pt(8),
                        )
                except Exception as e:
                    self._add_paragraph(
                        f"[Image could not be loaded: {e}]",
                        font_color=RGBColor(0xFF, 0x00, 0x00),
                    )
            else:
                self._add_paragraph(
                    f"[Image placeholder: {sec.image_path or 'no path provided'}]",
                    font_color=self.theme["text_light"],
                    italic=True,
                )

        elif sec.type == "page_break":
            self.doc.add_page_break()

    def _resolve_color(self, color_str: str | None):
        """Resolve a color string to an RGBColor or None."""
        if not color_str:
            return None
        try:
            hex_color = color_str.lstrip('#')
            return RGBColor(
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            )
        except Exception:
            return None

    def build(self) -> str:
        """Build the document and return the file path."""
        # Build title page
        if self.spec.title:
            self.build_title_page()

        # Build sections
        self.build_sections()

        # Set document properties
        self.doc.core_properties.title = self.spec.title
        self.doc.core_properties.author = self.spec.author
        self.doc.core_properties.created = datetime.now()

        # Save
        output = self.spec.output_path or str(
            Path(config.BASE_DIR) / "data" / "workspace" /
            f"{self.spec.title.replace(' ', '_')}_{int(time.time())}.docx"
        )
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(output_path))
        return str(output_path)


# ── Section Parsing ──────────────────────────────────────────────────────

def _parse_sections(sections_json: str | list) -> list[DocSection]:
    """Parse sections from JSON string or list."""
    if isinstance(sections_json, str):
        try:
            data = json.loads(sections_json)
        except json.JSONDecodeError:
            return []
    else:
        data = sections_json

    if not isinstance(data, list):
        return []

    sections = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sec = DocSection(
            type=item.get("type", "paragraph"),
            text=item.get("text", ""),
            level=item.get("level", 0),
            items=item.get("items", item.get("body", [])),
            bold=item.get("bold", False),
            italic=item.get("italic", False),
            font_size=item.get("font_size"),
            font_color=item.get("font_color"),
            alignment=item.get("alignment", "left"),
            table_data=item.get("table_data", item.get("table", [])),
            table_header_row=item.get("table_header_row", True),
            image_path=item.get("image_path", ""),
            image_width=item.get("image_width"),
            image_caption=item.get("image_caption", ""),
        )
        sections.append(sec)

    return sections


# ── Tool implementations ─────────────────────────────────────────────────

@tool(
    name="docx",
    description=(
        "Create professional Microsoft Word (.docx) documents with title page, "
        "headings, paragraphs, bullet lists, numbered lists, tables, images, "
        "and page breaks. Supports multiple themes (professional, dark, minimal, "
        "nature, corporate). Returns the path to the created file."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Document title (appears on title page and in header)",
            },
            "subtitle": {
                "type": "string",
                "description": "Document subtitle on title page",
            },
            "author": {
                "type": "string",
                "description": "Document author name",
                "default": "Hyper Nexus",
            },
            "theme": {
                "type": "string",
                "enum": list(THEMES.keys()),
                "description": "Visual color theme for the document",
                "default": "professional",
            },
            "sections": {
                "type": "array",
                "description": (
                    "Array of section objects. Each section has a type and content. "
                    "Supported types: title, heading1, heading2, heading3, paragraph, "
                    "bullet_list, numbered_list, table, image, page_break."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "title", "heading1", "heading2", "heading3",
                                "paragraph", "bullet_list", "numbered_list",
                                "table", "image", "page_break",
                            ],
                            "description": "Section type",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text content (for title, heading, paragraph types)",
                        },
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List items (for bullet_list, numbered_list types)",
                        },
                        "bold": {"type": "boolean", "description": "Bold text"},
                        "italic": {"type": "boolean", "description": "Italic text"},
                        "font_size": {
                            "type": "integer",
                            "description": "Font size in points",
                        },
                        "font_color": {
                            "type": "string",
                            "description": "Font color in hex (e.g. #FF0000)",
                        },
                        "alignment": {
                            "type": "string",
                            "enum": ["left", "center", "right", "justify"],
                            "description": "Text alignment",
                        },
                        "table_data": {
                            "type": "array",
                            "description": (
                                "2D array for table: "
                                "[[header1, header2], [row1col1, row1col2], ...]"
                            ),
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "table_header_row": {
                            "type": "boolean",
                            "description": "Whether first row is a header row",
                        },
                        "image_path": {
                            "type": "string",
                            "description": "Path to image file",
                        },
                        "image_width": {
                            "type": "number",
                            "description": "Image width in inches",
                        },
                        "image_caption": {
                            "type": "string",
                            "description": "Image caption text",
                        },
                    },
                },
            },
            "include_page_numbers": {
                "type": "boolean",
                "description": "Add page numbers in footer",
            },
            "include_header": {
                "type": "boolean",
                "description": "Show header on each page",
            },
            "header_text": {
                "type": "string",
                "description": "Custom header text (defaults to document title)",
            },
            "include_footer": {
                "type": "boolean",
                "description": "Show footer on each page",
            },
            "footer_text": {
                "type": "string",
                "description": "Custom footer text",
            },
            "page_size": {
                "type": "string",
                "enum": ["A4", "Letter"],
                "description": "Page size",
            },
            "output_path": {
                "type": "string",
                "description": "Custom output file path",
            },
        },
        "required": ["title", "sections"],
    },
)
async def create_docx(params: dict[str, Any]) -> ToolResult:
    """Create a professional Word document."""
    if not DOCX_AVAILABLE:
        return ToolResult(
            success=False,
            output="python-docx is not installed. Install it with: pip install python-docx"
        )

    try:
        title = params.get("title", "Document")
        sections = params.get("sections", [])
        subtitle = params.get("subtitle", "")
        author = params.get("author", "Hyper Nexus")
        theme = params.get("theme", "professional")
        include_page_numbers = params.get("include_page_numbers", True)
        include_header = params.get("include_header", True)
        include_footer = params.get("include_footer", False)
        header_text = params.get("header_text", "")
        footer_text = params.get("footer_text", "")
        page_size = params.get("page_size", "A4")
        output_path = params.get("output_path", "")

        parsed_sections = _parse_sections(sections)
        if not parsed_sections:
            return ToolResult(
                success=False,
                output="No valid sections provided. Each section needs at minimum a 'type' field."
            )

        spec = DocSpec(
            title=title,
            subtitle=subtitle,
            author=author,
            theme=theme if theme in THEMES else DEFAULT_THEME,
            sections=parsed_sections,
            include_page_numbers=include_page_numbers,
            include_header=include_header,
            header_text=header_text,
            include_footer=include_footer,
            footer_text=footer_text,
            page_size=page_size,
            output_path=output_path,
        )

        builder = DocBuilder(spec)
        file_path = builder.build()

        await emit("document_created", path=file_path, file_kind="docx")

        return ToolResult(
            success=True,
            output=json.dumps({
                "success": True,
                "file_path": file_path,
                "section_count": len(parsed_sections),
                "theme": theme,
                "message": f"Created '{title}' with {len(parsed_sections)} sections at {file_path}",
            })
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output=f"Failed to create document: {type(e).__name__}: {e}"
        )


@tool(
    name="list_docx_templates",
    description="List available Word document themes/templates",
    parameters_schema={
        "type": "object",
        "properties": {},
    },
)
async def list_docx_templates() -> ToolResult:
    """List available document themes."""
    themes = []
    for key, theme in THEMES.items():
        themes.append({"id": key, "name": theme["name"]})

    theme_names = "\n".join(f"  - {t['id']}: {t['name']}" for t in themes)

    return ToolResult(
        success=True,
        output=json.dumps({
            "success": True,
            "themes": themes,
            "message": (
                f"Available themes:\n{theme_names}\n\n"
                f"Section types: title, heading1, heading2, heading3, paragraph, "
                f"bullet_list, numbered_list, table, image, page_break\n\n"
                f"Example usage:\n"
                f'docx(title="My Document", sections=[\n'
                f'  {{"type": "heading1", "text": "Introduction"}},\n'
                f'  {{"type": "paragraph", "text": "This is the body text."}},\n'
                f'  {{"type": "bullet_list", "items": ["Point one", "Point two"]}},\n'
                f'  {{"type": "table", "table_data": [["Name", "Value"], ["A", "1"], ["B", "2"]]}},\n'
                f'], theme="professional")\n'
            ),
        })
    )

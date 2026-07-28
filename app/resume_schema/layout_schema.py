from typing import Literal

from pydantic import BaseModel, Field

# defaults so renderer never breaks
class GlobalStyleSchema(BaseModel):
    """Document-wide resume styling and layout settings."""

    max_pages: int = Field(default=1, ge=1)
    page_size: Literal["letter", "a4"] = "letter"

    font_family: str = "Arial, sans-serif"
    font_size: str = "9.5pt"
    line_height: str = "1.18"

    page_margin_top: str = "0.45in"
    page_margin_right: str = "0.5in"
    page_margin_bottom: str = "0.45in"
    page_margin_left: str = "0.5in"

    section_margin: str = "7px"
    section_heading_margin: str = "3px"
    item_margin: str = "4px"
    bullet_margin: str = "2px"

    heading_font_size: str = "11pt"
    heading_font_weight: int = Field(default=700, ge=100, le=900)

    text_color: str = "#111111"
    heading_color: str = "#111111"
    divider_color: str = "#444444"
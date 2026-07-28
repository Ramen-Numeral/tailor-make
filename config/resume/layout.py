"""Default global resume style configuration."""

from app.resume_schema.layout_schema import GlobalStyleSchema


LAYOUT_STYLE = GlobalStyleSchema(
    max_pages=1,
    page_size="letter",
    font_family="Arial, sans-serif",
    font_size="9.5pt",
    line_height="1.18",
    page_margin_top="0.45in",
    page_margin_right="0.5in",
    page_margin_bottom="0.45in",
    page_margin_left="0.5in",
    section_margin="7px",
    section_heading_margin="3px",
    item_margin="4px",
    bullet_margin="2px",
    heading_font_size="11pt",
    heading_font_weight=700,
    text_color="#111111",
    heading_color="#111111",
    divider_color="#444444",
)
"""Shared, report-shape-agnostic HTML email rendering helpers.

Extracted from AssetRotationStrategy (strategy.py) so CryptoRotationStrategy
can build its own daily report email using the same visual style without
duplicating ~80 lines of HTML/CSS table-building boilerplate. Each function
here only ever takes plain data in and HTML text out -- no report-shape
assumptions, no self/strategy coupling -- so both strategies assemble their
own report-specific sections and pass them in.
"""

import html
from typing import Any


def email_status_theme(status: str) -> tuple[str, str]:
    """Map a free-text status line to a (background, text) banner color."""
    lowered = status.lower()
    if "block" in lowered or "error" in lowered or "failed" in lowered:
        return "#fdecea", "#b3261e"
    if "pending" in lowered or "waiting" in lowered:
        return "#fff4e5", "#8a5300"
    if any(term in lowered for term in ("submitted", "complete", "filled", "finished", "top-up", "build")):
        return "#e6f4ea", "#1e7e34"
    return "#eceff1", "#455a64"


def email_value(value: Any, *, money: bool = False) -> str:
    if value is None:
        return "unavailable"
    if money and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"${float(value):,.2f}"
    return str(value)


def email_kv_section(title: str, rows: list[tuple[str, Any]]) -> str:
    """Render a titled two-column table of label/value rows."""
    body_rows = []
    for index, (label, value) in enumerate(rows):
        shade = "#ffffff" if index % 2 == 0 else "#f8f9fb"
        body_rows.append(
            '<tr style="background-color:{shade};">'
            '<td style="padding:8px 12px;font-size:13px;color:#5f6368;width:44%;'
            'border-bottom:1px solid #eceff1;vertical-align:top;">{label}</td>'
            '<td style="padding:8px 12px;font-size:13px;color:#1a1a2e;font-weight:500;'
            'border-bottom:1px solid #eceff1;vertical-align:top;">{value}</td>'
            "</tr>".format(
                shade=shade,
                label=html.escape(label),
                value=html.escape(email_value(value)),
            )
        )
    return (
        '<div style="font-size:12px;font-weight:700;color:#8a8f98;'
        'text-transform:uppercase;letter-spacing:0.05em;margin:20px 0 6px;">'
        f"{html.escape(title)}</div>"
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;">' + "".join(body_rows) + "</table>"
    )


def email_bullet_section(title: str, items: list[str]) -> str:
    """Render a titled bullet list, or a muted placeholder when empty."""
    heading = (
        '<div style="font-size:12px;font-weight:700;color:#8a8f98;'
        'text-transform:uppercase;letter-spacing:0.05em;margin:20px 0 6px;">'
        f"{html.escape(title)}</div>"
    )
    if not items:
        return heading + (
            '<div style="font-size:13px;color:#8a8f98;font-style:italic;">'
            "None reported</div>"
        )
    list_items = "".join(
        f'<li style="margin-bottom:4px;">{html.escape(str(item))}</li>' for item in items
    )
    return heading + (
        '<ul style="margin:0;padding-left:18px;color:#333333;font-size:13px;line-height:1.6;">'
        + list_items
        + "</ul>"
    )


def render_email_shell(
    *,
    report_date: str,
    mode_label: str,
    status: str,
    narrative: str,
    sections_html: str,
) -> str:
    """Wrap pre-rendered section HTML in the shared header/badge/footer chrome."""
    badge_bg, badge_fg = email_status_theme(status)
    narrative = (narrative or "").strip()
    narrative_block = (
        '<tr><td style="padding:14px 24px 0;color:#333333;font-size:13px;'
        f'line-height:1.6;font-style:italic;">{html.escape(narrative)}</td></tr>'
        if narrative
        else ""
    )
    return f"""\
<!doctype html>
<html>
<body style="margin:0;padding:0;background-color:#f2f4f6;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f2f4f6;">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background-color:#ffffff;border-radius:8px;overflow:hidden;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
<tr><td style="background-color:#1a1a2e;padding:20px 24px;">
<div style="color:#ffffff;font-size:18px;font-weight:600;">Raspberry Pi Trading Agent</div>
<div style="color:#b8bcc8;font-size:13px;margin-top:4px;">Daily Summary &middot; {html.escape(report_date)} &middot; {html.escape(mode_label)}</div>
</td></tr>
<tr><td style="padding:20px 24px 0;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:{badge_bg};border-radius:6px;">
<tr><td style="padding:14px 16px;color:{badge_fg};font-size:14px;font-weight:600;">{html.escape(status)}</td></tr>
</table>
</td></tr>
{narrative_block}
<tr><td style="padding:0 24px 8px;">
{sections_html}
</td></tr>
<tr><td style="padding:16px 24px 20px;color:#8a8f98;font-size:12px;line-height:1.5;border-top:1px solid #eceff1;">
Review all orders and positions in the Alpaca dashboard.<br>
This automated message is not financial advice.
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>
"""

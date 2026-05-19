"""Build a styled HTML version of the Markdown project report."""

from __future__ import annotations

import html
import re
from pathlib import Path


SOURCE = Path("report/project_report.md")
TARGET = Path("report/project_report.html")


def parse_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*(.*?)\*", r"<em>\1</em>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    table_lines = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        table_lines.append(lines[index].strip())
        index += 1

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    body_lines = table_lines[2:]
    output = ["<table>", "<thead><tr>"]
    output.extend(f"<th>{parse_inline(cell)}</th>" for cell in header)
    output.append("</tr></thead>")
    output.append("<tbody>")
    for row in body_lines:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        output.append("<tr>")
        output.extend(f"<td>{parse_inline(cell)}</td>" for cell in cells)
        output.append("</tr>")
    output.append("</tbody></table>")
    return "\n".join(output), index


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    in_list = False
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{parse_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            close_list()
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and lines[index + 1].strip().startswith("|"):
            flush_paragraph()
            close_list()
            table_html, index = parse_table(lines, index)
            output.append(table_html)
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            close_list()
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            text = stripped[level:].strip()
            output.append(f"<h{level}>{parse_inline(text)}</h{level}>")
            index += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{parse_inline(stripped[2:].strip())}</li>")
            index += 1
            continue

        close_list()
        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    close_list()
    return "\n".join(output)


def build_document(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Text Classification for Spam Detection</title>
  <style>
    @page {{
      size: A4;
      margin: 20mm 18mm 20mm 18mm;
    }}
    body {{
      color: #1f2933;
      font-family: "Liberation Serif", "Times New Roman", serif;
      font-size: 11.5pt;
      line-height: 1.45;
    }}
    main {{
      max-width: 100%;
    }}
    h1, h2, h3 {{
      color: #111827;
      font-family: "Liberation Sans", Arial, sans-serif;
      line-height: 1.2;
      margin: 1.05em 0 0.45em;
      page-break-after: avoid;
    }}
    h1 {{
      font-size: 21pt;
      margin-top: 0;
      text-align: center;
    }}
    h2 {{
      border-bottom: 1px solid #d1d5db;
      font-size: 15pt;
      padding-bottom: 0.15em;
    }}
    h3 {{
      font-size: 12.5pt;
    }}
    p {{
      margin: 0 0 0.7em;
      text-align: justify;
    }}
    ul {{
      margin: 0 0 0.8em 1.2em;
      padding: 0;
    }}
    li {{
      margin: 0.2em 0;
    }}
    table {{
      border-collapse: collapse;
      font-size: 9.4pt;
      margin: 0.6em 0 1em;
      page-break-inside: avoid;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 5px 6px;
      vertical-align: top;
    }}
    th {{
      background: #eef2f7;
      font-family: "Liberation Sans", Arial, sans-serif;
      font-weight: 700;
      text-align: left;
    }}
    code {{
      background: #f3f4f6;
      border-radius: 3px;
      font-family: "Liberation Mono", monospace;
      font-size: 0.9em;
      padding: 1px 3px;
    }}
    strong {{
      color: #111827;
    }}
  </style>
</head>
<body>
  <main>
{body}
  </main>
</body>
</html>
"""


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    TARGET.write_text(build_document(markdown_to_html(markdown)), encoding="utf-8")
    print(f"Generated {TARGET}")


if __name__ == "__main__":
    main()

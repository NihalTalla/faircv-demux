"""Export a markdown report to a printable PDF.

Uses the `markdown` package (tables extension) + headless Chrome print-to-pdf.
Usage: python export_pdf.py <input.md> <output.pdf>
"""
import os
import sys
import subprocess
import tempfile

import markdown

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
COMPACT = "--compact" in sys.argv

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{ size: A4; margin: 14mm 13mm 16mm 13mm; }}
  body {{ font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif;
         font-size: 9.5pt; line-height: 1.42; color: #1a1a1a; }}
  h1 {{ font-size: 17pt; border-bottom: 2px solid #444; padding-bottom: 4px; }}
  h2 {{ font-size: 13pt; border-bottom: 1px solid #bbb; padding-bottom: 2px;
        margin-top: 16px; page-break-after: avoid; }}
  h3 {{ font-size: 11pt; margin-top: 12px; page-break-after: avoid; }}
  h4 {{ font-size: 10pt; }}
  p, li {{ text-align: justify; }}
  code {{ font-family: Consolas, monospace; font-size: 8.2pt;
          background: #f3f3f3; padding: 0 2px; border-radius: 2px; }}
  pre {{ background: #f6f6f6; padding: 6px 8px; font-size: 8.2pt;
        overflow-x: hidden; white-space: pre-wrap; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0;
          font-size: 7.6pt; page-break-inside: auto; }}
  th, td {{ border: 0.6px solid #999; padding: 2.5px 4px; text-align: left;
            vertical-align: top; }}
  th {{ background: #ececec; font-weight: 600; }}
  tr {{ page-break-inside: avoid; }}
  blockquote {{ margin: 6px 0 6px 10px; padding: 4px 10px; color: #333;
                border-left: 3px solid #aaa; background: #f7f7f7; }}
  strong {{ color: #000; }}
  hr {{ border: none; border-top: 1px solid #ccc; margin: 12px 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main():
    args = [a for a in sys.argv[1:] if a != "--compact"]
    if len(args) != 2:
        print("usage: python export_pdf.py [--compact] <input.md> <output.pdf>")
        raise SystemExit(1)
    src, dst = args[0], args[1]
    with open(src, encoding="utf-8") as f:
        text = f.read()
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    title = os.path.splitext(os.path.basename(src))[0]
    if COMPACT:
        body = (
            "<style>"
            "@page { size: A4; margin: 9mm 10mm 10mm 10mm; }"
            "body { font-size: 8.8pt; line-height: 1.3; }"
            "h1 { font-size: 13.5pt; margin: 0 0 4px 0; }"
            "h2 { font-size: 10.5pt; margin: 8px 0 3px 0; }"
            "table { font-size: 7.2pt; margin: 4px 0; }"
            "th, td { padding: 1.5px 3px; }"
            "p, li { margin: 2px 0; }"
            "ul { margin: 2px 0 2px 0; padding-left: 16px; }"
            "hr { margin: 6px 0; }"
            "</style>"
        ) + body
    html = HTML_TEMPLATE.format(title=title, body=body)

    tmp_html = os.path.join(tempfile.gettempdir(), "faircv_report.html")
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(html)

    dst_abs = os.path.abspath(dst)
    url = "file:///" + tmp_html.replace("\\", "/")
    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={dst_abs}",
        url,
    ]
    print("running:", " ".join(cmd[:6]) + " ...")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    size = os.path.getsize(dst_abs)
    print(f"wrote {dst_abs} ({size:,} bytes)")


if __name__ == "__main__":
    main()

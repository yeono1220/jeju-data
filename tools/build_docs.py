"""docs/src/*.html(본문 조각) → docs/*.pdf. 그림은 base64로 넣고, 공통 스타일을 입힌다.

사용: python tools/build_docs.py [이름 ...]   (이름을 안 주면 docs/src의 모든 .html)
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataolle.report.document import to_pdf  # noqa: E402

SRC, OUT = ROOT / "docs" / "src", ROOT / "docs"
CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #898781; } }
* { box-sizing: border-box; }
body { font-family: "Malgun Gothic", "맑은 고딕", "Noto Sans CJK KR", sans-serif; color: #1b1b1a; font-size: 10.2pt;
  line-height: 1.65; margin: 0; word-break: keep-all; overflow-wrap: anywhere; background: #fff; }
.cover { border-bottom: 2px solid #1c5cab; padding: 2mm 0 5mm; margin-bottom: 6mm; }
.kicker { color: #1c5cab; font-weight: 700; font-size: 9pt; }
h1 { font-size: 21pt; margin: 1mm 0 1.5mm; line-height: 1.3; }
.sub { color: #52514e; }
h2 { font-size: 13.5pt; color: #0d366b; margin: 8mm 0 2.5mm; padding-bottom: 1mm; border-bottom: 1px solid #e1e0d9; break-after: avoid; }
h3 { font-size: 11pt; margin: 5mm 0 1.5mm; color: #1b1b1a; break-after: avoid; }
p { margin: 1.5mm 0; }
ol, ul { margin: 1.5mm 0 1.5mm 5.5mm; padding: 0; }
li { margin: 0.8mm 0; }
code, .mono { font-family: Consolas, "D2Coding", monospace; background: #f4f6fa; padding: 0.2mm 1.2mm; border-radius: 1mm; font-size: 9.2pt; }
pre { background: #f4f6fa; padding: 2.5mm 3.5mm; border-radius: 1.5mm; font-size: 8.8pt; white-space: pre-wrap; break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 2mm 0; break-inside: avoid; }
th, td { border-bottom: 1px solid #e1e0d9; padding: 1.4mm 2mm; text-align: left; vertical-align: top; }
th { background: #f4f6fa; font-weight: 600; }
figure { margin: 3mm 0; break-inside: avoid; }
figure img { width: 100%; border: 1px solid #e1e0d9; border-radius: 1.5mm; }
figcaption { font-size: 8.6pt; color: #52514e; margin-top: 1mm; }
.box { border-left: 3px solid #2a78d6; background: #f3f7fd; padding: 2.5mm 3.5mm; margin: 3mm 0; border-radius: 0 1.5mm 1.5mm 0; break-inside: avoid; }
.box.warn { border-left-color: #eda100; background: #fff8e8; }
.box.ok { border-left-color: #1baf7a; background: #eefaf4; }
.step { display: inline-block; min-width: 6mm; height: 6mm; border-radius: 3mm; background: #1c5cab; color: #fff; text-align: center;
  font-weight: 700; font-size: 9pt; line-height: 6mm; margin-right: 1.5mm; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
.small { font-size: 8.6pt; color: #52514e; }
.pagebreak { break-before: page; }
"""


def inline_images(html: str) -> str:
    def rep(m):
        path = SRC / m.group(1)
        if not path.exists():
            return m.group(0)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f'src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"'
    return re.sub(r'src="(img/[^"]+)"', rep, html)


def build(name: str) -> Path | None:
    src = SRC / f"{name}.html"
    body = inline_images(src.read_text(encoding="utf-8"))
    title = re.search(r"<h1>(.*?)</h1>", body)
    doc = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{title.group(1) if title else name}</title>"
           f"<style>{CSS}</style></head><body>{body}</body></html>")
    tmp = OUT / f".{name}.html"
    tmp.write_text(doc, encoding="utf-8")
    pdf = to_pdf(tmp, OUT / f"{name}.pdf")
    tmp.unlink(missing_ok=True)
    return pdf


if __name__ == "__main__":
    names = sys.argv[1:] or [p.stem for p in sorted(SRC.glob("*.html"))]
    for n in names:
        print(n, "→", build(n))

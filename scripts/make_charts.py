#!/usr/bin/env python3
"""Generate baseline-vs-LoRA accuracy bar charts as self-contained SVG.

Reads the per-dataset metric JSONs in results/ and writes
results/figures/accuracy_baseline_after.svg. SVG renders natively on GitHub
(via markdown image links) without any external service or binary asset.

Retired models (qwen3b) and models without after-results (qwen7b) are drawn
as baseline-only bars. Usage:
    python scripts/make_charts.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")

# Display order; only models present in results/ are rendered
MODELS = [
    ("qwen3vl2b", "Qwen3-VL-2B"),
    ("qwen3vl8b", "Qwen3-VL-8B"),
    ("molmo4b", "Molmo2-4B"),
    ("qwen7b", "Qwen2.5-VL-7B (baseline only)"),
]
DATASETS = [
    ("snli_ve", "SNLI-VE"),
    ("vqa_v2-third", "VQAv2 yes/no (1/3)"),
    ("visual7w", "Visual7W"),
]
BASE_COLOR = "#9aa7b4"      # gray baseline
AFTER_COLOR = "#2f6fed"     # blue LoRA after
GAIN_COLOR = "#1a9e5c"      # green delta marker
GRID_COLOR = "#e3e8ee"
TEXT = "#1f2733"
SUBTEXT = "#5b6b7b"


def load(phase, model, ds):
    path = os.path.join(RES, f"{phase}-{model}-{ds}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))["accuracy"]


def fmt(v):
    return f"{v*100:.1f}"


def make_group(x0, y0, w, h, title, rows):
    """rows: list of (model_label, baseline or None, after or None)."""
    parts = []
    # title
    parts.append(f'<text x="{x0}" y="{y0-14}" font-family="Arial" font-size="15" '
                 f'font-weight="700" fill="{TEXT}">{title}</text>')
    # plot area
    plot_x = x0 + 175
    plot_w = w - 175
    top, bottom = y0, y0 + h
    # gridlines at 0.6/0.7/0.8/0.9/1.0
    for gx in (0.6, 0.7, 0.8, 0.9, 1.0):
        gxp = plot_x + plot_w * (gx - 0.6) / 0.4
        parts.append(f'<line x1="{gxp:.1f}" y1="{top-6}" x2="{gxp:.1f}" y2="{bottom}" '
                     f'stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text x="{gxp:.1f}" y="{bottom+16}" text-anchor="middle" '
                     f'font-family="Arial" font-size="10" fill="{SUBTEXT}">{int(gx*100)}</text>')
    row_h = h / len(rows)
    bar_h = 13
    for i, (label, base, after) in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        # label
        parts.append(f'<text x="{x0}" y="{cy+4}" font-family="Arial" font-size="12" '
                     f'fill="{TEXT}">{label}</text>')
        if base is not None:
            bx = plot_x + plot_w * (base - 0.6) / 0.4
            parts.append(f'<rect x="{plot_x}" y="{cy-bar_h-2}" width="{bx-plot_x:.1f}" '
                         f'height="{bar_h}" rx="2.5" fill="{BASE_COLOR}"/>')
            parts.append(f'<text x="{bx+6:.1f}" y="{cy-3}" font-family="Arial" font-size="10.5" '
                         f'fill="{SUBTEXT}">{fmt(base)}</text>')
        if after is not None:
            axp = plot_x + plot_w * (after - 0.6) / 0.4
            parts.append(f'<rect x="{plot_x}" y="{cy+2}" width="{axp-plot_x:.1f}" '
                         f'height="{bar_h}" rx="2.5" fill="{AFTER_COLOR}"/>')
            # up-triangle marker at the after tip
            tri = f"{axp-4:.1f},{cy+2} {axp+5:.1f},{cy+2} {axp+0.5:.1f},{cy-3}"
            parts.append(f'<polygon points="{tri}" fill="{GAIN_COLOR}"/>')
            parts.append(f'<text x="{axp+10:.1f}" y="{cy+13}" font-family="Arial" '
                         f'font-size="10.5" font-weight="700" fill="{GAIN_COLOR}">'
                         f'&#9650;+{fmt(after-base)}</text>')
    # axis baseline
    parts.append(f'<line x1="{plot_x}" y1="{bottom}" x2="{plot_x+plot_w}" y2="{bottom}" '
                 f'stroke="{SUBTEXT}" stroke-width="1"/>')
    return "".join(parts)


def main():
    os.makedirs(OUT, exist_ok=True)
    W, H = 980, 3 * 240 + 110
    body = []
    body.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    body.append(f'<text x="30" y="34" font-family="Arial" font-size="20" font-weight="700" '
                f'fill="{TEXT}">VisProbe — Accuracy: baseline vs LoRA fine-tuned</text>')
    body.append(f'<text x="30" y="56" font-family="Arial" font-size="12" fill="{SUBTEXT}">'
                f'Gray bar = frozen baseline; blue bar = after LoRA; green &#9650; = absolute '
                f'gain (percentage points).</text>')
    # legend
    body.append(f'<rect x="620" y="26" width="14" height="10" rx="2" fill="{BASE_COLOR}"/>'
                f'<text x="640" y="35" font-family="Arial" font-size="11" fill="{TEXT}">'
                f'baseline</text>')
    body.append(f'<rect x="720" y="26" width="14" height="10" rx="2" fill="{AFTER_COLOR}"/>'
                f'<text x="740" y="35" font-family="Arial" font-size="11" fill="{TEXT}">'
                f'LoRA after</text>')
    body.append(f'<polygon points="835,35 847,35 841,27" fill="{GAIN_COLOR}"/>'
                f'<text x="853" y="35" font-family="Arial" font-size="11" fill="{TEXT}">'
                f'gain (pp)</text>')
    for j, (ds, title) in enumerate(DATASETS):
        rows = []
        for m, ml in MODELS:
            b = load("baseline", m, ds)
            a = load("after", m, ds)
            if b is not None:
                rows.append((ml, b, a))
        body.append(make_group(30, 95 + j * 240, W - 60, 175, title, rows))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}">' + "".join(body) + "</svg>\n")
    p = os.path.join(OUT, "accuracy_baseline_after.svg")
    with open(p, "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", p, f"({len(svg)//1024} KB)")


if __name__ == "__main__":
    main()

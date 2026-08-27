"""Render the demo sweep as one self-contained HTML page.

The heatmap carries the argument -- which block and which head each
attribute responds to -- and every cell links to the image that produced
its score. No external assets, so the output directory can be zipped and
opened anywhere.
"""

from __future__ import annotations

from pathlib import Path

from ..hasm import HASM
from .sweep import DemoPaths

_CSS = """
body { font: 15px/1.6 system-ui, sans-serif; margin: 0; padding: 2rem;
       background: #14151e; color: #eaeaf2; }
h1 { font-size: 1.6rem; margin: 0 0 .3rem; }
h2 { font-size: 1.15rem; margin: 2.4rem 0 .2rem; text-transform: capitalize; }
.sub { color: #9a9db3; margin: 0 0 2rem; }
.peak { color: #9089d8; font-weight: 600; }
table { border-collapse: collapse; margin-top: .6rem; }
caption { text-align: left; color: #9a9db3; font-size: .85rem;
          padding-bottom: .4rem; }
th { font: 500 11px monospace; color: #787c93; padding: 2px 4px; }
td { padding: 0; }
a.cell { display: block; width: 22px; height: 22px; border: 1px solid #14151e; }
a.cell:hover { outline: 2px solid #eaeaf2; outline-offset: -2px; }
a.cell.rejected { background: repeating-linear-gradient(45deg,
    #2a2c3a, #2a2c3a 3px, #4a3030 3px, #4a3030 6px); }
.scroll { overflow-x: auto; }
"""


def heat_color(score: float) -> str:
    """Dark indigo through to hot amber, for a score in [0, 1]."""
    score = max(0.0, min(1.0, float(score)))
    red = int(38 + score * (217 - 38))
    green = int(36 + score * (167 - 36))
    blue = int(61 + score * (91 - 61))
    return f"rgb({red}, {green}, {blue})"


def render_report(
    hasm: HASM,
    paths: DemoPaths,
    title: str,
    rejected: set | None = None,
) -> str:
    parts = [
        "<!doctype html>",
        f"<title>{title}</title>",
        f"<style>{_CSS}</style>",
        f"<h1>{title}</h1>",
        '<p class="sub">Each cell is one attention head. Colour is that '
        "head's measured sensitivity to the attribute; click a cell to see "
        "the image produced by swapping the attribute at that head alone.</p>",
    ]

    for attr in hasm.attributes:
        peak_unit, peak_score = hasm.top_k(attr, 1)[0]
        parts.append(f"<h2>{attr.value}</h2>")
        parts.append(
            f'<p class="sub">peak: <span class="peak">block {peak_unit.block}, '
            f"head {peak_unit.head}</span> at {peak_score:.3f}</p>"
        )
        parts.append('<div class="scroll"><table>')
        parts.append(
            "<caption>rows: blocks &nbsp;&middot;&nbsp; columns: heads</caption>"
        )

        parts.append(
            "<tr><th></th>"
            + "".join(f"<th>{head}</th>" for head in hasm.head_ids)
            + "</tr>"
        )
        for block in hasm.block_ids:
            cells = []
            for head in hasm.head_ids:
                from ..heads import HeadUnit

                unit = HeadUnit(block=block, head=head)
                score = hasm.score(unit, attr)
                href = paths.head_image(attr, unit).relative_to(paths.root).as_posix()

                if rejected and unit in rejected:
                    # Not a low score -- a destroyed frame. Rendering it as a
                    # cold cell would read as "this head does not matter",
                    # the opposite of what it means.
                    cells.append(
                        f'<td><a class="cell rejected" href="{href}" '
                        f'title="block {block}, head {head} -- rejected: '
                        f'the generation collapsed"></a></td>'
                    )
                    continue

                cells.append(
                    f'<td><a class="cell" href="{href}" '
                    f'style="background:{heat_color(score)}" '
                    f'title="block {block}, head {head} -- {score:.3f}"></a></td>'
                )
            parts.append(f"<tr><th>B{block}</th>" + "".join(cells) + "</tr>")
        parts.append("</table></div>")

    return "\n".join(parts)


def write_report(
    hasm: HASM, paths: DemoPaths, title: str, rejected: set | None = None
) -> Path:
    path = paths.root / "index.html"
    path.write_text(
        render_report(hasm, paths, title, rejected=rejected), encoding="utf-8"
    )
    return path
